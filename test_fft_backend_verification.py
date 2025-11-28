"""
Verify that FFT backend selection is actually working correctly.

This test:
1. Adds logging to track which FFT implementation is called
2. Verifies C++ and Python Cooley-Tukey are actually used (not NumPy)
3. Checks that the environment variable is being respected
"""

import sys
sys.path.insert(0, './python')

import numpy as np
import os

# Monkey-patch the FFT functions to add logging
def add_fft_logging():
    """Add logging to FFT implementations to verify which is being called."""

    import needle.backend_ndarray.ndarray_backend_numpy as np_backend

    # Track calls
    call_log = {
        'numpy_fft': 0,
        'cooley_tukey_python': 0,
        'cpp_cooley_tukey': 0,
        'cpp_cooley_tukey_complex': 0,
        'bluestein_wrapper': 0
    }

    # Wrap NumPy FFT
    original_numpy_fft = np.fft.fft
    def logged_numpy_fft(*args, **kwargs):
        call_log['numpy_fft'] += 1
        return original_numpy_fft(*args, **kwargs)
    np.fft.fft = logged_numpy_fft

    # Wrap Python Cooley-Tukey
    try:
        from needle.backend_ndarray.fft_cooley_tukey import cooley_tukey_fft_iterative
        original_ct = cooley_tukey_fft_iterative

        def logged_ct(*args, **kwargs):
            call_log['cooley_tukey_python'] += 1
            return original_ct(*args, **kwargs)

        import needle.backend_ndarray.fft_cooley_tukey as ct_module
        ct_module.cooley_tukey_fft_iterative = logged_ct
    except ImportError:
        pass

    # Wrap C++ backend
    if np_backend._cpp_backend is not None:
        cpp_backend = np_backend._cpp_backend

        original_cpp_fft = cpp_backend.cooley_tukey_fft
        def logged_cpp_fft(*args, **kwargs):
            call_log['cpp_cooley_tukey'] += 1
            return original_cpp_fft(*args, **kwargs)
        cpp_backend.cooley_tukey_fft = logged_cpp_fft

        original_cpp_fft_complex = cpp_backend.cooley_tukey_fft_complex
        def logged_cpp_fft_complex(*args, **kwargs):
            call_log['cpp_cooley_tukey_complex'] += 1
            return original_cpp_fft_complex(*args, **kwargs)
        cpp_backend.cooley_tukey_fft_complex = logged_cpp_fft_complex

    # Wrap Bluestein
    try:
        from needle.backend_ndarray.fft_arbitrary_size import fft_arbitrary_1d
        original_bluestein = fft_arbitrary_1d

        def logged_bluestein(*args, **kwargs):
            call_log['bluestein_wrapper'] += 1
            return original_bluestein(*args, **kwargs)

        import needle.backend_ndarray.fft_arbitrary_size as bluestein_module
        bluestein_module.fft_arbitrary_1d = logged_bluestein
    except ImportError:
        pass

    return call_log


def test_fft_implementation(fft_impl):
    """Test that the correct FFT implementation is being used."""

    print(f"\n{'='*80}")
    print(f"Testing FFT Implementation: {fft_impl}")
    print(f"{'='*80}")

    # Set environment variable
    os.environ['NEEDLE_FFT_IMPL'] = fft_impl

    # Reload the backend to pick up new environment variable
    import importlib
    import needle.backend_ndarray.ndarray_backend_numpy as np_backend
    importlib.reload(np_backend)

    # Add logging
    call_log = add_fft_logging()

    # Import after setting environment
    import needle
    from needle import Tensor
    import needle.ops as ops

    device = needle.cpu_numpy()

    # Test 1: Power-of-2 FFT (N=32)
    print(f"\nTest 1: Power-of-2 FFT (N=32)")
    x = Tensor(np.random.randn(4, 32).astype(np.float32), device=device)
    result = ops.fft(x, dim=-1, norm="backward")

    print(f"  NumPy FFT calls: {call_log['numpy_fft']}")
    print(f"  Python Cooley-Tukey calls: {call_log['cooley_tukey_python']}")
    print(f"  C++ Cooley-Tukey calls: {call_log['cpp_cooley_tukey']}")
    print(f"  C++ Complex FFT calls: {call_log['cpp_cooley_tukey_complex']}")
    print(f"  Bluestein wrapper calls: {call_log['bluestein_wrapper']}")

    # Reset counters
    for key in call_log:
        call_log[key] = 0

    # Test 2: Non-power-of-2 FFT (N=28)
    print(f"\nTest 2: Non-power-of-2 FFT (N=28, MNIST size)")
    x = Tensor(np.random.randn(4, 28).astype(np.float32), device=device)
    result = ops.fft(x, dim=-1, norm="backward")

    print(f"  NumPy FFT calls: {call_log['numpy_fft']}")
    print(f"  Python Cooley-Tukey calls: {call_log['cooley_tukey_python']}")
    print(f"  C++ Cooley-Tukey calls: {call_log['cpp_cooley_tukey']}")
    print(f"  C++ Complex FFT calls: {call_log['cpp_cooley_tukey_complex']}")
    print(f"  Bluestein wrapper calls: {call_log['bluestein_wrapper']}")

    # Verify correct implementation was called
    print(f"\nVerification:")
    if fft_impl == "numpy":
        if call_log['numpy_fft'] > 0:
            print(f"  ✓ NumPy FFT was called")
        else:
            print(f"  ✗ ERROR: NumPy FFT was NOT called!")
    elif fft_impl == "cooley_tukey":
        if call_log['cooley_tukey_python'] > 0:
            print(f"  ✓ Python Cooley-Tukey was called")
        else:
            print(f"  ✗ ERROR: Python Cooley-Tukey was NOT called!")
    elif fft_impl == "cpp":
        if call_log['cpp_cooley_tukey_complex'] > 0:
            print(f"  ✓ C++ FFT was called (via Bluestein for N=28)")
        elif call_log['cpp_cooley_tukey'] > 0:
            print(f"  ✓ C++ FFT was called")
        else:
            print(f"  ✗ ERROR: C++ FFT was NOT called!")

    if call_log['numpy_fft'] > 0 and fft_impl != "numpy":
        print(f"  ⚠ WARNING: NumPy FFT was called when {fft_impl} was expected!")

    return call_log


def main():
    """Run backend verification tests."""

    print("="*80)
    print("FFT Backend Verification Test")
    print("="*80)
    print("\nThis test verifies that NEEDLE_FFT_IMPL environment variable")
    print("correctly selects the FFT implementation (not always using NumPy).")

    results = {}

    # Test each implementation
    for impl in ['numpy', 'cooley_tukey', 'cpp']:
        try:
            call_log = test_fft_implementation(impl)
            results[impl] = call_log
        except Exception as e:
            print(f"\n✗ Test failed for {impl}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    print(f"\n{'Implementation':<20} | {'NumPy':<8} | {'Python C-T':<12} | {'C++':<8} | {'C++ Complex':<12} | {'Bluestein':<10}")
    print("-"*90)

    for impl, log in results.items():
        print(f"{impl:<20} | {log.get('numpy_fft', 0):<8} | {log.get('cooley_tukey_python', 0):<12} | "
              f"{log.get('cpp_cooley_tukey', 0):<8} | {log.get('cpp_cooley_tukey_complex', 0):<12} | "
              f"{log.get('bluestein_wrapper', 0):<10}")

    # Analysis
    print(f"\n\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")

    # Check if NumPy is being used when it shouldn't be
    numpy_log = results.get('numpy', {})
    ct_log = results.get('cooley_tukey', {})
    cpp_log = results.get('cpp', {})

    print(f"\n1. NumPy Implementation:")
    if numpy_log.get('numpy_fft', 0) > 0:
        print(f"   ✓ NumPy FFT is used when NEEDLE_FFT_IMPL=numpy")
    else:
        print(f"   ✗ ERROR: NumPy FFT not used!")

    print(f"\n2. Python Cooley-Tukey Implementation:")
    if ct_log.get('cooley_tukey_python', 0) > 0:
        print(f"   ✓ Python Cooley-Tukey is used when NEEDLE_FFT_IMPL=cooley_tukey")
    else:
        print(f"   ✗ ERROR: Python Cooley-Tukey not used!")

    if ct_log.get('numpy_fft', 0) > 0:
        print(f"   ⚠ WARNING: NumPy FFT was also called (should not happen!)")

    print(f"\n3. C++ Implementation:")
    if cpp_log.get('cpp_cooley_tukey_complex', 0) > 0 or cpp_log.get('cpp_cooley_tukey', 0) > 0:
        print(f"   ✓ C++ FFT is used when NEEDLE_FFT_IMPL=cpp")
        if cpp_log.get('cpp_cooley_tukey_complex', 0) > 0:
            print(f"   ✓ C++ complex FFT used (for Bluestein with N=28)")
    else:
        print(f"   ✗ ERROR: C++ FFT not used!")

    if cpp_log.get('numpy_fft', 0) > 0:
        print(f"   ⚠ WARNING: NumPy FFT was also called (should not happen!)")

    print(f"\n{'='*80}\n")

    return results


if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

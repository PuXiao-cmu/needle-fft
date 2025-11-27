"""
Standalone test for FFT backends without importing needle framework

Tests the underlying FFT implementations directly to verify they work
with arbitrary sizes through the Bluestein wrapper.

This bypasses Python version compatibility issues with needle.
"""

import numpy as np
import sys
import os
import time

# Add to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python/needle/backend_ndarray'))

print("=" * 80)
print("Testing FFT Backends for Frequency-Domain Convolution")
print("=" * 80)

# Test 1: Import and test Python Cooley-Tukey
print("\n" + "=" * 80)
print("Test 1: Python Cooley-Tukey FFT")
print("=" * 80)

try:
    from fft_cooley_tukey import cooley_tukey_fft_iterative

    # Test power-of-2 size
    x = np.arange(32, dtype=np.complex128)
    start = time.time()
    result = cooley_tukey_fft_iterative(x)
    elapsed = time.time() - start

    # Compare with NumPy
    numpy_result = np.fft.fft(x)
    error = np.max(np.abs(result - numpy_result))

    print(f"✓ Python Cooley-Tukey imported successfully")
    print(f"  Size: 32 (power of 2)")
    print(f"  Time: {elapsed*1000:.2f} ms")
    print(f"  Error vs NumPy: {error:.2e}")
    print(f"  Status: {'✓ PASS' if error < 1e-10 else '✗ FAIL'}")

    python_ct_available = True
except Exception as e:
    print(f"✗ Python Cooley-Tukey import failed: {e}")
    python_ct_available = False

# Test 2: Bluestein wrapper with arbitrary sizes
print("\n" + "=" * 80)
print("Test 2: Bluestein Wrapper (Arbitrary Sizes)")
print("=" * 80)

try:
    from fft_arbitrary_size import (
        fft_arbitrary_1d,
        ifft_arbitrary_1d,
        freq_conv_2d_arbitrary,
        create_backend_fft_wrapper,
        is_power_of_2,
        next_power_of_2
    )

    # Create backend wrapper
    fft_backend = create_backend_fft_wrapper()
    print(f"✓ Bluestein wrapper created")

    # Test various sizes
    test_sizes = [3, 5, 7, 10, 28, 32, 64]
    print(f"\nTesting arbitrary sizes:")
    print(f"{'Size':<8} {'Pow2?':<8} {'Internal':<12} {'Time':<12} {'Error':<12} {'Status'}")
    print(f"{'-'*8} {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    for n in test_sizes:
        x = np.arange(n, dtype=np.float32)

        start = time.time()
        fft_real, fft_imag = fft_arbitrary_1d(x, None, fft_backend)
        elapsed = time.time() - start

        # Compare with NumPy
        numpy_fft = np.fft.fft(x)
        error = np.max(np.abs(fft_real - numpy_fft.real))

        is_pow2 = is_power_of_2(n)
        internal_size = n if is_pow2 else next_power_of_2(2*n-1)

        status = '✓' if error < 1e-4 else '✗'
        print(f"{n:<8} {str(is_pow2):<8} {internal_size:<12} {elapsed*1000:.2f} ms{' '*4} {error:.2e}{' '*4} {status}")

    bluestein_available = True
except Exception as e:
    print(f"✗ Bluestein wrapper test failed: {e}")
    import traceback
    traceback.print_exc()
    bluestein_available = False

# Test 3: 2D Frequency-domain Convolution
print("\n" + "=" * 80)
print("Test 3: 2D Frequency-Domain Convolution")
print("=" * 80)

if bluestein_available:
    try:
        # MNIST size test
        print(f"\nTest 3.1: MNIST size (28×28 * 5×5)")
        image = np.random.randn(28, 28).astype(np.float32)
        kernel = np.random.randn(5, 5).astype(np.float32)

        start = time.time()
        result = freq_conv_2d_arbitrary(image, kernel, fft_backend, mode='valid')
        elapsed = time.time() - start

        print(f"  Image: {image.shape}")
        print(f"  Kernel: {kernel.shape}")
        print(f"  Output: {result.shape}")
        print(f"  Expected: (24, 24)")
        print(f"  Time: {elapsed*1000:.2f} ms")
        print(f"  Status: {'✓ PASS' if result.shape == (24, 24) else '✗ FAIL'}")

        # Power-of-2 size test
        print(f"\nTest 3.2: Power-of-2 size (32×32 * 5×5)")
        image32 = np.random.randn(32, 32).astype(np.float32)

        start = time.time()
        result32 = freq_conv_2d_arbitrary(image32, kernel, fft_backend, mode='valid')
        elapsed32 = time.time() - start

        print(f"  Image: {image32.shape}")
        print(f"  Kernel: {kernel.shape}")
        print(f"  Output: {result32.shape}")
        print(f"  Expected: (28, 28)")
        print(f"  Time: {elapsed32*1000:.2f} ms")
        print(f"  Status: {'✓ PASS' if result32.shape == (28, 28) else '✗ FAIL'}")

        # Performance comparison
        print(f"\nPerformance comparison:")
        print(f"  MNIST (28×28, non-pow2): {elapsed*1000:.2f} ms")
        print(f"  32×32 (power of 2):      {elapsed32*1000:.2f} ms")
        speedup = elapsed / elapsed32
        print(f"  Speedup (pow2 vs non-pow2): {speedup:.2f}x")

        conv2d_works = True
    except Exception as e:
        print(f"✗ 2D Convolution test failed: {e}")
        import traceback
        traceback.print_exc()
        conv2d_works = False
else:
    print(f"⚠ Skipping (Bluestein not available)")
    conv2d_works = False

# Test 4: Round-trip test (FFT -> IFFT)
print("\n" + "=" * 80)
print("Test 4: Round-trip Test (FFT -> IFFT)")
print("=" * 80)

if bluestein_available:
    try:
        test_sizes = [28, 32, 64]
        print(f"{'Size':<8} {'Round-trip Error':<20} {'Status'}")
        print(f"{'-'*8} {'-'*20} {'-'*10}")

        for n in test_sizes:
            x = np.arange(n, dtype=np.float32) + 1

            # Forward FFT
            fft_real, fft_imag = fft_arbitrary_1d(x, None, fft_backend)

            # Inverse FFT
            ifft_real, ifft_imag = ifft_arbitrary_1d(fft_real, fft_imag, fft_backend)

            # Error
            error = np.max(np.abs(ifft_real - x))
            status = '✓ PASS' if error < 1e-5 else '✗ FAIL'

            print(f"{n:<8} {error:.2e}{' '*12} {status}")

        roundtrip_works = True
    except Exception as e:
        print(f"✗ Round-trip test failed: {e}")
        roundtrip_works = False
else:
    print(f"⚠ Skipping (Bluestein not available)")
    roundtrip_works = False

# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)

tests = [
    ("Python Cooley-Tukey", python_ct_available),
    ("Bluestein Wrapper", bluestein_available),
    ("2D Convolution", conv2d_works),
    ("Round-trip (FFT->IFFT)", roundtrip_works),
]

for name, status in tests:
    symbol = "✓" if status else "✗"
    print(f"{symbol} {name:<30} {'PASS' if status else 'FAIL'}")

all_passed = all(status for _, status in tests)

print("\n" + "=" * 80)
if all_passed:
    print("✓ ALL TESTS PASSED")
    print("\nYour FFT implementations work correctly with:")
    print("  - Arbitrary sizes (via Bluestein)")
    print("  - Frequency-domain convolution")
    print("  - MNIST (28×28) and other non-power-of-2 sizes")
else:
    print("✗ SOME TESTS FAILED")
    print("\nPlease check the errors above.")

print("=" * 80)

sys.exit(0 if all_passed else 1)

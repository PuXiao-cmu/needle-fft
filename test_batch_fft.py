#!/usr/bin/env python3
"""
Test Batch FFT Functions in C++ Backend
"""

import sys
sys.path.insert(0, './python')

import numpy as np

# Import backend directly
try:
    from needle.backend_ndarray import ndarray_backend_cpu as backend
except ImportError:
    import needle.backend_ndarray.ndarray_backend_cpu as backend

def test_batch_fft():
    """Test batch FFT vs sequential FFT"""

    print("="*70)
    print("Testing Batch FFT Functions")
    print("="*70)

    # Parameters
    batch_size = 4
    n = 256  # FFT size (must be power of 2)

    # Generate random test data
    np.random.seed(42)
    test_data = np.random.randn(batch_size, n).astype(np.float32)

    print(f"\nTest configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  FFT size: {n}")
    print(f"  Input shape: {test_data.shape}")

    # Test 1: Sequential FFT (current approach)
    print(f"\n{'='*70}")
    print("1. Sequential FFT (current approach - many Python-C++ calls)")
    print(f"{'='*70}")

    results_seq_real = []
    results_seq_imag = []

    for i in range(batch_size):
        a = backend.Array(n)
        out_real = backend.Array(n)
        out_imag = backend.Array(n)

        backend.from_numpy(np.ascontiguousarray(test_data[i]), a)
        backend.cooley_tukey_fft(a, out_real, out_imag, n)

        r_real = backend.to_numpy(out_real, [n], [1], 0)
        r_imag = backend.to_numpy(out_imag, [n], [1], 0)

        results_seq_real.append(r_real)
        results_seq_imag.append(r_imag)

    results_seq_real = np.array(results_seq_real)
    results_seq_imag = np.array(results_seq_imag)

    print(f"✓ Completed {batch_size} sequential FFTs")
    print(f"  Output shape: {results_seq_real.shape}")

    # Test 2: Batch FFT (new approach)
    print(f"\n{'='*70}")
    print("2. Batch FFT (new approach - single Python-C++ call)")
    print(f"{'='*70}")

    # Flatten input for batch processing
    test_data_flat = test_data.flatten()
    total_size = batch_size * n

    # Allocate batch arrays
    a_batch = backend.Array(total_size)
    out_real_batch = backend.Array(total_size)
    out_imag_batch = backend.Array(total_size)

    # Copy data
    backend.from_numpy(np.ascontiguousarray(test_data_flat), a_batch)

    # Single batch FFT call
    backend.cooley_tukey_fft_batch(a_batch, out_real_batch, out_imag_batch, batch_size, n)

    # Extract results
    results_batch_real = backend.to_numpy(out_real_batch, [total_size], [1], 0)
    results_batch_imag = backend.to_numpy(out_imag_batch, [total_size], [1], 0)

    # Reshape to (batch_size, n)
    results_batch_real = results_batch_real.reshape(batch_size, n)
    results_batch_imag = results_batch_imag.reshape(batch_size, n)

    print(f"✓ Completed 1 batch FFT ({batch_size} FFTs)")
    print(f"  Output shape: {results_batch_real.shape}")

    # Test 3: Verify correctness
    print(f"\n{'='*70}")
    print("3. Verification")
    print(f"{'='*70}")

    # Compare results
    diff_real = np.abs(results_seq_real - results_batch_real)
    diff_imag = np.abs(results_seq_imag - results_batch_imag)

    max_diff_real = np.max(diff_real)
    max_diff_imag = np.max(diff_imag)

    print(f"\nMax difference (real part): {max_diff_real:.2e}")
    print(f"Max difference (imag part): {max_diff_imag:.2e}")

    tolerance = 1e-4
    if max_diff_real < tolerance and max_diff_imag < tolerance:
        print(f"\n✓ Batch FFT matches sequential FFT (within tolerance {tolerance})")
        print("\n" + "="*70)
        print("SUCCESS: Batch FFT implementation is correct!")
        print("="*70)
        return True
    else:
        print(f"\n✗ Batch FFT differs from sequential FFT")
        print("\n" + "="*70)
        print("FAILURE: Results do not match")
        print("="*70)
        return False

def test_batch_fft_complex():
    """Test batch FFT for complex input"""

    print("\n\n")
    print("="*70)
    print("Testing Batch FFT Complex")
    print("="*70)

    batch_size = 3
    n = 128

    np.random.seed(43)
    test_real = np.random.randn(batch_size, n).astype(np.float32)
    test_imag = np.random.randn(batch_size, n).astype(np.float32)

    print(f"\nTest configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  FFT size: {n}")

    # Sequential approach
    results_seq_real = []
    results_seq_imag = []

    for i in range(batch_size):
        a_real = backend.Array(n)
        a_imag = backend.Array(n)
        out_real = backend.Array(n)
        out_imag = backend.Array(n)

        backend.from_numpy(np.ascontiguousarray(test_real[i]), a_real)
        backend.from_numpy(np.ascontiguousarray(test_imag[i]), a_imag)
        backend.cooley_tukey_fft_complex(a_real, a_imag, out_real, out_imag, n)

        r_real = backend.to_numpy(out_real, [n], [1], 0)
        r_imag = backend.to_numpy(out_imag, [n], [1], 0)

        results_seq_real.append(r_real)
        results_seq_imag.append(r_imag)

    results_seq_real = np.array(results_seq_real)
    results_seq_imag = np.array(results_seq_imag)

    # Batch approach
    total_size = batch_size * n
    a_real_batch = backend.Array(total_size)
    a_imag_batch = backend.Array(total_size)
    out_real_batch = backend.Array(total_size)
    out_imag_batch = backend.Array(total_size)

    backend.from_numpy(np.ascontiguousarray(test_real.flatten()), a_real_batch)
    backend.from_numpy(np.ascontiguousarray(test_imag.flatten()), a_imag_batch)

    backend.cooley_tukey_fft_complex_batch(a_real_batch, a_imag_batch,
                                           out_real_batch, out_imag_batch,
                                           batch_size, n)

    results_batch_real = backend.to_numpy(out_real_batch, [total_size], [1], 0).reshape(batch_size, n)
    results_batch_imag = backend.to_numpy(out_imag_batch, [total_size], [1], 0).reshape(batch_size, n)

    # Verify
    diff_real = np.max(np.abs(results_seq_real - results_batch_real))
    diff_imag = np.max(np.abs(results_seq_imag - results_batch_imag))

    print(f"\nMax difference (real): {diff_real:.2e}")
    print(f"Max difference (imag): {diff_imag:.2e}")

    tolerance = 1e-4
    if diff_real < tolerance and diff_imag < tolerance:
        print(f"\n✓ Batch FFT Complex is correct!")
        return True
    else:
        print(f"\n✗ Batch FFT Complex failed")
        return False


if __name__ == "__main__":
    try:
        success1 = test_batch_fft()
        success2 = test_batch_fft_complex()

        if success1 and success2:
            print("\n\n" + "="*70)
            print("ALL TESTS PASSED")
            print("="*70)
            print("\nThe batch FFT functions are working correctly!")
            print("These functions can significantly reduce Python-C++ overhead")
            print("by processing multiple FFTs in a single call.")
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

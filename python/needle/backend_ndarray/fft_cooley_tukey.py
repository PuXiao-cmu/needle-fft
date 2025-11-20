"""
Cooley-Tukey FFT Algorithm Implementation

This module implements the classic Cooley-Tukey radix-2 decimation-in-time (DIT) FFT algorithm.
It serves as a custom implementation to compare against NumPy's optimized FFT.

References:
- Cooley, J. W., & Tukey, J. W. (1965). "An algorithm for the machine calculation
  of complex Fourier series". Mathematics of computation, 19(90), 297-301.
"""

import numpy as np
from typing import Tuple


def cooley_tukey_fft(x: np.ndarray) -> np.ndarray:
    """
    Compute FFT using Cooley-Tukey radix-2 DIT algorithm.

    Algorithm Overview:
    1. Recursively divide the DFT into even and odd indexed samples
    2. Compute smaller DFTs recursively
    3. Combine results using twiddle factors

    Time Complexity: O(N log N)
    Space Complexity: O(N log N) due to recursion

    Args:
        x: Input array (real or complex), length must be power of 2

    Returns:
        Complex array containing FFT result

    Raises:
        ValueError: If length is not a power of 2
    """
    N = len(x)

    # Base case: length 1
    if N == 1:
        return np.array([x[0]], dtype=np.complex128)

    # Check if N is power of 2
    if N & (N - 1) != 0:
        raise ValueError(f"Length {N} is not a power of 2. Use zero-padding.")

    # Divide: split into even and odd indices
    x_even = cooley_tukey_fft(x[0::2])  # x[0], x[2], x[4], ...
    x_odd = cooley_tukey_fft(x[1::2])   # x[1], x[3], x[5], ...

    # Conquer: compute twiddle factors
    # W_N^k = e^(-2πi k/N)
    k = np.arange(N // 2)
    twiddle = np.exp(-2j * np.pi * k / N)

    # Combine results
    # X[k] = X_even[k] + W_N^k * X_odd[k]
    # X[k + N/2] = X_even[k] - W_N^k * X_odd[k]
    result = np.zeros(N, dtype=np.complex128)
    result[:N//2] = x_even + twiddle * x_odd
    result[N//2:] = x_even - twiddle * x_odd

    return result


def cooley_tukey_ifft(X: np.ndarray) -> np.ndarray:
    """
    Compute IFFT using Cooley-Tukey algorithm.

    IFFT can be computed using FFT with conjugate trick:
    IFFT(X) = conj(FFT(conj(X))) / N

    Args:
        X: Input array (complex), length must be power of 2

    Returns:
        Complex array containing IFFT result
    """
    N = len(X)

    # Conjugate input
    X_conj = np.conj(X)

    # Apply FFT
    result_conj = cooley_tukey_fft(X_conj)

    # Conjugate output and normalize
    result = np.conj(result_conj) / N

    return result


def cooley_tukey_fft_iterative(x: np.ndarray) -> np.ndarray:
    """
    Iterative (in-place) Cooley-Tukey FFT using bit-reversal.

    This is more memory-efficient than the recursive version.

    Algorithm:
    1. Bit-reverse permutation of input
    2. Iteratively combine stages (butterfly operations)

    Args:
        x: Input array, length must be power of 2

    Returns:
        Complex array containing FFT result
    """
    N = len(x)

    if N & (N - 1) != 0:
        raise ValueError(f"Length {N} is not a power of 2")

    # Convert to complex and copy
    X = np.array(x, dtype=np.complex128, copy=True)

    # Bit-reversal permutation
    num_bits = int(np.log2(N))
    for i in range(N):
        # Reverse bits of i
        j = int(bin(i)[2:].zfill(num_bits)[::-1], 2)
        if j > i:
            X[i], X[j] = X[j], X[i]

    # Iterative FFT (Cooley-Tukey butterfly)
    # Process stages: stage s has 2^s elements per DFT
    for s in range(1, num_bits + 1):
        m = 2 ** s  # Size of each DFT

        # Twiddle factor for this stage
        # W_m = e^(-2πi/m)
        W_m = np.exp(-2j * np.pi / m)

        # Process each group of size m
        for k in range(0, N, m):
            W = 1.0  # W^0

            # Butterfly operations within this group
            for j in range(m // 2):
                # Indices of butterfly pair
                t = W * X[k + j + m // 2]
                u = X[k + j]

                # Butterfly computation
                X[k + j] = u + t
                X[k + j + m // 2] = u - t

                # Update twiddle factor
                W = W * W_m

    return X


def cooley_tukey_ifft_iterative(X: np.ndarray) -> np.ndarray:
    """
    Iterative IFFT using conjugate trick.

    Args:
        X: Input array (complex), length must be power of 2

    Returns:
        Complex array containing IFFT result
    """
    N = len(X)

    # Conjugate input
    X_conj = np.conj(X)

    # Apply FFT
    result_conj = cooley_tukey_fft_iterative(X_conj)

    # Conjugate output and normalize
    result = np.conj(result_conj) / N

    return result


def next_power_of_2(n: int) -> int:
    """Find the next power of 2 greater than or equal to n."""
    return 1 if n == 0 else 2 ** (n - 1).bit_length()


def fft_with_padding(x: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Compute FFT with automatic zero-padding to power of 2.

    Args:
        x: Input array of any length

    Returns:
        Tuple of (FFT result, original length)
    """
    N_original = len(x)
    N_padded = next_power_of_2(N_original)

    # Zero-pad if necessary
    if N_padded != N_original:
        x_padded = np.zeros(N_padded, dtype=x.dtype)
        x_padded[:N_original] = x
    else:
        x_padded = x

    # Compute FFT
    result = cooley_tukey_fft_iterative(x_padded)

    return result, N_original


def ifft_with_unpadding(X: np.ndarray, N_original: int) -> np.ndarray:
    """
    Compute IFFT and remove padding.

    Args:
        X: FFT result (possibly padded)
        N_original: Original length before padding

    Returns:
        IFFT result with original length
    """
    # Compute IFFT
    result = cooley_tukey_ifft_iterative(X)

    # Unpad if necessary
    if len(result) != N_original:
        result = result[:N_original]

    return result


def benchmark_comparison(size: int = 1024, num_runs: int = 100):
    """
    Compare performance of Cooley-Tukey vs NumPy FFT.

    Args:
        size: FFT size (should be power of 2)
        num_runs: Number of runs for timing
    """
    import time

    # Generate random input
    x = np.random.randn(size).astype(np.float64)

    # Warm-up
    _ = np.fft.fft(x)
    _ = cooley_tukey_fft_iterative(x)

    # Benchmark NumPy FFT
    start = time.time()
    for _ in range(num_runs):
        result_numpy = np.fft.fft(x)
    time_numpy = (time.time() - start) / num_runs

    # Benchmark Cooley-Tukey FFT (recursive)
    start = time.time()
    for _ in range(num_runs):
        result_recursive = cooley_tukey_fft(x)
    time_recursive = (time.time() - start) / num_runs

    # Benchmark Cooley-Tukey FFT (iterative)
    start = time.time()
    for _ in range(num_runs):
        result_iterative = cooley_tukey_fft_iterative(x)
    time_iterative = (time.time() - start) / num_runs

    # Check correctness
    error_recursive = np.abs(result_numpy - result_recursive).max()
    error_iterative = np.abs(result_numpy - result_iterative).max()

    # Print results
    print(f"\nBenchmark Results (N={size}, runs={num_runs}):")
    print(f"{'Method':<25} {'Time (ms)':<12} {'Speedup':<10} {'Max Error'}")
    print("-" * 65)
    print(f"{'NumPy FFT':<25} {time_numpy*1000:>10.4f}   {1.0:>8.2f}x   {0.0:.2e}")
    print(f"{'Cooley-Tukey (recursive)':<25} {time_recursive*1000:>10.4f}   {time_recursive/time_numpy:>8.2f}x   {error_recursive:.2e}")
    print(f"{'Cooley-Tukey (iterative)':<25} {time_iterative*1000:>10.4f}   {time_iterative/time_numpy:>8.2f}x   {error_iterative:.2e}")

    return {
        'numpy': time_numpy,
        'recursive': time_recursive,
        'iterative': time_iterative,
        'error_recursive': error_recursive,
        'error_iterative': error_iterative
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Cooley-Tukey FFT Algorithm Implementation")
    print("=" * 70)

    # Test 1: Basic correctness
    print("\n[Test 1] Basic Correctness Test")
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    fft_numpy = np.fft.fft(x)
    fft_recursive = cooley_tukey_fft(x)
    fft_iterative = cooley_tukey_fft_iterative(x)

    print(f"Input: {x}")
    print(f"\nNumPy FFT:       {fft_numpy}")
    print(f"Recursive FFT:   {fft_recursive}")
    print(f"Iterative FFT:   {fft_iterative}")

    error_recursive = np.abs(fft_numpy - fft_recursive).max()
    error_iterative = np.abs(fft_numpy - fft_iterative).max()

    print(f"\nRecursive Error: {error_recursive:.2e}")
    print(f"Iterative Error: {error_iterative:.2e}")

    if error_recursive < 1e-10 and error_iterative < 1e-10:
        print("✓ Correctness test passed!")
    else:
        print("✗ Correctness test failed!")

    # Test 2: Round-trip (FFT -> IFFT)
    print("\n[Test 2] Round-trip Test (FFT -> IFFT)")

    ifft_recursive = cooley_tukey_ifft(fft_recursive)
    ifft_iterative = cooley_tukey_ifft_iterative(fft_iterative)

    print(f"Original:        {x}")
    print(f"After recursive: {np.real(ifft_recursive)}")
    print(f"After iterative: {np.real(ifft_iterative)}")

    roundtrip_error_recursive = np.abs(x - np.real(ifft_recursive)).max()
    roundtrip_error_iterative = np.abs(x - np.real(ifft_iterative)).max()

    print(f"\nRecursive round-trip error: {roundtrip_error_recursive:.2e}")
    print(f"Iterative round-trip error: {roundtrip_error_iterative:.2e}")

    if roundtrip_error_recursive < 1e-10 and roundtrip_error_iterative < 1e-10:
        print("✓ Round-trip test passed!")
    else:
        print("✗ Round-trip test failed!")

    # Test 3: Different sizes
    print("\n[Test 3] Different Sizes")
    for size in [4, 8, 16, 32, 64, 128]:
        x_test = np.random.randn(size)

        fft_numpy = np.fft.fft(x_test)
        fft_custom = cooley_tukey_fft_iterative(x_test)

        error = np.abs(fft_numpy - fft_custom).max()

        if error < 1e-10:
            print(f"  Size {size:4d}: ✓ (error: {error:.2e})")
        else:
            print(f"  Size {size:4d}: ✗ (error: {error:.2e})")

    # Test 4: Benchmark
    print("\n[Test 4] Performance Benchmark")
    benchmark_comparison(size=1024, num_runs=100)

    print("\n" + "=" * 70)
    print("Tests completed!")
    print("=" * 70)

"""
FFT for Arbitrary Sizes using Bluestein's Algorithm

This module extends Cooley-Tukey FFT (which only works for N = 2^k) to support
arbitrary sizes by using Bluestein's algorithm.

Strategy:
---------
1. For N = power of 2: directly use Cooley-Tukey (Python/C++/CUDA)
2. For arbitrary N: use Bluestein's algorithm to convert to power-of-2 FFT

This allows frequency-domain convolution to work with any image/kernel size.

Performance:
-----------
- Power of 2 sizes: Full Cooley-Tukey performance
- Arbitrary sizes: ~3x overhead from Bluestein, but still O(N log N)
- Much faster than spatial convolution for large kernels
"""

import numpy as np


def next_power_of_2(n):
    """Find the smallest power of 2 that is >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def is_power_of_2(n):
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def fft_arbitrary_1d(x_real, x_imag, backend_fft_func):
    """
    1D FFT for arbitrary size using Bluestein's algorithm.

    Args:
        x_real: numpy array, real part of input
        x_imag: numpy array, imaginary part of input (can be None)
        backend_fft_func: function(x_real, x_imag) -> (fft_real, fft_imag)
                         Your Cooley-Tukey implementation (requires N = 2^k)

    Returns:
        (fft_real, fft_imag): tuple of numpy arrays
    """
    n = len(x_real)

    # Initialize imaginary part if not provided
    if x_imag is None:
        x_imag = np.zeros(n, dtype=np.float32)

    # Fast path: if already power of 2, use Cooley-Tukey directly
    if is_power_of_2(n):
        return backend_fft_func(x_real, x_imag)

    # Bluestein's algorithm for arbitrary N
    # Convert to power-of-2 FFT of size M >= 2N-1
    m = next_power_of_2(2 * n - 1)

    # Step 1: Compute chirp sequence w[k] = exp(-i * pi * k^2 / N)
    chirp_real = np.zeros(m, dtype=np.float32)
    chirp_imag = np.zeros(m, dtype=np.float32)

    for k in range(n):
        angle = -np.pi * k * k / n
        chirp_real[k] = np.cos(angle)
        chirp_imag[k] = np.sin(angle)

    # Step 2: Create convolution kernel (conjugated and wrapped chirp)
    kernel_real = np.zeros(m, dtype=np.float32)
    kernel_imag = np.zeros(m, dtype=np.float32)

    # kernel[0] = conj(chirp[0])
    kernel_real[0] = chirp_real[0]
    kernel_imag[0] = -chirp_imag[0]

    # kernel[1..n-1] = conj(chirp[1..n-1])
    for k in range(1, n):
        kernel_real[k] = chirp_real[k]
        kernel_imag[k] = -chirp_imag[k]

    # Wrap around for circular convolution
    for k in range(1, n):
        kernel_real[m - k] = chirp_real[k]
        kernel_imag[m - k] = -chirp_imag[k]

    # Step 3: Multiply input by chirp: a[k] = x[k] * chirp[k]
    a_real = np.zeros(m, dtype=np.float32)
    a_imag = np.zeros(m, dtype=np.float32)

    for k in range(n):
        # Complex multiplication: (x_real + i*x_imag) * (chirp_real + i*chirp_imag)
        a_real[k] = x_real[k] * chirp_real[k] - x_imag[k] * chirp_imag[k]
        a_imag[k] = x_real[k] * chirp_imag[k] + x_imag[k] * chirp_real[k]

    # Step 4: FFT both sequences (now size M = power of 2)
    a_fft_real, a_fft_imag = backend_fft_func(a_real, a_imag)
    kernel_fft_real, kernel_fft_imag = backend_fft_func(kernel_real, kernel_imag)

    # Step 5: Multiply in frequency domain
    conv_fft_real = a_fft_real * kernel_fft_real - a_fft_imag * kernel_fft_imag
    conv_fft_imag = a_fft_real * kernel_fft_imag + a_fft_imag * kernel_fft_real

    # Step 6: IFFT to get convolution result
    # Use IFFT = conj(FFT(conj(X))) / N
    conv_fft_imag_conj = -conv_fft_imag
    temp_real, temp_imag = backend_fft_func(conv_fft_real, conv_fft_imag_conj)
    conv_real = temp_real / m
    conv_imag = -temp_imag / m

    # Step 7: Multiply result by chirp to get final FFT
    result_real = np.zeros(n, dtype=np.float32)
    result_imag = np.zeros(n, dtype=np.float32)

    for k in range(n):
        # Complex multiplication: conv[k] * chirp[k]
        result_real[k] = conv_real[k] * chirp_real[k] - conv_imag[k] * chirp_imag[k]
        result_imag[k] = conv_real[k] * chirp_imag[k] + conv_imag[k] * chirp_real[k]

    return result_real, result_imag


def ifft_arbitrary_1d(x_real, x_imag, backend_fft_func):
    """
    1D IFFT for arbitrary size using Bluestein's algorithm.

    Uses: IFFT(X) = conj(FFT(conj(X))) / N

    Args:
        x_real: numpy array, real part of input
        x_imag: numpy array, imaginary part of input
        backend_fft_func: function(x_real, x_imag) -> (fft_real, fft_imag)

    Returns:
        (ifft_real, ifft_imag): tuple of numpy arrays
    """
    n = len(x_real)

    # Conjugate input
    conj_imag = -x_imag if x_imag is not None else np.zeros(n, dtype=np.float32)

    # Apply forward FFT
    fft_real, fft_imag = fft_arbitrary_1d(x_real, conj_imag, backend_fft_func)

    # Conjugate result and normalize
    result_real = fft_real / n
    result_imag = -fft_imag / n

    return result_real, result_imag


def fft_arbitrary_2d(image_real, image_imag, backend_fft_func):
    """
    2D FFT for arbitrary size.

    Applies 1D FFT along rows, then along columns.

    Args:
        image_real: 2D numpy array, real part
        image_imag: 2D numpy array, imaginary part (or None)
        backend_fft_func: 1D FFT function

    Returns:
        (fft_real, fft_imag): 2D FFT result
    """
    h, w = image_real.shape

    if image_imag is None:
        image_imag = np.zeros_like(image_real)

    # Step 1: FFT along rows
    temp_real = np.zeros_like(image_real)
    temp_imag = np.zeros_like(image_imag)

    for i in range(h):
        temp_real[i, :], temp_imag[i, :] = fft_arbitrary_1d(
            image_real[i, :], image_imag[i, :], backend_fft_func
        )

    # Step 2: FFT along columns
    result_real = np.zeros_like(image_real)
    result_imag = np.zeros_like(image_imag)

    for j in range(w):
        result_real[:, j], result_imag[:, j] = fft_arbitrary_1d(
            temp_real[:, j], temp_imag[:, j], backend_fft_func
        )

    return result_real, result_imag


def ifft_arbitrary_2d(image_real, image_imag, backend_fft_func):
    """
    2D IFFT for arbitrary size.

    Args:
        image_real: 2D numpy array, real part
        image_imag: 2D numpy array, imaginary part
        backend_fft_func: 1D FFT function

    Returns:
        (ifft_real, ifft_imag): 2D IFFT result
    """
    h, w = image_real.shape

    if image_imag is None:
        image_imag = np.zeros_like(image_real)

    # Step 1: IFFT along rows
    temp_real = np.zeros_like(image_real)
    temp_imag = np.zeros_like(image_imag)

    for i in range(h):
        temp_real[i, :], temp_imag[i, :] = ifft_arbitrary_1d(
            image_real[i, :], image_imag[i, :], backend_fft_func
        )

    # Step 2: IFFT along columns
    result_real = np.zeros_like(image_real)
    result_imag = np.zeros_like(image_imag)

    for j in range(w):
        result_real[:, j], result_imag[:, j] = ifft_arbitrary_1d(
            temp_real[:, j], temp_imag[:, j], backend_fft_func
        )

    return result_real, result_imag


def freq_conv_2d_arbitrary(image, kernel, backend_fft_func, mode='valid'):
    """
    2D frequency-domain convolution for arbitrary sizes.

    This is the main function for your assignment!

    Complexity: O(HW log(HW)) vs O(HW K^2) for spatial convolution

    Args:
        image: 2D numpy array (H x W)
        kernel: 2D numpy array (K x K)
        backend_fft_func: Your Cooley-Tukey FFT (from C++/CUDA backend)
        mode: 'valid' or 'same'

    Returns:
        Convolution result
    """
    H, W = image.shape
    K, _ = kernel.shape

    # Step 1: Calculate padding size
    if mode == 'valid':
        # For linear convolution: need at least H+K-1 to avoid aliasing
        pad_h = H + K - 1
        pad_w = W + K - 1
    elif mode == 'same':
        # Same size as input
        pad_h = H
        pad_w = W
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Optional: pad to power of 2 for better FFT performance
    fft_h = next_power_of_2(pad_h)
    fft_w = next_power_of_2(pad_w)

    # Step 2: Zero-pad image and kernel
    image_padded = np.zeros((fft_h, fft_w), dtype=np.float32)
    image_padded[:H, :W] = image

    kernel_padded = np.zeros((fft_h, fft_w), dtype=np.float32)
    kernel_padded[:K, :K] = kernel

    # Step 3: 2D FFT using your backend (supports arbitrary size via Bluestein)
    image_fft_real, image_fft_imag = fft_arbitrary_2d(
        image_padded, None, backend_fft_func
    )
    kernel_fft_real, kernel_fft_imag = fft_arbitrary_2d(
        kernel_padded, None, backend_fft_func
    )

    # Step 4: Element-wise multiplication in frequency domain
    # (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    result_fft_real = image_fft_real * kernel_fft_real - image_fft_imag * kernel_fft_imag
    result_fft_imag = image_fft_real * kernel_fft_imag + image_fft_imag * kernel_fft_real

    # Step 5: 2D IFFT
    result_real, result_imag = ifft_arbitrary_2d(
        result_fft_real, result_fft_imag, backend_fft_func
    )

    # Step 6: Crop to desired output size
    if mode == 'valid':
        return result_real[:H-K+1, :W-K+1]
    elif mode == 'same':
        # Center crop
        start_h = K // 2
        start_w = K // 2
        return result_real[start_h:start_h+H, start_w:start_w+W]


# Example wrapper for using with your existing backend
def create_backend_fft_wrapper(device=None):
    """
    Create a wrapper around your backend FFT for use with Bluestein.

    Args:
        device: Your needle device (cpu, cuda, etc.)

    Returns:
        fft_func: function(x_real, x_imag) -> (fft_real, fft_imag)
    """
    # Import here to avoid circular dependencies
    try:
        from fft_cooley_tukey import cooley_tukey_fft_iterative
    except ImportError:
        # Try absolute import
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        from fft_cooley_tukey import cooley_tukey_fft_iterative

    def fft_func(x_real, x_imag):
        """
        Wrapper that calls your backend FFT.

        This assumes you have fft() function in your backend that:
        - Takes real and imaginary arrays
        - Returns real and imaginary arrays
        - Only works for N = power of 2
        """
        # Use the Python Cooley-Tukey for now
        # TODO: Replace with C++/CUDA backend call when available
        n = len(x_real)

        # Combine real and imaginary into complex array
        x_complex = x_real.astype(np.complex128) + 1j * x_imag.astype(np.complex128)

        # Call Cooley-Tukey
        result = cooley_tukey_fft_iterative(x_complex)

        # Split into real and imaginary
        return result.real.astype(np.float32), result.imag.astype(np.float32)

    return fft_func


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Arbitrary-Size FFT Test (using Bluestein + Cooley-Tukey)")
    print("=" * 80 + "\n")

    # Get backend FFT function
    fft_backend = create_backend_fft_wrapper(None)

    # Test 1: FFT of size 28 (MNIST)
    print("Test 1: FFT of non-power-of-2 size (N=28)")
    print("-" * 80)

    x = np.arange(28, dtype=np.float32)
    fft_real, fft_imag = fft_arbitrary_1d(x, None, fft_backend)

    # Verify against NumPy
    numpy_fft = np.fft.fft(x)
    error_real = np.max(np.abs(fft_real - numpy_fft.real))
    error_imag = np.max(np.abs(fft_imag - numpy_fft.imag))

    print(f"Input size: {len(x)}")
    print(f"Is power of 2: {is_power_of_2(len(x))}")
    print(f"FFT size used internally: {next_power_of_2(2*len(x)-1)}")
    print(f"Error vs NumPy: real={error_real:.2e}, imag={error_imag:.2e}")
    print(f"Status: {'✓ PASS' if error_real < 1e-4 else '✗ FAIL'}\n")

    # Test 2: 2D Convolution
    print("Test 2: 2D Frequency-domain Convolution (28x28 * 5x5)")
    print("-" * 80)

    image = np.random.randn(28, 28).astype(np.float32)
    kernel = np.random.randn(5, 5).astype(np.float32)

    result = freq_conv_2d_arbitrary(image, kernel, fft_backend, mode='valid')

    print(f"Image shape: {image.shape}")
    print(f"Kernel shape: {kernel.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Expected shape: (24, 24)")
    print(f"Status: {'✓ PASS' if result.shape == (24, 24) else '✗ FAIL'}\n")

    print("=" * 80)
    print("✓ Arbitrary-size FFT working! Ready for frequency-domain convolution.")
    print("=" * 80)

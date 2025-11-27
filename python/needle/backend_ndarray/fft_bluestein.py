"""
Bluestein's FFT Algorithm - Supports arbitrary input sizes

This algorithm converts an arbitrary-size N FFT into a power-of-2 size FFT
by representing the DFT as a convolution. This allows us to use the existing
Cooley-Tukey implementation for any input size.

Algorithm:
---------
For DFT: X[k] = Σ(n=0 to N-1) x[n] * exp(-2πi*nk/N)

Using the identity: nk = (n² + k² - (n-k)²) / 2

We can rewrite as: X[k] = exp(-πik²/N) * Σ(n=0 to N-1) [x[n]*exp(-πin²/N)] * exp(πi(k-n)²/N)

This is a convolution that can be computed with FFT of size M ≥ 2N-1 (next power of 2).
"""

import numpy as np
from .fft_cooley_tukey import cooley_tukey_fft, cooley_tukey_ifft


def next_power_of_2(n):
    """Find the next power of 2 greater than or equal to n."""
    return 1 << (n - 1).bit_length()


def is_power_of_2(n):
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def bluestein_fft(x_real, x_imag=None):
    """
    Compute FFT of arbitrary size using Bluestein's algorithm.

    If input size is already a power of 2, directly use Cooley-Tukey.
    Otherwise, use Bluestein's algorithm to convert to power-of-2 FFT.

    Args:
        x_real: Real part of input (1D numpy array)
        x_imag: Imaginary part of input (1D numpy array, optional)

    Returns:
        (fft_real, fft_imag): FFT result as (real, imaginary) tuple
    """
    n = len(x_real)

    # Initialize imaginary part if not provided
    if x_imag is None:
        x_imag = np.zeros(n, dtype=np.float32)

    # If already power of 2, use Cooley-Tukey directly
    if is_power_of_2(n):
        return cooley_tukey_fft(x_real, x_imag)

    # Bluestein's algorithm for arbitrary N
    # Step 1: Find M ≥ 2N-1, next power of 2
    m = next_power_of_2(2 * n - 1)

    # Step 2: Compute chirp sequence: w[k] = exp(-πik²/N)
    # w[k] = cos(-πk²/N) + i*sin(-πk²/N)
    chirp_real = np.zeros(m, dtype=np.float32)
    chirp_imag = np.zeros(m, dtype=np.float32)

    for k in range(n):
        angle = -np.pi * k * k / n
        chirp_real[k] = np.cos(angle)
        chirp_imag[k] = np.sin(angle)

    # Step 3: Create the convolution kernel (conjugate of chirp, reversed)
    # Also need to wrap around for circular convolution
    kernel_real = np.zeros(m, dtype=np.float32)
    kernel_imag = np.zeros(m, dtype=np.float32)

    # kernel[0] = chirp[0]^*
    kernel_real[0] = chirp_real[0]
    kernel_imag[0] = -chirp_imag[0]

    # kernel[1..n-1] = chirp[1..n-1]^*
    for k in range(1, n):
        kernel_real[k] = chirp_real[k]
        kernel_imag[k] = -chirp_imag[k]

    # kernel[m-(n-1)..m-1] = chirp[n-1..1]^* (wrap around for circular conv)
    for k in range(1, n):
        kernel_real[m - k] = chirp_real[k]
        kernel_imag[m - k] = -chirp_imag[k]

    # Step 4: Multiply input by chirp: a[n] = x[n] * chirp[n]
    a_real = np.zeros(m, dtype=np.float32)
    a_imag = np.zeros(m, dtype=np.float32)

    for k in range(n):
        # Complex multiplication: (x_real + i*x_imag) * (chirp_real + i*chirp_imag)
        a_real[k] = x_real[k] * chirp_real[k] - x_imag[k] * chirp_imag[k]
        a_imag[k] = x_real[k] * chirp_imag[k] + x_imag[k] * chirp_real[k]

    # Step 5: FFT of both sequences (size M, power of 2)
    a_fft_real, a_fft_imag = cooley_tukey_fft(a_real, a_imag)
    kernel_fft_real, kernel_fft_imag = cooley_tukey_fft(kernel_real, kernel_imag)

    # Step 6: Multiply in frequency domain
    # (a_fft_real + i*a_fft_imag) * (kernel_fft_real + i*kernel_fft_imag)
    conv_fft_real = a_fft_real * kernel_fft_real - a_fft_imag * kernel_fft_imag
    conv_fft_imag = a_fft_real * kernel_fft_imag + a_fft_imag * kernel_fft_real

    # Step 7: IFFT to get convolution result
    conv_real, conv_imag = cooley_tukey_ifft(conv_fft_real, conv_fft_imag)

    # Step 8: Multiply result by chirp to get final FFT
    result_real = np.zeros(n, dtype=np.float32)
    result_imag = np.zeros(n, dtype=np.float32)

    for k in range(n):
        # Complex multiplication: conv[k] * chirp[k]
        result_real[k] = conv_real[k] * chirp_real[k] - conv_imag[k] * chirp_imag[k]
        result_imag[k] = conv_real[k] * chirp_imag[k] + conv_imag[k] * chirp_real[k]

    return result_real, result_imag


def bluestein_ifft(x_real, x_imag=None):
    """
    Compute IFFT of arbitrary size using Bluestein's algorithm.

    Uses the property: IFFT(X) = conj(FFT(conj(X))) / N

    Args:
        x_real: Real part of input (1D numpy array)
        x_imag: Imaginary part of input (1D numpy array, optional)

    Returns:
        (ifft_real, ifft_imag): IFFT result as (real, imaginary) tuple
    """
    n = len(x_real)

    # Initialize imaginary part if not provided
    if x_imag is None:
        x_imag = np.zeros(n, dtype=np.float32)

    # If already power of 2, use Cooley-Tukey IFFT directly
    if is_power_of_2(n):
        return cooley_tukey_ifft(x_real, x_imag)

    # Step 1: Conjugate input (negate imaginary part)
    conj_real = x_real.copy()
    conj_imag = -x_imag

    # Step 2: Apply forward FFT
    fft_real, fft_imag = bluestein_fft(conj_real, conj_imag)

    # Step 3: Conjugate result and normalize
    result_real = fft_real / n
    result_imag = -fft_imag / n

    return result_real, result_imag


def bluestein_fft_2d(x_real, x_imag=None):
    """
    Compute 2D FFT of arbitrary size using Bluestein's algorithm.

    Args:
        x_real: Real part of input (2D numpy array)
        x_imag: Imaginary part of input (2D numpy array, optional)

    Returns:
        (fft_real, fft_imag): 2D FFT result as (real, imaginary) tuple
    """
    h, w = x_real.shape

    if x_imag is None:
        x_imag = np.zeros_like(x_real)

    # Step 1: FFT along rows
    temp_real = np.zeros_like(x_real)
    temp_imag = np.zeros_like(x_imag)

    for i in range(h):
        temp_real[i, :], temp_imag[i, :] = bluestein_fft(x_real[i, :], x_imag[i, :])

    # Step 2: FFT along columns
    result_real = np.zeros_like(x_real)
    result_imag = np.zeros_like(x_imag)

    for j in range(w):
        result_real[:, j], result_imag[:, j] = bluestein_fft(temp_real[:, j], temp_imag[:, j])

    return result_real, result_imag


def bluestein_ifft_2d(x_real, x_imag=None):
    """
    Compute 2D IFFT of arbitrary size using Bluestein's algorithm.

    Args:
        x_real: Real part of input (2D numpy array)
        x_imag: Imaginary part of input (2D numpy array, optional)

    Returns:
        (ifft_real, ifft_imag): 2D IFFT result as (real, imaginary) tuple
    """
    h, w = x_real.shape

    if x_imag is None:
        x_imag = np.zeros_like(x_real)

    # Step 1: IFFT along rows
    temp_real = np.zeros_like(x_real)
    temp_imag = np.zeros_like(x_imag)

    for i in range(h):
        temp_real[i, :], temp_imag[i, :] = bluestein_ifft(x_real[i, :], x_imag[i, :])

    # Step 2: IFFT along columns
    result_real = np.zeros_like(x_real)
    result_imag = np.zeros_like(x_imag)

    for j in range(w):
        result_real[:, j], result_imag[:, j] = bluestein_ifft(temp_real[:, j], temp_imag[:, j])

    return result_real, result_imag

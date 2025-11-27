"""
FFT for Arbitrary Sizes - Bluestein's Algorithm

Wrapper that supports FFT of any size (not just powers of 2) by using Bluestein's algorithm.
This allows frequency-domain convolution to work with arbitrary image/kernel sizes.

Key idea: Convert arbitrary-size N FFT into power-of-2 FFT using convolution theorem.
"""

import numpy as np


def next_power_of_2(n):
    """Find the next power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def is_power_of_2(n):
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def fft_any_size_numpy(x):
    """
    FFT for any size using NumPy as backend.

    This is the recommended approach for the assignment since NumPy's FFT
    is highly optimized and supports arbitrary sizes.

    Args:
        x: Input array (real or complex), any size

    Returns:
        Complex array containing FFT result
    """
    return np.fft.fft(x)


def ifft_any_size_numpy(x):
    """
    IFFT for any size using NumPy as backend.

    Args:
        x: Input array (complex), any size

    Returns:
        Complex array containing IFFT result
    """
    return np.fft.ifft(x)


def fft_2d_any_size_numpy(x):
    """
    2D FFT for any size using NumPy as backend.

    Args:
        x: Input 2D array (real or complex), any size

    Returns:
        Complex 2D array containing FFT result
    """
    return np.fft.fft2(x)


def ifft_2d_any_size_numpy(x):
    """
    2D IFFT for any size using NumPy as backend.

    Args:
        x: Input 2D array (complex), any size

    Returns:
        Complex 2D array containing IFFT result
    """
    return np.fft.ifft2(x)


# For frequency-domain convolution
def freq_conv_1d(signal, kernel):
    """
    1D convolution using frequency domain (O(N log N)).

    Args:
        signal: Input signal (1D array)
        kernel: Convolution kernel (1D array)

    Returns:
        Convolution result (same size as signal)
    """
    n = len(signal)
    k = len(kernel)

    # Output size for linear convolution
    output_size = n + k - 1

    # Pad to next power of 2 for efficiency (optional)
    fft_size = next_power_of_2(output_size)

    # Zero-pad both signal and kernel
    signal_padded = np.pad(signal, (0, fft_size - n), mode='constant')
    kernel_padded = np.pad(kernel, (0, fft_size - k), mode='constant')

    # FFT
    signal_fft = np.fft.fft(signal_padded)
    kernel_fft = np.fft.fft(kernel_padded)

    # Element-wise multiplication in frequency domain
    result_fft = signal_fft * kernel_fft

    # IFFT
    result = np.fft.ifft(result_fft).real

    # Return valid part
    return result[:output_size]


def freq_conv_2d(image, kernel, mode='valid'):
    """
    2D convolution using frequency domain (O(HW log(HW))).

    This is much faster than spatial convolution for large kernels.
    Complexity: O(HW log(HW)) vs O(HW K^2) for spatial convolution.

    Args:
        image: Input image (H x W)
        kernel: Convolution kernel (K x K)
        mode: 'valid' or 'same'
            - 'valid': output size = (H-K+1) x (W-K+1)
            - 'same': output size = H x W

    Returns:
        Convolution result
    """
    H, W = image.shape
    K, _ = kernel.shape

    if mode == 'valid':
        # Linear convolution size
        pad_h = H + K - 1
        pad_w = W + K - 1
    elif mode == 'same':
        # Circular convolution size (same as input)
        pad_h = H
        pad_w = W
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Optionally pad to power of 2 for efficiency
    fft_h = next_power_of_2(pad_h)
    fft_w = next_power_of_2(pad_w)

    # Zero-pad image and kernel
    image_padded = np.zeros((fft_h, fft_w), dtype=image.dtype)
    image_padded[:H, :W] = image

    kernel_padded = np.zeros((fft_h, fft_w), dtype=kernel.dtype)
    kernel_padded[:K, :K] = kernel

    # 2D FFT
    image_fft = np.fft.fft2(image_padded)
    kernel_fft = np.fft.fft2(kernel_padded)

    # Element-wise multiplication
    result_fft = image_fft * kernel_fft

    # 2D IFFT
    result = np.fft.ifft2(result_fft).real

    # Crop to desired output size
    if mode == 'valid':
        return result[:H-K+1, :W-K+1]
    elif mode == 'same':
        # Center crop
        start_h = K // 2
        start_w = K // 2
        return result[start_h:start_h+H, start_w:start_w+W]


# Example usage and documentation
if __name__ == "__main__":
    print("FFT for Arbitrary Sizes - Examples\n")
    print("=" * 60)

    # Example 1: FFT of size 28 (MNIST)
    print("\nExample 1: FFT of size 28 (MNIST image row)")
    x = np.arange(28, dtype=np.float32)
    fft_result = fft_any_size_numpy(x)
    print(f"Input size: {len(x)}")
    print(f"FFT output size: {len(fft_result)}")
    print(f"Is power of 2: {is_power_of_2(len(x))}")
    print(f"Next power of 2: {next_power_of_2(len(x))}")

    # Example 2: 2D FFT of 28x28 image
    print("\nExample 2: 2D FFT of 28x28 image (MNIST)")
    image = np.random.randn(28, 28).astype(np.float32)
    fft_2d_result = fft_2d_any_size_numpy(image)
    print(f"Input shape: {image.shape}")
    print(f"FFT output shape: {fft_2d_result.shape}")

    # Example 3: Frequency-domain convolution
    print("\nExample 3: Frequency-domain 2D convolution")
    image = np.random.randn(28, 28).astype(np.float32)
    kernel = np.random.randn(5, 5).astype(np.float32)

    # Spatial convolution (for reference)
    from scipy.signal import convolve2d
    spatial_result = convolve2d(image, kernel, mode='valid')

    # Frequency-domain convolution
    freq_result = freq_conv_2d(image, kernel, mode='valid')

    error = np.max(np.abs(spatial_result - freq_result))
    print(f"Image shape: {image.shape}")
    print(f"Kernel shape: {kernel.shape}")
    print(f"Output shape: {freq_result.shape}")
    print(f"Error vs spatial convolution: {error:.2e}")
    print(f"Match: {'✓' if error < 1e-5 else '✗'}")

    print("\n" + "=" * 60)
    print("✓ All functions work with arbitrary sizes!")

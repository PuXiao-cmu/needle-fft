"""
Standalone test for Bluestein's FFT Algorithm
Tests without importing needle module
"""

import numpy as np
import sys
import os

# Directly import the required functions
project_root = os.path.join(os.path.dirname(__file__), '../..')
sys.path.insert(0, os.path.join(project_root, 'python/needle/backend_ndarray'))

# Import Cooley-Tukey first (needed by Bluestein)
from fft_cooley_tukey import cooley_tukey_fft, cooley_tukey_ifft


def next_power_of_2(n):
    """Find the next power of 2 greater than or equal to n."""
    return 1 << (n - 1).bit_length()


def is_power_of_2(n):
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def bluestein_fft(x_real, x_imag=None):
    """Compute FFT of arbitrary size using Bluestein's algorithm."""
    n = len(x_real)

    if x_imag is None:
        x_imag = np.zeros(n, dtype=np.float32)

    # If already power of 2, use Cooley-Tukey directly
    if is_power_of_2(n):
        return cooley_tukey_fft(x_real, x_imag)

    # Bluestein's algorithm
    m = next_power_of_2(2 * n - 1)

    # Step 1: Compute chirp sequence
    chirp_real = np.zeros(m, dtype=np.float32)
    chirp_imag = np.zeros(m, dtype=np.float32)

    for k in range(n):
        angle = -np.pi * k * k / n
        chirp_real[k] = np.cos(angle)
        chirp_imag[k] = np.sin(angle)

    # Step 2: Create convolution kernel
    kernel_real = np.zeros(m, dtype=np.float32)
    kernel_imag = np.zeros(m, dtype=np.float32)

    kernel_real[0] = chirp_real[0]
    kernel_imag[0] = -chirp_imag[0]

    for k in range(1, n):
        kernel_real[k] = chirp_real[k]
        kernel_imag[k] = -chirp_imag[k]
        kernel_real[m - k] = chirp_real[k]
        kernel_imag[m - k] = -chirp_imag[k]

    # Step 3: Multiply input by chirp
    a_real = np.zeros(m, dtype=np.float32)
    a_imag = np.zeros(m, dtype=np.float32)

    for k in range(n):
        a_real[k] = x_real[k] * chirp_real[k] - x_imag[k] * chirp_imag[k]
        a_imag[k] = x_real[k] * chirp_imag[k] + x_imag[k] * chirp_real[k]

    # Step 4: FFT of both sequences
    a_fft_real, a_fft_imag = cooley_tukey_fft(a_real, a_imag)
    kernel_fft_real, kernel_fft_imag = cooley_tukey_fft(kernel_real, kernel_imag)

    # Step 5: Multiply in frequency domain
    conv_fft_real = a_fft_real * kernel_fft_real - a_fft_imag * kernel_fft_imag
    conv_fft_imag = a_fft_real * kernel_fft_imag + a_fft_imag * kernel_fft_real

    # Step 6: IFFT
    conv_real, conv_imag = cooley_tukey_ifft(conv_fft_real, conv_fft_imag)

    # Step 7: Multiply by chirp to get final result
    result_real = np.zeros(n, dtype=np.float32)
    result_imag = np.zeros(n, dtype=np.float32)

    for k in range(n):
        result_real[k] = conv_real[k] * chirp_real[k] - conv_imag[k] * chirp_imag[k]
        result_imag[k] = conv_real[k] * chirp_imag[k] + conv_imag[k] * chirp_real[k]

    return result_real, result_imag


def bluestein_ifft(x_real, x_imag=None):
    """Compute IFFT of arbitrary size."""
    n = len(x_real)

    if x_imag is None:
        x_imag = np.zeros(n, dtype=np.float32)

    if is_power_of_2(n):
        return cooley_tukey_ifft(x_real, x_imag)

    conj_real = x_real.copy()
    conj_imag = -x_imag

    fft_real, fft_imag = bluestein_fft(conj_real, conj_imag)

    result_real = fft_real / n
    result_imag = -fft_imag / n

    return result_real, result_imag


print("\n" + "=" * 80)
print("Bluestein FFT 算法测试 (独立测试)")
print("=" * 80 + "\n")

# Test 1: Arbitrary sizes
print("测试 1: 任意大小 FFT (非 2 的次方)")
print("-" * 80)

test_sizes = [3, 5, 7, 10, 12, 15, 20, 28, 30]

for n in test_sizes:
    x = np.array([float(i + 1) for i in range(n)], dtype=np.float32)

    # Bluestein FFT
    fft_real, fft_imag = bluestein_fft(x)

    # NumPy reference
    numpy_fft = np.fft.fft(x)

    # Error
    error_real = np.max(np.abs(fft_real - numpy_fft.real))
    error_imag = np.max(np.abs(fft_imag - numpy_fft.imag))

    status = "✓" if (error_real < 1e-4 and error_imag < 1e-4) else "✗"
    print(f"N={n:3d}: 实部误差={error_real:.2e}, 虚部误差={error_imag:.2e} {status}")

print()

# Test 2: Round-trip test
print("测试 2: IFFT 往返测试")
print("-" * 80)

for n in [3, 5, 7, 10, 15, 20, 28]:
    x = np.array([float(i + 1) for i in range(n)], dtype=np.float32)

    fft_real, fft_imag = bluestein_fft(x)
    ifft_real, ifft_imag = bluestein_ifft(fft_real, fft_imag)

    error = np.max(np.abs(ifft_real - x))
    status = "✓" if error < 1e-5 else "✗"
    print(f"N={n:3d}: 往返误差={error:.2e} {status}")

print()

# Test 3: MNIST size (28)
print("测试 3: MNIST 图像大小 (28)")
print("-" * 80)

x = np.arange(28, dtype=np.float32) + 1
fft_real, fft_imag = bluestein_fft(x)

numpy_fft = np.fft.fft(x)
error_real = np.max(np.abs(fft_real - numpy_fft.real))
error_imag = np.max(np.abs(fft_imag - numpy_fft.imag))

print(f"FFT 误差: 实部={error_real:.2e}, 虚部={error_imag:.2e}")

ifft_real, ifft_imag = bluestein_ifft(fft_real, fft_imag)
error = np.max(np.abs(ifft_real - x))

print(f"往返误差: {error:.2e}")

if error < 1e-5:
    print("✓ MNIST 大小 (28) 测试通过!")
else:
    print("✗ 测试失败")

print()

# Test 4: Common image sizes
print("测试 4: 常见图像大小")
print("-" * 80)

image_sizes = [28, 32, 64, 224, 256]

for n in image_sizes:
    is_pow2 = is_power_of_2(n)
    next_pow2 = next_power_of_2(n)

    x = np.arange(n, dtype=np.float32)
    fft_real, fft_imag = bluestein_fft(x)

    numpy_fft = np.fft.fft(x)
    error = np.max(np.abs(fft_real - numpy_fft.real))

    status = "✓" if error < 1e-4 else "✗"
    pow2_str = "2的次方" if is_pow2 else f"非2的次方 (下一个2的次方: {next_pow2})"

    print(f"N={n:3d} ({pow2_str:35s}): 误差={error:.2e} {status}")

print()
print("=" * 80)
print("测试完成! Bluestein 算法可以处理任意大小的 FFT")
print("=" * 80)

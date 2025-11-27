"""
Test Bluestein's FFT Algorithm for arbitrary input sizes
"""

import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '../..')
sys.path.insert(0, os.path.join(project_root, 'python'))

from needle.backend_ndarray.fft_bluestein import (
    bluestein_fft, bluestein_ifft, bluestein_fft_2d, bluestein_ifft_2d,
    is_power_of_2, next_power_of_2
)


def test_power_of_2_detection():
    """Test power of 2 detection and next power of 2 calculation."""
    print("=" * 80)
    print("测试 1: Power of 2 检测")
    print("=" * 80)

    # Test power of 2 detection
    assert is_power_of_2(1) == True
    assert is_power_of_2(2) == True
    assert is_power_of_2(4) == True
    assert is_power_of_2(8) == True
    assert is_power_of_2(3) == False
    assert is_power_of_2(5) == False
    assert is_power_of_2(7) == False

    print("✓ Power of 2 检测正确")

    # Test next power of 2
    assert next_power_of_2(1) == 1
    assert next_power_of_2(2) == 2
    assert next_power_of_2(3) == 4
    assert next_power_of_2(5) == 8
    assert next_power_of_2(7) == 8
    assert next_power_of_2(9) == 16
    assert next_power_of_2(28) == 32

    print("✓ Next power of 2 计算正确")
    print()


def test_bluestein_power_of_2():
    """Test Bluestein with power of 2 sizes (should match Cooley-Tukey)."""
    print("=" * 80)
    print("测试 2: Bluestein FFT (2的次方大小)")
    print("=" * 80)

    sizes = [8, 16, 32, 64]

    for n in sizes:
        # Generate test data
        x = np.array([float(i + 1) for i in range(n)], dtype=np.float32)

        # Compute FFT with Bluestein
        fft_real, fft_imag = bluestein_fft(x)

        # Compare with NumPy
        numpy_fft = np.fft.fft(x)
        numpy_real = numpy_fft.real.astype(np.float32)
        numpy_imag = numpy_fft.imag.astype(np.float32)

        # Check error
        error_real = np.max(np.abs(fft_real - numpy_real))
        error_imag = np.max(np.abs(fft_imag - numpy_imag))

        print(f"N={n:3d}: 实部误差={error_real:.2e}, 虚部误差={error_imag:.2e}", end="")

        if error_real < 1e-4 and error_imag < 1e-4:
            print(" ✓")
        else:
            print(" ✗ 误差过大!")

    print()


def test_bluestein_arbitrary_sizes():
    """Test Bluestein with non-power-of-2 sizes."""
    print("=" * 80)
    print("测试 3: Bluestein FFT (任意大小 - 非2的次方)")
    print("=" * 80)

    # Test various non-power-of-2 sizes
    sizes = [3, 5, 6, 7, 10, 12, 15, 20, 28, 30, 50, 100]

    for n in sizes:
        # Generate test data
        x = np.array([float(i + 1) for i in range(n)], dtype=np.float32)

        # Compute FFT with Bluestein
        fft_real, fft_imag = bluestein_fft(x)

        # Compare with NumPy
        numpy_fft = np.fft.fft(x)
        numpy_real = numpy_fft.real.astype(np.float32)
        numpy_imag = numpy_fft.imag.astype(np.float32)

        # Check error
        error_real = np.max(np.abs(fft_real - numpy_real))
        error_imag = np.max(np.abs(fft_imag - numpy_imag))

        print(f"N={n:3d}: 实部误差={error_real:.2e}, 虚部误差={error_imag:.2e}", end="")

        if error_real < 1e-4 and error_imag < 1e-4:
            print(" ✓")
        else:
            print(f" ✗ 误差过大!")
            print(f"  Bluestein 实部: {fft_real[:5]}")
            print(f"  NumPy 实部:     {numpy_real[:5]}")

    print()


def test_bluestein_ifft():
    """Test Bluestein IFFT with round-trip."""
    print("=" * 80)
    print("测试 4: Bluestein IFFT 往返测试")
    print("=" * 80)

    sizes = [3, 5, 7, 10, 15, 20, 28, 32, 50]

    for n in sizes:
        # Generate test data
        x = np.array([float(i + 1) for i in range(n)], dtype=np.float32)

        # Forward FFT
        fft_real, fft_imag = bluestein_fft(x)

        # Inverse FFT
        ifft_real, ifft_imag = bluestein_ifft(fft_real, fft_imag)

        # Check round-trip error
        error = np.max(np.abs(ifft_real - x))

        print(f"N={n:3d}: 往返误差={error:.2e}", end="")

        if error < 1e-5:
            print(" ✓")
        else:
            print(" ✗ 误差过大!")
            print(f"  原始数据: {x[:5]}")
            print(f"  恢复数据: {ifft_real[:5]}")

    print()


def test_bluestein_2d():
    """Test 2D Bluestein FFT."""
    print("=" * 80)
    print("测试 5: Bluestein 2D FFT (任意大小)")
    print("=" * 80)

    # Test various 2D sizes
    sizes = [(8, 8), (10, 10), (15, 15), (28, 28), (7, 11), (13, 17)]

    for (h, w) in sizes:
        # Generate test data
        x = np.arange(h * w, dtype=np.float32).reshape(h, w)

        # Compute 2D FFT with Bluestein
        fft_real, fft_imag = bluestein_fft_2d(x)

        # Compare with NumPy
        numpy_fft = np.fft.fft2(x)
        numpy_real = numpy_fft.real.astype(np.float32)
        numpy_imag = numpy_fft.imag.astype(np.float32)

        # Check error
        error_real = np.max(np.abs(fft_real - numpy_real))
        error_imag = np.max(np.abs(fft_imag - numpy_imag))

        print(f"Size {h:2d}×{w:2d}: 实部误差={error_real:.2e}, 虚部误差={error_imag:.2e}", end="")

        if error_real < 1e-4 and error_imag < 1e-4:
            print(" ✓")
        else:
            print(" ✗ 误差过大!")

    print()


def test_mnist_size():
    """Test with MNIST image size (28x28)."""
    print("=" * 80)
    print("测试 6: MNIST 图像大小 (28×28)")
    print("=" * 80)

    # Create a 28x28 test image
    x = np.random.randn(28, 28).astype(np.float32)

    # 2D FFT
    fft_real, fft_imag = bluestein_fft_2d(x)

    # Compare with NumPy
    numpy_fft = np.fft.fft2(x)
    error_real = np.max(np.abs(fft_real - numpy_fft.real))
    error_imag = np.max(np.abs(fft_imag - numpy_fft.imag))

    print(f"FFT 误差: 实部={error_real:.2e}, 虚部={error_imag:.2e}")

    # Round-trip test
    ifft_real, ifft_imag = bluestein_ifft_2d(fft_real, fft_imag)
    error = np.max(np.abs(ifft_real - x))

    print(f"往返误差: {error:.2e}")

    if error < 1e-4:
        print("✓ MNIST 大小测试通过!")
    else:
        print("✗ MNIST 大小测试失败!")

    print()


if __name__ == "__main__":
    print("\n")
    print("=" * 80)
    print("Bluestein FFT 算法测试")
    print("支持任意大小的 FFT (不限于 2 的次方)")
    print("=" * 80)
    print()

    test_power_of_2_detection()
    test_bluestein_power_of_2()
    test_bluestein_arbitrary_sizes()
    test_bluestein_ifft()
    test_bluestein_2d()
    test_mnist_size()

    print("=" * 80)
    print("所有测试完成!")
    print("=" * 80)

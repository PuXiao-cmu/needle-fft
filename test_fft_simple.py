"""
Simple test script for FFT/IFFT operations - no external dependencies except needle itself.
"""

import sys
sys.path.append('./python')

# Check if we can import needle
try:
    import needle as ndl
    print("✓ Successfully imported needle")
except ImportError as e:
    print(f"✗ Failed to import needle: {e}")
    sys.exit(1)

# Check if we can import numpy (needed by needle)
try:
    import numpy as np
    print("✓ Successfully imported numpy")
except ImportError as e:
    print(f"✗ Failed to import numpy: {e}")
    print("Note: Please install numpy (pip install numpy) to run tests")
    sys.exit(1)

from needle import backend_ndarray as nd


def test_fft_basic():
    """Test basic FFT functionality."""
    print("\n" + "=" * 60)
    print("Test 1: Basic FFT Forward Pass")
    print("=" * 60)

    # Create a simple input
    _A = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    A = ndl.Tensor(nd.array(_A), device=ndl.cpu())

    print(f"Input shape: {A.shape}")
    print(f"Input data: {A.numpy()}")

    # Compute FFT
    fft_result = ndl.ops.fft(A, dim=-1, norm="backward")
    print(f"\nFFT output shape: {fft_result.shape}")
    print(f"FFT output: {fft_result.numpy()}")

    # Compare with numpy
    numpy_fft = np.fft.fft(_A, norm="backward")
    numpy_fft_real = np.real(numpy_fft).astype(np.float32)
    print(f"\nNumPy FFT (real part): {numpy_fft_real}")

    # Check if they match
    diff = np.abs(fft_result.numpy() - numpy_fft_real)
    max_diff = np.max(diff)
    print(f"\nMax difference: {max_diff}")

    if max_diff < 1e-5:
        print("✓ FFT forward pass TEST PASSED")
        return True
    else:
        print("✗ FFT forward pass TEST FAILED")
        return False


def test_ifft_basic():
    """Test basic IFFT functionality."""
    print("\n" + "=" * 60)
    print("Test 2: Basic IFFT Forward Pass")
    print("=" * 60)

    # Create a simple input
    _A = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    A = ndl.Tensor(nd.array(_A), device=ndl.cpu())

    print(f"Input shape: {A.shape}")
    print(f"Input data: {A.numpy()}")

    # Compute IFFT
    ifft_result = ndl.ops.ifft(A, dim=-1, norm="backward")
    print(f"\nIFFT output shape: {ifft_result.shape}")
    print(f"IFFT output: {ifft_result.numpy()}")

    # Compare with numpy
    numpy_ifft = np.fft.ifft(_A, norm="backward")
    numpy_ifft_real = np.real(numpy_ifft).astype(np.float32)
    print(f"\nNumPy IFFT (real part): {numpy_ifft_real}")

    # Check if they match
    diff = np.abs(ifft_result.numpy() - numpy_ifft_real)
    max_diff = np.max(diff)
    print(f"\nMax difference: {max_diff}")

    if max_diff < 1e-5:
        print("✓ IFFT forward pass TEST PASSED")
        return True
    else:
        print("✗ IFFT forward pass TEST FAILED")
        return False


def test_roundtrip():
    """Test FFT -> IFFT round-trip."""
    print("\n" + "=" * 60)
    print("Test 3: FFT -> IFFT Round-trip")
    print("=" * 60)

    # Create input
    _A = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    A = ndl.Tensor(nd.array(_A), device=ndl.cpu())

    print(f"Original input: {A.numpy()}")

    # FFT followed by IFFT
    fft_result = ndl.ops.fft(A, dim=-1, norm="backward")
    recovered = ndl.ops.ifft(fft_result, dim=-1, norm="backward")

    print(f"After FFT->IFFT: {recovered.numpy()}")

    # Check if we recovered the input
    diff = np.abs(recovered.numpy() - _A)
    max_diff = np.max(diff)
    print(f"\nMax difference from original: {max_diff}")

    if max_diff < 1e-4:
        print("✓ Round-trip TEST PASSED")
        return True
    else:
        print("✗ Round-trip TEST FAILED")
        return False


def test_fft_gradient():
    """Test FFT gradient with simple finite difference check."""
    print("\n" + "=" * 60)
    print("Test 4: FFT Gradient Check (Finite Difference)")
    print("=" * 60)

    # Use smaller input for gradient check
    _A = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    A = ndl.Tensor(nd.array(_A), device=ndl.cpu())

    print(f"Input: {A.numpy()}")

    # Compute FFT
    fft_result = ndl.ops.fft(A, dim=-1, norm="backward")

    # Create a random gradient
    out_grad = np.array([0.5, 0.3, 0.2, 0.1], dtype=np.float32)
    c = ndl.Tensor(nd.array(out_grad), device=ndl.cpu())

    print(f"Output gradient: {out_grad}")

    # Compute analytical gradient
    analytical_grad = fft_result.op.gradient_as_tuple(c, fft_result)[0]
    print(f"\nAnalytical gradient: {analytical_grad.numpy()}")

    # Compute numerical gradient using finite differences
    eps = 1e-5
    numerical_grad = np.zeros_like(_A)

    for i in range(len(_A)):
        # f(x + eps)
        _A[i] += eps
        A_plus = ndl.Tensor(nd.array(_A.copy()), device=ndl.cpu())
        f_plus = ndl.ops.fft(A_plus, dim=-1, norm="backward")
        val_plus = np.sum(f_plus.numpy() * out_grad)

        # f(x - eps)
        _A[i] -= 2 * eps
        A_minus = ndl.Tensor(nd.array(_A.copy()), device=ndl.cpu())
        f_minus = ndl.ops.fft(A_minus, dim=-1, norm="backward")
        val_minus = np.sum(f_minus.numpy() * out_grad)

        # Restore
        _A[i] += eps

        # Finite difference
        numerical_grad[i] = (val_plus - val_minus) / (2 * eps)

    print(f"Numerical gradient: {numerical_grad}")

    # Compare gradients
    grad_diff = np.abs(analytical_grad.numpy() - numerical_grad)
    max_grad_diff = np.max(grad_diff)
    print(f"\nMax gradient difference: {max_grad_diff}")

    if max_grad_diff < 1e-2:
        print("✓ FFT gradient TEST PASSED")
        return True
    else:
        print("✗ FFT gradient TEST FAILED")
        print(f"Note: Gradient error {max_grad_diff} exceeds tolerance 1e-2")
        return False


def test_ifft_gradient():
    """Test IFFT gradient with simple finite difference check."""
    print("\n" + "=" * 60)
    print("Test 5: IFFT Gradient Check (Finite Difference)")
    print("=" * 60)

    # Use smaller input for gradient check
    _A = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    A = ndl.Tensor(nd.array(_A), device=ndl.cpu())

    print(f"Input: {A.numpy()}")

    # Compute IFFT
    ifft_result = ndl.ops.ifft(A, dim=-1, norm="backward")

    # Create a random gradient
    out_grad = np.array([0.5, 0.3, 0.2, 0.1], dtype=np.float32)
    c = ndl.Tensor(nd.array(out_grad), device=ndl.cpu())

    print(f"Output gradient: {out_grad}")

    # Compute analytical gradient
    analytical_grad = ifft_result.op.gradient_as_tuple(c, ifft_result)[0]
    print(f"\nAnalytical gradient: {analytical_grad.numpy()}")

    # Compute numerical gradient using finite differences
    eps = 1e-5
    numerical_grad = np.zeros_like(_A)

    for i in range(len(_A)):
        # f(x + eps)
        _A[i] += eps
        A_plus = ndl.Tensor(nd.array(_A.copy()), device=ndl.cpu())
        f_plus = ndl.ops.ifft(A_plus, dim=-1, norm="backward")
        val_plus = np.sum(f_plus.numpy() * out_grad)

        # f(x - eps)
        _A[i] -= 2 * eps
        A_minus = ndl.Tensor(nd.array(_A.copy()), device=ndl.cpu())
        f_minus = ndl.ops.ifft(A_minus, dim=-1, norm="backward")
        val_minus = np.sum(f_minus.numpy() * out_grad)

        # Restore
        _A[i] += eps

        # Finite difference
        numerical_grad[i] = (val_plus - val_minus) / (2 * eps)

    print(f"Numerical gradient: {numerical_grad}")

    # Compare gradients
    grad_diff = np.abs(analytical_grad.numpy() - numerical_grad)
    max_grad_diff = np.max(grad_diff)
    print(f"\nMax gradient difference: {max_grad_diff}")

    if max_grad_diff < 1e-2:
        print("✓ IFFT gradient TEST PASSED")
        return True
    else:
        print("✗ IFFT gradient TEST FAILED")
        print(f"Note: Gradient error {max_grad_diff} exceeds tolerance 1e-2")
        return False


def main():
    print("=" * 60)
    print("FFT/IFFT Operator Tests for Needle Framework")
    print("=" * 60)

    results = []

    # Run all tests
    results.append(("FFT Forward", test_fft_basic()))
    results.append(("IFFT Forward", test_ifft_basic()))
    results.append(("Round-trip", test_roundtrip()))
    results.append(("FFT Gradient", test_fft_gradient()))
    results.append(("IFFT Gradient", test_ifft_gradient()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

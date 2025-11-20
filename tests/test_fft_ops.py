"""
Test file for FFT/IFFT operations with gradient verification.

This test suite verifies:
1. Forward FFT/IFFT computation correctness
2. Gradient computation using finite-difference checks
3. Round-trip consistency (FFT -> IFFT should recover input)
4. Different normalization modes
"""

import sys
sys.path.append('./python')

import numpy as np
import pytest
import needle as ndl
from needle import backend_ndarray as nd


def backward_check(f, *args, **kwargs):
    """
    Numerical gradient checking using finite differences.

    Compares analytical gradients (from autograd) with numerical gradients
    computed using the finite difference method: df/dx ≈ (f(x+ε) - f(x-ε)) / (2ε)
    """
    eps = 1e-5
    out = f(*args, **kwargs)
    c = np.random.randn(*out.shape)
    numerical_grad = [np.zeros(a.shape) for a in args]
    num_args = len(args)

    for i in range(num_args):
        for j in range(args[i].realize_cached_data().size):
            # Compute f(x + eps)
            args[i].realize_cached_data().flat[j] += eps
            f1 = (f(*args, **kwargs).numpy() * c).sum()

            # Compute f(x - eps)
            args[i].realize_cached_data().flat[j] -= 2 * eps
            f2 = (f(*args, **kwargs).numpy() * c).sum()

            # Restore original value
            args[i].realize_cached_data().flat[j] += eps

            # Finite difference approximation
            numerical_grad[i].flat[j] = (f1 - f2) / (2 * eps)

    # Compute analytical gradient using autograd
    backward_grad = out.op.gradient_as_tuple(ndl.Tensor(c, device=args[0].device), out)

    # Compare numerical and analytical gradients
    error = sum(
        np.linalg.norm(backward_grad[i].numpy() - numerical_grad[i])
        for i in range(len(args))
    )

    # Allow some numerical error due to finite precision
    assert error < 1e-2, f"Gradient error {error} exceeds tolerance"
    return [g.numpy() for g in backward_grad]


# Test configurations
_DEVICES = [ndl.cpu()]  # Can add CUDA if available

# Test different input shapes
FFT_SHAPES = [
    (8,),      # 1D FFT
    (16,),     # 1D FFT (power of 2)
    (4, 8),    # 2D tensor, FFT on last dim
    (2, 4, 8), # 3D tensor, FFT on last dim
]

NORMALIZATION_MODES = ["backward", "forward", "ortho"]


@pytest.mark.parametrize("shape", FFT_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu"])
def test_fft_forward(shape, device):
    """Test forward FFT computation matches numpy."""
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)

    # Compute FFT using Needle
    fft_result = ndl.ops.fft(A, dim=-1, norm="backward")

    # Compute FFT using NumPy
    numpy_fft = np.fft.fft(_A, axis=-1, norm="backward")
    # Take real part for comparison (since we only return real part)
    numpy_fft_real = np.real(numpy_fft).astype(np.float32)

    # Compare results
    np.testing.assert_allclose(
        fft_result.numpy(),
        numpy_fft_real,
        atol=1e-5,
        rtol=1e-5,
        err_msg=f"FFT forward pass failed for shape {shape}"
    )


@pytest.mark.parametrize("shape", FFT_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu"])
def test_ifft_forward(shape, device):
    """Test forward IFFT computation matches numpy."""
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)

    # Compute IFFT using Needle
    ifft_result = ndl.ops.ifft(A, dim=-1, norm="backward")

    # Compute IFFT using NumPy
    numpy_ifft = np.fft.ifft(_A, axis=-1, norm="backward")
    numpy_ifft_real = np.real(numpy_ifft).astype(np.float32)

    # Compare results
    np.testing.assert_allclose(
        ifft_result.numpy(),
        numpy_ifft_real,
        atol=1e-5,
        rtol=1e-5,
        err_msg=f"IFFT forward pass failed for shape {shape}"
    )


@pytest.mark.parametrize("shape", [(8,), (4, 8)])
@pytest.mark.parametrize("norm", NORMALIZATION_MODES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu"])
def test_fft_normalization_modes(shape, norm, device):
    """Test different normalization modes for FFT."""
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)

    # Compute FFT with specific normalization
    fft_result = ndl.ops.fft(A, dim=-1, norm=norm)

    # Compare with NumPy
    numpy_fft = np.fft.fft(_A, axis=-1, norm=norm)
    numpy_fft_real = np.real(numpy_fft).astype(np.float32)

    np.testing.assert_allclose(
        fft_result.numpy(),
        numpy_fft_real,
        atol=1e-5,
        rtol=1e-5,
        err_msg=f"FFT normalization mode '{norm}' failed"
    )


@pytest.mark.parametrize("shape", [(8,), (4, 8)])
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu"])
def test_fft_roundtrip(shape, device):
    """Test that FFT -> IFFT recovers the original input."""
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)

    # FFT followed by IFFT should recover input (approximately)
    fft_result = ndl.ops.fft(A, dim=-1, norm="backward")
    recovered = ndl.ops.ifft(fft_result, dim=-1, norm="backward")

    # Check if we recover the original
    np.testing.assert_allclose(
        recovered.numpy(),
        _A,
        atol=1e-4,
        rtol=1e-4,
        err_msg=f"FFT->IFFT round-trip failed for shape {shape}"
    )


@pytest.mark.parametrize("shape", [(8,), (4, 8)])
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu"])
def test_ifft_roundtrip(shape, device):
    """Test that IFFT -> FFT recovers the original input."""
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)

    # IFFT followed by FFT should recover input (approximately)
    ifft_result = ndl.ops.ifft(A, dim=-1, norm="backward")
    recovered = ndl.ops.fft(ifft_result, dim=-1, norm="backward")

    # Check if we recover the original
    np.testing.assert_allclose(
        recovered.numpy(),
        _A,
        atol=1e-4,
        rtol=1e-4,
        err_msg=f"IFFT->FFT round-trip failed for shape {shape}"
    )


@pytest.mark.parametrize("shape", [(8,), (4, 8)])
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu"])
def test_fft_gradient(shape, device):
    """Test FFT gradient using finite-difference check."""
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)

    # Define function for gradient checking
    def fft_fn(x):
        return ndl.ops.fft(x, dim=-1, norm="backward")

    # Perform backward check
    print(f"\nTesting FFT gradient for shape {shape}...")
    backward_check(fft_fn, A)
    print(f"✓ FFT gradient check passed for shape {shape}")


@pytest.mark.parametrize("shape", [(8,), (4, 8)])
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu"])
def test_ifft_gradient(shape, device):
    """Test IFFT gradient using finite-difference check."""
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)

    # Define function for gradient checking
    def ifft_fn(x):
        return ndl.ops.ifft(x, dim=-1, norm="backward")

    # Perform backward check
    print(f"\nTesting IFFT gradient for shape {shape}...")
    backward_check(ifft_fn, A)
    print(f"✓ IFFT gradient check passed for shape {shape}")


@pytest.mark.parametrize("norm", NORMALIZATION_MODES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu"])
def test_fft_gradient_normalization(norm, device):
    """Test FFT gradient with different normalization modes."""
    shape = (8,)
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)

    # Define function for gradient checking
    def fft_fn(x):
        return ndl.ops.fft(x, dim=-1, norm=norm)

    # Perform backward check
    print(f"\nTesting FFT gradient with norm='{norm}'...")
    backward_check(fft_fn, A)
    print(f"✓ FFT gradient check passed for norm='{norm}'")


@pytest.mark.parametrize("norm", NORMALIZATION_MODES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu"])
def test_ifft_gradient_normalization(norm, device):
    """Test IFFT gradient with different normalization modes."""
    shape = (8,)
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)

    # Define function for gradient checking
    def ifft_fn(x):
        return ndl.ops.ifft(x, dim=-1, norm=norm)

    # Perform backward check
    print(f"\nTesting IFFT gradient with norm='{norm}'...")
    backward_check(ifft_fn, A)
    print(f"✓ IFFT gradient check passed for norm='{norm}'")


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu"])
def test_fft_in_computation_graph(device):
    """Test FFT in a more complex computation graph."""
    shape = (4, 8)
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)

    # Complex computation: (FFT(x) + x) * 2
    def complex_fn(x):
        fft_x = ndl.ops.fft(x, dim=-1, norm="backward")
        return (fft_x + x) * 2

    print(f"\nTesting FFT in computation graph...")
    backward_check(complex_fn, A)
    print(f"✓ FFT computation graph gradient check passed")


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu"])
def test_ifft_in_computation_graph(device):
    """Test IFFT in a more complex computation graph."""
    shape = (4, 8)
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)

    # Complex computation: (IFFT(x) * 3 + x)
    def complex_fn(x):
        ifft_x = ndl.ops.ifft(x, dim=-1, norm="backward")
        return ifft_x * 3 + x

    print(f"\nTesting IFFT in computation graph...")
    backward_check(complex_fn, A)
    print(f"✓ IFFT computation graph gradient check passed")


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu"])
def test_fft_ifft_chain_gradient(device):
    """Test gradient through FFT->IFFT chain."""
    shape = (8,)
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)

    # Chain FFT and IFFT
    def chain_fn(x):
        return ndl.ops.ifft(ndl.ops.fft(x, dim=-1, norm="backward"), dim=-1, norm="backward")

    print(f"\nTesting FFT->IFFT chain gradient...")
    backward_check(chain_fn, A)
    print(f"✓ FFT->IFFT chain gradient check passed")


if __name__ == "__main__":
    # Run tests manually
    print("=" * 80)
    print("FFT/IFFT Operator Tests")
    print("=" * 80)

    device = ndl.cpu()

    # Test forward operations
    print("\n1. Testing forward FFT...")
    test_fft_forward((8,), device)
    print("✓ Forward FFT passed")

    print("\n2. Testing forward IFFT...")
    test_ifft_forward((8,), device)
    print("✓ Forward IFFT passed")

    print("\n3. Testing FFT normalization modes...")
    for norm in NORMALIZATION_MODES:
        test_fft_normalization_modes((8,), norm, device)
    print("✓ All normalization modes passed")

    print("\n4. Testing round-trip consistency...")
    test_fft_roundtrip((8,), device)
    test_ifft_roundtrip((8,), device)
    print("✓ Round-trip tests passed")

    print("\n5. Testing FFT gradients...")
    test_fft_gradient((8,), device)
    print("✓ FFT gradients passed")

    print("\n6. Testing IFFT gradients...")
    test_ifft_gradient((8,), device)
    print("✓ IFFT gradients passed")

    print("\n7. Testing gradients with different normalization modes...")
    for norm in NORMALIZATION_MODES:
        test_fft_gradient_normalization(norm, device)
        test_ifft_gradient_normalization(norm, device)
    print("✓ All gradient normalization tests passed")

    print("\n8. Testing FFT in computation graphs...")
    test_fft_in_computation_graph(device)
    test_ifft_in_computation_graph(device)
    print("✓ Computation graph tests passed")

    print("\n9. Testing FFT->IFFT chain gradients...")
    test_fft_ifft_chain_gradient(device)
    print("✓ Chain gradient tests passed")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)

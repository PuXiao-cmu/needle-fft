"""
Test FFT/IFFT with separate real and imaginary parts
"""

import sys
sys.path.insert(0, './python')
import needle as ndl
import numpy as np

print("=" * 60)
print("FFT/IFFT Test with Real/Imaginary Parts")
print("=" * 60)

# Test 1: Basic FFT returns real and imaginary parts
print("\n[Test 1] FFT returns real and imaginary parts")
try:
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    x = ndl.Tensor(data, device=ndl.cpu_numpy())

    print(f"  Input: {x.numpy()}")

    # FFT should return a TensorTuple with (real, imag)
    fft_result = ndl.ops.fft(x, dim=-1, norm="backward")

    print(f"  FFT type: {type(fft_result)}")
    print(f"  FFT is TensorTuple: {isinstance(fft_result, ndl.TensorTuple)}")

    # Extract real and imaginary parts
    fft_real = ndl.ops.tuple_get_item(fft_result, 0)
    fft_imag = ndl.ops.tuple_get_item(fft_result, 1)

    print(f"  Real part: {fft_real.numpy()}")
    print(f"  Imag part: {fft_imag.numpy()}")

    # Verify with numpy
    expected_fft = np.fft.fft(data, norm='backward')
    expected_real = np.real(expected_fft)
    expected_imag = np.imag(expected_fft)

    real_diff = np.abs(fft_real.numpy() - expected_real).max()
    imag_diff = np.abs(fft_imag.numpy() - expected_imag).max()

    print(f"  Real part error: {real_diff:.2e}")
    print(f"  Imag part error: {imag_diff:.2e}")

    if real_diff < 1e-5 and imag_diff < 1e-5:
        print("  ✓ FFT test passed")
    else:
        print(f"  ✗ FFT test failed")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: IFFT from real and imaginary parts
print("\n[Test 2] IFFT from real and imaginary parts")
try:
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    x = ndl.Tensor(data, device=ndl.cpu_numpy())

    # Get FFT
    fft_result = ndl.ops.fft(x, dim=-1, norm="backward")
    fft_real = ndl.ops.tuple_get_item(fft_result, 0)
    fft_imag = ndl.ops.tuple_get_item(fft_result, 1)

    # IFFT
    recovered = ndl.ops.ifft(fft_real, fft_imag, dim=-1, norm="backward")

    print(f"  Original:  {x.numpy()}")
    print(f"  Recovered: {recovered.numpy()}")

    diff = np.abs(recovered.numpy() - x.numpy()).max()
    print(f"  Max difference: {diff:.2e}")

    if diff < 1e-5:
        print("  ✓ Round-trip test passed!")
    else:
        print(f"  ✗ Round-trip test failed: diff = {diff}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Different normalization modes
print("\n[Test 3] Different normalization modes")
for norm_mode in ["backward", "forward", "ortho"]:
    try:
        data = [1.0, 2.0, 3.0, 4.0]
        x = ndl.Tensor(data, device=ndl.cpu_numpy())

        # FFT -> IFFT
        fft_result = ndl.ops.fft(x, norm=norm_mode)
        fft_real = ndl.ops.tuple_get_item(fft_result, 0)
        fft_imag = ndl.ops.tuple_get_item(fft_result, 1)
        recovered = ndl.ops.ifft(fft_real, fft_imag, norm=norm_mode)

        diff = np.abs(recovered.numpy() - x.numpy()).max()

        if diff < 1e-5:
            print(f"  ✓ norm='{norm_mode}': diff = {diff:.2e}")
        else:
            print(f"  ✗ norm='{norm_mode}': diff = {diff:.2e} (FAILED)")
    except Exception as e:
        print(f"  ✗ norm='{norm_mode}': Error: {e}")

# Test 4: Gradient check
print("\n[Test 4] Gradient check")
try:
    data = [1.0, 2.0, 3.0, 4.0]
    x = ndl.Tensor(data, device=ndl.cpu_numpy())

    # Compute FFT
    fft_result = ndl.ops.fft(x)
    fft_real = ndl.ops.tuple_get_item(fft_result, 0)
    fft_imag = ndl.ops.tuple_get_item(fft_result, 1)

    # Define loss as sum of squares of real and imaginary parts
    loss = (fft_real ** 2).sum() + (fft_imag ** 2).sum()

    # Backward
    loss.backward()

    print(f"  Input: {x.numpy()}")
    print(f"  Gradient: {x.grad.numpy()}")
    print(f"  ✓ Gradient computed successfully")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Multi-dimensional tensor
print("\n[Test 5] Multi-dimensional tensor")
try:
    data = np.random.randn(3, 8).astype(np.float32)
    x = ndl.Tensor(data, device=ndl.cpu_numpy())

    print(f"  Input shape: {x.shape}")

    # FFT along last axis
    fft_result = ndl.ops.fft(x, dim=-1)
    fft_real = ndl.ops.tuple_get_item(fft_result, 0)
    fft_imag = ndl.ops.tuple_get_item(fft_result, 1)

    print(f"  FFT real shape: {fft_real.shape}")
    print(f"  FFT imag shape: {fft_imag.shape}")

    # IFFT
    recovered = ndl.ops.ifft(fft_real, fft_imag, dim=-1)

    print(f"  Recovered shape: {recovered.shape}")

    diff = np.abs(recovered.numpy() - x.numpy()).max()
    print(f"  Max difference: {diff:.2e}")

    if diff < 1e-5:
        print("  ✓ Multi-dimensional test passed")
    else:
        print(f"  ✗ Multi-dimensional test failed: diff = {diff}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Tests completed!")
print("=" * 60)

"""
Simple test for C++ FFT implementation
"""

import sys
import os
import numpy as np

print("=" * 80)
print("C++ Cooley-Tukey FFT Test")
print("=" * 80)

# Add path
sys.path.insert(0, './python')

# Test if C++ backend is available
try:
    from needle.backend_ndarray import ndarray_backend_cpu
    print(f"\n✓ C++ backend loaded")
    print(f"  Module: {ndarray_backend_cpu}")

    # Check for FFT functions
    has_fft = hasattr(ndarray_backend_cpu, 'cooley_tukey_fft')
    has_ifft = hasattr(ndarray_backend_cpu, 'cooley_tukey_ifft')

    print(f"  Has cooley_tukey_fft: {has_fft}")
    print(f"  Has cooley_tukey_ifft: {has_ifft}")

    if not has_fft:
        print("\n✗ FFT functions not found in C++ backend")
        print("Available functions:")
        for name in dir(ndarray_backend_cpu):
            if not name.startswith('_'):
                print(f"    - {name}")
        sys.exit(1)

except Exception as e:
    print(f"\n✗ C++ backend not available: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test data
test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)

print("\n" + "=" * 80)
print("Testing C++ Cooley-Tukey FFT")
print("=" * 80)

try:
    # Create arrays
    from needle.backend_ndarray.ndarray_backend_cpu import Array as CppArray

    input_arr = CppArray(8)
    output_real = CppArray(8)
    output_imag = CppArray(8)
    recovered = CppArray(8)

    # Copy data to C++ array
    for i in range(8):
        input_arr.ptr[i] = test_data[i]

    # FFT
    print(f"\nInput:     {test_data}")
    ndarray_backend_cpu.cooley_tukey_fft(input_arr, output_real, output_imag, 8)

    # Get results
    fft_real = np.array([output_real.ptr[i] for i in range(8)])
    fft_imag = np.array([output_imag.ptr[i] for i in range(8)])

    print(f"FFT Real:  {fft_real}")
    print(f"FFT Imag:  {fft_imag}")

    # Compare with NumPy
    numpy_fft = np.fft.fft(test_data, norm='backward')
    numpy_real = np.real(numpy_fft)
    numpy_imag = np.imag(numpy_fft)

    real_error = np.abs(fft_real - numpy_real).max()
    imag_error = np.abs(fft_imag - numpy_imag).max()

    print(f"\nAccuracy vs NumPy:")
    print(f"  Real part error: {real_error:.2e}")
    print(f"  Imag part error: {imag_error:.2e}")

    if real_error < 1e-5 and imag_error < 1e-5:
        print("  ✓ C++ FFT is correct!")
    else:
        print("  ✗ C++ FFT has errors")

    # IFFT
    ndarray_backend_cpu.cooley_tukey_ifft(output_real, output_imag, recovered, 8)

    recovered_data = np.array([recovered.ptr[i] for i in range(8)])
    print(f"\nRecovered: {recovered_data}")

    roundtrip_error = np.abs(recovered_data - test_data).max()
    print(f"Round-trip error: {roundtrip_error:.2e}")

    if roundtrip_error < 1e-5:
        print("\n✓ C++ FFT implementation works correctly!")
    else:
        print(f"\n✗ Round-trip test failed: error = {roundtrip_error}")

except Exception as e:
    print(f"\n✗ C++ FFT test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("Test Complete!")
print("=" * 80)

"""
Direct test of C++ FFT implementation without importing needle module
"""

import sys
import numpy as np

print("=" * 80)
print("Direct C++ Cooley-Tukey FFT Test")
print("=" * 80)

# Import C++ backend directly
try:
    sys.path.insert(0, './python/needle/backend_ndarray')
    import ndarray_backend_cpu
    print(f"\n✓ C++ backend loaded successfully")
except Exception as e:
    print(f"\n✗ Failed to load C++ backend: {e}")
    sys.exit(1)

# Check for FFT functions
has_fft = hasattr(ndarray_backend_cpu, 'cooley_tukey_fft')
has_ifft = hasattr(ndarray_backend_cpu, 'cooley_tukey_ifft')

print(f"  Has cooley_tukey_fft: {has_fft}")
print(f"  Has cooley_tukey_ifft: {has_ifft}")

if not has_fft:
    print("\n✗ FFT functions not found!")
    print("Available functions:")
    for name in dir(ndarray_backend_cpu):
        if not name.startswith('_'):
            print(f"    - {name}")
    sys.exit(1)

# Test data
test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)

print("\n" + "=" * 80)
print("Testing FFT Functionality")
print("=" * 80)

try:
    # Create arrays using the Array class from C++ backend
    Array = ndarray_backend_cpu.Array

    input_arr = Array(8)
    output_real = Array(8)
    output_imag = Array(8)
    recovered = Array(8)

    # Copy test data to input array using from_numpy
    ndarray_backend_cpu.from_numpy(test_data, input_arr)

    # Perform FFT
    print(f"\nInput:     {test_data}")
    ndarray_backend_cpu.cooley_tukey_fft(input_arr, output_real, output_imag, 8)

    # Extract results using to_numpy
    fft_real = ndarray_backend_cpu.to_numpy(output_real, [8], [1], 0)
    fft_imag = ndarray_backend_cpu.to_numpy(output_imag, [8], [1], 0)

    print(f"FFT Real:  {fft_real}")
    print(f"FFT Imag:  {fft_imag}")

    # Compare with NumPy FFT (ground truth)
    numpy_fft = np.fft.fft(test_data, norm='backward')
    numpy_real = np.real(numpy_fft)
    numpy_imag = np.imag(numpy_fft)

    print(f"\nNumPy FFT:")
    print(f"  Real:    {numpy_real}")
    print(f"  Imag:    {numpy_imag}")

    # Calculate errors
    real_error = np.abs(fft_real - numpy_real).max()
    imag_error = np.abs(fft_imag - numpy_imag).max()

    print(f"\nAccuracy:")
    print(f"  Real part max error: {real_error:.2e}")
    print(f"  Imag part max error: {imag_error:.2e}")

    # Test IFFT
    print("\n" + "-" * 80)
    print("Testing IFFT (Round-trip)")
    print("-" * 80)

    ndarray_backend_cpu.cooley_tukey_ifft(output_real, output_imag, recovered, 8)

    recovered_data = ndarray_backend_cpu.to_numpy(recovered, [8], [1], 0)
    print(f"\nRecovered: {recovered_data}")
    print(f"Original:  {test_data}")

    roundtrip_error = np.abs(recovered_data - test_data).max()
    print(f"\nRound-trip max error: {roundtrip_error:.2e}")

    # Summary
    print("\n" + "=" * 80)
    print("Test Results")
    print("=" * 80)

    if real_error < 1e-5 and imag_error < 1e-5:
        print("✓ FFT accuracy: PASS")
    else:
        print(f"✗ FFT accuracy: FAIL (errors too large)")

    if roundtrip_error < 1e-5:
        print("✓ IFFT round-trip: PASS")
    else:
        print(f"✗ IFFT round-trip: FAIL (error = {roundtrip_error})")

    if real_error < 1e-5 and imag_error < 1e-5 and roundtrip_error < 1e-5:
        print("\n🎉 All tests PASSED! C++ Cooley-Tukey FFT works correctly!")
    else:
        print("\n⚠️  Some tests failed")

except Exception as e:
    print(f"\n✗ Test failed with exception: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)

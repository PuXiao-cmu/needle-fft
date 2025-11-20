import numpy as np
import os

# FFT implementation selector
# Set environment variable NEEDLE_FFT_IMPL to choose implementation:
#   "numpy" (default): Use NumPy's optimized FFT
#   "cooley_tukey": Use custom Cooley-Tukey Python implementation
#   "cpp": Use C++ Cooley-Tukey implementation (requires compilation)
FFT_IMPLEMENTATION = os.environ.get("NEEDLE_FFT_IMPL", "numpy")

__device_name__ = "numpy"
_datatype = np.float32
_datetype_size = np.dtype(_datatype).itemsize


class Array:
    def __init__(self, size):
        self.array = np.empty(size, dtype=np.float32)

    @property
    def size(self):
        return self.array.size


def to_numpy(a, shape, strides, offset):
    return np.lib.stride_tricks.as_strided(
        a.array[offset:], shape, tuple([s * _datetype_size for s in strides])
    )


def from_numpy(a, out):
    out.array[:] = a.flatten()


def fill(out, val):
    out.array.fill(val)


def compact(a, out, shape, strides, offset):
    out.array[:] = to_numpy(a, shape, strides, offset).flatten()


def ewise_setitem(a, out, shape, strides, offset):
    to_numpy(out, shape, strides, offset)[:] = a.array.reshape(shape)


def scalar_setitem(size, val, out, shape, strides, offset):
    to_numpy(out, shape, strides, offset)[:] = val


def ewise_add(a, b, out):
    out.array[:] = a.array + b.array


def scalar_add(a, val, out):
    out.array[:] = a.array + val


def ewise_mul(a, b, out):
    out.array[:] = a.array * b.array


def scalar_mul(a, val, out):
    out.array[:] = a.array * val


def ewise_div(a, b, out):
    out.array[:] = a.array / b.array


def scalar_div(a, val, out):
    out.array[:] = a.array / val


def scalar_power(a, val, out):
    out.array[:] = a.array**val


def ewise_maximum(a, b, out):
    out.array[:] = np.maximum(a.array, b.array)


def scalar_maximum(a, val, out):
    out.array[:] = np.maximum(a.array, val)


def ewise_eq(a, b, out):
    out.array[:] = (a.array == b.array).astype(np.float32)


def scalar_eq(a, val, out):
    out.array[:] = (a.array == val).astype(np.float32)


def ewise_ge(a, b, out):
    out.array[:] = (a.array >= b.array).astype(np.float32)


def scalar_ge(a, val, out):
    out.array[:] = (a.array >= val).astype(np.float32)


def ewise_log(a, out):
    out.array[:] = np.log(a.array)


def ewise_exp(a, out):
    out.array[:] = np.exp(a.array)


def ewise_tanh(a, out):
    out.array[:] = np.tanh(a.array)


def matmul(a, b, out, m, n, p):
    out.array[:] = (a.array.reshape(m, n) @ b.array.reshape(n, p)).reshape(-1)


def reduce_max(a, out, reduce_size):
    out.array[:] = a.array[:].reshape(-1, reduce_size).max(axis=1)


def reduce_sum(a, out, reduce_size):
    out.array[:] = a.array[:].reshape(-1, reduce_size).sum(axis=1)


def fft(a, out_real, out_imag, shape, axis):
    """
    Compute FFT along specified axis, returning real and imaginary parts separately.

    Implementation can be selected via NEEDLE_FFT_IMPL environment variable:
    - "numpy" (default): NumPy's optimized FFT (fast, production-ready)
    - "cooley_tukey": Custom Cooley-Tukey implementation (educational, slower)

    Args:
        a: Input Array (flattened, real-valued)
        out_real: Output Array for real part (flattened, same size as input)
        out_imag: Output Array for imaginary part (flattened, same size as input)
        shape: Shape of the array
        axis: Axis along which to compute FFT
    """
    # Reshape input to proper shape
    a_reshaped = a.array.reshape(shape)

    # Choose FFT implementation
    if FFT_IMPLEMENTATION == "cooley_tukey":
        # Import Cooley-Tukey implementation
        try:
            from .fft_cooley_tukey import cooley_tukey_fft_iterative
        except (ImportError, ValueError):
            # Fallback for direct module loading
            import sys
            module_dir = os.path.dirname(os.path.abspath(__file__))
            if module_dir not in sys.path:
                sys.path.insert(0, module_dir)
            from fft_cooley_tukey import cooley_tukey_fft_iterative

        # Apply FFT along specified axis
        fft_result = np.apply_along_axis(
            cooley_tukey_fft_iterative, axis, a_reshaped
        )
    else:
        # Use NumPy's optimized FFT (default)
        fft_result = np.fft.fft(a_reshaped, axis=axis, norm='backward')

    # Split into real and imaginary parts
    out_real.array[:] = np.real(fft_result).astype(np.float32).flatten()
    out_imag.array[:] = np.imag(fft_result).astype(np.float32).flatten()


def ifft(a_real, a_imag, out, shape, axis):
    """
    Compute IFFT along specified axis from real and imaginary parts.

    Implementation can be selected via NEEDLE_FFT_IMPL environment variable:
    - "numpy" (default): NumPy's optimized IFFT (fast, production-ready)
    - "cooley_tukey": Custom Cooley-Tukey implementation (educational, slower)

    Args:
        a_real: Input Array for real part (flattened)
        a_imag: Input Array for imaginary part (flattened)
        out: Output Array (flattened, real-valued, same size as input)
        shape: Shape of the array
        axis: Axis along which to compute IFFT
    """
    # Reshape inputs to proper shape
    a_real_reshaped = a_real.array.reshape(shape)
    a_imag_reshaped = a_imag.array.reshape(shape)

    # Reconstruct complex array
    a_complex = a_real_reshaped + 1j * a_imag_reshaped

    # Choose IFFT implementation
    if FFT_IMPLEMENTATION == "cooley_tukey":
        # Import Cooley-Tukey implementation
        try:
            from .fft_cooley_tukey import cooley_tukey_ifft_iterative
        except (ImportError, ValueError):
            # Fallback for direct module loading
            import sys
            module_dir = os.path.dirname(os.path.abspath(__file__))
            if module_dir not in sys.path:
                sys.path.insert(0, module_dir)
            from fft_cooley_tukey import cooley_tukey_ifft_iterative

        # Apply IFFT along specified axis
        ifft_result = np.apply_along_axis(
            cooley_tukey_ifft_iterative, axis, a_complex
        )
    else:
        # Use NumPy's optimized IFFT (default)
        ifft_result = np.fft.ifft(a_complex, axis=axis, norm='backward')

    # Take real part and flatten to output
    # (Imaginary part should be ~0 for real input signals)
    out.array[:] = np.real(ifft_result).astype(np.float32).flatten()

import numpy as np
import os

# FFT implementation selector
# Set environment variable NEEDLE_FFT_IMPL to choose implementation:
#   "numpy" (default): Use NumPy's optimized FFT
#   "cooley_tukey": Use custom Cooley-Tukey Python implementation
#   "cpp": Use C++ Cooley-Tukey implementation (requires compilation)
#   "cuda": Use CUDA Cooley-Tukey implementation (requires CUDA)
FFT_IMPLEMENTATION = os.environ.get("NEEDLE_FFT_IMPL", "numpy")

# Try to import C++ and CUDA backends for FFT
_cpp_backend = None
_cuda_backend = None

try:
    from . import ndarray_backend_cpu
    _cpp_backend = ndarray_backend_cpu
except ImportError:
    pass

try:
    from . import ndarray_backend_cuda
    _cuda_backend = ndarray_backend_cuda
except ImportError:
    pass

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


def _call_cpp_fft(x, backend):
    """
    Helper function to call C++ FFT backend with Bluestein support.

    Args:
        x: 1D numpy array (real-valued)
        backend: C++ backend module

    Returns:
        Complex array with FFT result
    """
    n = len(x)

    # Check if n is a power of 2
    is_pow2 = (n > 0) and ((n & (n - 1)) == 0)

    if is_pow2:
        # Direct C++ Cooley-Tukey for power of 2
        a = backend.Array(n)
        out_real = backend.Array(n)
        out_imag = backend.Array(n)

        backend.from_numpy(np.ascontiguousarray(x.astype(np.float32)), a)
        backend.cooley_tukey_fft(a, out_real, out_imag, n)

        result_real = backend.to_numpy(out_real, [n], [1], 0)
        result_imag = backend.to_numpy(out_imag, [n], [1], 0)

        result = result_real + 1j * result_imag
        return result
    else:
        # Use Bluestein algorithm for non-power-of-2 sizes
        from .fft_arbitrary_size import fft_arbitrary_1d

        # Create a wrapper that calls C++ backend for power-of-2 FFT
        def cpp_fft_wrapper(x_real, x_imag):
            n_inner = len(x_real)
            # Use the complex FFT version to handle both real and imaginary parts
            a_real = backend.Array(n_inner)
            a_imag = backend.Array(n_inner)
            out_real = backend.Array(n_inner)
            out_imag = backend.Array(n_inner)

            backend.from_numpy(np.ascontiguousarray(x_real.astype(np.float32)), a_real)
            backend.from_numpy(np.ascontiguousarray(x_imag.astype(np.float32)), a_imag)
            backend.cooley_tukey_fft_complex(a_real, a_imag, out_real, out_imag, n_inner)

            return (backend.to_numpy(out_real, [n_inner], [1], 0),
                    backend.to_numpy(out_imag, [n_inner], [1], 0))

        # Use Bluestein with C++ backend for internal FFT
        result_real, result_imag = fft_arbitrary_1d(x, None, cpp_fft_wrapper)
        return result_real + 1j * result_imag


def _call_cuda_fft(x, backend):
    """
    Helper function to call CUDA FFT backend with Bluestein support.

    Args:
        x: 1D numpy array (real-valued)
        backend: CUDA backend module

    Returns:
        Complex array with FFT result
    """
    n = len(x)

    # Check if n is a power of 2
    is_pow2 = (n > 0) and ((n & (n - 1)) == 0)

    if is_pow2:
        # Direct CUDA Cooley-Tukey for power of 2
        a = backend.CudaArray(n)
        out_real = backend.CudaArray(n)
        out_imag = backend.CudaArray(n)

        backend.from_numpy(np.ascontiguousarray(x.astype(np.float32)), a)
        backend.cooley_tukey_fft_cuda(a, out_real, out_imag, n)

        result_real = backend.to_numpy(out_real, [n], [1], 0)
        result_imag = backend.to_numpy(out_imag, [n], [1], 0)

        result = result_real + 1j * result_imag
        return result
    else:
        # Use Bluestein algorithm for non-power-of-2 sizes
        from .fft_arbitrary_size import fft_arbitrary_1d

        # Create a wrapper that calls CUDA backend for power-of-2 FFT
        def cuda_fft_wrapper(x_real, x_imag):
            n_inner = len(x_real)
            # Note: x_imag is ignored for real input, but required by Bluestein interface
            a = backend.CudaArray(n_inner)
            out_real = backend.CudaArray(n_inner)
            out_imag = backend.CudaArray(n_inner)

            backend.from_numpy(np.ascontiguousarray(x_real.astype(np.float32)), a)
            backend.cooley_tukey_fft_cuda(a, out_real, out_imag, n_inner)

            return (backend.to_numpy(out_real, [n_inner], [1], 0),
                    backend.to_numpy(out_imag, [n_inner], [1], 0))

        # Use Bluestein with CUDA backend for internal FFT
        result_real, result_imag = fft_arbitrary_1d(x, None, cuda_fft_wrapper)
        return result_real + 1j * result_imag


def fft(a, out_real, out_imag, shape, axis):
    """
    Compute FFT along specified axis, returning real and imaginary parts separately.

    Implementation can be selected via NEEDLE_FFT_IMPL environment variable:
    - "numpy" (default): NumPy's optimized FFT (fast, production-ready)
    - "cooley_tukey": Custom Cooley-Tukey Python implementation (educational, slower)
    - "cpp": C++ Cooley-Tukey implementation (fast, requires compilation)
    - "cuda": CUDA Cooley-Tukey implementation (GPU, requires CUDA)

    Args:
        a: Input Array (flattened, real-valued)
        out_real: Output Array for real part (flattened, same size as input)
        out_imag: Output Array for imaginary part (flattened, same size as input)
        shape: Shape of the array
        axis: Axis along which to compute FFT
    """
    # Debug: Print which implementation is being used (only once per process)
    global _fft_impl_logged
    if '_fft_impl_logged' not in globals():
        _fft_impl_logged = True
        print(f"[FFT Backend] Using NEEDLE_FFT_IMPL={FFT_IMPLEMENTATION}")

    # Reshape input to proper shape
    a_reshaped = a.array.reshape(shape)

    # Choose FFT implementation
    if FFT_IMPLEMENTATION == "cooley_tukey":
        # Python Cooley-Tukey implementation
        if '_fft_impl_logged' not in globals() or not globals()['_fft_impl_logged']:
            print(f"[FFT Backend] → Calling Python Cooley-Tukey implementation (with Bluestein for N≠2^k)")
        try:
            from .fft_cooley_tukey import cooley_tukey_fft_iterative
        except (ImportError, ValueError):
            # Fallback for direct module loading
            import sys
            module_dir = os.path.dirname(os.path.abspath(__file__))
            if module_dir not in sys.path:
                sys.path.insert(0, module_dir)
            from fft_cooley_tukey import cooley_tukey_fft_iterative

        # Create a wrapper that uses Bluestein for non-power-of-2 sizes
        def python_ct_with_bluestein(x):
            n = len(x)
            is_pow2 = (n > 0) and ((n & (n - 1)) == 0)

            if is_pow2:
                # Direct Cooley-Tukey for power of 2
                return cooley_tukey_fft_iterative(x)
            else:
                # Use Bluestein algorithm for non-power-of-2 sizes
                from .fft_arbitrary_size import fft_arbitrary_1d

                # Wrapper that calls Python Cooley-Tukey for internal power-of-2 FFTs
                def python_ct_fft_wrapper(x_real, x_imag):
                    n_inner = len(x_real)
                    # Combine real and imaginary parts
                    x_complex = x_real + 1j * x_imag
                    result = cooley_tukey_fft_iterative(x_complex)
                    return np.real(result), np.imag(result)

                result_real, result_imag = fft_arbitrary_1d(x, None, python_ct_fft_wrapper)
                return result_real + 1j * result_imag

        # Apply FFT along specified axis
        fft_result = np.apply_along_axis(
            python_ct_with_bluestein, axis, a_reshaped
        )
    elif FFT_IMPLEMENTATION == "cpp":
        # C++ Cooley-Tukey implementation
        if '_fft_impl_logged' not in globals() or not globals()['_fft_impl_logged']:
            print(f"[FFT Backend] → Calling C++ Cooley-Tukey implementation (with Bluestein for N≠2^k)")
        if _cpp_backend is None:
            raise ImportError("C++ backend not available. Please compile it with 'make'.")

        # Apply FFT along specified axis using C++ backend
        fft_result = np.apply_along_axis(
            lambda x: _call_cpp_fft(x, _cpp_backend), axis, a_reshaped
        )
    elif FFT_IMPLEMENTATION == "cuda":
        # CUDA Cooley-Tukey implementation
        if '_fft_impl_logged' not in globals() or not globals()['_fft_impl_logged']:
            print(f"[FFT Backend] → Calling CUDA Cooley-Tukey implementation")
        if _cuda_backend is None:
            raise ImportError("CUDA backend not available. Please compile it with CUDA support.")

        # Apply FFT along specified axis using CUDA backend
        fft_result = np.apply_along_axis(
            lambda x: _call_cuda_fft(x, _cuda_backend), axis, a_reshaped
        )
    else:
        # Use NumPy's optimized FFT (default)
        if '_fft_impl_logged' not in globals() or not globals()['_fft_impl_logged']:
            print(f"[FFT Backend] → Calling NumPy FFT (np.fft.fft)")
        fft_result = np.fft.fft(a_reshaped, axis=axis, norm='backward')

    # Split into real and imaginary parts
    out_real.array[:] = np.real(fft_result).astype(np.float32).flatten()
    out_imag.array[:] = np.imag(fft_result).astype(np.float32).flatten()


def _call_cpp_ifft(x_real, x_imag, backend):
    """
    Helper function to call C++ IFFT backend with Bluestein support.

    Args:
        x_real: 1D numpy array (real part)
        x_imag: 1D numpy array (imaginary part)
        backend: C++ backend module

    Returns:
        Real-valued array with IFFT result
    """
    n = len(x_real)

    # Check if n is a power of 2
    is_pow2 = (n > 0) and ((n & (n - 1)) == 0)

    if is_pow2:
        # Direct C++ Cooley-Tukey IFFT for power of 2
        a_real = backend.Array(n)
        a_imag = backend.Array(n)
        out = backend.Array(n)

        backend.from_numpy(np.ascontiguousarray(x_real.astype(np.float32)), a_real)
        backend.from_numpy(np.ascontiguousarray(x_imag.astype(np.float32)), a_imag)

        backend.cooley_tukey_ifft(a_real, a_imag, out, n)

        result = backend.to_numpy(out, [n], [1], 0)
        return result
    else:
        # Use Bluestein algorithm for non-power-of-2 sizes
        from .fft_arbitrary_size import ifft_arbitrary_1d

        # Create a wrapper that calls C++ backend for power-of-2 FFT
        def cpp_fft_wrapper(x_real_inner, x_imag_inner):
            n_inner = len(x_real_inner)
            # Use the complex FFT version
            a_real = backend.Array(n_inner)
            a_imag = backend.Array(n_inner)
            out_real = backend.Array(n_inner)
            out_imag = backend.Array(n_inner)

            backend.from_numpy(np.ascontiguousarray(x_real_inner.astype(np.float32)), a_real)
            backend.from_numpy(np.ascontiguousarray(x_imag_inner.astype(np.float32)), a_imag)
            backend.cooley_tukey_fft_complex(a_real, a_imag, out_real, out_imag, n_inner)

            return (backend.to_numpy(out_real, [n_inner], [1], 0),
                    backend.to_numpy(out_imag, [n_inner], [1], 0))

        # Use Bluestein IFFT with C++ backend for internal FFT
        result_real, result_imag = ifft_arbitrary_1d(x_real, x_imag, cpp_fft_wrapper)
        # IFFT should return real values (imaginary part ~0)
        return result_real


def _call_cuda_ifft(x_real, x_imag, backend):
    """
    Helper function to call CUDA IFFT backend with Bluestein support.

    Args:
        x_real: 1D numpy array (real part)
        x_imag: 1D numpy array (imaginary part)
        backend: CUDA backend module

    Returns:
        Real-valued array with IFFT result
    """
    n = len(x_real)

    # Check if n is a power of 2
    is_pow2 = (n > 0) and ((n & (n - 1)) == 0)

    if is_pow2:
        # Direct CUDA Cooley-Tukey IFFT for power of 2
        a_real = backend.CudaArray(n)
        a_imag = backend.CudaArray(n)
        out = backend.CudaArray(n)

        backend.from_numpy(np.ascontiguousarray(x_real.astype(np.float32)), a_real)
        backend.from_numpy(np.ascontiguousarray(x_imag.astype(np.float32)), a_imag)

        backend.cooley_tukey_ifft_cuda(a_real, a_imag, out, n)

        result = backend.to_numpy(out, [n], [1], 0)
        return result
    else:
        # Use Bluestein algorithm for non-power-of-2 sizes
        from .fft_arbitrary_size import ifft_arbitrary_1d

        # Create a wrapper that calls CUDA backend for power-of-2 FFT
        def cuda_fft_wrapper(x_real_inner, x_imag_inner):
            n_inner = len(x_real_inner)
            a = backend.CudaArray(n_inner)
            out_real = backend.CudaArray(n_inner)
            out_imag = backend.CudaArray(n_inner)

            backend.from_numpy(np.ascontiguousarray(x_real_inner.astype(np.float32)), a)
            backend.cooley_tukey_fft_cuda(a, out_real, out_imag, n_inner)

            return (backend.to_numpy(out_real, [n_inner], [1], 0),
                    backend.to_numpy(out_imag, [n_inner], [1], 0))

        # Use Bluestein IFFT with CUDA backend for internal FFT
        result_real, result_imag = ifft_arbitrary_1d(x_real, x_imag, cuda_fft_wrapper)
        # IFFT should return real values (imaginary part ~0)
        return result_real


def ifft(a_real, a_imag, out, shape, axis):
    """
    Compute IFFT along specified axis from real and imaginary parts.

    Implementation can be selected via NEEDLE_FFT_IMPL environment variable:
    - "numpy" (default): NumPy's optimized IFFT (fast, production-ready)
    - "cooley_tukey": Custom Cooley-Tukey Python implementation (educational, slower)
    - "cpp": C++ Cooley-Tukey implementation (fast, requires compilation)
    - "cuda": CUDA Cooley-Tukey implementation (GPU, requires CUDA)

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

    # Choose IFFT implementation
    if FFT_IMPLEMENTATION == "cooley_tukey":
        # Python Cooley-Tukey implementation
        try:
            from .fft_cooley_tukey import cooley_tukey_ifft_iterative
        except (ImportError, ValueError):
            # Fallback for direct module loading
            import sys
            module_dir = os.path.dirname(os.path.abspath(__file__))
            if module_dir not in sys.path:
                sys.path.insert(0, module_dir)
            from fft_cooley_tukey import cooley_tukey_ifft_iterative

        # Reconstruct complex array
        a_complex = a_real_reshaped + 1j * a_imag_reshaped

        # Apply IFFT along specified axis
        ifft_result = np.apply_along_axis(
            cooley_tukey_ifft_iterative, axis, a_complex
        )
    elif FFT_IMPLEMENTATION == "cpp":
        # C++ Cooley-Tukey implementation
        if _cpp_backend is None:
            raise ImportError("C++ backend not available. Please compile it with 'make'.")

        # Apply IFFT along specified axis using C++ backend
        ifft_result = _apply_ifft_along_axis_cpp(a_real_reshaped, a_imag_reshaped, axis, _cpp_backend)
    elif FFT_IMPLEMENTATION == "cuda":
        # CUDA Cooley-Tukey implementation
        if _cuda_backend is None:
            raise ImportError("CUDA backend not available. Please compile it with CUDA support.")

        # Apply IFFT along specified axis using CUDA backend
        ifft_result = _apply_ifft_along_axis_cuda(a_real_reshaped, a_imag_reshaped, axis, _cuda_backend)
    else:
        # Use NumPy's optimized IFFT (default)
        # Reconstruct complex array
        a_complex = a_real_reshaped + 1j * a_imag_reshaped
        ifft_result = np.fft.ifft(a_complex, axis=axis, norm='backward')

    # Take real part and flatten to output
    # (Imaginary part should be ~0 for real input signals)
    out.array[:] = np.real(ifft_result).astype(np.float32).flatten()


def _apply_ifft_along_axis_cpp(a_real, a_imag, axis, backend):
    """Apply C++ IFFT along a specific axis of multi-dimensional arrays."""
    # Move the target axis to the end
    a_real_moved = np.moveaxis(a_real, axis, -1)
    a_imag_moved = np.moveaxis(a_imag, axis, -1)

    # Reshape to (batch, n)
    original_shape = a_real_moved.shape
    n = original_shape[-1]
    a_real_flat = a_real_moved.reshape(-1, n)
    a_imag_flat = a_imag_moved.reshape(-1, n)

    # Apply IFFT to each batch element
    result_flat = np.empty_like(a_real_flat)
    for i in range(a_real_flat.shape[0]):
        result_flat[i] = _call_cpp_ifft(a_real_flat[i], a_imag_flat[i], backend)

    # Reshape back
    result = result_flat.reshape(original_shape)

    # Move axis back to original position
    result = np.moveaxis(result, -1, axis)

    return result


def _apply_ifft_along_axis_cuda(a_real, a_imag, axis, backend):
    """Apply CUDA IFFT along a specific axis of multi-dimensional arrays."""
    # Move the target axis to the end
    a_real_moved = np.moveaxis(a_real, axis, -1)
    a_imag_moved = np.moveaxis(a_imag, axis, -1)

    # Reshape to (batch, n)
    original_shape = a_real_moved.shape
    n = original_shape[-1]
    a_real_flat = a_real_moved.reshape(-1, n)
    a_imag_flat = a_imag_moved.reshape(-1, n)

    # Apply IFFT to each batch element
    result_flat = np.empty_like(a_real_flat)
    for i in range(a_real_flat.shape[0]):
        result_flat[i] = _call_cuda_ifft(a_real_flat[i], a_imag_flat[i], backend)

    # Reshape back
    result = result_flat.reshape(original_shape)

    # Move axis back to original position
    result = np.moveaxis(result, -1, axis)

    return result

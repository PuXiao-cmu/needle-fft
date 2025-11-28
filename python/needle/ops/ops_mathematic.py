"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a ** b
        
    def gradient(self, out_grad, node):
        x, y = node.inputs
        # d(x^y)/dx = y * x^(y-1)
        gx = out_grad * (y * (x ** (y - 1)))
        # d(x^y)/dy = x^y * log(x)
        gy = out_grad * ((x ** y) * log(x))
        return gx, gy

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad, node):
        x, = node.inputs
        k = self.scalar
        # d(x^k)/dx = k * x^(k-1)
        return out_grad * (k * (x ** (k - 1))),


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        return out_grad / b, (-out_grad * a) / (b * b)


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return (out_grad / self.scalar),


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes=None):
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
            # swap last two axes
            i, j = a.ndim - 2, a.ndim - 1
            order = list(range(a.ndim))
            order[i], order[j] = order[j], order[i]
            return a.permute(tuple(order))

        axes = tuple(self.axes)
        if len(axes) == 2:
            # swap exactly these two axes
            i, j = axes
            order = list(range(a.ndim))
            order[i], order[j] = order[j], order[i]
            return a.permute(tuple(order))
        elif len(axes) == a.ndim:
            # full permutation provided
            return a.permute(axes)
        else:
            raise AssertionError("transpose: axes must be length 2 (swap) or equal to ndim (permute)")

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        if self.axes is None:
            # swap last two back
            i, j = len(a.shape) - 2, len(a.shape) - 1
            inv = list(range(len(a.shape)))
            inv[i], inv[j] = inv[j], inv[i]
            return transpose(out_grad, tuple(inv))

        axes = tuple(self.axes)
        if len(axes) == 2:
            # swap back the same two
            i, j = axes
            inv = list(range(len(a.shape)))
            inv[i], inv[j] = inv[j], inv[i]
            return transpose(out_grad, tuple(inv))
        elif len(axes) == len(a.shape):
            # inverse permutation
            inv = [0] * len(a.shape)
            for i, ax in enumerate(axes):
                inv[ax] = i
            return transpose(out_grad, tuple(inv))
        else:
            raise AssertionError("transpose: axes must be length 2 (swap) or equal to ndim (permute)")


def transpose(a: Tensor, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.compact().reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        in_shape = node.inputs[0].shape
        return reshape(out_grad, in_shape)
        ### END YOUR SOLUTION



def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.broadcast_to(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        in_shape = node.inputs[0].shape
        out_shape = self.shape
        nd_in, nd_out = len(in_shape), len(out_shape)
        lead = nd_out - nd_in
        axes_to_sum = list(range(lead))
        for i in range(nd_in):
            if in_shape[i] == 1 and out_shape[lead + i] != 1:
                axes_to_sum.append(lead + i)
        if axes_to_sum:
            g = summation(out_grad, axes=tuple(sorted(axes_to_sum)))
        else:
            g = out_grad
        return reshape(g, in_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            return a.sum()
        if isinstance(self.axes, int):
            return a.sum(axis=self.axes)
        res = a
        for ax in sorted(self.axes, reverse=True):
            res = res.sum(axis=ax)
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        in_shape = node.inputs[0].shape
        if self.axes is None:
            return broadcast_to(reshape(out_grad, (1,) * len(in_shape)), in_shape)

        if isinstance(self.axes, int):
            axes = (self.axes,)
        else:
            axes = tuple(self.axes)

        shape = list(in_shape)
        for ax in axes:
            shape[ax] = 1
        return broadcast_to(reshape(out_grad, tuple(shape)), in_shape)
        ### END YOUR SOLUTION



def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a @ b

    def gradient(self, out_grad, node):
        a, b = node.inputs

        # Basic matrix calculus:
        # dL/da = dL/dout @ b^T
        # dL/db = a^T @ dL/dout
        grad_a = out_grad @ transpose(b)
        grad_b = transpose(a) @ out_grad

        # Helper: reduce extra broadcasted dimensions
        def reduce_like(g, target_shape):
            # Sum out extra leading dimensions
            while len(g.shape) > len(target_shape):
                g = summation(g, axes=0)

            # For each axis where the target had size 1
            # but the gradient axis is >1, sum over that axis
            for i in range(len(target_shape)):
                if target_shape[i] == 1 and g.shape[i] != 1:
                    g = summation(g, axes=i)

            # Ensure final shape matches exactly
            if g.shape != target_shape:
                g = reshape(g, target_shape)
            return g

        # Apply reduction to match original input shapes
        grad_a = reduce_like(grad_a, a.shape)
        grad_b = reduce_like(grad_b, b.shape)
        return grad_a, grad_b


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return (-out_grad),


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        x, = node.inputs
        return out_grad / x,


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0]),


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
      x_data = node.inputs[0].realize_cached_data()
      return out_grad * (x_data > 0)


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        # Use array_api's tanh function
        return array_api.tanh(a)

    def gradient(self, out_grad, node):
        # d(tanh(x))/dx = 1 - tanh^2(x)
        # node is the output tanh(x), so we can use it directly
        return out_grad * (-node ** 2 + 1)


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        assert len(args) > 0
        arrs = list(args)
        base_shape = arrs[0].shape
        for a in arrs:
            assert a.shape == base_shape
        axis = self.axis if self.axis >= 0 else self.axis + len(base_shape) + 1
        out_shape = base_shape[:axis] + (len(arrs),) + base_shape[axis:]
        out = array_api.empty(out_shape, device=arrs[0].device)
        for i, a in enumerate(arrs):
            idx = [slice(None)] * axis + [i] + [slice(None)] * (len(base_shape) - axis)
            out[tuple(idx)] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        axis = self.axis if self.axis >= 0 else self.axis + A.ndim
        n = A.shape[axis]
        before = A.shape[:axis]
        after = A.shape[axis + 1 :]
        out = []
        for i in range(n):
            idx = [slice(None)] * axis + [i] + [slice(None)] * (A.ndim - axis - 1)
            out.append(A[tuple(idx)].compact().reshape(before + after))
        return tuple(out)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        parts = [out_grad[i] for i in range(len(out_grad))]
        return stack(parts, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)

class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a: NDArray):
        return a.flip(self.axes)

    def gradient(self, out_grad, node):
        return flip(out_grad, self.axes)


def flip(a, axes):
    return Flip(axes)(a)

class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = int(dilation)

    def compute(self, a: NDArray):
        d = self.dilation
        axes = self.axes if isinstance(self.axes, tuple) else (self.axes,)
        nd = len(a.shape)
        ax_set = set(ax if ax >= 0 else ax + nd for ax in axes)

        new_shape = list(a.shape)
        for i in range(nd):
            if i in ax_set:
                new_shape[i] = a.shape[i] * (d + 1)

        out = type(a).make(tuple(new_shape), device=a.device)
        out.fill(0.0)

        slices = []
        for i in range(nd):
            step = (d + 1) if i in ax_set else 1
            slices.append(slice(0, new_shape[i], step))
        out[tuple(slices)] = a
        return out

    def gradient(self, out_grad, node):
        return undilate(out_grad, self.axes, self.dilation)

def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = int(dilation)

    def compute(self, a: NDArray):
        d = self.dilation
        axes = self.axes if isinstance(self.axes, tuple) else (self.axes,)
        nd = len(a.shape)
        ax_set = set(ax if ax >= 0 else ax + nd for ax in axes)

        slices = []
        for i in range(nd):
            step = (d + 1) if i in ax_set else 1
            slices.append(slice(0, a.shape[i], step))
        view = a[tuple(slices)]
        return view.compact()

    def gradient(self, out_grad, node):
        return dilate(out_grad, self.axes, self.dilation)

def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)

class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        """
        A: (N, H, W, Cin)  NHWC
        B: (K, K, Cin, Cout)
        return: (N, Hout, Wout, Cout)
        """
        N, H, W, Cin = A.shape
        K, K2, Cin_b, Cout = B.shape
        assert K == K2 and Cin == Cin_b, "conv weight shape must be (K,K,Cin,Cout) matching input Cin"

        p = int(self.padding) if self.padding is not None else 0
        s = int(self.stride) if self.stride is not None else 1

        if p > 0:
            A_pad = A.pad(((0, 0), (p, p), (p, p), (0, 0)))
        else:
            A_pad = A

        A_pad = A_pad.compact()
        sN, sH, sW, sC = A_pad.strides
        Hp, Wp = A_pad.shape[1], A_pad.shape[2]

        Hout = (Hp - K) // s + 1
        Wout = (Wp - K) // s + 1
        assert Hout > 0 and Wout > 0, "Invalid conv: (H+2p-K)/s+1 and (W+2p-K)/s+1 must be positive"

        patches = A_pad.as_strided(
            (N, Hout, Wout, K, K, Cin),
            (sN, s * sH, s * sW, sH, sW, sC),
        )

        Xcol = patches.compact().reshape((N * Hout * Wout, K * K * Cin))

        Wcol = B.compact().reshape((K * K * Cin, Cout))

        out2d = Xcol @ Wcol
        out = out2d.reshape((N, Hout, Wout, Cout))
        return out

    def gradient(self, out_grad, node):
        """
        out_grad: (N, Hout, Wout, Cout)
        X: node.inputs[0] -> (N, H, W, Cin)
        W: node.inputs[1] -> (K, K, Cin, Cout)
        """
        X, W = node.inputs
        s = int(self.stride) if self.stride is not None else 1
        p = int(self.padding) if self.padding is not None else 0
        K = W.shape[0]
        Cin = W.shape[2]
        Cout = W.shape[3]

        g = out_grad
        if s > 1:
            g = dilate(g, axes=(1, 2), dilation=s - 1)  # (N, H', W', Cout)

        W_flip_T = transpose(flip(W, axes=(0, 1)))     # (K, K, Cout, Cin)

        pad_x = K - 1 - p
        dx = conv(g, W_flip_T, stride=1, padding=pad_x)

        X_nd = X.realize_cached_data()
        if p > 0:
            X_pad = X_nd.pad(((0, 0), (p, p), (p, p), (0, 0)))
        else:
            X_pad = X_nd

        X_pad = X_pad.compact()
        N, Hp, Wp, Cin_check = X_pad.shape
        assert Cin_check == Cin

        sN, sH, sW, sC = X_pad.strides
        Hout = (Hp - K) // s + 1
        Wout = (Wp - K) // s + 1

        patches = X_pad.as_strided(
            (N, Hout, Wout, K, K, Cin),
            (sN, s * sH, s * sW, sH, sW, sC),
        )

        Xcol = patches.compact().reshape((N * Hout * Wout, K * K * Cin))

        dY_nd = out_grad.realize_cached_data()
        dYcol = dY_nd.compact().reshape((N * Hout * Wout, Cout))

        Xcol_T = Xcol.permute((1, 0)).compact()
        dWcol = Xcol_T @ dYcol

        dW_nd = dWcol.reshape((K, K, Cin, Cout))

        dW = Tensor.make_const(dW_nd)

        return dx, dW


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


class FFT(TensorTupleOp):
    def __init__(self, dim: int = -1, norm: str = "backward"):
        """
        Fast Fourier Transform operator that returns real and imaginary parts.

        Parameters:
        -----------
        dim : int
            Dimension along which to compute FFT (default: -1, last dimension)
        norm : str
            Normalization mode: "backward" (default), "forward", or "ortho"
            - "backward": no normalization on forward FFT, 1/n on inverse
            - "forward": 1/n normalization on forward FFT, no normalization on inverse
            - "ortho": 1/sqrt(n) normalization on both forward and inverse
        """
        self.dim = dim
        self.norm = norm

    def compute(self, a: NDArray) -> tuple[NDArray, NDArray]:
        """
        Forward FFT computation using backend API.

        Returns:
            Tuple of (real_part, imag_part)
        """
        # Check if device supports FFT
        if hasattr(a, 'device'):
            device_name = getattr(a.device, 'name', 'unknown')
            if device_name not in ['cpu_numpy', 'numpy', 'cpu', 'cuda']:
                raise NotImplementedError(
                    f"FFT is not implemented for device '{device_name}'. "
                    f"Currently only 'cpu_numpy' backend is supported. "
                    f"Please use: needle.Tensor(data, device=needle.cpu_numpy())"
                )

        # Normalize axis
        axis = self.dim if self.dim >= 0 else len(a.shape) + self.dim

        # Call backend FFT (returns real and imaginary parts)
        result_real, result_imag = array_api.fft(a, axis=axis)

        # Handle different normalization modes
        if self.norm == "forward":
            # Apply 1/n normalization
            n = a.shape[axis]
            result_real = result_real / n
            result_imag = result_imag / n
        elif self.norm == "ortho":
            # Apply 1/sqrt(n) normalization
            n = a.shape[axis]
            import math
            scale = math.sqrt(n)
            result_real = result_real / scale
            result_imag = result_imag / scale
        # For "backward" mode, no additional normalization needed

        return result_real, result_imag

    def gradient(self, out_grad: TensorTuple, node: Tensor) -> Tuple[Tensor]:
        """
        Backward pass for FFT.

        The gradient of FFT is IFFT (with appropriate normalization).
        out_grad is a TensorTuple containing (grad_real, grad_imag)
        """
        # Extract gradients for real and imaginary parts
        grad_real = out_grad[0]
        grad_imag = out_grad[1]

        # The gradient computation depends on the normalization mode
        if self.norm == "backward":
            # Forward FFT has no normalization, so gradient is IFFT with 1/n
            return (ifft(grad_real, grad_imag, self.dim, norm="forward"),)
        elif self.norm == "forward":
            # Forward FFT has 1/n normalization
            return (ifft(grad_real, grad_imag, self.dim, norm="backward"),)
        elif self.norm == "ortho":
            # Symmetric normalization
            return (ifft(grad_real, grad_imag, self.dim, norm="ortho"),)
        else:
            raise ValueError(f"Unknown normalization mode: {self.norm}")


def fft(a, dim=-1, norm="backward"):
    """
    Compute the one-dimensional discrete Fourier Transform.

    Parameters:
    -----------
    a : Tensor
        Input tensor (real-valued)
    dim : int
        Dimension along which to compute FFT (default: -1)
    norm : str
        Normalization mode (default: "backward")

    Returns:
    --------
    TensorTuple
        Tuple of (real_part, imag_part) of the FFT
    """
    return FFT(dim, norm)(a)


class IFFT(TensorOp):
    def __init__(self, dim: int = -1, norm: str = "backward"):
        """
        Inverse Fast Fourier Transform operator.

        Parameters:
        -----------
        dim : int
            Dimension along which to compute IFFT (default: -1)
        norm : str
            Normalization mode (default: "backward")
        """
        self.dim = dim
        self.norm = norm

    def compute(self, a_real: NDArray, a_imag: NDArray) -> NDArray:
        """
        Forward IFFT computation using backend API.

        Args:
            a_real: Real part of complex input
            a_imag: Imaginary part of complex input

        Returns:
            Real-valued result of IFFT
        """
        # Check if device supports IFFT
        if hasattr(a_real, 'device'):
            device_name = getattr(a_real.device, 'name', 'unknown')
            if device_name not in ['cpu_numpy', 'numpy']:
                raise NotImplementedError(
                    f"IFFT is not implemented for device '{device_name}'. "
                    f"Currently only 'cpu_numpy' backend is supported. "
                    f"Please use: needle.Tensor(data, device=needle.cpu_numpy())"
                )

        # Normalize axis
        axis = self.dim if self.dim >= 0 else len(a_real.shape) + self.dim

        # Call backend IFFT (takes real and imaginary parts)
        result = array_api.ifft(a_real, a_imag, axis=axis)

        # Handle different normalization modes
        # Backend IFFT uses "backward" norm by default (1/n normalization)
        if self.norm == "forward":
            # Remove the 1/n normalization (multiply by n)
            n = a_real.shape[axis]
            result = result * n
        elif self.norm == "ortho":
            # Adjust to 1/sqrt(n) normalization
            n = a_real.shape[axis]
            import math
            result = result * math.sqrt(n)
        # For "backward" mode, no additional adjustment needed

        return result

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Backward pass for IFFT.

        The gradient of IFFT is FFT (with appropriate normalization).
        Returns gradients for both real and imaginary inputs.
        """
        # The gradient computation depends on the normalization mode
        if self.norm == "backward":
            # IFFT has 1/n normalization, so gradient is FFT with forward norm
            grad_fft = fft(out_grad, self.dim, norm="forward")
        elif self.norm == "forward":
            # IFFT has no normalization
            grad_fft = fft(out_grad, self.dim, norm="backward")
        elif self.norm == "ortho":
            # Symmetric normalization
            grad_fft = fft(out_grad, self.dim, norm="ortho")
        else:
            raise ValueError(f"Unknown normalization mode: {self.norm}")

        # grad_fft is a TensorTuple containing (grad_real, grad_imag)
        return tuple_get_item(grad_fft, 0), tuple_get_item(grad_fft, 1)


def ifft(a_real, a_imag, dim=-1, norm="backward"):
    """
    Compute the one-dimensional inverse discrete Fourier Transform.

    Parameters:
    -----------
    a_real : Tensor
        Real part of complex input
    a_imag : Tensor
        Imaginary part of complex input
    dim : int
        Dimension along which to compute IFFT (default: -1)
    norm : str
        Normalization mode (default: "backward")

    Returns:
    --------
    Tensor
        IFFT of the input along the specified dimension (real-valued)
    """
    return IFFT(dim, norm)(a_real, a_imag)


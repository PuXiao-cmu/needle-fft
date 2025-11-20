from typing import Optional
from ..autograd import NDArray, Tensor, TensorOp
from .ops_mathematic import *
from ..backend_selection import BACKEND


class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        last = Z.ndim - 1
        m = Z.max(axis=last, keepdims=True)
        shifted = Z - m.broadcast_to(Z.shape)
        lse = shifted.exp().sum(axis=last, keepdims=True).log() + m
        return Z - lse.broadcast_to(Z.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        # dL/dZ = dL/dY - softmax(Z) * sum(dL/dY, axis=last, keepdims=True)
        Y = node
        sm = exp(Y)  # softmax
        last = Y.ndim - 1
        s = summation(out_grad, axes=(last,))
        s = reshape(s, tuple(list(out_grad.shape[:-1]) + [1]))
        return out_grad - sm * s


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes
    def compute(self, Z: NDArray) -> NDArray:
        axes = self.axes
        if axes is None:
            m = Z.max(axis=None, keepdims=True)              # (1,)
            shifted = Z - m.broadcast_to(Z.shape)
            out = shifted.exp().sum(axis=None, keepdims=True).log() + m
            return out                                       # (1,)

        if isinstance(axes, int):
            axes = (axes,)

        m = Z
        for ax in sorted(axes, reverse=True):
            m = m.max(axis=ax, keepdims=True)

        shifted = Z - m.broadcast_to(Z.shape)
        s = shifted.exp()
        for ax in sorted(axes, reverse=True):
            s = s.sum(axis=ax, keepdims=True)
        out = s.log() + m

        out_shape = tuple(d for i, d in enumerate(Z.shape) if i not in set(axes))
        return out.compact().reshape(out_shape if out_shape else (1,))

    def gradient(self, out_grad: Tensor, node: Tensor):
        Z = node.inputs[0]
        axes = self.axes
        if axes is None:
            lse = reshape(node, (1,) * len(Z.shape))
            coeff = broadcast_to(reshape(out_grad, (1,)), Z.shape)
        else:
            if isinstance(axes, int):
                axes = (axes,)
            bshape = list(Z.shape)
            for ax in axes:
                bshape[ax] = 1
            lse = reshape(node, tuple(bshape))
            coeff = broadcast_to(reshape(out_grad, tuple(bshape)), Z.shape)

        return coeff * exp(Z - broadcast_to(lse, Z.shape))


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)

"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
import math


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        W = init.kaiming_uniform(in_features, out_features, nonlinearity="relu",
                                 device=device, dtype=dtype)

        self.weight = Parameter(W.numpy(), device=device, dtype=dtype, requires_grad=True)

        self.bias = None
        if bias:
            gain = math.sqrt(2.0)
            bound = gain * math.sqrt(3.0 / out_features)
            b = init.rand(1, out_features, low=-bound, high=bound, device=device, dtype=dtype)
            self.bias = Parameter(b.numpy(), device=device, dtype=dtype, requires_grad=True)

    def forward(self, X: Tensor) -> Tensor:
        # X: (N, in), weight: (in, out)  ->  (N, out)
        Y = X @ self.weight
        if self.bias is not None:
            out_shape = tuple(list(X.shape[:-1]) + [self.out_features])
            b = ops.broadcast_to(self.bias, out_shape)
            Y = Y + b
        return Y


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        shape = X.shape
        N = shape[0]
        if len(shape) == 2:
            return X
        D = 1
        for s in shape[1:]:
            D *= s
        return ops.reshape(X, (N, D))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        if hasattr(ops, "relu"):
            return ops.relu(x)
        return ops.maximum(x, 0.0)

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for m in self.modules:
            x = m(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        N, C = logits.shape
        lse = ops.logsumexp(logits, axes=(1,))
        Y = init.one_hot(C, y, device=logits.device, dtype=logits.dtype)
        true_logit = ops.summation(logits * Y, axes=(1,))
        loss_vec = lse - true_logit
        return ops.summation(loss_vec, axes=(0,)) / N


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        g = init.ones(dim, device=device, dtype=dtype)
        b = init.zeros(dim, device=device, dtype=dtype)
        self.weight = Parameter(g.numpy(), device=device, dtype=dtype, requires_grad=True)
        self.bias   = Parameter(b.numpy(), device=device, dtype=dtype, requires_grad=True)

        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var  = init.ones(dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        N, D = x.shape
        assert D == self.dim

        ones_row = init.ones(1, D, device=x.device, dtype=x.dtype)
        ones_col = init.ones(N, 1, device=x.device, dtype=x.dtype)

        if self.training:
            mean_vec = ops.summation(x, axes=(0,)) / N               # (D,)
            mean_full = ones_col @ ops.reshape(mean_vec, (1, D))      # (N,D)
            xc = x - mean_full                                       # (N,D)
            var_vec = ops.summation(xc * xc, axes=(0,)) / N          # (D,)

            denom_full = ones_col @ ops.reshape(
                ops.power_scalar(var_vec + self.eps, 0.5), (1, D)
            )                                                        # (N,D)
            x_hat = xc / denom_full                                  # (N,D)

            m = self.momentum
            self.running_mean.data = ((1 - m) * self.running_mean + m * mean_vec).data
            self.running_var.data  = ((1 - m) * self.running_var  + m * var_vec ).data

        else:
            mean_full = ones_col @ ops.reshape(self.running_mean, (1, D))   # (N,D)
            xc = x - mean_full
            denom_full = ones_col @ ops.reshape(
                ops.power_scalar(self.running_var + self.eps, 0.5), (1, D)
            )
            x_hat = xc / denom_full

        gamma_full = ones_col @ ops.reshape(self.weight, (1, D))     # (N,D)
        beta_full  = ones_col @ ops.reshape(self.bias,   (1, D))     # (N,D)
        return x_hat * gamma_full + beta_full



class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        g = init.ones(dim, device=device, dtype=dtype)
        b = init.zeros(dim, device=device, dtype=dtype)
        self.weight = Parameter(g.numpy(), device=device, dtype=dtype, requires_grad=True)
        self.bias   = Parameter(b.numpy(), device=device, dtype=dtype, requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        N, D = x.shape
        mean = ops.summation(x, axes=(1,)) / D
        mean = ops.reshape(mean, (N, 1))
        xc = x - ops.broadcast_to(mean, x.shape)

        var = ops.summation(xc * xc, axes=(1,)) / D
        var = ops.reshape(var, (N, 1))

        denom = ops.power_scalar(var + self.eps, 0.5)
        x_hat = xc / ops.broadcast_to(denom, x.shape)

        gamma = ops.broadcast_to(self.weight, (N, D))
        beta  = ops.broadcast_to(self.bias, (N, D))
        return x_hat * gamma + beta


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if (not self.training) or self.p == 0.0:
            return x

        if self.p == 1.0:
            return x * 0.0

        keep = 1.0 - self.p
        mask = init.randb(*x.shape, p=keep, device=x.device, dtype=x.dtype)  # 0/1 float
        return x * (mask / keep)


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        y = self.fn(x)
        return y + x

class BatchNorm2d(BatchNorm1d):
    def forward(self, x: Tensor):
        # x: (N, C, H, W)
        N, C, H, W = x.shape
        M = N * H * W

        # NCHW -> NHWC
        x_nhwc = x.transpose((0, 2, 3, 1))          # (N, H, W, C)
        # (N, H, W, C) -> (M, C)
        x_flat = x_nhwc.reshape((M, C))

        y_flat = super().forward(x_flat)             # (M, C)

        # (M, C) -> (N, H, W, C)
        y_nhwc = y_flat.reshape((N, H, W, C))
        # NHWC -> NCHW
        y = y_nhwc.transpose((0, 3, 1, 2))           # (N, C, H, W)
        return y


import math
from functools import reduce
from operator import mul
from .init_basic import *
from typing import Any, Tuple


def _prod(xs):
    return reduce(mul, xs, 1)


def _calc_fans_from_shape(shape: Tuple[int, ...]) -> Tuple[int, int]:
    if len(shape) < 2:
        return 1, 1
    receptive = _prod(shape[:-2]) if len(shape) > 2 else 1
    fan_in = shape[-2] * receptive
    fan_out = shape[-1] * receptive
    return fan_in, fan_out


def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    shape = (fan_in, fan_out)
    return rand(*shape, **kwargs) * (2 * a) - a


def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    shape = (fan_in, fan_out)
    return randn(*shape, **kwargs) * std


def kaiming_uniform(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    bound = math.sqrt(6.0 / fan_in)  # = sqrt(3)*gain/sqrt(fan_in)，gain=√2
    shape = (fan_in, fan_out)
    return rand(*shape, **kwargs) * (2 * bound) - bound


def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    if shape is None:
        bound = math.sqrt(6.0 / fan_in)
        shape = (fan_in, fan_out)
    else:
        fi, fo = _calc_fans_from_shape(tuple(shape))
        bound = math.sqrt(6.0 / fi)
    return rand(*shape, **kwargs) * (2 * bound) - bound


def kaiming_normal(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    std = math.sqrt(2.0 / fan_in)  # gain/sqrt(fan_in) with gain=√2
    shape = (fan_in, fan_out)
    return randn(*shape, **kwargs) * std

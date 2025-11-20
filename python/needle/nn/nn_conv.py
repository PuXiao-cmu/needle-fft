"""The module.
"""
from typing import List, Callable, Any
import math
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.device = device
        self.dtype = dtype
        self.use_bias = bias

        k = self.kernel_size
        w_shape = (k, k, in_channels, out_channels)
        W = init.kaiming_uniform(None, None, shape=w_shape, nonlinearity="relu",
                                 device=device, dtype=dtype)
        self.weight = Parameter(W)

        if self.use_bias:
            bound = 1.0 / math.sqrt(in_channels * k * k)
            B = init.rand(out_channels, device=device, dtype=dtype) * (2 * bound) - bound
            self.bias = Parameter(B)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # x: NCHW -> NHWC
        x_nhwc = ops.transpose(x, (0, 2, 3, 1))

        pad = self.kernel_size // 2

        y_nhwc = ops.conv(x_nhwc, self.weight, stride=self.stride, padding=pad)

        y = ops.transpose(y_nhwc, (0, 3, 1, 2))

        if self.use_bias:
            b = ops.reshape(self.bias, (1, self.out_channels, 1, 1))
            y = y + ops.broadcast_to(b, y.shape)

        return y

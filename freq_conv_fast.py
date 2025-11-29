"""
Fast Frequency-Domain Convolution Layer

Key optimization: Avoid 5D tensor broadcasting
Instead: Process each output channel separately
"""

import sys
sys.path.insert(0, './python')

import numpy as np
import needle
from needle import Tensor
import needle.init as init
import needle.ops as ops
from needle.nn import Module

# Make numpy available for weight slicing
np = np


class FastFrequencyConv2D(Module):
    """
    Fast frequency-domain 2D convolution.

    Optimizations:
    1. Avoid 5D broadcasting - process channels iteratively
    2. Cache weight FFT
    3. Minimize tensor operations
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.device = device
        self.dtype = dtype

        # Weight: (K_h, K_w, in_c, out_c)
        self.weight = init.kaiming_uniform(
            self.kernel_size[0] * self.kernel_size[1] * in_channels,
            self.kernel_size[0] * self.kernel_size[1] * out_channels,
            shape=(self.kernel_size[0], self.kernel_size[1], in_channels, out_channels),
            device=device,
            dtype=dtype,
            requires_grad=True
        )

        # Bias
        if bias:
            self.bias = init.kaiming_uniform(
                out_channels, 1,
                shape=(out_channels,),
                device=device,
                dtype=dtype,
                requires_grad=True
            )
        else:
            self.bias = None

        # Cache for weight FFT
        self._cached_weight_fft = None
        self._cached_fft_size = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass using FFT.

        Args:
            x: Input tensor (batch, in_c, H, W)

        Returns:
            Output tensor (batch, out_c, H, W)
        """
        batch, in_c, H, W = x.shape
        K_h, K_w = self.kernel_size
        out_c = self.out_channels

        # Padding for FFT
        pad_h = K_h - 1
        pad_w = K_w - 1
        H_pad = H + 2 * pad_h
        W_pad = W + 2 * pad_w

        # Pad input
        x_padded = self._pad_tensor(x, [(0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)])

        # FFT of input (batch, in_c, H_pad, W_pad)
        x_freq_real, x_freq_imag = self._fft2d(x_padded)

        # Get weight FFT (out_c, in_c, H_pad, W_pad) for each output channel
        w_fft_list = self._get_weight_fft_list(H_pad, W_pad)

        # Process each output channel separately (avoid 5D broadcasting)
        output_channels = []

        for oc in range(out_c):
            # Get weight FFT for this output channel: (in_c, H_pad, W_pad)
            w_r, w_i = w_fft_list[oc]

            # Broadcast weight to batch dimension: (batch, in_c, H_pad, W_pad)
            w_r_bc = ops.broadcast_to(
                ops.reshape(w_r, (1, in_c, H_pad, W_pad)),
                (batch, in_c, H_pad, W_pad)
            )
            w_i_bc = ops.broadcast_to(
                ops.reshape(w_i, (1, in_c, H_pad, W_pad)),
                (batch, in_c, H_pad, W_pad)
            )

            # Complex multiply: (batch, in_c, H_pad, W_pad)
            prod_real = x_freq_real * w_r_bc - x_freq_imag * w_i_bc
            prod_imag = x_freq_real * w_i_bc + x_freq_imag * w_r_bc

            # Sum over input channels: (batch, H_pad, W_pad)
            conv_real = ops.summation(prod_real, axes=(1,))
            conv_imag = ops.summation(prod_imag, axes=(1,))

            # IFFT: (batch, H_pad, W_pad)
            out_ch = self._ifft2d(conv_real, conv_imag)

            output_channels.append(out_ch)

        # Stack output channels: (batch, out_c, H_pad, W_pad)
        output = ops.stack(output_channels, axis=1)

        # Crop to original size
        output = self._crop_center(output, H, W)

        # Apply stride if needed
        if self.stride[0] > 1 or self.stride[1] > 1:
            output = output[:, :, ::self.stride[0], ::self.stride[1]]

        # Add bias
        if self.bias is not None:
            bias_reshaped = ops.reshape(self.bias, (1, out_c, 1, 1))
            bias_broadcast = ops.broadcast_to(bias_reshaped, output.shape)
            output = output + bias_broadcast

        return output

    def _get_weight_fft_list(self, H_pad, W_pad):
        """
        Get FFT of weights for each output channel.

        Returns:
            List of (w_freq_real, w_freq_imag) tuples, one per output channel
            Each is (in_c, H_pad, W_pad)
        """
        fft_size = (H_pad, W_pad)

        # Check cache
        if self._cached_weight_fft is not None and self._cached_fft_size == fft_size:
            return self._cached_weight_fft

        # Compute FFT for each output channel
        w_fft_list = []

        # Weight shape: (K_h, K_w, in_c, out_c)
        # Process weight for each output channel separately
        w_np = self.weight.numpy()  # Convert to numpy to slice
        K_h, K_w, in_c, out_c = w_np.shape

        for oc in range(out_c):
            # Get weight for this output channel: (K_h, K_w, in_c)
            w_oc_np = w_np[:, :, :, oc]  # (K_h, K_w, in_c)

            # Transpose to (in_c, K_h, K_w)
            w_oc_np = np.transpose(w_oc_np, (2, 0, 1))

            # Convert back to Tensor
            w_oc = Tensor(w_oc_np, device=self.device, dtype=self.dtype)

            # Pad to (in_c, H_pad, W_pad)
            w_padded = self._pad_tensor(
                w_oc,
                [(0, 0), (0, H_pad - K_h), (0, W_pad - K_w)]
            )

            # FFT: (in_c, H_pad, W_pad)
            w_r, w_i = self._fft2d(w_padded)

            w_fft_list.append((w_r, w_i))

        # Cache
        self._cached_weight_fft = w_fft_list
        self._cached_fft_size = fft_size

        return w_fft_list

    def _fft2d(self, x):
        """2D FFT along last two dimensions."""
        # FFT along H (axis=-2)
        fft_h = ops.fft(x, dim=-2, norm="backward")
        x_fft_h_real = ops.tuple_get_item(fft_h, 0)
        x_fft_h_imag = ops.tuple_get_item(fft_h, 1)

        # FFT along W (axis=-1) for real part
        fft_w_real = ops.fft(x_fft_h_real, dim=-1, norm="backward")
        real_real = ops.tuple_get_item(fft_w_real, 0)
        real_imag = ops.tuple_get_item(fft_w_real, 1)

        # FFT along W (axis=-1) for imag part
        fft_w_imag = ops.fft(x_fft_h_imag, dim=-1, norm="backward")
        imag_real = ops.tuple_get_item(fft_w_imag, 0)
        imag_imag = ops.tuple_get_item(fft_w_imag, 1)

        # Combine: (a+bi) * i = -b + ai
        final_real = real_real - imag_imag
        final_imag = real_imag + imag_real

        return final_real, final_imag

    def _ifft2d(self, freq_real, freq_imag):
        """2D IFFT along last two dimensions."""
        # IFFT along W (axis=-1)
        # ops.ifft returns a single Tensor (real-valued), not a TensorTuple
        ifft_w_real = ops.ifft(freq_real, freq_imag, dim=-1, norm="backward")

        # IFFT along H (axis=-2)
        # For real result, we only need the real part of second IFFT
        zero_imag = init.zeros_like(ifft_w_real)
        result_real = ops.ifft(ifft_w_real, zero_imag, dim=-2, norm="backward")

        return result_real

    def _pad_tensor(self, x, padding):
        """
        Pad tensor using NDArray.pad().

        Args:
            x: Input tensor
            padding: List of (before, after) for each dimension

        Returns:
            Padded tensor
        """
        # Convert to NDArray, pad, convert back to Tensor
        x_data = x.realize_cached_data()
        padded_data = x_data.pad(*padding)
        return Tensor(padded_data, device=self.device, dtype=self.dtype)

    def _crop_center(self, x, target_h, target_w):
        """Crop center of tensor to target size."""
        batch, channels, H, W = x.shape

        start_h = (H - target_h) // 2
        start_w = (W - target_w) // 2

        # Convert to numpy, crop, convert back
        x_np = x.numpy()
        cropped_np = x_np[:, :, start_h:start_h+target_h, start_w:start_w+target_w]

        return Tensor(cropped_np, device=self.device, dtype=self.dtype)

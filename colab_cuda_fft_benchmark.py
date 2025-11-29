#!/usr/bin/env python3
"""
CUDA FFT Convolution Benchmark for Google Colab

This script tests CUDA-accelerated frequency-domain convolution on Colab GPUs.
It performs forward-only benchmarks to measure speedup vs spatial convolution.

Usage in Colab:
    1. Upload this script and your needle codebase to Colab
    2. Set runtime to GPU (Runtime -> Change runtime type -> GPU)
    3. Run: !python colab_cuda_fft_benchmark.py
"""

import os
import sys
import time

# Force CUDA FFT implementation
os.environ['NEEDLE_FFT_IMPL'] = 'cuda'

# Add needle to path (adjust if needed)
sys.path.insert(0, './python')

import numpy as np
import needle
from needle import Tensor
import needle.init as init
import needle.ops as ops
import needle.nn as nn


print("="*80)
print("CUDA FFT Convolution Benchmark for Google Colab")
print("="*80)

# Check CUDA availability
try:
    device = needle.cuda()
    print(f"\n✓ CUDA device available: {device}")
except:
    print("\n✗ CUDA not available! Please set runtime to GPU.")
    print("  Runtime -> Change runtime type -> GPU")
    sys.exit(1)


class SpatialConv2D(nn.Module):
    """Standard spatial convolution for comparison."""

    def __init__(self, in_channels, out_channels, kernel_size, device=None):
        super().__init__()
        self.conv = nn.Conv(in_channels, out_channels, kernel_size, device=device)

    def forward(self, x):
        return self.conv(x)


class FrequencyConv2D(nn.Module):
    """
    Frequency-domain convolution using CUDA FFT.

    Forward-only implementation (no gradient support needed for benchmark).
    """

    def __init__(self, in_channels, out_channels, kernel_size, device=None):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.device = device

        # Initialize weights
        K_h, K_w = kernel_size
        self.weight = init.kaiming_uniform(
            K_h * K_w * in_channels,
            K_h * K_w * out_channels,
            shape=(K_h, K_w, in_channels, out_channels),
            device=device,
            requires_grad=False  # Forward-only
        )

        self.bias = init.kaiming_uniform(
            out_channels, 1,
            shape=(out_channels,),
            device=device,
            requires_grad=False
        )

    def forward(self, x):
        """Forward pass using FFT on CUDA."""
        batch, in_c, H, W = x.shape
        K_h, K_w = self.kernel_size
        out_c = self.out_channels

        # FFT size (for valid convolution)
        H_fft = H + K_h - 1
        W_fft = W + K_w - 1

        # Pad input
        pad_h = K_h - 1
        pad_w = K_w - 1
        x_padded = self._pad_tensor(x, [(0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)])

        # FFT of input
        x_freq_r, x_freq_i = self._fft2d(x_padded)

        # Process each output channel
        output_channels = []

        # Get weight for each output channel
        w_np = self.weight.numpy()

        for oc in range(out_c):
            # Extract weight for this output channel
            w_oc_np = w_np[:, :, :, oc]  # (K_h, K_w, in_c)
            w_oc_np = np.transpose(w_oc_np, (2, 0, 1))  # (in_c, K_h, K_w)

            w_oc = Tensor(w_oc_np, device=self.device)

            # Pad weight
            w_padded = self._pad_tensor(
                w_oc,
                [(0, 0), (0, H_fft - K_h), (0, W_fft - K_w)]
            )

            # FFT of weight
            w_freq_r, w_freq_i = self._fft2d(w_padded)

            # Broadcast to batch dimension
            w_r = ops.broadcast_to(
                ops.reshape(w_freq_r, (1, in_c, H_fft, W_fft)),
                (batch, in_c, H_fft, W_fft)
            )
            w_i = ops.broadcast_to(
                ops.reshape(w_freq_i, (1, in_c, H_fft, W_fft)),
                (batch, in_c, H_fft, W_fft)
            )

            # Complex multiply
            prod_r = x_freq_r * w_r - x_freq_i * w_i
            prod_i = x_freq_r * w_i + x_freq_i * w_r

            # Sum over input channels
            conv_r = ops.summation(prod_r, axes=(1,))
            conv_i = ops.summation(prod_i, axes=(1,))

            # IFFT
            out_ch = self._ifft2d(conv_r, conv_i)

            output_channels.append(out_ch)

        # Stack output channels
        output = ops.stack(output_channels, axis=1)

        # Crop to original size
        output = self._crop_center(output, H, W)

        # Add bias
        bias_reshaped = ops.reshape(self.bias, (1, out_c, 1, 1))
        bias_broadcast = ops.broadcast_to(bias_reshaped, output.shape)
        output = output + bias_broadcast

        return output

    def _fft2d(self, x):
        """2D FFT using CUDA backend."""
        # FFT along H
        fft_h = ops.fft(x, dim=-2, norm="backward")
        x_r = ops.tuple_get_item(fft_h, 0)
        x_i = ops.tuple_get_item(fft_h, 1)

        # FFT along W of real part
        fft_w_r = ops.fft(x_r, dim=-1, norm="backward")
        rr = ops.tuple_get_item(fft_w_r, 0)
        ri = ops.tuple_get_item(fft_w_r, 1)

        # FFT along W of imag part
        fft_w_i = ops.fft(x_i, dim=-1, norm="backward")
        ir = ops.tuple_get_item(fft_w_i, 0)
        ii = ops.tuple_get_item(fft_w_i, 1)

        # Combine
        final_r = rr - ii
        final_i = ri + ir

        return final_r, final_i

    def _ifft2d(self, freq_r, freq_i):
        """2D IFFT using CUDA backend."""
        # IFFT along W
        ifft_w = ops.ifft(freq_r, freq_i, dim=-1, norm="backward")

        # IFFT along H
        zero_i = init.zeros_like(ifft_w, device=self.device)
        result = ops.ifft(ifft_w, zero_i, dim=-2, norm="backward")

        return result

    def _pad_tensor(self, x, padding):
        """Pad tensor."""
        x_data = x.realize_cached_data()
        padded_data = x_data.pad(*padding)
        return Tensor(padded_data, device=self.device)

    def _crop_center(self, x, target_h, target_w):
        """Crop center region."""
        batch, channels, H, W = x.shape
        start_h = (H - target_h) // 2
        start_w = (W - target_w) // 2

        x_np = x.numpy()
        cropped_np = x_np[:, :, start_h:start_h+target_h, start_w:start_w+target_w]

        return Tensor(cropped_np, device=self.device)


def benchmark_forward(model, input_tensor, warmup=3, iterations=10):
    """Benchmark forward pass only."""

    # Warmup
    for _ in range(warmup):
        _ = model(input_tensor)

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        output = model(input_tensor)
    elapsed = (time.time() - start) / iterations

    return elapsed * 1000  # Convert to ms


def main():
    """Run CUDA FFT benchmark."""

    device = needle.cuda()

    # Test configurations
    configs = [
        # (H, W, in_c, out_c, K, batch_size, description)
        (64, 64, 3, 16, 5, 4, "Small: 64×64, K=5"),
        (128, 128, 3, 16, 7, 2, "Medium: 128×128, K=7"),
        (256, 256, 3, 16, 11, 1, "Large: 256×256, K=11"),
        (256, 256, 3, 16, 21, 1, "Large kernel: 256×256, K=21"),
    ]

    print("\n" + "="*80)
    print("CUDA Forward-Only Benchmark")
    print("="*80)

    results = []

    for H, W, in_c, out_c, K, batch, desc in configs:
        print(f"\n{desc}")
        print(f"  Input: ({batch}, {in_c}, {H}, {W}), Kernel: {K}×{K}, Output channels: {out_c}")

        # Create models
        spatial_model = SpatialConv2D(in_c, out_c, K, device=device)
        fft_model = FrequencyConv2D(in_c, out_c, K, device=device)

        # Create input
        np.random.seed(42)
        input_np = np.random.randn(batch, in_c, H, W).astype(np.float32)
        input_tensor = Tensor(input_np, device=device)

        # Benchmark spatial convolution
        print("  Testing spatial convolution...", end=" ", flush=True)
        try:
            spatial_time = benchmark_forward(spatial_model, input_tensor, warmup=2, iterations=5)
            print(f"✓ {spatial_time:.2f} ms/iter")
        except Exception as e:
            print(f"✗ Failed: {e}")
            spatial_time = None

        # Benchmark FFT convolution
        print("  Testing FFT convolution (CUDA)...", end=" ", flush=True)
        try:
            fft_time = benchmark_forward(fft_model, input_tensor, warmup=2, iterations=5)
            print(f"✓ {fft_time:.2f} ms/iter")
        except Exception as e:
            print(f"✗ Failed: {e}")
            fft_time = None

        # Calculate speedup
        if spatial_time and fft_time:
            speedup = spatial_time / fft_time
            print(f"  Speedup: {speedup:.2f}x {'(FFT faster)' if speedup > 1 else '(Spatial faster)'}")

            results.append({
                'config': desc,
                'spatial_time': spatial_time,
                'fft_time': fft_time,
                'speedup': speedup
            })
        else:
            results.append({
                'config': desc,
                'spatial_time': spatial_time,
                'fft_time': fft_time,
                'speedup': None
            })

    # Summary
    print("\n\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    print(f"\n{'Configuration':<30} | {'Spatial':<12} | {'FFT (CUDA)':<12} | {'Speedup':<10}")
    print("-"*80)

    for r in results:
        spatial_str = f"{r['spatial_time']:.2f} ms" if r['spatial_time'] else "Failed"
        fft_str = f"{r['fft_time']:.2f} ms" if r['fft_time'] else "Failed"
        speedup_str = f"{r['speedup']:.2f}x" if r['speedup'] else "N/A"

        print(f"{r['config']:<30} | {spatial_str:<12} | {fft_str:<12} | {speedup_str:<10}")

    print("\n" + "="*80)

    # Check if FFT was faster for large kernels
    large_kernel_results = [r for r in results if 'Large kernel' in r['config'] and r['speedup']]
    if large_kernel_results and all(r['speedup'] > 1 for r in large_kernel_results):
        print("\n✓ SUCCESS: CUDA FFT convolution is faster for large kernels!")
        print("  This demonstrates the advantage of frequency-domain convolution on GPU.")
    else:
        print("\n⚠ Note: Results may vary based on GPU type and problem size.")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

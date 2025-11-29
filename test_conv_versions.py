#!/usr/bin/env python3
"""
Compare different convolution implementations.
"""

import sys
import os

os.environ['NEEDLE_FFT_IMPL'] = 'cpp'
sys.path.insert(0, './python')

import numpy as np
import time

import needle
from needle import Tensor
import needle.nn as nn

# Import different versions
from freq_conv_layer import FrequencyDomainConv2D as OriginalFreqConv
from freq_conv_optimized import OptimizedFrequencyConv2D as OptimizedFreqConv
from freq_conv_fast import FastFrequencyConv2D as FastFreqConv

print("="*70)
print("Convolution Implementation Comparison")
print("="*70)

device = needle.cpu_numpy()

# Test configuration
batch = 4
in_c = 3
out_c = 16
H, W = 96, 96
kernel_size = 15
num_iter = 3

# Generate test data
print(f"\nTest configuration:")
print(f"  Input: ({batch}, {in_c}, {H}, {W})")
print(f"  Kernel: {kernel_size}×{kernel_size}")
print(f"  Output channels: {out_c}")
print(f"  Iterations: {num_iter}")

x_np = np.random.randn(batch, in_c, H, W).astype(np.float32) * 0.1
x = Tensor(x_np, device=device)

# Test 1: Spatial Conv
print(f"\n{'='*70}")
print("1. Spatial Convolution (Baseline)")
print(f"{'='*70}")

spatial_conv = nn.Conv(in_c, out_c, kernel_size, device=device)

spatial_times = []
for i in range(num_iter):
    start = time.time()
    out = spatial_conv(x)
    _ = out.numpy()
    elapsed = time.time() - start
    spatial_times.append(elapsed)
    print(f"   Run {i+1}: {elapsed*1000:.2f} ms")

spatial_avg = np.mean(spatial_times)
print(f"   Average: {spatial_avg*1000:.2f} ms")
print(f"   Output shape: {out.shape}")

# Test 2: Original Frequency Conv
print(f"\n{'='*70}")
print("2. Original Frequency Convolution")
print(f"{'='*70}")

try:
    orig_conv = OriginalFreqConv(in_c, out_c, kernel_size, device=device)

    orig_times = []
    for i in range(num_iter):
        start = time.time()
        out = orig_conv(x)
        _ = out.numpy()
        elapsed = time.time() - start
        orig_times.append(elapsed)
        print(f"   Run {i+1}: {elapsed*1000:.2f} ms")

    orig_avg = np.mean(orig_times)
    print(f"   Average: {orig_avg*1000:.2f} ms")
    print(f"   Speedup vs spatial: {spatial_avg/orig_avg:.2f}x")

except Exception as e:
    print(f"   ❌ Failed: {e}")
    orig_avg = float('inf')

# Test 3: Optimized Frequency Conv (with 5D broadcasting)
print(f"\n{'='*70}")
print("3. Optimized Frequency Convolution (5D broadcast)")
print(f"{'='*70}")

try:
    opt_conv = OptimizedFreqConv(in_c, out_c, kernel_size, device=device)

    opt_times = []
    for i in range(num_iter):
        start = time.time()
        out = opt_conv(x)
        _ = out.numpy()
        elapsed = time.time() - start
        opt_times.append(elapsed)
        print(f"   Run {i+1}: {elapsed*1000:.2f} ms")

    opt_avg = np.mean(opt_times)
    print(f"   Average: {opt_avg*1000:.2f} ms")
    print(f"   Speedup vs spatial: {spatial_avg/opt_avg:.2f}x")
    print(f"   Speedup vs original: {orig_avg/opt_avg:.2f}x")

except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    opt_avg = float('inf')

# Test 4: Fast Frequency Conv (channel-wise, no 5D)
print(f"\n{'='*70}")
print("4. Fast Frequency Convolution (channel-wise)")
print(f"{'='*70}")

try:
    fast_conv = FastFreqConv(in_c, out_c, kernel_size, device=device)

    fast_times = []
    for i in range(num_iter):
        start = time.time()
        out = fast_conv(x)
        _ = out.numpy()
        elapsed = time.time() - start
        fast_times.append(elapsed)
        print(f"   Run {i+1}: {elapsed*1000:.2f} ms")

    fast_avg = np.mean(fast_times)
    print(f"   Average: {fast_avg*1000:.2f} ms")
    print(f"   Speedup vs spatial: {spatial_avg/fast_avg:.2f}x")
    print(f"   Speedup vs optimized (5D): {opt_avg/fast_avg:.2f}x")

except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    fast_avg = float('inf')

# Summary
print(f"\n\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")

print(f"\n{'Method':<30} | {'Time (ms)':<12} | {'vs Spatial':<12}")
print("-"*60)
print(f"{'Spatial Conv':<30} | {spatial_avg*1000:>10.2f}ms | {'1.00x':<12}")
if orig_avg < float('inf'):
    print(f"{'Original FFT':<30} | {orig_avg*1000:>10.2f}ms | {spatial_avg/orig_avg:>10.2f}x")
if opt_avg < float('inf'):
    print(f"{'Optimized FFT (5D)':<30} | {opt_avg*1000:>10.2f}ms | {spatial_avg/opt_avg:>10.2f}x")
if fast_avg < float('inf'):
    print(f"{'Fast FFT (channel-wise)':<30} | {fast_avg*1000:>10.2f}ms | {spatial_avg/fast_avg:>10.2f}x")

print(f"\n{'='*70}")
print("ANALYSIS")
print(f"{'='*70}")

if fast_avg < spatial_avg:
    print(f"\n✓ Fast FFT is faster than spatial!")
    print(f"  Speedup: {spatial_avg/fast_avg:.2f}x")
elif fast_avg < opt_avg:
    print(f"\n✓ Fast FFT is {opt_avg/fast_avg:.2f}x faster than 5D broadcast version")
    print(f"  But still {fast_avg/spatial_avg:.2f}x slower than spatial")
    print(f"  Need larger images or GPU for FFT advantage")
else:
    print(f"\n⚠ FFT still slower than spatial on {H}×{W} images")
    print(f"  Try larger images (≥256×256) or GPU implementation")

print(f"\n{'='*70}\n")

#!/usr/bin/env python3
"""
Debug FFT implementation to find numerical issues.
"""

import sys
import os

os.environ['NEEDLE_FFT_IMPL'] = 'numpy'
sys.path.insert(0, './python')

import numpy as np
import needle
from needle import Tensor
import needle.ops as ops

print("Testing basic FFT operations...")

device = needle.cpu_numpy()

# Test 1: Simple small tensor
print("\n1. Testing small tensor FFT...")
x_np = np.random.randn(2, 3, 8, 8).astype(np.float32) * 0.1
x = Tensor(x_np, device=device)
print(f"   Input shape: {x.shape}")
print(f"   Input range: [{x.numpy().min():.3f}, {x.numpy().max():.3f}]")

try:
    x_freq_real, x_freq_imag = ops.fft(x, dim=-1, norm="backward")
    print(f"   ✓ FFT successful")
    print(f"   Output real range: [{x_freq_real.numpy().min():.3f}, {x_freq_real.numpy().max():.3f}]")
    print(f"   Output imag range: [{x_freq_imag.numpy().min():.3f}, {x_freq_imag.numpy().max():.3f}]")

    # Check for NaN/Inf
    if np.any(np.isnan(x_freq_real.numpy())) or np.any(np.isinf(x_freq_real.numpy())):
        print(f"   ❌ FFT output contains NaN/Inf!")

except Exception as e:
    print(f"   ❌ FFT failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: IFFT
print("\n2. Testing IFFT...")
try:
    x_reconstructed = ops.ifft(x_freq_real, x_freq_imag, dim=-1, norm="backward")
    print(f"   ✓ IFFT successful")
    print(f"   Reconstructed range: [{x_reconstructed.numpy().min():.3f}, {x_reconstructed.numpy().max():.3f}]")

    # Check reconstruction error
    error = np.abs(x_reconstructed.numpy() - x.numpy()).max()
    print(f"   Reconstruction error: {error:.6f}")

    if error > 1e-3:
        print(f"   ⚠ Large reconstruction error!")

except Exception as e:
    print(f"   ❌ IFFT failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test with Conv layer weights
print("\n3. Testing Conv layer initialization...")
try:
    import needle.nn as nn

    conv = nn.Conv(3, 16, 11, device=device)
    print(f"   ✓ Conv layer created")
    print(f"   Weight shape: {conv.weight.shape}")
    print(f"   Weight range: [{conv.weight.numpy().min():.3f}, {conv.weight.numpy().max():.3f}]")

    # Check for NaN/Inf in weights
    if np.any(np.isnan(conv.weight.numpy())) or np.any(np.isinf(conv.weight.numpy())):
        print(f"   ❌ Conv weights contain NaN/Inf!")

except Exception as e:
    print(f"   ❌ Conv layer failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Simple forward pass with spatial conv
print("\n4. Testing spatial Conv forward pass...")
try:
    x_small = Tensor(np.random.randn(2, 3, 16, 16).astype(np.float32) * 0.1, device=device)
    conv_small = nn.Conv(3, 8, 5, device=device)

    out = conv_small(x_small)
    print(f"   ✓ Spatial conv successful")
    print(f"   Output shape: {out.shape}")
    print(f"   Output range: [{out.numpy().min():.3f}, {out.numpy().max():.3f}]")

    if np.any(np.isnan(out.numpy())) or np.any(np.isinf(out.numpy())):
        print(f"   ❌ Conv output contains NaN/Inf!")

except Exception as e:
    print(f"   ❌ Spatial conv failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Frequency conv
print("\n5. Testing Frequency Conv forward pass...")
try:
    from freq_conv_optimized import OptimizedFrequencyConv2D

    freq_conv = OptimizedFrequencyConv2D(3, 8, 5, device=device)
    print(f"   Created FrequencyConv2D")

    # Use same input
    print(f"   Running forward pass...")
    out_freq = freq_conv(x_small)

    print(f"   ✓ Frequency conv successful")
    print(f"   Output shape: {out_freq.shape}")
    print(f"   Output range: [{out_freq.numpy().min():.3f}, {out_freq.numpy().max():.3f}]")

    if np.any(np.isnan(out_freq.numpy())) or np.any(np.isinf(out_freq.numpy())):
        print(f"   ❌ Frequency conv output contains NaN/Inf!")

except Exception as e:
    print(f"   ❌ Frequency conv failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("All basic tests passed!")
print("="*70)

# Now test with actual STL-10 images
print("\n6. Testing with STL-10 images...")
try:
    import os
    bin_file = './data/stl10/stl10_binary/train_X.bin'

    if not os.path.exists(bin_file):
        print(f"   ⚠ STL-10 not found, skipping")
    else:
        with open(bin_file, 'rb') as f:
            images = np.fromfile(f, dtype=np.uint8, count=96*96*3*4)  # Just 4 images

        images = images.reshape(4, 3, 96, 96)
        images = images.astype(np.float32) / 255.0

        print(f"   Loaded 4 STL-10 images")
        print(f"   Image range: [{images.min():.3f}, {images.max():.3f}]")

        x_stl = Tensor(images, device=device)

        # Try spatial conv
        conv_stl = nn.Conv(3, 8, 11, device=device)
        print(f"   Testing spatial conv on STL-10...")
        out_spatial = conv_stl(x_stl)
        print(f"   ✓ Spatial conv output: {out_spatial.shape}")

        # Try frequency conv
        freq_conv_stl = OptimizedFrequencyConv2D(3, 8, 11, device=device)
        print(f"   Testing frequency conv on STL-10...")
        out_freq_stl = freq_conv_stl(x_stl)
        print(f"   ✓ Frequency conv output: {out_freq_stl.shape}")

        print(f"\n   ✓ STL-10 tests passed!")

except Exception as e:
    print(f"   ❌ STL-10 test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("Debug test complete!")
print("="*70)

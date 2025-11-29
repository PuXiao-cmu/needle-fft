#!/usr/bin/env python3
"""
Simple FFT convolution test on STL-10 dataset (no scipy needed).
Uses only the original 96x96 images to avoid dependencies.
"""

import sys
import os

# Set FFT implementation
os.environ['NEEDLE_FFT_IMPL'] = 'cpp'

sys.path.insert(0, './python')

import numpy as np
import time

import needle
from needle import Tensor
import needle.nn as nn

# Import optimized version
try:
    from freq_conv_optimized import OptimizedFrequencyConv2D as FrequencyConv
    print("Using optimized FFT convolution")
except ImportError:
    from freq_conv_layer import FrequencyDomainConv2D as FrequencyConv
    print("Using standard FFT convolution")


def load_stl10_images(data_dir='./data/stl10', split='train', max_images=100):
    """
    Load STL-10 images.

    Args:
        data_dir: Path to STL-10 data
        split: 'train' or 'test'
        max_images: Maximum number of images to load

    Returns:
        numpy array of shape (N, 3, 96, 96)
    """
    if split == 'train':
        bin_file = os.path.join(data_dir, 'stl10_binary', 'train_X.bin')
    else:
        bin_file = os.path.join(data_dir, 'stl10_binary', 'test_X.bin')

    if not os.path.exists(bin_file):
        print(f"❌ STL-10 data not found at {bin_file}")
        print(f"   Please download from: http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz")
        return None

    print(f"Loading images from {bin_file}...")

    # STL-10 format: each image is 96*96*3 = 27648 bytes
    with open(bin_file, 'rb') as f:
        images = np.fromfile(f, dtype=np.uint8)

    # Reshape: STL-10 uses channels-first format
    num_images = len(images) // (96 * 96 * 3)
    images = images.reshape(num_images, 3, 96, 96)

    # Take subset
    images = images[:max_images]

    # Convert to float32 and normalize to [0, 1]
    images = images.astype(np.float32) / 255.0

    print(f"✓ Loaded {len(images)} images, shape: {images.shape}")

    return images


def test_convolution_comparison(images, kernel_size, num_iterations=5):
    """
    Compare spatial vs FFT convolution on real images.

    Args:
        images: numpy array (N, C, H, W)
        kernel_size: Size of square kernel
        num_iterations: Number of forward passes

    Returns:
        dict with timing results
    """
    N, C, H, W = images.shape

    print(f"\n{'='*70}")
    print(f"Testing: {H}×{W} images, {kernel_size}×{kernel_size} kernel")
    print(f"{'='*70}")

    device = needle.cpu_numpy()
    batch_size = min(4, N)
    in_channels = C
    out_channels = 16

    # Use first batch
    x_np = images[:batch_size]
    x = Tensor(x_np, device=device)

    print(f"Input shape: {x.shape}")
    print(f"Data size: {x.numpy().nbytes / 1024 / 1024:.2f} MB")

    # Test 1: Spatial Convolution
    print(f"\n📊 Testing Spatial Convolution...")

    spatial_conv = nn.Conv(in_channels, out_channels, kernel_size, device=device)

    spatial_times = []
    for i in range(num_iterations):
        start = time.time()
        out_spatial = spatial_conv(x)
        _ = out_spatial.numpy()  # Force computation
        elapsed = time.time() - start
        spatial_times.append(elapsed)

        if i == 0:
            print(f"   First run: {elapsed*1000:.2f} ms")

    spatial_avg = np.mean(spatial_times[1:])
    spatial_std = np.std(spatial_times[1:])

    print(f"   Average: {spatial_avg*1000:.2f} ± {spatial_std*1000:.2f} ms")
    print(f"   Output shape: {out_spatial.shape}")

    # Test 2: FFT Convolution
    print(f"\n📊 Testing FFT Convolution...")

    freq_conv = FrequencyConv(in_channels, out_channels, kernel_size, device=device)

    freq_times = []
    for i in range(num_iterations):
        start = time.time()
        out_freq = freq_conv(x)
        _ = out_freq.numpy()  # Force computation
        elapsed = time.time() - start
        freq_times.append(elapsed)

        if i == 0:
            print(f"   First run: {elapsed*1000:.2f} ms (includes weight FFT)")

    freq_avg = np.mean(freq_times[1:])
    freq_std = np.std(freq_times[1:])

    print(f"   Average: {freq_avg*1000:.2f} ± {freq_std*1000:.2f} ms")
    print(f"   Output shape: {out_freq.shape}")

    # Analysis
    speedup = spatial_avg / freq_avg

    print(f"\n📈 Performance Analysis:")
    print(f"   Speedup: {speedup:.2f}x ({'FFT faster' if speedup > 1 else 'Spatial faster'})")

    # Theoretical analysis
    H_pad = H + 2 * (kernel_size - 1)
    W_pad = W + 2 * (kernel_size - 1)

    spatial_ops = H * W * kernel_size * kernel_size * in_channels * out_channels
    fft_ops_per_channel = 2 * H_pad * W_pad * np.log2(max(H_pad, W_pad))
    fft_ops = fft_ops_per_channel * (in_channels + out_channels)

    theoretical_advantage = spatial_ops / fft_ops

    print(f"\n🔬 Theoretical Analysis:")
    print(f"   Spatial ops: ~{spatial_ops/1e6:.1f}M")
    print(f"   FFT ops: ~{fft_ops/1e6:.1f}M")
    print(f"   Theoretical speedup: {theoretical_advantage:.2f}x")

    if speedup > 1:
        efficiency = (speedup / theoretical_advantage) * 100
        print(f"   Implementation efficiency: {efficiency:.1f}% of theoretical")
    else:
        overhead = freq_avg / spatial_avg
        print(f"   Overhead factor: {overhead:.2f}x")

    return {
        'image_size': H,
        'kernel_size': kernel_size,
        'spatial_time': spatial_avg,
        'freq_time': freq_avg,
        'speedup': speedup,
        'theoretical': theoretical_advantage
    }


def main():
    """Run STL-10 tests."""

    print("="*70)
    print("FFT Convolution Test on STL-10 Dataset (96×96)")
    print("="*70)

    # Load images
    print("\nStep 1: Load STL-10 images")
    images = load_stl10_images(data_dir='./data/stl10', split='train', max_images=100)

    if images is None:
        print("\n❌ Cannot proceed without data")
        return []

    # Test configurations
    print("\nStep 2: Run performance tests")

    results = []

    # Test different kernel sizes on 96×96 images
    configs = [
        11,  # Small-medium kernel
        15,  # Medium kernel
        31,  # Very large kernel (skip 21 to save time)
    ]

    for kernel_size in configs:
        print(f"\n{'#'*70}")
        print(f"# TEST: 96×96 images, {kernel_size}×{kernel_size} kernel")
        print(f"{'#'*70}")

        try:
            result = test_convolution_comparison(images, kernel_size, num_iterations=3)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if len(results) == 0:
        print("\n❌ All tests failed")
        return results

    print(f"\n{'Image Size':<12} | {'Kernel':<8} | {'Spatial':<12} | {'FFT':<12} | {'Speedup':<10} | {'Theory':<10}")
    print("-"*75)

    for r in results:
        img_str = f"{r['image_size']}×{r['image_size']}"
        kernel_str = f"{r['kernel_size']}×{r['kernel_size']}"
        spatial_ms = r['spatial_time'] * 1000
        freq_ms = r['freq_time'] * 1000
        speedup = r['speedup']
        theory = r['theoretical']

        speedup_str = f"{speedup:.2f}x"
        theory_str = f"{theory:.2f}x"

        print(f"{img_str:<12} | {kernel_str:<8} | {spatial_ms:>10.1f}ms | {freq_ms:>10.1f}ms | {speedup_str:<10} | {theory_str:<10}")

    # Analysis
    print("\n\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    max_speedup = max(r['speedup'] for r in results)
    best = [r for r in results if r['speedup'] == max_speedup][0]

    if max_speedup > 1:
        print(f"\n✓ FFT shows advantage with {best['kernel_size']}×{best['kernel_size']} kernel")
        print(f"  Speedup: {max_speedup:.2f}x (actual) vs {best['theoretical']:.2f}x (theoretical)")
        print(f"  Efficiency: {(max_speedup/best['theoretical'])*100:.1f}%")
    else:
        print(f"\n⚠ No FFT speedup observed (max={max_speedup:.2f}x)")
        print(f"  At 96×96, FFT overhead dominates")
        print(f"  FFT shows advantage at larger image sizes (≥256×256)")

    print("\n💡 Insights:")
    print(f"  - Image size (96×96) is moderate for FFT")
    print(f"  - FFT advantage increases with kernel size")
    print(f"  - For clear advantage, use ≥256×256 images or GPU")

    print("\n" + "="*70 + "\n")

    return results


if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

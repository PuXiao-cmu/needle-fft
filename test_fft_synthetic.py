#!/usr/bin/env python3
"""
Test FFT convolution advantage with synthetic large images.

This test creates large synthetic images to demonstrate where FFT
convolution has clear advantages over spatial convolution.

No dataset download needed - generates data on the fly!
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
import needle.ops as ops

# Import optimized version if available, else use original
try:
    from freq_conv_optimized import OptimizedFrequencyConv2D as FrequencyConv
    print("Using optimized FFT convolution")
except ImportError:
    from freq_conv_layer import FrequencyDomainConv2D as FrequencyConv
    print("Using standard FFT convolution")


def gaussian_blur_numpy(image, sigma=2.0, kernel_size=None):
    """
    Simple Gaussian blur using numpy (no scipy needed).

    Args:
        image: 2D numpy array
        sigma: Standard deviation for Gaussian kernel
        kernel_size: Size of kernel (if None, auto-compute)

    Returns:
        Blurred image
    """
    if kernel_size is None:
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)

    # Create Gaussian kernel
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)

    # Pad image
    pad_size = kernel_size // 2
    padded = np.pad(image, pad_size, mode='reflect')

    # Convolve
    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = np.sum(padded[i:i+kernel_size, j:j+kernel_size] * kernel)

    return output


def generate_synthetic_images(batch_size, channels, height, width, complexity='medium'):
    """
    Generate synthetic images with various patterns.

    Args:
        batch_size, channels, height, width: Image dimensions
        complexity: 'simple', 'medium', or 'complex'

    Returns:
        numpy array of shape (batch_size, channels, height, width)
    """
    images = np.zeros((batch_size, channels, height, width), dtype=np.float32)

    for b in range(batch_size):
        for c in range(channels):
            # Create base noise
            noise = np.random.randn(height, width).astype(np.float32)

            if complexity == 'simple':
                # Just smoothed noise
                img = gaussian_blur_numpy(noise, sigma=2.0)

            elif complexity == 'medium':
                # Multi-scale patterns
                img = np.zeros((height, width), dtype=np.float32)
                for scale in [1, 2, 4]:
                    filtered = gaussian_blur_numpy(noise, sigma=scale)
                    img += filtered * (1.0 / scale)

            else:  # complex
                # Gabor-like patterns
                img = np.zeros((height, width), dtype=np.float32)
                for theta in np.linspace(0, np.pi, 4):
                    for freq in [0.1, 0.2, 0.3]:
                        y, x = np.mgrid[0:height, 0:width]
                        sinusoid = np.sin(2 * np.pi * freq * (x * np.cos(theta) + y * np.sin(theta)))
                        envelope = np.exp(-((x - width/2)**2 + (y - height/2)**2) / (2 * (width/4)**2))
                        img += sinusoid * envelope * noise

            # Normalize
            img = (img - img.mean()) / (img.std() + 1e-8)
            images[b, c] = img

    return images


def test_convolution_comparison(image_size, kernel_size, num_iterations=5):
    """
    Compare spatial vs FFT convolution on large synthetic images.

    Args:
        image_size: Size of square images (e.g., 512 for 512×512)
        kernel_size: Size of square kernel (e.g., 15 for 15×15)
        num_iterations: Number of forward passes to average

    Returns:
        dict with timing results
    """
    print(f"\n{'='*70}")
    print(f"Testing: {image_size}×{image_size} images, {kernel_size}×{kernel_size} kernel")
    print(f"{'='*70}")

    device = needle.cpu_numpy()
    batch_size = 4
    in_channels = 16
    out_channels = 16

    # Generate synthetic data
    print(f"\nGenerating synthetic images...")
    images_np = generate_synthetic_images(batch_size, in_channels, image_size, image_size, complexity='medium')
    x = Tensor(images_np, device=device)

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
            print(f"   First run: {elapsed*1000:.2f} ms (cold start)")

    spatial_avg = np.mean(spatial_times[1:])  # Exclude first run
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
            print(f"   First run: {elapsed*1000:.2f} ms (cold start, includes weight FFT)")

    freq_avg = np.mean(freq_times[1:])  # Exclude first run
    freq_std = np.std(freq_times[1:])

    print(f"   Average: {freq_avg*1000:.2f} ± {freq_std*1000:.2f} ms")
    print(f"   Output shape: {out_freq.shape}")

    # Analysis
    speedup = spatial_avg / freq_avg

    print(f"\n📈 Performance Analysis:")
    print(f"   Speedup: {speedup:.2f}x ({'FFT faster' if speedup > 1 else 'Spatial faster'})")

    # Theoretical analysis
    H_pad = image_size + 2 * (kernel_size - 1)
    W_pad = H_pad

    spatial_ops = image_size * image_size * kernel_size * kernel_size * in_channels * out_channels
    # FFT: input FFT + weight FFT + multiply + IFFT
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
        print(f"   Overhead factor: {overhead:.2f}x (implementation not efficient yet)")

    return {
        'image_size': image_size,
        'kernel_size': kernel_size,
        'spatial_time': spatial_avg,
        'freq_time': freq_avg,
        'speedup': speedup,
        'theoretical': theoretical_advantage
    }


def main():
    """Run tests on different image/kernel sizes."""

    print("="*70)
    print("FFT Convolution Advantage Test - Synthetic Large Images")
    print("="*70)

    print("\nThis test generates large synthetic images to demonstrate")
    print("where FFT convolution has clear advantages.")
    print("\nNo dataset download needed!")

    # Test configurations
    configs = [
        # (image_size, kernel_size)
        (128, 15),   # Small-ish image, medium kernel
        (256, 15),   # Medium image, medium kernel
        (256, 31),   # Medium image, large kernel
        (512, 15),   # Large image, medium kernel
        (512, 31),   # Large image, large kernel
    ]

    results = []

    for image_size, kernel_size in configs:
        try:
            result = test_convolution_comparison(image_size, kernel_size, num_iterations=3)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Test failed for {image_size}×{image_size}, K={kernel_size}: {e}")
            continue

    # Summary
    print("\n\n" + "="*70)
    print("SUMMARY")
    print("="*70)

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
    print("ANALYSIS")
    print("="*70)

    print("\n1. When does FFT become faster?")

    crossover_found = False
    for r in results:
        if r['speedup'] > 1.0 and not crossover_found:
            print(f"   ✓ FFT becomes faster at {r['image_size']}×{r['image_size']}, K={r['kernel_size']}×{r['kernel_size']}")
            print(f"     Speedup: {r['speedup']:.2f}x (actual) vs {r['theoretical']:.2f}x (theory)")
            crossover_found = True

    if not crossover_found:
        print(f"   ⚠ FFT not faster in tested configurations")
        print(f"     Try larger images (≥1024) or larger kernels (≥31)")

    print("\n2. Scaling trends:")

    if len(results) >= 2:
        # Compare similar kernel sizes at different image sizes
        print("   As image size increases (same kernel):")
        for i in range(len(results)-1):
            if results[i]['kernel_size'] == results[i+1]['kernel_size']:
                ratio = results[i+1]['speedup'] / results[i]['speedup']
                print(f"     {results[i]['image_size']} → {results[i+1]['image_size']}: "
                      f"speedup improves by {ratio:.2f}x")

    print("\n3. Recommendations:")

    max_speedup = max(r['speedup'] for r in results)
    best = [r for r in results if r['speedup'] == max_speedup][0]

    if max_speedup > 1:
        print(f"   ✓ Best performance: {best['image_size']}×{best['image_size']}, "
              f"K={best['kernel_size']}, speedup={max_speedup:.2f}x")
        print(f"   ✓ For better speedup, use:")
        print(f"     - Larger images (≥{best['image_size']})")
        print(f"     - Larger kernels (≥{best['kernel_size']})")
        print(f"     - GPU implementation (cuFFT)")
    else:
        print(f"   ⚠ No speedup observed (max={max_speedup:.2f}x)")
        print(f"   → Current implementation has too much overhead")
        print(f"   → Need larger test cases or GPU implementation")

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)

    print("\n1. To see clear FFT advantage, try:")
    print("   - Image size: 1024×1024 or larger")
    print("   - Kernel size: 31×31 or larger")
    print("   - GPU implementation with cuFFT")

    print("\n2. Recommended datasets:")
    print("   - DIV2K (high-resolution images)")
    print("   - Places365 (scene classification)")
    print("   - Medical images (CT/MRI scans)")
    print("   See FFT_OPTIMAL_SCENARIOS.md for details")

    print("\n3. Optimization opportunities:")
    print("   - Cache weight FFT (reuse across batches)")
    print("   - Batch FFT operations")
    print("   - Use cuFFT on GPU")
    print("   - Reduce tensor copies/reshapes")

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

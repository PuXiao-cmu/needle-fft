"""
Test Frequency-Domain Convolution with Different FFT Backends

This script tests the frequency-domain convolution layer with:
1. NumPy FFT (default, fastest)
2. Python Cooley-Tukey (educational, slow)
3. C++ Cooley-Tukey (fast for small arrays)
4. CUDA Cooley-Tukey (fast for large arrays on GPU)

Usage:
    # Test all backends
    python test_freq_conv_all_backends.py

    # Test specific backend
    NEEDLE_FFT_IMPL=numpy python test_freq_conv_all_backends.py
    NEEDLE_FFT_IMPL=cooley_tukey python test_freq_conv_all_backends.py
    NEEDLE_FFT_IMPL=cpp python test_freq_conv_all_backends.py
    NEEDLE_FFT_IMPL=cuda python test_freq_conv_all_backends.py
"""

import sys
sys.path.insert(0, './python')

import numpy as np
import os
import time
from typing import Tuple

import needle
from needle import Tensor
import needle.init as init
import needle.ops as ops
from needle.autograd import TensorOp


# Import the frequency-domain convolution layer
# (Assuming it's saved as freq_conv_layer.py)
try:
    from freq_conv_layer import FrequencyDomainConv2D
except ImportError:
    print("Error: Could not import FrequencyDomainConv2D")
    print("Please ensure freq_conv_layer.py is in the current directory")
    sys.exit(1)


def test_backend(backend_name, device, input_size=(32, 32), kernel_size=3,
                 batch_size=2, in_channels=3, out_channels=16):
    """
    Test frequency-domain convolution with a specific FFT backend.

    Args:
        backend_name: Name of the FFT implementation
        device: Needle device (cpu_numpy or cuda)
        input_size: (height, width) of input images
        kernel_size: Size of convolution kernel
        batch_size: Batch size
        in_channels: Number of input channels
        out_channels: Number of output channels

    Returns:
        dict with test results
    """
    print(f"\n{'='*80}")
    print(f"Testing: {backend_name}")
    print(f"{'='*80}")

    h, w = input_size

    # Check if size is power of 2 for non-NumPy backends
    is_pow2 = (h & (h - 1)) == 0 and (w & (w - 1)) == 0

    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Input size: {batch_size} × {in_channels} × {h} × {w}")
    print(f"  Kernel size: {kernel_size}")
    print(f"  Output channels: {out_channels}")
    print(f"  Is power of 2: {is_pow2}")

    # Warning for non-power-of-2 with Cooley-Tukey
    if not is_pow2 and backend_name in ['Python Cooley-Tukey', 'C++ Cooley-Tukey', 'CUDA Cooley-Tukey']:
        print(f"\n⚠️  WARNING: {backend_name} only supports power-of-2 sizes!")
        print(f"    Current size {h}×{w} is not a power of 2.")
        print(f"    Consider using Bluestein algorithm or padding to power of 2.")
        print(f"    This test may fail or fall back to NumPy.")

    results = {
        'backend': backend_name,
        'device': str(device),
        'input_size': input_size,
        'is_power_of_2': is_pow2,
        'success': False,
        'forward_time': None,
        'backward_time': None,
        'output_shape': None,
        'weight_grad_ok': False,
        'bias_grad_ok': False,
        'error': None
    }

    try:
        # Create layer
        print(f"\nCreating FrequencyDomainConv2D...")
        conv = FrequencyDomainConv2D(in_channels, out_channels, kernel_size=kernel_size, device=device)

        # Create input
        x = Tensor(np.random.randn(batch_size, in_channels, h, w).astype(np.float32), device=device)

        # Forward pass with timing
        print(f"\nForward pass...")
        start_time = time.time()
        output = conv(x)
        forward_time = time.time() - start_time

        results['output_shape'] = output.shape
        results['forward_time'] = forward_time
        print(f"  Output shape: {output.shape}")
        print(f"  Forward time: {forward_time*1000:.2f} ms")

        # Get device info
        device_info = conv.get_device_info()
        print(f"\nDevice execution:")
        print(f"  Intended device: {device_info['intended_device']}")
        print(f"  Actual device: {device_info['actual_device_used']}")
        print(f"  FFT on CUDA: {device_info['fft_on_cuda']}")
        if device_info['was_fallback']:
            print(f"  ⚠️  Fallback reason: {device_info['fallback_reason']}")

        # Backward pass with timing
        print(f"\nBackward pass...")
        start_time = time.time()
        loss = ops.summation(output)
        loss.backward()
        backward_time = time.time() - start_time

        results['backward_time'] = backward_time
        print(f"  Backward time: {backward_time*1000:.2f} ms")

        # Check gradients
        print(f"\nGradient check:")
        weight_grad = getattr(conv.weight, 'grad', None)
        bias_grad = getattr(conv.bias, 'grad', None)

        if weight_grad is not None:
            weight_grad_norm = np.linalg.norm(weight_grad.numpy())
            results['weight_grad_ok'] = True
            print(f"  ✓ Weight gradient: {weight_grad.shape}, norm={weight_grad_norm:.2e}")
        else:
            print(f"  ⚠️  Weight gradient: None (padding breaks graph)")

        if bias_grad is not None:
            bias_grad_norm = np.linalg.norm(bias_grad.numpy())
            results['bias_grad_ok'] = True
            print(f"  ✓ Bias gradient: {bias_grad.shape}, norm={bias_grad_norm:.2e}")
        else:
            print(f"  ✗ Bias gradient: None")

        results['success'] = True
        print(f"\n✓ Test PASSED")

    except Exception as e:
        results['error'] = str(e)
        print(f"\n✗ Test FAILED")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

    return results


def print_summary(all_results):
    """Print a summary table of all test results."""
    print(f"\n\n{'='*80}")
    print(f"SUMMARY: Frequency-Domain Convolution Test Results")
    print(f"{'='*80}\n")

    # Print table header
    print(f"{'Backend':<25} {'Size':<10} {'Success':<8} {'Forward':<12} {'Backward':<12} {'Gradients':<15}")
    print(f"{'-'*25} {'-'*10} {'-'*8} {'-'*12} {'-'*12} {'-'*15}")

    # Print each result
    for result in all_results:
        backend = result['backend']
        size = f"{result['input_size'][0]}×{result['input_size'][1]}"
        success = "✓" if result['success'] else "✗"

        if result['forward_time']:
            forward = f"{result['forward_time']*1000:.2f} ms"
        else:
            forward = "N/A"

        if result['backward_time']:
            backward = f"{result['backward_time']*1000:.2f} ms"
        else:
            backward = "N/A"

        grads = []
        if result['weight_grad_ok']:
            grads.append("W")
        if result['bias_grad_ok']:
            grads.append("B")
        grad_str = "+".join(grads) if grads else "None"

        print(f"{backend:<25} {size:<10} {success:<8} {forward:<12} {backward:<12} {grad_str:<15}")

        # Print error if failed
        if not result['success'] and result['error']:
            print(f"  Error: {result['error'][:60]}...")

    print(f"\n{'='*80}\n")


def main():
    """Run tests for all FFT backends."""

    print("=" * 80)
    print("Frequency-Domain Convolution: Multi-Backend Test Suite")
    print("=" * 80)

    # Test configurations
    test_configs = [
        # (backend_name, env_var_value, device, input_size)
        ("NumPy FFT (default)", "numpy", needle.cpu_numpy(), (32, 32)),
        ("NumPy FFT (non-pow2)", "numpy", needle.cpu_numpy(), (28, 28)),
        ("Python Cooley-Tukey", "cooley_tukey", needle.cpu_numpy(), (32, 32)),
        ("C++ Cooley-Tukey", "cpp", needle.cpu_numpy(), (32, 32)),
    ]

    # Add CUDA tests if available
    try:
        cuda_device = needle.cuda()
        test_configs.extend([
            ("CUDA Cooley-Tukey", "cuda", cuda_device, (32, 32)),
            ("CUDA Cooley-Tukey (large)", "cuda", cuda_device, (256, 256)),
        ])
        print("\n✓ CUDA device available, will test GPU backends")
    except:
        print("\n⚠️  CUDA device not available, skipping GPU tests")

    # Run all tests
    all_results = []

    for backend_name, env_value, device, input_size in test_configs:
        # Set environment variable
        original_env = os.environ.get('NEEDLE_FFT_IMPL', None)
        os.environ['NEEDLE_FFT_IMPL'] = env_value

        try:
            result = test_backend(
                backend_name=backend_name,
                device=device,
                input_size=input_size,
                kernel_size=3,
                batch_size=2,
                in_channels=3,
                out_channels=16
            )
            all_results.append(result)
        finally:
            # Restore original environment
            if original_env is not None:
                os.environ['NEEDLE_FFT_IMPL'] = original_env
            else:
                os.environ.pop('NEEDLE_FFT_IMPL', None)

    # Print summary
    print_summary(all_results)

    # Print recommendations
    print("\n📊 Performance Analysis:")
    print("-" * 80)

    successful_results = [r for r in all_results if r['success']]

    if successful_results:
        # Find fastest
        fastest = min(successful_results, key=lambda r: r['forward_time'] or float('inf'))
        print(f"✓ Fastest forward pass: {fastest['backend']}")
        print(f"  Time: {fastest['forward_time']*1000:.2f} ms")
        print(f"  Size: {fastest['input_size'][0]}×{fastest['input_size'][1]}")

        # Print recommendations
        print(f"\n💡 Recommendations:")
        print(f"  - For general use (any size): NumPy FFT")
        print(f"  - For small power-of-2 arrays: C++ Cooley-Tukey")
        print(f"  - For large power-of-2 arrays with GPU: CUDA Cooley-Tukey")
        print(f"  - For non-power-of-2 sizes: Use Bluestein wrapper")

    # Count successes
    num_success = sum(1 for r in all_results if r['success'])
    num_total = len(all_results)

    print(f"\n{'='*80}")
    print(f"Final Result: {num_success}/{num_total} tests passed")
    print(f"{'='*80}\n")

    return num_success == num_total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

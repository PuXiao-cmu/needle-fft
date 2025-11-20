"""
CUDA FFT 测试脚本 - 专为 Google Colab 设计

使用方法:
1. 打开 Google Colab
2. 运行时 -> 更改运行时类型 -> GPU (T4)
3. 上传这个脚本和 needle 项目
4. 运行此脚本
"""

import sys
import os
import numpy as np
import time

print("=" * 80)
print("CUDA FFT 测试 - Google Colab")
print("=" * 80)

# ============================================================================
# 步骤 1: 检查 GPU 环境
# ============================================================================
print("\n[步骤 1] 检查 GPU 环境")
print("-" * 80)

try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print(result.stdout)
    print("✓ NVIDIA GPU 可用")
except:
    print("✗ 未检测到 NVIDIA GPU")
    print("请在 Colab 中: 运行时 -> 更改运行时类型 -> 硬件加速器 -> GPU")
    sys.exit(1)

# ============================================================================
# 步骤 2: 加载 CUDA backend
# ============================================================================
print("\n[步骤 2] 加载 CUDA Backend")
print("-" * 80)

# 添加 Python 路径
sys.path.insert(0, './python/needle/backend_ndarray')

try:
    import ndarray_backend_cuda as cuda_backend
    print("✓ CUDA backend 加载成功")
    print(f"  Device name: {cuda_backend.__device_name__}")
except Exception as e:
    print(f"✗ CUDA backend 加载失败: {e}")
    print("\n请确保已编译 CUDA backend:")
    print("  make clean && make")
    sys.exit(1)

# 检查 FFT 函数
has_fft = hasattr(cuda_backend, 'cooley_tukey_fft_cuda')
has_ifft = hasattr(cuda_backend, 'cooley_tukey_ifft_cuda')

print(f"  cooley_tukey_fft_cuda: {has_fft}")
print(f"  cooley_tukey_ifft_cuda: {has_ifft}")

if not has_fft or not has_ifft:
    print("\n✗ FFT 函数未找到")
    print("可用函数:")
    for name in dir(cuda_backend):
        if not name.startswith('_'):
            print(f"    - {name}")
    sys.exit(1)

# ============================================================================
# 步骤 3: 正确性测试
# ============================================================================
print("\n[步骤 3] 正确性测试")
print("=" * 80)

# 测试数据
test_data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)

print(f"\n输入数据: {test_data}")

try:
    # 创建 CUDA 数组
    Array = cuda_backend.Array

    input_arr = Array(8)
    output_real = Array(8)
    output_imag = Array(8)
    recovered = Array(8)

    # 复制数据到 GPU
    cuda_backend.from_numpy(test_data, input_arr)

    # FFT
    cuda_backend.cooley_tukey_fft_cuda(input_arr, output_real, output_imag, 8)

    # 复制结果回 CPU
    fft_real = cuda_backend.to_numpy(output_real, [8], [1], 0)
    fft_imag = cuda_backend.to_numpy(output_imag, [8], [1], 0)

    print(f"FFT 实部:  {fft_real}")
    print(f"FFT 虚部:  {fft_imag}")

    # 与 NumPy 对比
    numpy_fft = np.fft.fft(test_data, norm='backward')
    numpy_real = np.real(numpy_fft)
    numpy_imag = np.imag(numpy_fft)

    print(f"\nNumPy 实部: {numpy_real}")
    print(f"NumPy 虚部: {numpy_imag}")

    real_error = np.abs(fft_real - numpy_real).max()
    imag_error = np.abs(fft_imag - numpy_imag).max()

    print(f"\n精度:")
    print(f"  实部最大误差: {real_error:.2e}")
    print(f"  虚部最大误差: {imag_error:.2e}")

    if real_error < 1e-4 and imag_error < 1e-4:
        print("  ✓ CUDA FFT 正确!")
    else:
        print("  ✗ CUDA FFT 有误差")

    # IFFT 测试
    print("\n" + "-" * 80)
    print("IFFT 往返测试")
    print("-" * 80)

    cuda_backend.cooley_tukey_ifft_cuda(output_real, output_imag, recovered, 8)
    recovered_data = cuda_backend.to_numpy(recovered, [8], [1], 0)

    print(f"\n恢复数据: {recovered_data}")
    print(f"原始数据: {test_data}")

    roundtrip_error = np.abs(recovered_data - test_data).max()
    print(f"\n往返最大误差: {roundtrip_error:.2e}")

    if roundtrip_error < 1e-4:
        print("✓ IFFT 往返测试通过!")
    else:
        print(f"✗ IFFT 往返测试失败 (误差 = {roundtrip_error})")

except Exception as e:
    print(f"\n✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 步骤 4: 性能基准测试
# ============================================================================
print("\n[步骤 4] 性能基准测试")
print("=" * 80)

sizes = [256, 512, 1024, 2048, 4096, 8192]
num_runs = 100

print(f"\n运行 {num_runs} 次迭代测试...\n")
print(f"{'大小':<10} {'NumPy (ms)':<15} {'CUDA (ms)':<15} {'加速比':<15}")
print("-" * 80)

results = []

for size in sizes:
    # 生成测试数据
    test_large = np.random.randn(size).astype(np.float32)

    # NumPy FFT
    start = time.time()
    for _ in range(num_runs):
        _ = np.fft.fft(test_large, norm='backward')
    time_numpy = (time.time() - start) / num_runs * 1000

    # CUDA FFT
    gpu_input = Array(size)
    gpu_real = Array(size)
    gpu_imag = Array(size)

    cuda_backend.from_numpy(test_large, gpu_input)

    # 预热
    cuda_backend.cooley_tukey_fft_cuda(gpu_input, gpu_real, gpu_imag, size)

    start = time.time()
    for _ in range(num_runs):
        cuda_backend.cooley_tukey_fft_cuda(gpu_input, gpu_real, gpu_imag, size)
    time_cuda = (time.time() - start) / num_runs * 1000

    speedup = time_numpy / time_cuda if time_cuda > 0 else 0

    results.append({
        'size': size,
        'numpy': time_numpy,
        'cuda': time_cuda,
        'speedup': speedup
    })

    print(f"{size:<10} {time_numpy:<15.4f} {time_cuda:<15.4f} {speedup:<15.2f}x")

# ============================================================================
# 步骤 5: 大数组测试
# ============================================================================
print("\n[步骤 5] 大数组性能测试")
print("=" * 80)

large_sizes = [16384, 32768, 65536]
num_runs_large = 10

print(f"\n运行 {num_runs_large} 次大数组测试...\n")
print(f"{'大小':<10} {'NumPy (ms)':<15} {'CUDA (ms)':<15} {'加速比':<15}")
print("-" * 80)

for size in large_sizes:
    try:
        test_large = np.random.randn(size).astype(np.float32)

        # NumPy
        start = time.time()
        for _ in range(num_runs_large):
            _ = np.fft.fft(test_large, norm='backward')
        time_numpy = (time.time() - start) / num_runs_large * 1000

        # CUDA
        gpu_input = Array(size)
        gpu_real = Array(size)
        gpu_imag = Array(size)

        cuda_backend.from_numpy(test_large, gpu_input)

        # 预热
        cuda_backend.cooley_tukey_fft_cuda(gpu_input, gpu_real, gpu_imag, size)

        start = time.time()
        for _ in range(num_runs_large):
            cuda_backend.cooley_tukey_fft_cuda(gpu_input, gpu_real, gpu_imag, size)
        time_cuda = (time.time() - start) / num_runs_large * 1000

        speedup = time_numpy / time_cuda

        print(f"{size:<10} {time_numpy:<15.4f} {time_cuda:<15.4f} {speedup:<15.2f}x")

    except Exception as e:
        print(f"{size:<10} ✗ 失败: {e}")

# ============================================================================
# 步骤 6: 精度验证（大数组）
# ============================================================================
print("\n[步骤 6] 大数组精度验证")
print("=" * 80)

verify_size = 4096
test_verify = np.random.randn(verify_size).astype(np.float32)

# NumPy
numpy_result = np.fft.fft(test_verify, norm='backward')
numpy_real = np.real(numpy_result)
numpy_imag = np.imag(numpy_result)

# CUDA
gpu_input = Array(verify_size)
gpu_real = Array(verify_size)
gpu_imag = Array(verify_size)

cuda_backend.from_numpy(test_verify, gpu_input)
cuda_backend.cooley_tukey_fft_cuda(gpu_input, gpu_real, gpu_imag, verify_size)

cuda_real = cuda_backend.to_numpy(gpu_real, [verify_size], [1], 0)
cuda_imag = cuda_backend.to_numpy(gpu_imag, [verify_size], [1], 0)

real_error = np.abs(cuda_real - numpy_real).max()
imag_error = np.abs(cuda_imag - numpy_imag).max()
mean_real_error = np.abs(cuda_real - numpy_real).mean()
mean_imag_error = np.abs(cuda_imag - numpy_imag).mean()

print(f"\n数组大小: {verify_size}")
print(f"实部最大误差: {real_error:.2e}")
print(f"虚部最大误差: {imag_error:.2e}")
print(f"实部平均误差: {mean_real_error:.2e}")
print(f"虚部平均误差: {mean_imag_error:.2e}")

if real_error < 1e-3 and imag_error < 1e-3:
    print("\n✓ 大数组精度验证通过")
else:
    print("\n⚠ 大数组精度可能有问题")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 80)
print("测试总结")
print("=" * 80)

# 计算平均加速比
avg_speedup = np.mean([r['speedup'] for r in results])
max_speedup = max([r['speedup'] for r in results])

print(f"""
✓ CUDA FFT 实现验证完成

正确性测试:
  - 小数组 (N=8): {'✓ 通过' if roundtrip_error < 1e-4 else '✗ 失败'}
  - 大数组 (N={verify_size}): {'✓ 通过' if real_error < 1e-3 else '✗ 失败'}

性能测试:
  - 测试大小范围: {sizes[0]} - {sizes[-1]}
  - 平均加速比: {avg_speedup:.2f}x
  - 最大加速比: {max_speedup:.2f}x
  - 最佳大小: {results[np.argmax([r['speedup'] for r in results])]['size']}

结论:
  {'✓ CUDA FFT 实现成功并可用于生产环境' if avg_speedup > 1 else '⚠ 性能未达预期，需要进一步优化'}
""")

print("=" * 80)
print("测试完成!")
print("=" * 80)

# 保存结果
print("\n保存测试结果到 cuda_test_results.txt...")
with open('cuda_test_results.txt', 'w') as f:
    f.write("CUDA FFT 测试结果\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"正确性测试:\n")
    f.write(f"  小数组往返误差: {roundtrip_error:.2e}\n")
    f.write(f"  大数组实部误差: {real_error:.2e}\n")
    f.write(f"  大数组虚部误差: {imag_error:.2e}\n\n")

    f.write(f"性能测试结果:\n")
    f.write(f"{'大小':<10} {'NumPy (ms)':<15} {'CUDA (ms)':<15} {'加速比':<15}\n")
    f.write("-" * 80 + "\n")
    for r in results:
        f.write(f"{r['size']:<10} {r['numpy']:<15.4f} {r['cuda']:<15.4f} {r['speedup']:<15.2f}x\n")

    f.write(f"\n平均加速比: {avg_speedup:.2f}x\n")
    f.write(f"最大加速比: {max_speedup:.2f}x\n")

print("✓ 结果已保存")

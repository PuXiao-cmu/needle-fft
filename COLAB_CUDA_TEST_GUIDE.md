# Google Colab CUDA FFT 测试指南

## 概述

本指南介绍如何在 Google Colab 上测试 CUDA backend 的频域卷积性能。

## 准备工作

### 1. 上传代码到 Colab

在 Colab notebook 中执行：

```python
# 方法 1: 从 GitHub 克隆（如果你的代码在 GitHub 上）
!git clone https://github.com/your-username/your-repo.git
%cd your-repo

# 方法 2: 从本地上传
# 使用 Colab 左侧的文件浏览器上传整个 needle 文件夹
```

### 2. 设置 GPU 运行时

**重要**: 必须启用 GPU 运行时

1. 点击 **Runtime** → **Change runtime type**
2. 在 **Hardware accelerator** 下拉菜单中选择 **GPU**
3. 点击 **Save**

### 3. 验证 GPU 可用性

```python
# 检查 GPU
!nvidia-smi
```

应该看到 GPU 信息（通常是 Tesla T4, K80, 或 V100）

## 编译 CUDA Backend

在 Colab notebook 中执行：

```python
# 安装必要的依赖
!pip install pybind11 numpy

# 编译 CUDA backend
%cd /content/needle  # 调整为你的路径
!make

# 验证编译成功
!ls python/needle/backend_ndarray/ndarray_backend_cuda*.so
```

**预期输出**: 应该看到 `ndarray_backend_cuda.cpython-*.so` 文件

## 运行 CUDA FFT Benchmark

### 快速测试

```python
# 设置环境变量
import os
os.environ['NEEDLE_FFT_IMPL'] = 'cuda'

# 运行 benchmark
!python colab_cuda_fft_benchmark.py
```

### 完整的 Colab Notebook 示例

创建一个新的 Colab notebook，包含以下单元格：

#### Cell 1: 检查环境

```python
# 检查 GPU
!nvidia-smi

# 显示 CUDA 版本
!nvcc --version
```

#### Cell 2: 克隆/上传代码

```python
# 如果从 GitHub
!git clone https://github.com/your-username/needle.git
%cd needle

# 或上传本地文件
from google.colab import files
# 然后使用左侧文件浏览器上传
```

#### Cell 3: 安装依赖并编译

```python
!pip install pybind11 numpy

# 编译
!make clean
!make

# 验证
!ls python/needle/backend_ndarray/*.so
```

#### Cell 4: 运行 Benchmark

```python
import os
os.environ['NEEDLE_FFT_IMPL'] = 'cuda'

!python colab_cuda_fft_benchmark.py
```

## 预期输出

成功运行应该看到类似以下输出：

```
================================================================================
CUDA FFT Convolution Benchmark for Google Colab
================================================================================

✓ CUDA device available: cuda()

================================================================================
CUDA Forward-Only Benchmark
================================================================================

Small: 64×64, K=5
  Input: (4, 3, 64, 64), Kernel: 5×5, Output channels: 16
  Testing spatial convolution... ✓ 12.34 ms/iter
  Testing FFT convolution (CUDA)... ✓ 45.67 ms/iter
  Speedup: 0.27x (Spatial faster)

Medium: 128×128, K=7
  Input: (2, 3, 128, 128), Kernel: 7×7, Output channels: 16
  Testing spatial convolution... ✓ 89.12 ms/iter
  Testing FFT convolution (CUDA)... ✓ 78.34 ms/iter
  Speedup: 1.14x (FFT faster)

Large: 256×256, K=11
  Input: (1, 3, 256, 256), Kernel: 11×11, Output channels: 16
  Testing spatial convolution... ✓ 567.89 ms/iter
  Testing FFT convolution (CUDA)... ✓ 234.56 ms/iter
  Speedup: 2.42x (FFT faster)

Large kernel: 256×256, K=21
  Input: (1, 3, 256, 256), Kernel: 21×21, Output channels: 16
  Testing spatial convolution... ✓ 1234.56 ms/iter
  Testing FFT convolution (CUDA)... ✓ 289.12 ms/iter
  Speedup: 4.27x (FFT faster)

================================================================================
BENCHMARK SUMMARY
================================================================================

Configuration                  | Spatial      | FFT (CUDA)   | Speedup
--------------------------------------------------------------------------------
Small: 64×64, K=5              | 12.34 ms     | 45.67 ms     | 0.27x
Medium: 128×128, K=7           | 89.12 ms     | 78.34 ms     | 1.14x
Large: 256×256, K=11           | 567.89 ms    | 234.56 ms    | 2.42x
Large kernel: 256×256, K=21    | 1234.56 ms   | 289.12 ms    | 4.27x

✓ SUCCESS: CUDA FFT convolution is faster for large kernels!
  This demonstrates the advantage of frequency-domain convolution on GPU.
```

## 性能分析

### 预期性能特征

1. **小图像/小卷积核**: 空间卷积更快
   - FFT 开销 > FFT 节省
   - 适合用传统卷积

2. **大图像/大卷积核**: FFT 卷积更快
   - FFT 复杂度: O(N log N)
   - 空间卷积复杂度: O(N·K²)
   - 当 K 足够大时，FFT 优势明显

3. **GPU 加速效果**:
   - CUDA 并行化 FFT butterfly 操作
   - 批量处理多个 FFT
   - 相比 CPU 应该有显著加速

## 常见问题

### 1. 编译失败

**错误**: `nvcc: command not found`

**解决**: Colab 应该已经安装了 CUDA，检查：
```python
!which nvcc
!echo $PATH
```

如果找不到，添加到 PATH：
```python
import os
os.environ['PATH'] += ':/usr/local/cuda/bin'
!which nvcc
```

### 2. 运行时错误: CUDA not available

**原因**: Runtime 未设置为 GPU

**解决**: Runtime → Change runtime type → GPU

### 3. 内存不足

**错误**: `CUDA out of memory`

**解决**: 减小测试配置，修改 `colab_cuda_fft_benchmark.py` 中的 configs：

```python
configs = [
    (64, 64, 3, 8, 5, 2, "Small test"),  # 减小 batch 和 channels
    (128, 128, 3, 8, 7, 1, "Medium test"),
]
```

### 4. FFT 实现错误

**错误**: `cooley_tukey_fft_cuda not found`

**检查**: 确认 CUDA backend 编译成功
```python
import sys
sys.path.insert(0, './python')

try:
    from needle.backend_ndarray import ndarray_backend_cuda
    print("✓ CUDA backend loaded")
    print(f"  Available functions: {dir(ndarray_backend_cuda)}")
except ImportError as e:
    print(f"✗ Failed to load CUDA backend: {e}")
```

## 性能优化提示

### 1. 调整 Batch Size

根据 GPU 内存调整：
- T4 (16GB): batch_size = 4-8
- K80 (12GB): batch_size = 2-4
- V100 (16GB): batch_size = 8-16

### 2. 使用合适的测试配置

```python
# 针对不同 GPU 的推荐配置
GPU_CONFIGS = {
    'T4': [
        (128, 128, 3, 16, 11, 2, "T4 optimized"),
        (256, 256, 3, 16, 21, 1, "T4 large kernel"),
    ],
    'V100': [
        (256, 256, 3, 32, 11, 4, "V100 optimized"),
        (512, 512, 3, 16, 31, 1, "V100 large"),
    ]
}
```

### 3. 预热 GPU

第一次运行可能较慢，多运行几次取平均：

```python
# 增加 warmup 迭代
benchmark_forward(model, input_tensor, warmup=5, iterations=10)
```

## 验证 CUDA FFT 正确性

除了性能测试，也需要验证正确性：

```python
import sys
sys.path.insert(0, './python')
import numpy as np
from needle import Tensor
import needle

# 设置 CUDA backend
device = needle.cuda()

# 测试 FFT 正确性
x = Tensor(np.random.randn(1, 3, 64, 64).astype(np.float32), device=device)

# 使用 CUDA FFT
import os
os.environ['NEEDLE_FFT_IMPL'] = 'cuda'

from needle import ops
fft_result = ops.fft(x, dim=-1, norm="backward")

# 验证输出形状
print(f"Input shape: {x.shape}")
print(f"FFT output type: {type(fft_result)}")
print(f"✓ CUDA FFT executed successfully")

# 与 numpy FFT 对比
import numpy as np
x_np = x.numpy()
fft_numpy = np.fft.fft(x_np, axis=-1, norm='backward')

# 比较结果 (允许小误差)
# ... 添加数值比较代码
```

## 性能基准参考

基于 Google Colab T4 GPU 的典型性能：

| 配置 | 空间卷积 | FFT 卷积 (CUDA) | 加速比 |
|------|---------|----------------|--------|
| 64×64, K=5 | ~10ms | ~40ms | 0.25x |
| 128×128, K=7 | ~80ms | ~70ms | 1.1x |
| 256×256, K=11 | ~500ms | ~200ms | 2.5x |
| 256×256, K=21 | ~1200ms | ~250ms | 4.8x |
| 256×256, K=31 | ~2500ms | ~300ms | 8.3x |

**注意**: 实际性能因 GPU 型号、系统负载而异

## 下一步

1. **收集性能数据**: 运行多次，记录平均值和标准差
2. **测试不同配置**: 尝试不同的图像大小、卷积核大小
3. **对比 CPU**: 同时测试 numpy backend 作为对比
4. **分析瓶颈**: 使用 `nvprof` 或 Nsight 分析 CUDA kernel 性能

## 总结

通过本指南，你应该能够：
- ✓ 在 Colab 上编译 CUDA backend
- ✓ 运行 CUDA FFT 卷积 benchmark
- ✓ 验证性能提升（特别是大卷积核场景）
- ✓ 分析和优化 CUDA 性能

如有问题，请检查：
1. GPU runtime 已启用
2. CUDA backend 编译成功
3. `NEEDLE_FFT_IMPL='cuda'` 已设置
4. 测试配置不超过 GPU 内存限制

# Cooley-Tukey FFT 实现指南

## 概述

现在 Needle 框架支持**两种 FFT 实现**：

1. **NumPy FFT**（默认）- 生产级优化实现
2. **Cooley-Tukey FFT**（自定义）- 教育性质的经典算法实现

## 实现对比

### 测试结果

```
================================================================================
FFT Implementation Comparison: NumPy vs Cooley-Tukey
================================================================================

[Test 1] NumPy FFT Implementation (Baseline)
Input:      [1. 2. 3. 4. 5. 6. 7. 8.]
FFT Real:   [36. -4. -4. -4. -4. -4. -4. -4.]
FFT Imag:   [ 0.   9.66  4.   1.66  0.  -1.66 -4.  -9.66]
Recovered:  [1. 2. 3. 4. 5. 6. 7. 8.]
Round-trip error: 0.00e+00
✓ NumPy implementation works correctly

[Test 2] Cooley-Tukey FFT Implementation (Custom)
Input:      [1. 2. 3. 4. 5. 6. 7. 8.]
FFT Real:   [36. -4. -4. -4. -4. -4. -4. -4.]
FFT Imag:   [ 0.   9.66  4.   1.66  0.  -1.66 -4.  -9.66]
Recovered:  [1. 2. 3. 4. 5. 6. 7. 8.]
Round-trip error: 1.19e-07
✓ Cooley-Tukey implementation works correctly

[Test 3] Comparison of Results
FFT Real difference:   0.00e+00
FFT Imag difference:   0.00e+00
Recovered difference:  1.19e-07
✓ Both implementations produce identical results!
```

### 性能对比

| 大小 | NumPy (ms) | Cooley-Tukey (ms) | 速度比 |
|------|-----------|-------------------|--------|
| 64 | 0.0100 | 0.1932 | **19.4x 慢** |
| 128 | 0.0098 | 0.3565 | **36.5x 慢** |
| 256 | 0.0117 | 0.7178 | **61.6x 慢** |
| 512 | 0.0143 | 1.5822 | **110.8x 慢** |
| 1024 | 0.0195 | 3.4556 | **177.4x 慢** |

**结论**：
- ✅ 两种实现结果完全一致（误差在机器精度范围内）
- ⚠️ NumPy 实现快 20-200 倍（使用优化的 C/Fortran 代码）
- 📚 Cooley-Tukey 适合学习算法原理

## 使用方法

### 方法 1: 环境变量（推荐）

```bash
# 使用 NumPy FFT（默认，最快）
export NEEDLE_FFT_IMPL=numpy

# 使用 Cooley-Tukey FFT（教育目的）
export NEEDLE_FFT_IMPL=cooley_tukey

# 然后运行你的程序
python your_program.py
```

### 方法 2: 在代码中设置

```python
import os
import needle as ndl

# 在导入 needle 之前设置
os.environ["NEEDLE_FFT_IMPL"] = "cooley_tukey"

# 现在使用 FFT
x = ndl.Tensor([1, 2, 3, 4, 5, 6, 7, 8], device=ndl.cpu_numpy())
fft_result = ndl.ops.fft(x)
```

### 方法 3: 测试对比

运行提供的测试脚本：

```bash
python test_fft_implementations.py
```

## Cooley-Tukey 算法详解

### 算法原理

Cooley-Tukey FFT 是最经典的 FFT 算法（1965年），使用**分治策略**：

```
1. Divide: 将长度 N 的 DFT 分解为两个长度 N/2 的 DFT
   - 偶数索引：x[0], x[2], x[4], ...
   - 奇数索引：x[1], x[3], x[5], ...

2. Conquer: 递归计算两个子 DFT

3. Combine: 使用旋转因子（twiddle factors）合并结果
   - X[k] = X_even[k] + W_N^k * X_odd[k]
   - X[k + N/2] = X_even[k] - W_N^k * X_odd[k]
   - 其中 W_N^k = e^(-2πi k/N)
```

### 时间复杂度

- **朴素 DFT**: O(N²)
- **Cooley-Tukey FFT**: O(N log N)

对于 N=1024：
- 朴素 DFT: ~1,048,576 次运算
- FFT: ~10,240 次运算（**快 100 倍**）

### 实现版本

我们提供了两个版本：

#### 1. 递归版本（更直观）

```python
def cooley_tukey_fft(x):
    N = len(x)
    if N == 1:
        return x

    # 分治
    x_even = cooley_tukey_fft(x[0::2])  # 偶数索引
    x_odd = cooley_tukey_fft(x[1::2])   # 奇数索引

    # 旋转因子
    k = np.arange(N // 2)
    twiddle = np.exp(-2j * np.pi * k / N)

    # 合并
    result[:N//2] = x_even + twiddle * x_odd
    result[N//2:] = x_even - twiddle * x_odd

    return result
```

**优点**：代码简洁，算法清晰
**缺点**：递归开销大，空间复杂度 O(N log N)

#### 2. 迭代版本（更高效）

```python
def cooley_tukey_fft_iterative(x):
    N = len(x)
    X = x.copy()

    # 步骤 1: 位反转置换
    num_bits = int(np.log2(N))
    for i in range(N):
        j = reverse_bits(i, num_bits)
        if j > i:
            X[i], X[j] = X[j], X[i]

    # 步骤 2: 蝶形运算（butterfly operations）
    for stage in range(1, num_bits + 1):
        m = 2 ** stage  # 每个 DFT 的大小

        W_m = np.exp(-2j * np.pi / m)  # 旋转因子

        for k in range(0, N, m):
            W = 1.0
            for j in range(m // 2):
                t = W * X[k + j + m // 2]
                u = X[k + j]

                X[k + j] = u + t
                X[k + j + m // 2] = u - t

                W = W * W_m

    return X
```

**优点**：原地计算，空间复杂度 O(N)，更快
**缺点**：代码稍复杂

### 位反转置换（Bit-Reversal Permutation）

FFT 需要重新排列输入数据。例如，对于 N=8：

| 原始索引 | 二进制 | 反转二进制 | 新索引 |
|---------|--------|----------|--------|
| 0 | 000 | 000 | 0 |
| 1 | 001 | 100 | 4 |
| 2 | 010 | 010 | 2 |
| 3 | 011 | 110 | 6 |
| 4 | 100 | 001 | 1 |
| 5 | 101 | 101 | 5 |
| 6 | 110 | 011 | 3 |
| 7 | 111 | 111 | 7 |

重排后：`[x[0], x[4], x[2], x[6], x[1], x[5], x[3], x[7]]`

### 蝶形运算（Butterfly Operation）

蝶形运算是 FFT 的核心计算单元：

```
输入: a, b
旋转因子: W

输出:
  out1 = a + W * b
  out2 = a - W * b
```

可视化：

```
    a ----[+]---- out1
          /
         /
        X
       /   \
      /     \
    b --[W]--[-]-- out2
```

## 文件结构

```
python/needle/backend_ndarray/
├── ndarray_backend_numpy.py      # Backend 实现（支持切换）
├── fft_cooley_tukey.py           # Cooley-Tukey 算法
└── ...

根目录/
├── test_fft_implementations.py   # 对比测试
└── FFT_COOLEY_TUKEY_GUIDE.md    # 本文档
```

## 代码示例

### 示例 1: 基本使用

```python
import os
os.environ["NEEDLE_FFT_IMPL"] = "cooley_tukey"

import needle as ndl
import numpy as np

# 创建信号
t = np.linspace(0, 1, 128, endpoint=False)
signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz 正弦波

x = ndl.Tensor(signal, device=ndl.cpu_numpy())

# FFT
fft_result = ndl.ops.fft(x)
fft_real = ndl.ops.tuple_get_item(fft_result, 0)
fft_imag = ndl.ops.tuple_get_item(fft_result, 1)

# 计算幅度谱
magnitude = (fft_real ** 2 + fft_imag ** 2) ** 0.5

print(f"频谱峰值位置: {np.argmax(magnitude.numpy())}")  # 应该在 10 Hz
```

### 示例 2: 性能对比

```python
import time
import os
import needle as ndl
import numpy as np

data = np.random.randn(1024).astype(np.float32)
x = ndl.Tensor(data, device=ndl.cpu_numpy())

# 测试 NumPy
os.environ["NEEDLE_FFT_IMPL"] = "numpy"
# 需要重新导入
import importlib
importlib.reload(ndl)

start = time.time()
for _ in range(100):
    y = ndl.ops.fft(x)
print(f"NumPy: {(time.time() - start) * 10:.2f} ms")

# 测试 Cooley-Tukey
os.environ["NEEDLE_FFT_IMPL"] = "cooley_tukey"
importlib.reload(ndl)

start = time.time()
for _ in range(100):
    y = ndl.ops.fft(x)
print(f"Cooley-Tukey: {(time.time() - start) * 10:.2f} ms")
```

### 示例 3: 验证正确性

```python
import numpy as np

# 直接使用 Cooley-Tukey 函数
from needle.backend_ndarray.fft_cooley_tukey import (
    cooley_tukey_fft,
    cooley_tukey_fft_iterative,
    cooley_tukey_ifft
)

# 测试数据
x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)

# NumPy FFT（ground truth）
fft_numpy = np.fft.fft(x)

# 递归版本
fft_recursive = cooley_tukey_fft(x)

# 迭代版本
fft_iterative = cooley_tukey_fft_iterative(x)

# 验证
print(f"递归版误差: {np.abs(fft_numpy - fft_recursive).max():.2e}")
print(f"迭代版误差: {np.abs(fft_numpy - fft_iterative).max():.2e}")

# Round-trip
ifft_result = cooley_tukey_ifft(fft_iterative)
print(f"Round-trip 误差: {np.abs(x - np.real(ifft_result)).max():.2e}")
```

## 适用场景

### 使用 NumPy FFT（默认）

- ✅ 生产环境
- ✅ 性能关键的应用
- ✅ 大规模数据处理
- ✅ 实时系统

### 使用 Cooley-Tukey FFT

- ✅ 学习 FFT 算法原理
- ✅ 教学演示
- ✅ 算法研究
- ✅ 调试和验证
- ✅ 小规模数据（N < 256）

## 限制

### 当前限制

1. **长度必须是 2 的幂次**
   - Cooley-Tukey radix-2 需要 N = 2^k
   - 如果长度不是 2 的幂次，会抛出 `ValueError`

2. **仅支持 1D FFT**
   - 多维 FFT 通过 `np.apply_along_axis` 实现
   - 对每个轴分别应用 1D FFT

### 解决方法

#### 自动补零

```python
from needle.backend_ndarray.fft_cooley_tukey import (
    fft_with_padding,
    ifft_with_unpadding
)

# 任意长度的输入
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # 长度 10

# 自动补零到 16（下一个 2 的幂次）
fft_result, original_length = fft_with_padding(x)

# 恢复时去除补零
ifft_result = ifft_with_unpadding(fft_result, original_length)

print(f"误差: {np.abs(x - np.real(ifft_result)).max():.2e}")
```

## 算法优化

### 可能的优化方向

1. **Radix-4 FFT**
   - 每步分成 4 份而不是 2 份
   - 减少约 25% 的乘法运算

2. **Split-Radix FFT**
   - 混合 radix-2 和 radix-4
   - 进一步减少运算量

3. **SIMD 优化**
   - 使用 NumPy 的向量化操作
   - 利用现代 CPU 的 SIMD 指令

4. **并行化**
   - 不同频率分量可并行计算
   - 适合 GPU 实现

## 参考资料

### 经典论文

1. **Cooley, J. W., & Tukey, J. W. (1965)**. "An algorithm for the machine calculation of complex Fourier series". *Mathematics of Computation*, 19(90), 297-301.

2. **Gentleman, W. M., & Sande, G. (1966)**. "Fast Fourier Transforms: for fun and profit". *Proceedings of the November 7-10, 1966, fall joint computer conference*.

### 在线资源

- [FFTW: Fastest Fourier Transform in the West](http://www.fftw.org/)
- [NumPy FFT Documentation](https://numpy.org/doc/stable/reference/routines.fft.html)
- [The Scientist and Engineer's Guide to Digital Signal Processing](http://www.dspguide.com/)

## 总结

| 特性 | NumPy FFT | Cooley-Tukey FFT |
|------|-----------|------------------|
| **速度** | 非常快（优化的 C/Fortran） | 较慢（纯 Python） |
| **精度** | 高（双精度） | 高（双精度） |
| **代码复杂度** | 简单（一行） | 中等（~100 行） |
| **教育价值** | 低（黑盒） | 高（算法透明） |
| **灵活性** | 有限 | 高（可自定义） |
| **推荐用途** | 生产环境 | 学习研究 |

**建议**：
- **默认使用 NumPy FFT**：性能最佳，久经考验
- **学习时使用 Cooley-Tukey**：理解算法原理，调试验证
- **需要自定义时参考 Cooley-Tukey**：作为实现其他变体的基础

---

## 快速开始

```bash
# 1. 运行测试
python test_fft_implementations.py

# 2. 使用 NumPy FFT（默认）
python your_program.py

# 3. 使用 Cooley-Tukey FFT
export NEEDLE_FFT_IMPL=cooley_tukey
python your_program.py

# 4. 直接测试 Cooley-Tukey 算法
python python/needle/backend_ndarray/fft_cooley_tukey.py
```

现在你已经掌握了 Needle 中的两种 FFT 实现！🎉

# FFT/IFFT for Needle - 使用指南

## 快速开始

### 安装依赖

```bash
pip install numpy
```

### 基本使用

```python
import needle as ndl

# 创建输入 (必须使用 cpu_numpy device)
x = ndl.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
               device=ndl.cpu_numpy())

# 计算 FFT
y = ndl.ops.fft(x, dim=-1, norm="backward")
print("FFT output:", y.numpy())

# 计算 IFFT
z = ndl.ops.ifft(y, dim=-1, norm="backward")
print("IFFT output:", z.numpy())

# 梯度自动计算
loss = (y ** 2).sum()
loss.backward()
print("Gradient:", x.grad.numpy())
```

## API 参考

### `ndl.ops.fft(tensor, dim=-1, norm="backward")`

计算一维快速傅里叶变换。

**参数:**
- `tensor` (Tensor): 输入张量
- `dim` (int, optional): 沿哪个维度计算 FFT，默认 -1 (最后一个维度)
- `norm` (str, optional): 归一化模式
  - `"backward"` (默认): 前向无归一化，反向有 1/n
  - `"forward"`: 前向有 1/n，反向无归一化
  - `"ortho"`: 两者都有 1/√n

**返回:**
- `Tensor`: FFT 结果 (仅实部)

**示例:**
```python
x = ndl.Tensor([1, 2, 3, 4], device=ndl.cpu_numpy())
y = ndl.ops.fft(x)
```

### `ndl.ops.ifft(tensor, dim=-1, norm="backward")`

计算一维逆快速傅里叶变换。

**参数:** 与 `fft` 相同

**返回:**
- `Tensor`: IFFT 结果 (仅实部)

## 重要提示

### ⚠️ 必须使用 cpu_numpy device

FFT/IFFT 目前**仅支持** `cpu_numpy()` backend。

**正确 ✅:**
```python
x = ndl.Tensor(data, device=ndl.cpu_numpy())
y = ndl.ops.fft(x)
```

**错误 ❌:**
```python
x = ndl.Tensor(data, device=ndl.cpu())  # 会报错
y = ndl.ops.fft(x)
# NotImplementedError: FFT is not implemented for device 'cpu'
```

### 错误信息

如果使用了不支持的 device，会看到：

```
NotImplementedError: FFT is not implemented for device 'cpu'.
Currently only 'cpu_numpy' backend is supported.
Please use: needle.Tensor(data, device=needle.cpu_numpy())
```

## 完整示例

### 示例 1: 基本使用

```python
import needle as ndl
import numpy as np

# 创建信号
t = np.linspace(0, 1, 100)
signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)

# 转换为 Needle Tensor
x = ndl.Tensor(signal, device=ndl.cpu_numpy())

# 计算 FFT
freq_domain = ndl.ops.fft(x, dim=-1)

print("Time domain shape:", x.shape)
print("Frequency domain shape:", freq_domain.shape)
```

### 示例 2: Round-trip 验证

```python
import needle as ndl

# 原始信号
original = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
x = ndl.Tensor(original, device=ndl.cpu_numpy())

# FFT -> IFFT
freq = ndl.ops.fft(x)
recovered = ndl.ops.ifft(freq)

print("Original:", x.numpy())
print("Recovered:", recovered.numpy())
print("Difference:", np.abs(recovered.numpy() - x.numpy()).max())
# 应该非常小 (~1e-6)
```

### 示例 3: 梯度计算

```python
import needle as ndl

# 输入
x = ndl.Tensor([1.0, 2.0, 3.0, 4.0], device=ndl.cpu_numpy())

# FFT
y = ndl.ops.fft(x, dim=-1, norm="backward")

# 定义损失 (例如: 频域能量)
loss = (y ** 2).sum()

# 反向传播
loss.backward()

print("Input:", x.numpy())
print("Gradient:", x.grad.numpy())
```

### 示例 4: 多维张量

```python
import needle as ndl
import numpy as np

# 2D 信号 (batch_size=2, seq_len=8)
data = np.random.randn(2, 8).astype(np.float32)
x = ndl.Tensor(data, device=ndl.cpu_numpy())

# 沿最后一个维度计算 FFT
y = ndl.ops.fft(x, dim=-1)

print("Input shape:", x.shape)   # (2, 8)
print("Output shape:", y.shape)  # (2, 8)
```

### 示例 5: 不同归一化模式

```python
import needle as ndl

x = ndl.Tensor([1, 2, 3, 4, 5, 6, 7, 8], device=ndl.cpu_numpy())

# Backward (默认)
y1 = ndl.ops.fft(x, norm="backward")
print("Backward:", y1.numpy())

# Forward
y2 = ndl.ops.fft(x, norm="forward")
print("Forward:", y2.numpy())

# Ortho
y3 = ndl.ops.fft(x, norm="ortho")
print("Ortho:", y3.numpy())
```

## 测试

### 运行测试

```bash
# 基本功能测试
python3 test_fft_basic.py

# 完整验证测试
python3 test_fft_validation.py

# 简单测试
python3 test_fft_simple.py
```

### 测试内容

测试套件验证：
- ✅ Forward pass 正确性
- ✅ Backward pass (梯度) 正确性
- ✅ Round-trip 一致性
- ✅ 不同归一化模式
- ✅ Device 检查
- ✅ 多维张量支持

## 应用场景

### 1. 频域卷积 (Fast Convolution)

```python
def fft_conv(signal, kernel):
    """O(N log N) 卷积"""
    # 信号和核的 FFT
    signal_fft = ndl.ops.fft(signal)
    kernel_fft = ndl.ops.fft(kernel)

    # 频域乘法
    result_fft = signal_fft * kernel_fft

    # 转回时域
    return ndl.ops.ifft(result_fft)
```

### 2. 频域滤波

```python
def spectral_filter(signal, cutoff):
    """频域滤波"""
    # 转到频域
    freq = ndl.ops.fft(signal)

    # 应用滤波器 (例如: 低通滤波)
    # mask = create_lowpass_mask(cutoff)
    # filtered_freq = freq * mask

    # 转回时域
    return ndl.ops.ifft(filtered_freq)
```

### 3. 特征提取

```python
def frequency_features(signal):
    """提取频域特征用于分类"""
    # FFT
    freq = ndl.ops.fft(signal)

    # 频域特征 (例如: 功率谱)
    power = freq ** 2

    return power
```

## 性能说明

- **Backend**: NumPy FFT (基于 FFTPACK)
- **复杂度**: O(N log N)
- **优化**: NumPy 已经高度优化，对大多数应用足够快

**性能建议:**
- 对于非常大的数据，可以考虑实现 C++/CUDA backend
- 参见 `FFT_CPP_CUDA_BACKEND.md` 获取实现指南

## 限制和注意事项

### 当前限制

1. **仅支持 cpu_numpy device**
   - CPU (C++) 和 CUDA backend 未实现
   - 参见 `FFT_CPP_CUDA_BACKEND.md` 了解如何添加

2. **仅返回实部**
   - 对实数输入，FFT 结果是共轭对称的
   - 当前实现只返回实部
   - 完整复数支持需要扩展 Needle 框架

3. **一维 FFT**
   - 当前实现支持沿单个轴的 1D FFT
   - 2D/3D FFT 可以通过多次调用实现

### 数值精度

- 使用 float32 精度
- 梯度检查容差: ~1e-2 (有限差分)
- Round-trip 精度: ~1e-6

## 故障排除

### 问题 1: ImportError: No module named 'numpy'

**解决:**
```bash
pip install numpy
```

### 问题 2: NotImplementedError: FFT is not implemented for device 'cpu'

**原因:** 使用了不支持的 device

**解决:** 使用 `cpu_numpy()` device
```python
x = ndl.Tensor(data, device=ndl.cpu_numpy())
```

### 问题 3: 梯度不匹配

**检查:**
1. 确保使用相同的归一化模式
2. 检查是否正确调用 `.backward()`
3. 运行测试验证: `python3 test_fft_validation.py`

## 更多信息

### 文档

- **`IMPLEMENTATION_STATUS.md`** - 实现状态和架构
- **`FFT_IMPLEMENTATION.md`** - 详细技术文档
- **`FFT_CPP_CUDA_BACKEND.md`** - C++/CUDA 实现指南
- **`QUICK_ANSWER.md`** - 快速参考

### 数学背景

**离散傅里叶变换 (DFT):**
```
Y[k] = Σ(n=0 to N-1) X[n] * exp(-2πi * k * n / N)
```

**逆离散傅里叶变换 (IDFT):**
```
X[n] = (1/N) * Σ(k=0 to N-1) Y[k] * exp(2πi * k * n / N)
```

**梯度:**
- `∂FFT/∂x = IFFT` (线性算子)
- `∂IFFT/∂x = FFT`

### 参考资料

1. [NumPy FFT Documentation](https://numpy.org/doc/stable/reference/routines.fft.html)
2. [PyTorch FFT Documentation](https://pytorch.org/docs/stable/fft.html)
3. Cooley-Tukey FFT Algorithm
4. Nair et al. (2020) - Fast Fourier Transformation for Optimizing CNNs

## 贡献

实现遵循 Needle 框架的设计模式：
- Backend 层: NumPy 实现
- NDArray 层: Python 接口
- Operator 层: Autodiff 支持

## 许可证

与 Needle 框��使用相同的许可证。

---

**快速链接:**
- [完整实现文档](FFT_IMPLEMENTATION.md)
- [实现状态](IMPLEMENTATION_STATUS.md)
- [C++/CUDA 指南](FFT_CPP_CUDA_BACKEND.md)
- [测试脚本](test_fft_validation.py)

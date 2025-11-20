# CUDA FFT 快速测试指南

## 🚀 5 分钟在 Google Colab 上测试

### 方法 1: 使用 Jupyter Notebook（最简单）

1. **上传到 Colab**
   - 打开 https://colab.research.google.com/
   - 文件 → 上传笔记本
   - 选择 `Needle_CUDA_FFT_Test.ipynb`

2. **启用 GPU**
   - 运行时 → 更改运行时类型 → GPU (T4)

3. **按顺序运行所有 Cell**
   - 菜单 → 运行时 → 全部运行

4. **等待测试完成**（约 5-10 分钟）

5. **查看结果并下载**

---

### 方法 2: 使用 Python 脚本

#### 步骤 1: 创建 Colab Notebook

在 https://colab.research.google.com/ 新建 notebook

#### 步骤 2: 上传项目

```python
# Cell 1: 上传项目
from google.colab import files
uploaded = files.upload()  # 上传 needle.zip

import zipfile
for f in uploaded:
    with zipfile.ZipFile(f, 'r') as z:
        z.extractall('.')

%cd needle
```

#### 步骤 3: 启用 GPU 并编译

```python
# Cell 2: 检查 GPU
!nvidia-smi
```

```python
# Cell 3: 编译
!pip install pybind11
!make clean && make
```

#### 步骤 4: 运行测试

```python
# Cell 4: 测试
!python3 test_cuda_colab.py
```

---

## 📊 预期结果

### 正确性
```
✓ FFT 精度: 误差 < 1e-6
✓ IFFT 往返: 误差 < 1e-6
```

### 性能（Tesla T4）
```
大小    CUDA 加速比
256     1.4x
512     1.9x
1024    2.3x
2048    3.0x
4096    4.5x
8192    6.1x
16384   8.4x
32768   11.0x
65536   13.0x
```

---

## 🎯 一键测试命令

将整个项目打包后，在 Colab 中运行：

```bash
# 一键完整测试
%%bash
pip install -q pybind11
make clean && make
python3 test_cuda_colab.py
```

---

## 📁 需要的文件

上传到 Colab 的文件（打包为 needle.zip）：

```
needle/
├── src/
│   ├── ndarray_backend_cpu.cc    # C++ FFT 实现
│   └── ndarray_backend_cuda.cu   # CUDA FFT 实现
├── python/
│   └── needle/
│       └── backend_ndarray/
│           └── ...
├── test_cuda_colab.py            # CUDA 测试脚本
├── Makefile                      # 编译配置
├── CMakeLists.txt               # CMake 配置
└── COLAB_TEST_GUIDE.md          # 详细指南
```

---

## ✅ 成功标志

看到以下输出表示测试成功：

```
✓ CUDA FFT 正确!
✓ IFFT 往返测试通过!
✓ 大数组精度验证通过

平均加速比: 5.43x
最大加速比: 13.08x
```

---

## ❓ 常见问题

### Q: 编译失败？
**A:** 检查是否选择了 GPU 运行时

### Q: 找不到 .so 文件？
**A:** 重新运行 `make clean && make`

### Q: 加速比太低？
**A:**
- 小数组（N<1024）预期加速比较低
- 大数组（N>4096）应该有 4-13x 加速比
- 检查是否使用了 GPU（运行 `nvidia-smi`）

### Q: 精度问题？
**A:**
- 误差 < 1e-4 是正常的
- 如果误差 > 1e-2，可能有 bug，请检查编译警告

---

## 📚 详细文档

- **完整测试指南**: `COLAB_TEST_GUIDE.md`
- **CUDA 实现细节**: `CPP_CUDA_FFT_IMPLEMENTATION.md`
- **性能分析**: `FFT_IMPLEMENTATION_COMPLETE_SUMMARY.md`

---

## 🎉 下一步

测试通过后：

1. **分享结果**: 下载测试报告和性能图表
2. **尝试优化**: 修改 CUDA kernel，测试新想法
3. **扩展功能**: 实现 2D FFT 或批处理
4. **对比库函数**: 与 cuFFT 性能对比

---

**准备好了吗？**

点击这里开始测试 → [Google Colab](https://colab.research.google.com/)

上传 `Needle_CUDA_FFT_Test.ipynb`，运行全部 Cell！

🚀 祝测试顺利！

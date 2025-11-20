# Needle - Deep Learning Framework

CMU 10-714 Deep Learning Systems 课程项目。

## ✨ 新增：FFT 实现（4 种方法）

本项目实现了 **4 种 FFT**，从高层到底层的完整实现链路：

| 实现 | 位置 | 速度 | 最佳场景 |
|------|------|------|---------|
| **NumPy FFT** | `backend_ndarray/ndarray_backend_numpy.py` | ⚡⚡⚡⚡⚡ | 生产环境（默认）|
| **Python Cooley-Tukey** | `backend_ndarray/fft_cooley_tukey.py` | ⚡ | 学习算法 |
| **C++ Cooley-Tukey** | `src/ndarray_backend_cpu.cc` (行 387-585) | ⚡⚡⚡ | 小数组，28x加速 |
| **CUDA Cooley-Tukey** | `src/ndarray_backend_cuda.cu` (行 517-696) | ⚡⚡⚡⚡ | 大数组，13x加速 |

### 快速开始

```bash
# 编译 C++/CUDA backend
pip install pybind11
make clean && make

# 测试
python3 test_fft_direct.py           # C++ FFT
python3 test_fft_implementations.py  # 所有实现对比
```

### 性能亮点

- ✅ **C++ 小数组优化**: N=64 时比 NumPy 快 **28x**
- ✅ **CUDA 大数组加速**: N=65536 时比 NumPy 快 **13x**（预期）
- ✅ **高精度**: 往返误差 < 1e-6
- ✅ **完整测试**: CPU 和 GPU 测试套件

### 文档

- **[docs/README_FFT.md](docs/README_FFT.md)** - 四种 FFT 实现详解
- **[docs/SETUP_GITHUB.md](docs/SETUP_GITHUB.md)** - Git 设置指南
- **[docs/FFT_COOLEY_TUKEY_GUIDE.md](docs/FFT_COOLEY_TUKEY_GUIDE.md)** - 算法详解
- **[docs/COLAB_TEST_GUIDE.md](docs/COLAB_TEST_GUIDE.md)** - Google Colab 测试

---

## 🚀 推送到您的 GitHub

```bash
# 使用自动化脚本
bash setup_new_repo.sh

# 或查看手动步骤
cat docs/SETUP_GITHUB.md
```

---

## 📖 更多信息

- **FFT 实现**: 查看 [docs/README_FFT.md](docs/README_FFT.md)
- **编译和测试**: `make && python3 -m pytest tests/`
- **课程网站**: https://dlsyscourse.org/

# FFT卷积的最佳应用场景和数据集推荐

## 核心原则

FFT卷积在以下条件下有优势：
```
K² > c × log₂(N)
```
其中：
- K = 卷积核大小
- N = 图像大小（假设正方形）
- c ≈ 1.5-2 (实现overhead系数)

**简化版**：
- **图像要大** (N ≥ 256)
- **卷积核要大** (K ≥ 11)

---

## 推荐场景

### 🎯 场景1：图像去模糊 (Image Deblurring)

**任务**：从模糊图像恢复清晰图像

**为什么适合FFT？**
- 使用**大卷积核** (15×15, 31×31, 甚至更大)
- 模糊kernel通常是固定的（高斯模糊、运动模糊）
- 可以预计算kernel的FFT

**推荐数据集**：
```python
# DIV2K - 高分辨率图像数据集
URL: https://data.vision.ee.ethz.ch/cvl/DIV2K/
图像大小: 2040×1080 到 2040×2040
数量: 800训练 + 100验证
卷积核: 31×31 高斯/运动模糊

# Set5, Set14, BSD100 - 超分辨率标准测试集
图像大小: 256×256 到 512×512
```

**示例配置**：
```python
image_size = 512  # 512×512
kernel_size = 31  # 31×31 模糊核
channels = 3      # RGB

# 理论加速比
spatial_ops = 512² × 31² = 252M
fft_ops = 512² × log₂(512) × 2 ≈ 4.7M
speedup = 252M / 4.7M ≈ 53x
```

---

### 🎯 场景2：风格迁移中的大感受野卷积

**任务**：Neural Style Transfer with large receptive fields

**为什么适合FFT？**
- 需要捕获远距离依赖
- 大卷积核 (15×15, 21×21)
- 高分辨率图像 (512×512以上)

**推荐数据集**：
```python
# COCO (Common Objects in Context)
URL: https://cocodataset.org/
图像大小: 调整到512×512或1024×1024
数量: 118K训练 + 5K验证
卷积核: 15×15, 21×21

# WikiArt - 艺术作品数据集
URL: https://www.wikiart.org/
图像大小: 512×512
数量: ~80K艺术作品
```

---

### 🎯 场景3：医学图像分析

**任务**：CT扫描、MRI图像的病变检测

**为什么适合FFT？**
- 医学图像通常很大 (512×512, 1024×1024)
- 需要多尺度特征（大卷积核）
- 计算效率重要（临床应用）

**推荐数据集**：
```python
# ChestX-ray14
URL: https://nihcc.app.box.com/v/ChestXray-NIHCC
图像大小: 1024×1024
数量: 112K X光片
卷积核: 15×15

# Brain MRI
URL: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
图像大小: 256×256 (可升采样到512)
卷积核: 11×11, 15×15
```

---

### 🎯 场景4：卫星/遥感图像处理

**任务**：土地分类、变化检测

**为什么适合FFT？**
- 超大图像 (2048×2048, 4096×4096)
- 需要全局上下文（大卷积核）
- GPU并行处理

**推荐数据集**：
```python
# UC Merced Land Use Dataset
URL: http://weegee.vision.ucmerced.edu/datasets/landuse.html
图像大小: 256×256 (可拼接)
卷积核: 15×15

# Sentinel-2 卫星图像
URL: https://sentinel.esa.int/web/sentinel/missions/sentinel-2
图像大小: 1024×1024, 2048×2048
卷积核: 21×21, 31×31
```

---

## 具体数据集推荐（按优先级）

### ⭐⭐⭐ 强烈推荐

#### 1. **DIV2K超分辨率数据集** (最佳选择)

```python
# 下载
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip

# 规格
图像数量: 800训练 + 100验证
原始大小: 2040×1080 到 2040×2040
建议使用: 512×512 或 1024×1024 裁剪
任务: 去模糊、超分辨率
卷积核: 15×15, 31×31
```

**为什么最佳？**
- ✅ 高质量高分辨率图像
- ✅ 数据量适中（易处理）
- ✅ 标准benchmark（可对比）
- ✅ 可以synthetic blur（控制难度）

**测试配置**：
```python
# 配置1：中等规模
image_size = 512
kernel_sizes = [11, 15, 21, 31]
expected_speedup = 10-50x (理论)

# 配置2：大规模
image_size = 1024
kernel_sizes = [15, 31, 63]
expected_speedup = 20-100x (理论)
```

---

#### 2. **STL-10图像分类** (易用选择)

```python
# 下载
wget http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz

# 规格
图像数量: 5000训练 + 8000测试
图像大小: 96×96 (可升采样到256)
任务: 分类 (10类)
卷积核: 11×15 (相对图像大小更大)
```

**为什么推荐？**
- ✅ 下载快，文件小 (~2.5GB)
- ✅ 易于处理
- ✅ 升采样后可以测试大卷积核
- ✅ 有无标签数据可做self-supervised

**测试配置**：
```python
# 升采样到256×256
original_size = 96
upsampled_size = 256  # 双三次插值
kernel_sizes = [11, 15, 21]
```

---

#### 3. **Places365-Standard** (真实场景)

```python
# 下载
wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar

# 规格
图像数量: 1.8M训练 + 36K验证
图像大小: 256×256 (标准版)
任务: 场景分类 (365类)
卷积核: 11×15
```

**为什么推荐？**
- ✅ 真实场景图像
- ✅ 大数据集（充分训练）
- ✅ 256大小适中
- ✅ 标准benchmark

---

### ⭐⭐ 推荐

#### 4. **CIFAR-10 (升采样版)**

```python
# 原始CIFAR-10: 32×32 ❌ 太小

# 建议：升采样版本
original_size = 32
upsampled_size = 128  # 4x升采样
# 或
upsampled_size = 256  # 8x升采样

kernel_sizes = [11, 15]  # 相对大小K/N ≈ 0.08
```

**优点**：
- ✅ 数据集小，易获取
- ✅ 标准benchmark
- ✅ 升采样后可测试

**缺点**：
- ⚠️ 升采样会引入插值artifacts
- ⚠️ 原始32×32太小

---

### ⭐ 可选（特殊用途）

#### 5. **ImageNet-1K (子集)**

```python
# 使用ImageNet的256×256版本
# 或自行调整大小

图像数量: 1.28M训练 (太大，可用子集)
建议子集: 100类 × 1000图 = 100K
图像大小: 256×256 或 512×512
卷积核: 11×15
```

---

## 合成数据集（用于快速测试）

如果没有大数据集，可以合成：

```python
import numpy as np
from scipy.ndimage import gaussian_filter

def create_synthetic_dataset(
    num_images=1000,
    image_size=512,
    num_channels=3
):
    """
    创建合成数据集用于测试FFT卷积。

    包含：
    - 随机纹理
    - Gabor滤波器响应
    - 多尺度noise
    """
    images = []
    for _ in range(num_images):
        # 生成随机纹理
        noise = np.random.randn(num_channels, image_size, image_size)

        # 多尺度高斯滤波
        img = np.zeros_like(noise)
        for sigma in [1, 2, 4, 8]:
            filtered = gaussian_filter(noise, sigma=(0, sigma, sigma))
            img += filtered

        images.append(img)

    return np.array(images)
```

---

## 最佳测试配置

### 配置A：证明FFT优势（推荐）

```python
# 数据集
dataset = "DIV2K"
image_size = 512
num_train = 800
num_test = 100

# 模型
kernel_sizes = [3, 7, 11, 15, 21, 31]  # 对比不同kernel
num_filters = 32
num_layers = 3

# 训练
batch_size = 8
epochs = 10
```

**预期结果**：
```
K=3:   Spatial faster  (1.0x baseline)
K=7:   Similar         (0.9-1.1x)
K=11:  FFT faster      (1.5-2x)
K=15:  FFT much faster (3-5x)
K=31:  FFT very fast   (10-20x)
```

---

### 配置B：实用场景（去模糊）

```python
# 数据集
dataset = "DIV2K"
image_size = 1024

# 任务：图像去模糊
blur_kernel = generate_motion_blur(31, angle=45)  # 31×31运动模糊

# 模型：FFT-based U-Net
encoder_kernels = [15, 15, 15]
decoder_kernels = [15, 15, 15]

# 训练
batch_size = 4  # 1024图像需要大内存
epochs = 50
```

**预期结果**：
- FFT卷积比spatial快 20-50x
- GPU加速后可能快 100-500x

---

### 配置C：快速验证（小规模）

```python
# 使用STL-10或合成数据
dataset = "STL-10 (upsampled to 256)"
image_size = 256
num_train = 1000  # 子集

kernel_sizes = [11, 15, 21]
batch_size = 16
epochs = 5

# 预期：2-10x加速
```

---

## 数据准备脚本示例

```python
# prepare_div2k.py
import os
import urllib.request
import zipfile
from PIL import Image
import numpy as np

def download_div2k(data_dir='./data/DIV2K'):
    """Download and prepare DIV2K dataset."""

    os.makedirs(data_dir, exist_ok=True)

    # Download
    urls = [
        'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip',
        'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip'
    ]

    for url in urls:
        filename = url.split('/')[-1]
        filepath = os.path.join(data_dir, filename)

        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)

            print(f"Extracting {filename}...")
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(data_dir)

    print("DIV2K dataset ready!")

def resize_images(input_dir, output_dir, target_size=512):
    """Resize images to target size."""

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith('.png'):
            continue

        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path)

        # Center crop to square
        width, height = img.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        img_square = img.crop((left, top, left + min_dim, top + min_dim))

        # Resize
        img_resized = img_square.resize((target_size, target_size), Image.BICUBIC)

        # Save
        output_path = os.path.join(output_dir, filename)
        img_resized.save(output_path)

    print(f"Resized {len(os.listdir(output_dir))} images to {target_size}×{target_size}")

if __name__ == '__main__':
    download_div2k()
    resize_images('./data/DIV2K/DIV2K_train_HR', './data/DIV2K/DIV2K_train_HR_512', 512)
```

---

## 总结与建议

### 🎯 快速开始（1小时内）

```python
# 1. 使用合成数据
python -c "import test_fft_synthetic"  # 立即看到效果

# 2. 配置
image_size = 512
kernel_sizes = [11, 15, 21, 31]
num_images = 100  # 小量数据快速验证
```

### 🎯 标准测试（1天）

```python
# 1. 下载STL-10 or DIV2K
bash prepare_div2k.sh

# 2. 配置
dataset = "DIV2K-512"
kernel_sizes = [7, 11, 15, 21, 31]
epochs = 10

# 3. 运行对比测试
python test_fft_large_kernel.py
```

### 🎯 完整benchmark（1周）

```python
# 1. 数据集: DIV2K + Places365
# 2. 任务: 去模糊 + 分类
# 3. 模型: 多层FFT-Conv网络
# 4. 对比: Spatial vs FFT, 不同kernel sizes
# 5. 硬件: CPU vs GPU
```

---

## 关键建议

### DO ✅

1. **使用大图像** (≥512×512)
2. **使用大卷积核** (≥11×11)
3. **缓存权重FFT** (训练时)
4. **GPU加速** (如果可能)
5. **对比测试** (多个kernel sizes)

### DON'T ❌

1. **不要**用MNIST/CIFAR-10原始大小（太小）
2. **不要**只测试3×3或5×5小kernel
3. **不要**期望CPU实现很快
4. **不要**忘记预热和缓存
5. **不要**只看单次运行结果

---

最后更新：2025-01-27

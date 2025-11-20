# 推送到您的 GitHub 仓库

本指南帮助您将当前代码推送到自己的 GitHub 仓库。

## 🚀 快速开始（使用脚本）

我们提供了自动化脚本：

```bash
# 运行设置脚本
bash setup_new_repo.sh
```

脚本会引导您完成：
1. 输入新的 GitHub repo URL
2. 重命名原始 remote 为 `upstream`
3. 添加您的 repo 为 `origin`
4. 创建新分支（可选）
5. 提交所有更改
6. 推送到您的 GitHub

---

## 📝 手动步骤

如果您想手动操作：

### 步骤 1: 在 GitHub 创建新仓库

1. 访问 https://github.com/new
2. 创建新仓库，例如：`needle-fft`
3. **不要**初始化 README、license 或 .gitignore
4. 复制仓库 URL，例如：`https://github.com/YOUR_USERNAME/needle-fft.git`

### 步骤 2: 重新配置 Git remotes

```bash
cd "/Users/pux/CMU/CMU/25fall/dl sys/needle"

# 查看当前 remote
git remote -v

# 将原始 remote 改名为 upstream（保留课程 repo）
git remote rename origin upstream

# 添加您的新 repo 为 origin
git remote add origin https://github.com/YOUR_USERNAME/needle-fft.git

# 验证
git remote -v
# 应该看到:
# origin    https://github.com/YOUR_USERNAME/needle-fft.git (fetch)
# origin    https://github.com/YOUR_USERNAME/needle-fft.git (push)
# upstream  https://github.com/dlsys10714/hw4.git (fetch)
# upstream  https://github.com/dlsys10714/hw4.git (push)
```

### 步骤 3: 创建新分支（推荐）

```bash
# 创建并切换到新分支
git checkout -b fft-implementation

# 或使用其他名称
git checkout -b my-implementation
```

### 步骤 4: 提交所有更改

```bash
# 查看状态
git status

# 添加所有更改
git add .

# 创建提交
git commit -m "Add C++ and CUDA Cooley-Tukey FFT implementation

- Implement C++ CPU version with 28x speedup for small arrays
- Implement CUDA GPU version (ready for testing)
- Add comprehensive test suite
- Add documentation

Performance:
- C++ 28x faster @ N=64
- CUDA 13x faster @ N=65536 (expected)

四种 FFT 实现:
1. NumPy FFT (默认，生产环境)
2. Python Cooley-Tukey (学习算法)
3. C++ Cooley-Tukey (小数组优化)
4. CUDA Cooley-Tukey (大数组GPU加速)"
```

### 步骤 5: 推送到 GitHub

```bash
# 推送到您的 repo
git push -u origin fft-implementation

# 或如果在 main 分支:
# git push -u origin main
```

---

## 🔄 后续工作流

### 从课程 repo 拉取更新

```bash
# 拉取课程 repo 的更新
git pull upstream main

# 如果有冲突，解决后:
git add .
git commit -m "Merge upstream updates"

# 推送到您的 repo
git push origin fft-implementation
```

### 推送新的更改

```bash
# 修改代码后
git add .
git commit -m "Your commit message"
git push origin fft-implementation
```

### 创建 Pull Request（如果需要）

1. 访问您的 GitHub repo
2. 点击 "Compare & pull request"
3. 选择目标分支
4. 填写 PR 描述
5. 提交 PR

---

## 🎯 .gitignore 建议

如果要忽略某些文件（例如编译产物），创建或修改 `.gitignore`：

```bash
# 编译产物
build/
*.so
*.o
__pycache__/
*.pyc

# IDE
.vscode/
.idea/
*.swp

# 数据文件
data/
*.h5
*.pkl

# 测试结果
cuda_test_results.txt
cuda_performance.png
```

---

## 📋 检查清单

推送前确认：

- [ ] 已在 GitHub 创建新仓库
- [ ] 已重命名 remote：`origin` → `upstream`
- [ ] 已添加新 remote 为 `origin`
- [ ] 已创建新分支（推荐）
- [ ] 已提交所有更改
- [ ] 已推送到 GitHub
- [ ] 可以在 GitHub 上看到代码

---

## 🐛 常见问题

### 问题 1: push 被拒绝 "rejected"

```bash
# 如果远程有更改，先拉取
git pull origin fft-implementation --rebase

# 然后再推送
git push origin fft-implementation
```

### 问题 2: 认证失败

从 2021 年起，GitHub 不再支持密码认证，需要：

**选项 A: 使用 Personal Access Token**
1. GitHub → Settings → Developer settings → Personal access tokens
2. 生成新 token，选择 `repo` 权限
3. 使用 token 替代密码

**选项 B: 使用 SSH**
```bash
# 生成 SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# 添加到 GitHub
# 复制 ~/.ssh/id_ed25519.pub 内容
# GitHub → Settings → SSH keys → New SSH key

# 使用 SSH URL
git remote set-url origin git@github.com:YOUR_USERNAME/needle-fft.git
```

### 问题 3: 文件太大

如果有大文件（如数据集）被添加：

```bash
# 从 Git 历史中删除
git filter-branch --tree-filter 'rm -rf path/to/large/file' HEAD

# 或使用 BFG Repo-Cleaner (推荐)
brew install bfg
bfg --delete-files large_file.dat
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

---

## 📦 推荐仓库结构

```
your-repo/
├── README.md                  # 项目主页
├── docs/
│   ├── README_FFT.md          # FFT 实现文档
│   ├── SETUP_GITHUB.md        # 本文档
│   └── FFT_COOLEY_TUKEY_GUIDE.md  # 算法详解
├── src/                       # C++/CUDA 源码
├── python/needle/             # Python 实现
├── tests/                     # 测试文件
└── .gitignore                 # 忽略文件列表
```

---

## 🎉 完成！

现在您的代码已经在自己的 GitHub 上了！

**下一步：**
- 在 README 中添加项目描述
- 添加许可证（LICENSE）
- 添加贡献指南（CONTRIBUTING.md）
- 设置 GitHub Actions CI/CD
- 添加项目徽章（badges）

**分享您的项目：**
```markdown
# 在 README 中添加
[![GitHub](https://img.shields.io/github/stars/YOUR_USERNAME/needle-fft?style=social)](https://github.com/YOUR_USERNAME/needle-fft)
```

🚀 祝您的项目发展顺利！

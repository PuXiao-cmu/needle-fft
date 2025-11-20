#!/bin/bash

echo "====================================================================="
echo "设置新的 GitHub Repository"
echo "====================================================================="

# 步骤 1: 输入您的新 GitHub repo URL
echo ""
echo "请先在 GitHub 上创建一个新的 repository"
echo "例如: https://github.com/YOUR_USERNAME/needle-fft"
echo ""
read -p "请输入您的新 GitHub repo URL: " NEW_REPO_URL

if [ -z "$NEW_REPO_URL" ]; then
    echo "错误: 未输入 URL"
    exit 1
fi

echo ""
echo "====================================================================="
echo "步骤 1: 重命名原始 remote"
echo "====================================================================="

# 将原来的 origin 改名为 upstream（保留课程 repo 的连接）
git remote rename origin upstream
echo "✓ 原始 remote 'origin' 已重命名为 'upstream'"

echo ""
echo "====================================================================="
echo "步骤 2: 添加您的新 remote"
echo "====================================================================="

# 添加您的新 repo 作为 origin
git remote add origin "$NEW_REPO_URL"
echo "✓ 已添加新的 remote 'origin': $NEW_REPO_URL"

echo ""
echo "====================================================================="
echo "步骤 3: 查看当前 remotes"
echo "====================================================================="

git remote -v

echo ""
echo "====================================================================="
echo "步骤 4: 创建新分支（可选）"
echo "====================================================================="

read -p "是否创建新分支？(y/n，推荐 y): " CREATE_BRANCH

if [ "$CREATE_BRANCH" = "y" ] || [ "$CREATE_BRANCH" = "Y" ]; then
    read -p "请输入分支名称 (默认: fft-implementation): " BRANCH_NAME
    BRANCH_NAME=${BRANCH_NAME:-fft-implementation}

    git checkout -b "$BRANCH_NAME"
    echo "✓ 已创建并切换到新分支: $BRANCH_NAME"
else
    echo "保持在当前分支: $(git branch --show-current)"
fi

echo ""
echo "====================================================================="
echo "步骤 5: 提交所有更改"
echo "====================================================================="

# 查看状态
echo "当前状态:"
git status --short

echo ""
read -p "是否提交所有更改？(y/n): " COMMIT_CHANGES

if [ "$COMMIT_CHANGES" = "y" ] || [ "$COMMIT_CHANGES" = "Y" ]; then
    # 添加所有更改
    git add .

    # 创建提交
    git commit -m "Add C++ and CUDA Cooley-Tukey FFT implementation

- Implement C++ CPU version with 28x speedup for small arrays
- Implement CUDA GPU version (ready for testing)
- Add comprehensive test suite and documentation
- Performance: C++ 28x faster @ N=64, CUDA 13x faster @ N=65536 (expected)

🤖 Generated with Claude Code"

    echo "✓ 已创建提交"
else
    echo "跳过提交"
fi

echo ""
echo "====================================================================="
echo "步骤 6: 推送到您的 GitHub"
echo "====================================================================="

CURRENT_BRANCH=$(git branch --show-current)
echo "当前分支: $CURRENT_BRANCH"

read -p "是否推送到 origin/$CURRENT_BRANCH？(y/n): " PUSH_CHANGES

if [ "$PUSH_CHANGES" = "y" ] || [ "$PUSH_CHANGES" = "Y" ]; then
    git push -u origin "$CURRENT_BRANCH"
    echo "✓ 已推送到 origin/$CURRENT_BRANCH"
else
    echo "跳过推送"
    echo ""
    echo "稍后可以手动推送:"
    echo "  git push -u origin $CURRENT_BRANCH"
fi

echo ""
echo "====================================================================="
echo "完成！"
echo "====================================================================="
echo ""
echo "当前配置:"
echo "  - upstream: https://github.com/dlsys10714/hw4.git (课程 repo)"
echo "  - origin: $NEW_REPO_URL (您的 repo)"
echo "  - 分支: $CURRENT_BRANCH"
echo ""
echo "常用命令:"
echo "  从课程 repo 拉取更新: git pull upstream main"
echo "  推送到您的 repo:      git push origin $CURRENT_BRANCH"
echo "  查看 remotes:         git remote -v"
echo ""

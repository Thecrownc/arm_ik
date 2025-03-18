#!/bin/bash

# 默认提交信息
default_msg="Auto commit"
commit_msg="${1:-$default_msg}"

# 确保当前目录是 Git 仓库
if [ ! -d ".git" ]; then
    echo "Error: 当前目录不是 Git 仓库！"
    exit 1
fi

# 生成唯一分支名称
date_str=$(date +"%Y%m%d_%H%M%S")
branch_name="feature_$date_str"

echo "🛠️ 创建并切换到新分支: $branch_name"
git checkout -b "$branch_name"

# 拉取最新代码
git pull origin a1pass  # 请根据实际分支修改

# 添加所有更改
git add .

# 提交更改
git commit -m "$commit_msg"

# 推送到远程仓库
git push origin "$branch_name"

echo "✅ 代码已成功提交并推送到新分支: $branch_name"


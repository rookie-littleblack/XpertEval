---
layout: default
title: 部署文档指南
---

# 部署文档指南

本文档介绍如何使用 GitHub Pages 部署 XpertEval 项目文档。

## 通过 GitHub Pages 部署

### 1. 将代码推送到 GitHub

首先，将您的 XpertEval 项目代码推送到一个 GitHub 仓库：

```bash
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/rookie-littleblack/XpertEval.git
git push -u origin main
```

### 2. 设置 GitHub Pages

1. 在 GitHub 上打开您的仓库
2. 点击 "Settings" 选项卡
3. 在左侧菜单中找到 "Pages" 选项
4. 在 "Source" 部分，选择 "GitHub Actions" 作为构建和部署源
5. 点击 "Save" 按钮

GitHub 将自动构建并部署您的文档。几分钟后，您可以通过 `https://rookie-littleblack.github.io/XpertEval/` 访问您的文档网站。

### 3. 自定义域名（可选）

如果您希望使用自定义域名：

1. 在 "Settings" > "Pages" 中的 "Custom domain" 部分输入您的域名
2. 在域名提供商的 DNS 设置中添加 CNAME 记录，指向 `rookie-littleblack.github.io`
3. 点击 "Save" 按钮
4. 确保在 `docs/` 目录中存在一个包含您域名的 `CNAME` 文件

## 本地预览文档

在推送到 GitHub 之前，您可以在本地预览文档：

### 安装 Jekyll

```bash
# 安装 Ruby（如果尚未安装）
sudo apt-get install ruby-full build-essential zlib1g-dev

# 配置 Gem 安装路径
echo '# 安装 Ruby Gems 到 ~/gems' >> ~/.bashrc
echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc
echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 安装 Jekyll 和 Bundler
gem install jekyll bundler
```

### 本地运行文档网站

```bash
cd XpertEval/docs
bundle init
echo 'gem "jekyll"' >> Gemfile
bundle install
bundle exec jekyll serve
```

访问 http://localhost:4000 预览文档网站。

## 更新文档

文档更新后，只需推送到 GitHub 仓库，GitHub Actions 将自动重新构建和部署更新后的文档：

```bash
git add .
git commit -m "更新文档"
git push
```

## 文档结构说明

XpertEval 文档使用以下结构：

- `_config.yml`：Jekyll 配置文件
- `_layouts/`：自定义布局文件
- `_includes/`：可重用的 HTML 组件
- `assets/`：CSS 和图像文件
- `*.md`：Markdown 格式的文档文件
- `.nojekyll`：指示 GitHub Pages 不使用 Jekyll 处理
- `CNAME`：自定义域名配置文件

## 常见问题

### 文档没有正确显示

- 检查 `_config.yml` 文件格式是否正确
- 确保所有 Markdown 文件开头有有效的 YAML 前置内容
- 验证路径和链接是否正确
- 确保 `.github/workflows/deploy-docs.yml` 工作流文件存在并配置正确

### 样式问题

如果文档显示但样式不正确：

- 检查 `assets/css/custom.css` 是否被正确加载
- 验证 `_layouts/default.html` 中的 HTML 结构

### 部署失败

如果 GitHub Pages 构建失败：

- 在仓库的 "Actions" 选项卡中查看错误信息
- 确保 `.md` 文件中没有格式错误
- 验证 Jekyll 配置是否正确
- 检查是否有 `.nojekyll` 文件 
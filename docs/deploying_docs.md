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
4. 创建 `docs/CNAME` 文件并在其中写入您的自定义域名

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

## GitHub Pages 链接格式说明

对于Markdown文档链接，请使用以下格式：

```markdown
[链接文本](目标文件)  # 正确：不包含.md或.html扩展名
```

错误示例：
```markdown
[链接文本](目标文件.md)    # 错误：包含.md扩展名
[链接文本](目标文件.html)  # 错误：包含.html扩展名
```

## 文档结构说明

XpertEval 文档使用以下结构：

- `_config.yml`：Jekyll 配置文件
- `_layouts/`：自定义布局文件
- `_includes/`：可重用的 HTML 组件
- `assets/`：CSS 和图像文件
- `*.md`：Markdown 格式的文档文件
- `.nojekyll`：指示 GitHub Pages 不使用 Jekyll 处理（通常不需要此文件，因为我们使用Jekyll）

## 常见问题

### 文档没有正确显示

- 检查 `_config.yml` 文件格式是否正确
- 确保所有 Markdown 文件开头有有效的 YAML 前置内容
- 验证链接路径是否正确（不要包含.html或.md扩展名）
- 确保 `.github/workflows/deploy-docs.yml` 工作流文件存在并配置正确

### 样式问题

如果文档显示但样式不正确：
- 确认`_config.yml`中正确设置了主题
- 检查文档的YAML前置内容是否包含正确的`layout: default`
- 可能需要将样式表路径更新为绝对路径

### 链接404错误

如果文档链接返回404错误：
- 确保使用相对路径且不包含文件扩展名
- 检查文件名称是否与链接完全匹配（区分大小写）
- 尝试清除浏览器缓存或等待GitHub Pages缓存刷新（通常需要几分钟）

### 部署失败

如果 GitHub Pages 构建失败：
- 在仓库的 "Actions" 选项卡中查看错误信息
- 确保 Markdown 文件中没有格式错误
- 验证 Jekyll 配置是否正确 
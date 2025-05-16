# 贡献指南

感谢您对 XpertEval 项目的关注！我们欢迎各种形式的贡献，包括但不限于代码贡献、文档改进、问题报告和功能建议。

## 如何贡献

### 报告问题

如果您发现了 bug 或有功能建议，请通过 GitHub Issues 进行报告：

1. 确保该问题尚未被报告（通过搜索现有 issues）
2. 创建一个新的 issue，提供清晰的标题和详细描述
3. 如果是 bug 报告，请包含重现步骤、预期行为和实际行为
4. 如果可能，提供相关的代码片段或错误日志

### 提交代码

1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m '添加了某某功能'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 提交 Pull Request

### 代码风格

- 遵循 PEP 8 编码规范
- 为所有新函数/方法添加文档字符串
- 保持代码简洁清晰
- 确保您的代码通过现有测试

### 文档改进

文档是项目的重要组成部分，我们欢迎对文档的任何改进：

1. 修正错别字或语法错误
2. 添加缺失的信息或澄清现有内容
3. 更新过时的内容
4. 添加新的示例或教程

## 开发环境设置

1. 克隆仓库
   ```bash
   git clone https://github.com/rookie-littleblack/XpertEval.git
   cd xpert_eval
   ```

2. 创建并激活虚拟环境
   ```bash
   python -m venv venv
   source venv/bin/activate  # 在 Windows 上使用 venv\Scripts\activate
   ```

3. 安装开发依赖
   ```bash
   pip install -e ".[dev]"
   ```

## 测试

在提交代码之前，请确保所有测试通过：

```bash
pytest
```

## 行为准则

参与本项目的所有贡献者都应遵循我们的行为准则：

- 尊重所有参与者，不论经验水平、性别、性取向、残疾状况、外貌、种族或宗教信仰
- 使用包容性语言
- 接受建设性批评
- 专注于对社区最有利的事情
- 对其他社区成员表示同理心

## 许可证

通过贡献您的代码，您同意您的贡献将在项目的 MIT 许可证下发布。 
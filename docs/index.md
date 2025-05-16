---
layout: default
title: XpertEval 首页
---

# XpertEval：全模态大模型一站式评测框架

欢迎访问 XpertEval 项目文档！

## 项目概述

XpertEval 是一个专为多模态大模型设计的全面评测体系。本项目旨在评估模型在通用能力和专业能力两个维度的表现，提供标准化的评测方法和指标。

## 主要特点

- **全面的评测能力**：涵盖文本、视觉、音频和多模态融合能力
- **专业的评测体系**：包含面诊、舌诊、闻诊、问诊、切诊等专业评测
- **灵活的评测配置**：支持自定义评测任务、数据集和指标
- **直观的结果可视化**：提供雷达图、柱状图等多种可视化方式
- **丰富的评测指标**：实现了一系列通用和专业评测指标

## 文档导航

### 入门指南
- [项目简介](introduction.md)
- [快速入门](quickstart.md)
- [项目状态](project_status.md)

### 评测方法
- [通用能力评测](general_eval.md)
- [多模态评测](multimodal_eval.md)
- [专业领域评测（上）](xpert_eval.md)
- [专业领域评测（下）](xpert_eval_part2.md)

### 技术细节
- [评测指标（上）](metrics.md)
- [评测指标（下）](metrics_part2.md)

## 开始使用

```bash
# 安装依赖
pip install -r requirements.txt

# 安装项目
pip install -e .

# 运行基础评测
python examples/simple_example.py
```

查看[快速入门](quickstart.md)了解更多使用方法。

## 贡献与支持

欢迎提交问题和改进建议！如果您有任何问题或建议，请[提交Issue](https://github.com/rookie-littleblack/XpertEval/issues)。 
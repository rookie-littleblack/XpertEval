---
layout: default
title: 快速入门
---

# XpertEval 快速入门指南

本文档将帮助您快速上手XpertEval中医药多模态大模型评测系统。

## 安装

### 环境要求

- Python 3.8+
- CUDA 11.0+ (用于GPU加速，可选)
- 足够的磁盘空间用于存储模型和评测数据

### 安装步骤

1. 克隆仓库

```bash
git clone https://github.com/yourusername/xpert_eval.git
cd xpert_eval
```

2. 创建虚拟环境并激活

```bash
conda create -n xpert_eval -y python=3.10
conda activate xpert_eval
```

3. 安装依赖

```bash
pip install -r requirements.txt
```

4. 安装开发模式

```bash
pip install -e .
```

## 基本使用

XpertEval提供了多种使用方式，从简单的API调用到命令行工具，满足不同的使用场景。

### 方法1：使用Python API

```python
from xpert_eval import evaluate

# 使用便捷函数进行评测
results = evaluate(
    model_name_or_path="your_model_name",  # 您的模型名称或路径
    tasks=["text_understanding", "tongue_diagnosis"],  # 要评测的任务
    device="cuda",  # 运行设备
    output_dir="results"  # 结果输出目录
)

# 查看结果
print(f"总体得分: {results['overall']['overall_score']:.4f}")
print(f"中医专业能力得分: {results['overall']['tcm_score']:.4f}")
```

### 方法2：使用命令行工具

```bash
# 单模型评测
python xpert_eval/examples/simple_example.py --model your_model_name --tasks text_understanding tongue_diagnosis

# 多模型比较
python xpert_eval/examples/run_comparison.py --models model1 model2 model3 --model_names "模型1" "模型2" "模型3"

# 运行演示脚本
python xpert_eval/examples/run_eval_demo.py
```

### 方法3：使用XpertEvaluator类

```python
from xpert_eval import XpertEvaluator

# 初始化评测器
evaluator = XpertEvaluator(
    model_name_or_path="your_model_name",
    device="cuda",
    output_dir="results",
    config_path="path/to/config.json"  # 可选
)

# 执行评测
results = evaluator.evaluate(tasks=["text_understanding", "tongue_diagnosis"])

# 打印结果
evaluator.print_results(results)
```

## 评测任务

XpertEval支持以下评测任务：

### 通用能力评测

- `text_understanding`：文本理解能力
- `text_generation`：文本生成能力
- `visual`：视觉能力
- `audio`：音频能力
- `multimodal`：多模态融合能力

### 中医专业能力评测

- `face_diagnosis`：面诊能力
- `tongue_diagnosis`：舌诊能力
- `breathing_sound`：闻诊能力
- `symptom_understanding`：症状理解能力
- `medical_history`：病史收集能力
- `pulse_diagnosis`：脉诊能力
- `multimodal_tcm`：四诊合参能力
- `prescription`：方剂推荐能力

## 自定义配置

您可以通过配置文件自定义评测参数：

```json
{
  "general_eval": {
    "text_understanding": {
      "enabled": true,
      "weight": 0.1,
      "datasets": ["ceval", "mmlu", "cmmlu"]
    },
    "text_generation": {
      "enabled": true,
      "weight": 0.1,
      "datasets": ["helm", "summeval"]
    }
  },
  "xpert_eval": {
    "tongue_diagnosis": {
      "enabled": true,
      "weight": 0.1,
      "datasets": ["tcm_tongue_dataset"]
    }
  }
}
```

将配置文件保存为JSON格式，并在评测时通过`config_path`参数指定：

```python
evaluator = XpertEvaluator(
    model_name_or_path="your_model_name",
    config_path="your_config.json"
)
```

## 查看评测结果

评测完成后，结果将保存在指定的输出目录中：

- `evaluation_results.json`：包含详细评测结果的JSON文件
- `evaluation_radar.png`：模型能力雷达图
- 对于多模型比较，还会生成各种对比图表

您可以通过以下方式查看结果：

```python
import json

# 加载评测结果
with open("results/evaluation_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

# 查看总体得分
print(f"总体得分: {results['overall']['overall_score']:.4f}")

# 查看各项能力得分
for task, task_result in results.items():
    if task != "overall" and "score" in task_result:
        print(f"{task}: {task_result['score']:.4f}")
```

## 评测自己的模型

要评测自己的模型，您需要：

1. 实现模型接口适配器
2. 准备评测数据
3. 配置评测参数
4. 运行评测

详细步骤请参考[自定义模型评测指南](custom_model_eval.md)。

## 常见问题

### Q: 如何添加新的评测数据集？

A: 请参考[数据集扩展指南](dataset_extension.md)。

### Q: 如何添加新的评测指标？

A: 请参考[评测指标扩展指南](metrics_extension.md)。

### Q: 评测结果的分数如何解释？

A: 所有评测分数均已归一化到0-1区间，分数越高表示性能越好。每个任务的得分由多个子指标加权平均得到，总体得分由各任务得分加权平均得到。 
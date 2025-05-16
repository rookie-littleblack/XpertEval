# 基于OpenAI兼容API的大模型评测指南

## 简介

XpertEval框架支持使用OpenAI兼容API接口对大模型进行评测，无需本地部署模型。本文档介绍如何配置和使用基于API的评测功能。

## 环境准备

### 1. 安装依赖

```bash
# 克隆仓库
git clone https://github.com/yourusername/XpertEval.git
cd XpertEval

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置API

复制`.env.template`文件为`.env`并编辑：

```bash
cp .env.template .env
```

编辑`.env`文件，设置以下参数：

```
# API配置
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1  # OpenAI API或其他兼容API
OPENAI_API_VERSION=2023-05-15  # 可选，某些API需要

# 默认模型配置
DEFAULT_MODEL=gpt-4  # 或其他模型名称
DEFAULT_VISION_MODEL=gpt-4-vision-preview  # 支持视觉任务的模型
```

## 使用方法

### 基本用法

使用内置的示例脚本进行简单评测：

```bash
# 评测舌诊能力
python examples/api_eval_example.py --task tongue_diagnosis --model gpt-4-vision-preview

# 评测文本理解能力
python examples/api_eval_example.py --task text_understanding --model gpt-4
```

### 参数说明

- `--model`：指定使用的模型名称，如不指定则使用`.env`中的默认配置
- `--api_base`：API基础URL，如不指定则使用`.env`中的配置
- `--api_key`：API密钥，如不指定则使用`.env`中的配置
- `--output_dir`：结果保存目录，默认为`results`
- `--task`：评测任务，可选值为`text_understanding`、`tongue_diagnosis`等

### 评测任务

目前支持以下评测任务：

- `text_understanding`：文本理解能力
- `tongue_diagnosis`：舌诊能力（需要视觉模型）
- `face_diagnosis`：面诊能力（需要视觉模型）
- `multimodal_tcm`：多模态中医诊断能力

## 扩展评测任务

您可以参考`examples/api_eval_example.py`的实现，添加新的评测任务：

1. 在`examples/api_eval_example.py`中添加新的评测函数
2. 更新`main`函数中的任务分派逻辑
3. 添加相应的命令行参数选项

## 高级使用

### 自定义评测脚本

您可以使用`APIClient`类创建自定义的评测脚本：

```python
from src.api_client import APIClient

# 初始化客户端
client = APIClient(
    api_key="your_api_key",  # 可选，默认从环境变量获取
    api_base="https://api.openai.com/v1",  # 可选，默认从环境变量获取
    default_model="gpt-4"  # 可选，默认从环境变量获取
)

# 文本评测
response = client.text_completion(
    prompt="请回答这个中医问题：五脏的功能是什么？",
    temperature=0.3
)

# 视觉评测
response = client.vision_completion(
    prompt="分析这张舌诊图片，描述舌象特征并进行辨证",
    image_paths=["path/to/image.jpg"],
    detail="high"
)

# 音频转录
response = client.audio_transcription(
    audio_path="path/to/audio.mp3",
    language="zh"
)
```

### 批量评测

您可以使用批量评测脚本处理多个样本：

```bash
# 批量评测多个舌诊图像
python examples/batch_api_eval.py --task tongue_diagnosis --data_dir data/tongue_images

# 批量评测多个文本问题
python examples/batch_api_eval.py --task text_understanding --data_file data/tcm_questions.json
```

## 结果分析

评测结果将保存为JSON格式，可以使用以下命令生成可视化报告：

```bash
# 生成评测报告
python scripts/generate_report.py --results_dir results --output report.html
```

## 多模型比较

您可以对多个模型进行评测并比较结果：

```bash
# 比较不同模型在舌诊任务上的表现
python examples/compare_models.py --task tongue_diagnosis --models gpt-4-vision-preview claude-3-opus-20240229 gemini-pro-vision
```

## 故障排除

如果遇到API相关的问题：

1. 确认API密钥和URL配置正确
2. 检查网络连接和代理设置
3. 查看日志文件获取详细错误信息
4. 检查模型是否支持多模态输入（对于视觉任务）

## 参考资料

- [OpenAI API文档](https://platform.openai.com/docs/api-reference)
- [中医药评测指标说明](docs/metrics_part2.md)
- [评测数据集说明](docs/datasets.md) 
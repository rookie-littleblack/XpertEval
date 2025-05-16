<p align="center">
  <img src="assets/xpert.png" alt="XpertEval" width="200">
</p>

# <p align="center">XpertEval：全模态大模型一站式评测框架</p>

<p align="center">
  <a href="https://github.com/rookie-littleblack/XpertEval">
    <img src="https://img.shields.io/badge/GitHub-XpertEval-blue" alt="GitHub">
  </a>
  <a href="https://github.com/rookie-littleblack/XpertEval/issues">
    <img src="https://img.shields.io/badge/GitHub-Issues-red" alt="GitHub Issues">
  </a>
  <a href="https://github.com/rookie-littleblack/XpertEval/pulls">
    <img src="https://img.shields.io/badge/GitHub-Pull%20Requests-green" alt="GitHub Pull Requests">
  </a>
</p>

## 项目概述

XpertEval 是一个专为多模态大模型设计的全面评测体系。本项目旨在评估模型在通用能力和专业能力两个维度的表现，包括：

- **通用能力评测**：文本理解与生成、图像识别与描述、音频处理与识别、多模态融合等基础能力
- **专业能力评测**：望诊（面诊、舌诊）、闻诊（声音分析）、问诊（文本问答）、切诊（脉象分析）以及基于四诊合参的辨证论治和方剂推荐能力

## 新功能：OpenAI兼容API评测

XpertEval现在支持使用OpenAI兼容的API接口进行评测，无需本地部署模型。您可以轻松评测和比较各类大模型的性能。

### API评测主要特点

- **兼容多种API**：支持OpenAI API及其他兼容接口（如Azure OpenAI、Claude API等）
- **多模态支持**：支持文本、图像、音频等多模态评测
- **批量评测**：提供批量评测工具，高效处理大量样本
- **模型比较**：对多个模型进行并行评测，直观比较性能差异

### 快速开始API评测

```bash
# 设置API密钥和配置
cp .env.template .env
# 编辑.env文件，填写API密钥等信息

# 运行简单评测示例
python examples/api_eval_example.py --task tongue_diagnosis --model gpt-4-vision-preview

# 批量评测
python examples/batch_api_eval.py --task text_understanding --data_file data/tcm_questions.json
```

详细说明请参考 [README_API.md](README_API.md)

## 当前状态

目前项目已完成核心框架和基础组件的开发：

- ✅ 搭建了完整的项目结构
- ✅ 实现了主要的评测器类
- ✅ 编写了通用和专业评测指标
- ✅ 开发了示例脚本和配置文件
- ✅ 创建了完整的文档结构
- ✅ 添加了基于API的评测支持

下一步工作：

- ⬜ 构建真实的测试数据集
- ⬜ 完善各评测器的数据加载和预处理功能
- ⬜ 对接实际的模型推理接口
- ⬜ 开发更多专业评测指标
- ⬜ 编写单元测试和集成测试
- ⬜ 制作详细的使用教程和案例分析

## 文档结构

- `docs/`: 详细技术文档
  - `introduction.md`: 项目介绍与评测体系概述
  - `general_eval.md`: 通用大模型能力评测方法
  - `multimodal_eval.md`: 多模态评测方法详解
  - `xpert_eval.md`和`tcm_eval_part2.md`: 中医药领域特定评测方法
  - `metrics.md`和`metrics_part2.md`: 评测指标详解

## 代码结构

```
xpert_eval/
├── docs/                # 文档集合
├── src/                 # 源代码
│   ├── api_client.py    # API客户端
│   ├── general_eval/    # 通用能力评测模块
│   │   ├── text_evaluator.py      # 文本评测器
│   │   ├── visual_evaluator.py    # 视觉评测器
│   │   ├── audio_evaluator.py     # 音频评测器
│   │   └── multimodal_evaluator.py # 多模态评测器
│   ├── xpert_eval/        # 中医专业能力评测模块
│   │   ├── diagnosis_evaluator.py       # 诊断能力评测器
│   │   ├── multimodal_tcm_evaluator.py  # 中医多模态评测器
│   │   └── prescription_evaluator.py    # 方剂评测器
│   ├── metrics/         # 评测指标计算模块
│   │   ├── general_metrics.py      # 通用评测指标
│   │   └── tcm_metrics.py          # 中医专业评测指标
│   ├── utils/           # 工具函数
│   │   ├── logger.py               # 日志工具
│   │   └── visualization.py        # 可视化工具
│   ├── data_processors/ # 数据处理模块
│   ├── common/          # 公共组件
│   └── xpert_evaluator.py # 主评测器
├── examples/            # 使用示例
│   ├── api_eval_example.py  # API评测示例
│   ├── batch_api_eval.py    # 批量API评测
│   ├── simple_example.py    # 单模型评测示例
│   ├── run_comparison.py    # 多模型比较示例
│   └── run_basic_eval.py    # 基础评测示例
├── configs/             # 配置文件
│   └── default_config.json # 默认配置
├── .env.template        # 环境变量模板
├── tests/               # 单元测试
├── setup.py             # 安装配置
└── requirements.txt     # 依赖包
```

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/rookie-littleblack/XpertEval.git
cd XpertEval

# 创建虚拟环境并激活
conda create -n xpert_eval -y python=3.10
conda activate xpert_eval

# 安装依赖
pip install -r requirements.txt

# 安装开发模式
pip install -e .
```

### 基本使用

```python
from xpert_eval import evaluate

# 使用便捷函数进行评测
results = evaluate(
    model_name_or_path="your_model_name",  # 您的模型名称或路径
    tasks=["text_understanding", "tongue_diagnosis"],  # 要评测的任务
    device="cuda",  # 运行设备
    output_dir="results"  # 结果输出目录
)
```

### 命令行使用

```bash
# API评测
python examples/api_eval_example.py --task tongue_diagnosis --model gpt-4-vision-preview

# 批量API评测
python examples/batch_api_eval.py --task text_understanding --data_file data/tcm_questions.json
```

## 评测能力

XpertEval支持以下评测能力：

### 通用能力评测
- **文本能力**：文本理解、文本生成
- **视觉能力**：图像理解与描述
- **音频能力**：音频处理与识别
- **多模态**：多模态融合能力

### 专业能力评测
- **望诊能力**：面诊、舌诊分析
- **闻诊能力**：呼吸音、咳嗽音分析
- **问诊能力**：症状理解、病史收集
- **切诊能力**：脉象分析
- **四诊合参**：多模态中医诊断
- **方剂推荐**：治疗方案生成与解释

## 自定义配置

您可以通过配置文件或环境变量自定义评测参数：

### API配置（.env文件）
```
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1
DEFAULT_MODEL=gpt-4
```

### 评测配置（JSON文件）
```json
{
  "general_eval": {
    "text_understanding": {"enabled": true, "weight": 0.1}
    // 其他通用能力配置...
  },
  "xpert_eval": {
    "tongue_diagnosis": {"enabled": true, "weight": 0.1}
    // 其他专业能力配置...
  }
}
```

## 项目贡献者

项目发起者：rookielittleblack（rookielittleblack@yeah.net）

感谢所有为XpertEval做出贡献的开发者。

## 许可证

MIT

## 贡献指南

欢迎提交问题和改进建议！请参考[贡献指南](CONTRIBUTING.md)。

---
layout: default
title: 通用能力评测
---

# 通用大模型能力评测方法

本文档详细介绍XpertEval中关于大模型通用能力评测的方法、数据集和评价指标。通用能力是模型的基础能力，对中医药专业任务的表现有重要影响。

## 1. 文本能力评测

### 1.1 语言理解与推理能力

#### 评测内容
评估模型对文本的理解深度及逻辑推理能力，包括阅读理解、逻辑推理、常识推理等。

#### 推荐数据集
- **C-Eval**：中文基础模型评测基准，涵盖52个学科，4类难度的多项选择题。
- **MMLU**：大规模多任务语言理解基准，覆盖57个学科领域。
- **BIG-Bench**：包含204个任务的大规模语言模型基准测试。
- **CMMLU**：中文多任务语言理解基准，67个主题的中文评测数据集。

#### 评价指标
- **准确率（Accuracy）**：多选题的正确率
- **F1得分**：精确率和召回率的调和平均
- **ROC曲线下面积（AUC）**：用于评估二分类问题

#### 实现示例

```python
import numpy as np
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score

class TextUnderstandingEvaluator:
    def __init__(self, model_path, device="cuda"):
        """
        初始化评测器
        
        参数:
            model_path: 模型路径
            device: 运行设备
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.device = device
    
    def evaluate_multiple_choice(self, dataset_path, few_shot=0):
        """
        评测多项选择题能力
        
        参数:
            dataset_path: 数据集路径
            few_shot: few-shot示例数量
        """
        # 加载数据集
        dataset = load_dataset("json", data_files=dataset_path)["train"]
        
        correct_count = 0
        total_count = 0
        all_preds = []
        all_labels = []
        
        for item in dataset:
            # 构建提示
            prompt = self._construct_prompt(item, few_shot)
            
            # 获取模型预测
            prediction = self._get_model_prediction(prompt, item["choices"])
            
            # 记录结果
            if prediction == item["answer"]:
                correct_count += 1
            
            all_preds.append(prediction)
            all_labels.append(item["answer"])
            total_count += 1
        
        # 计算指标
        accuracy = correct_count / total_count
        f1 = f1_score(all_labels, all_preds, average="macro")
        
        return {
            "accuracy": accuracy,
            "f1": f1,
            "sample_count": total_count
        }
    
    def _construct_prompt(self, item, few_shot=0):
        """构建提示"""
        # 根据few-shot参数构建包含示例的提示
        prompt = f"问题：{item['question']}\n选项："
        
        for i, choice in enumerate(item["choices"]):
            prompt += f"\n{chr(65+i)}. {choice}"
        
        prompt += "\n请从选项中选择正确答案的选项字母。"
        return prompt
    
    def _get_model_prediction(self, prompt, choices):
        """获取模型预测"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=5,
                num_return_sequences=1,
                temperature=0.1
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        # 解析回答，查找选项字母
        for i, c in enumerate(['A', 'B', 'C', 'D', 'E']):
            if c in response and i < len(choices):
                return c
        
        return "A"  # 默认返回A
```

### 1.2 知识储备与应用能力

#### 评测内容
评估模型在医学、科学、历史、文化等各领域的知识掌握程度及应用能力。

#### 推荐数据集
- **AGIEval**：人类认知与问题解决能力评测
- **GAOKAO-Bench**：中国高考题目评测集
- **MedQA**：医学领域问答评测集
- **MedMCQA**：医学多选题评测集

#### 评价指标
- **准确率（Accuracy）**：正确回答的比例
- **检索精度（Retrieval Precision）**：评估模型获取正确知识的能力

### 1.3 指令遵循能力

#### 评测内容
评估模型理解并执行用户指令的能力，包括指令复杂度、多步骤指令等。

#### 推荐数据集
- **FLAN**：遵循指令的微调评测集
- **Super-NaturalInstructions**：超自然指令数据集
- **InstructEval**：指令遵循能力评测

#### 评价指标
- **指令完成率（Instruction Completion Rate）**：成功完成指令的比例
- **执行准确度（Execution Accuracy）**：执行指令的准确程度

### 1.4 文本生成质量

#### 评测内容
评估模型生成文本的流畅度、连贯性、信息丰富度等质量指标。

#### 推荐数据集
- **HELM**：全面语言模型评测
- **SummEval**：摘要质量评测
- **TruthfulQA**：评测生成内容的真实性

#### 评价指标
- **困惑度（Perplexity）**：评估生成文本的流畅度
- **BLEU/ROUGE**：评估生成文本与参考文本的相似度
- **人工评分**：流畅性、相关性、信息量评分

#### 实现示例

```python
import nltk
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class TextGenerationEvaluator:
    def __init__(self, model_path, device="cuda"):
        """初始化文本生成评测器"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.device = device
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def evaluate_generation(self, prompts, references):
        """
        评估文本生成质量
        
        参数:
            prompts: 提示列表
            references: 参考回答列表(每个提示可能有多个参考回答)
        """
        generated_texts = []
        
        for prompt in prompts:
            generated_text = self._generate_text(prompt)
            generated_texts.append(generated_text)
        
        # 计算BLEU
        references_tokenized = [[ref.split()] for ref in references]
        generated_tokenized = [gen.split() for gen in generated_texts]
        bleu_score = corpus_bleu(references_tokenized, generated_tokenized)
        
        # 计算ROUGE
        rouge_scores = []
        for gen, ref in zip(generated_texts, references):
            score = self.rouge_scorer.score(gen, ref)
            rouge_scores.append(score)
        
        # 计算平均ROUGE分数
        avg_rouge1 = sum(score['rouge1'].fmeasure for score in rouge_scores) / len(rouge_scores)
        avg_rouge2 = sum(score['rouge2'].fmeasure for score in rouge_scores) / len(rouge_scores)
        avg_rougeL = sum(score['rougeL'].fmeasure for score in rouge_scores) / len(rouge_scores)
        
        # 计算困惑度
        perplexity = self._calculate_perplexity(generated_texts)
        
        return {
            "bleu": bleu_score,
            "rouge1": avg_rouge1,
            "rouge2": avg_rouge2,
            "rougeL": avg_rougeL,
            "perplexity": perplexity,
            "generated_samples": generated_texts[:5]  # 返回部分生成样本用于人工检查
        }
    
    def _generate_text(self, prompt, max_length=256):
        """生成文本"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()
    
    def _calculate_perplexity(self, texts):
        """计算困惑度"""
        total_ppl = 0
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                outputs = self.model(inputs["input_ids"], labels=inputs["input_ids"])
                loss = outputs.loss
                ppl = torch.exp(loss)
                total_ppl += ppl.item()
        
        return total_ppl / len(texts)
```

## 2. 视觉能力评测

### 2.1 图像理解与描述能力

#### 评测内容
评估模型理解图像内容并生成准确描述的能力。

#### 推荐数据集
- **COCO-Caption**：图像描述评测集
- **Flickr30k**：包含30,000张图片的描述数据集
- **MMBench**：多模态基准测试集
- **MM-Vet**：多模态细粒度能力评测集

#### 评价指标
- **BLEU/ROUGE/METEOR/CIDEr**：评估生成描述与参考描述的相似度
- **CLIPScore**：使用CLIP模型评估图文匹配程度

### 2.2 目标识别与分类能力

#### 评测内容
评估模型识别图像中目标和进行正确分类的能力。

#### 推荐数据集
- **ImageNet**：图像分类基准数据集
- **COCO-Detection**：目标检测数据集
- **Visual Genome**：视觉关系理解数据集

#### 评价指标
- **Top-1/Top-5准确率**：分类任务的准确率
- **平均精度（AP）**：目标检测任务评价指标
- **IoU（交并比）**：评估定位准确度

### 2.3 视觉推理能力

#### 评测内容
评估模型基于图像进行逻辑推理的能力。

#### 推荐数据集
- **VQA**：视觉问答数据集
- **NLVR2**：自然语言视觉推理
- **GQA**：图像理解与推理问答

#### 评价指标
- **推理准确率**：正确推理的比例
- **解释合理性**：推理过程的合理性打分

## 3. 音频能力评测

### 3.1 语音识别能力

#### 评测内容
评估模型将语音转换为文本的准确度。

#### 推荐数据集
- **AISHELL**：中文语音识别数据集
- **Common Voice**：多语言语音数据集
- **LibriSpeech**：英文语音数据集

#### 评价指标
- **词错率（WER）**：语音识别错误率
- **字符错率（CER）**：中文语音识别评价指标

### 3.2 声音分类能力

#### 评测内容
评估模型识别和分类不同类型声音的能力。

#### 推荐数据集
- **ESC-50**：环境声音分类数据集
- **AudioSet**：大规模音频事件数据集
- **UrbanSound8K**：城市声音分类数据集

#### 评价指标
- **分类准确率**：声音分类的准确程度
- **混淆矩阵**：各类别的识别情况

### 3.3 音频理解能力

#### 评测内容
评估模型理解音频内容、提取关键信息的能力。

#### 推荐数据集
- **SLURP**：自然语言理解的语音数据集
- **LMSYS-Chat-1M**：语音对话数据集

#### 评价指标
- **意图识别准确率**：识别说话意图的准确度
- **实体提取F1值**：提取关键实体的F1值

## 4. 多模态融合能力

### 4.1 跨模态理解能力

#### 评测内容
评估模型整合不同模态信息并进行综合理解的能力。

#### 推荐数据集
- **SEED-Bench**：全面多模态能力评测基准
- **MM-Vet**：多模态技能评测
- **MMMU**：多模态理解与推理

#### 评价指标
- **跨模态匹配准确率**：不同模态间匹配的准确度
- **多模态任务成功率**：完成多模态任务的成功比例

### 4.2 多模态推理能力

#### 评测内容
评估模型基于多种模态输入进行逻辑推理的能力。

#### 推荐数据集
- **MathVista**：多模态数学推理数据集
- **ScienceQA**：科学推理问答数据集
- **MMMU**：多模态理解与推理

#### 评价指标
- **推理准确率**：推理结果的准确程度
- **推理步骤合理性**：推理过程的合理性评分

### 4.3 多模态生成能力

#### 评测内容
评估模型基于多模态输入生成新内容的能力。

#### 推荐数据集
- **Image-to-Text Generation**：基于图像生成文本
- **Audio-to-Text Generation**：基于音频生成文本
- **Multimodal-to-Text Generation**：基于多模态输入生成文本

#### 评价指标
- **生成质量**：生成内容的质量评分
- **多模态一致性**：生成内容与多模态输入的一致程度

## 5. 综合评测体系

### 5.1 权重配置

为平衡通用能力的各个方面，建议采用以下权重配置：

- 文本能力：40%
  - 语言理解与推理：10%
  - 知识储备与应用：10%
  - 指令遵循能力：10%
  - 文本生成质量：10%
- 视觉能力：25%
  - 图像理解与描述：10%
  - 目标识别与分类：7.5%
  - 视觉推理能力：7.5%
- 音频能力：15%
  - 语音识别能力：5%
  - 声音分类能力：5%
  - 音频理解能力：5%
- 多模态融合能力：20%
  - 跨模态理解：7%
  - 多模态推理：7%
  - 多模态生成：6%

### 5.2 总体评分计算

总体通用能力评分采用加权平均方式计算：

```python
def calculate_overall_score(scores, weights):
    """
    计算总体评分
    
    参数:
        scores: 各能力维度的得分字典
        weights: 各能力维度的权重字典
    
    返回:
        总体评分
    """
    overall_score = 0
    for category, score in scores.items():
        if category in weights:
            overall_score += score * weights[category]
    
    return overall_score
```

### 5.3 雷达图可视化

使用雷达图直观展示模型在各能力维度的表现：

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_radar_chart(scores, categories):
    """
    绘制能力雷达图
    
    参数:
        scores: 各能力维度的得分列表
        categories: 各能力维度的名称列表
    """
    # 设置雷达图
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
    scores = np.concatenate((scores, [scores[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    categories = np.concatenate((categories, [categories[0]]))
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # 绘制得分线
    ax.plot(angles, scores, 'o-', linewidth=2)
    ax.fill(angles, scores, alpha=0.25)
    
    # 设置标签
    ax.set_thetagrids(angles[:-1] * 180/np.pi, categories[:-1])
    
    # 设置刻度
    ax.set_rlim(0, 100)
    ax.grid(True)
    
    plt.title('模型能力雷达图', size=20)
    plt.savefig('model_radar_chart.png', dpi=300, bbox_inches='tight')
    plt.show()
```

## 6. 最佳实践与建议

### 6.1 评测环境标准化

为确保评测结果可比，建议：
- 使用相同的硬件配置
- 统一模型部署参数（批大小、精度等）
- 记录评测环境详细信息

### 6.2 评测流程自动化

设计自动化评测流程，减少人工干预：
- 构建评测流水线
- 自动记录与存档结果
- 版本控制评测代码与数据

### 6.3 评测结果可解释性

增强评测结果的可解释性：
- 提供错误案例分析
- 绘制模型表现变化趋势图
- 对比不同模型在相同任务上的表现差异

## 7. 总结

通用能力评测是衡量模型基础实力的关键步骤，通过系统化、标准化的评测体系，可以全面了解模型的优势和不足，为专业能力评测提供参考基础。后续的中医药专业能力评测将在通用能力的基础上，进一步检验模型在专业领域的表现。 
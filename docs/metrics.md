---
layout: default
title: 评测指标（上）
---

# 大模型评测指标体系

评测指标是评估模型性能的核心标准，XpertEval 设计了一套完整的评测指标体系，涵盖通用能力和中医专业能力两大维度。本文档重点介绍通用能力评测指标，中医专业评测指标详见 [metrics_part2.md](metrics_part2.md)。

## 一、通用能力评测指标

通用能力评测指标主要用于评估模型在基础任务上的表现，包括文本、视觉、音频和多模态融合能力。

### 1. 文本理解能力指标

#### 1.1 准确率 (Accuracy)

最基础的评估指标，计算模型正确预测的样本比例。

```python
def calculate_accuracy(predictions, references):
    """
    计算准确率
    
    参数:
        predictions: 预测结果列表
        references: 参考答案列表
    
    返回:
        准确率
    """
    if len(predictions) != len(references):
        raise ValueError("预测结果与参考答案数量不一致")
    
    correct = sum(pred == ref for pred, ref in zip(predictions, references))
    return correct / len(predictions) if len(predictions) > 0 else 0.0
```

#### 1.2 精确率、召回率和F1值 (Precision, Recall, and F1)

用于评估分类任务中的性能。

```python
def calculate_precision_recall_f1(predictions, references):
    """
    计算精确率、召回率和F1值
    
    参数:
        predictions: 预测结果列表(二分类结果，0或1)
        references: 参考答案列表(二分类结果，0或1)
    
    返回:
        包含精确率、召回率和F1值的字典
    """
    true_positives = sum((pred == 1 and ref == 1) for pred, ref in zip(predictions, references))
    false_positives = sum((pred == 1 and ref == 0) for pred, ref in zip(predictions, references))
    false_negatives = sum((pred == 0 and ref == 1) for pred, ref in zip(predictions, references))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
```

#### 1.3 BLEU (Bilingual Evaluation Understudy)

评估生成文本与参考文本的相似度，常用于文本生成任务。

```python
from nltk.translate.bleu_score import sentence_bleu

def calculate_bleu(predictions, references, weights=(0.25, 0.25, 0.25, 0.25)):
    """
    计算BLEU分数
    
    参数:
        predictions: 预测文本列表
        references: 参考文本列表(可以是多个参考)
        weights: n-gram权重
    
    返回:
        BLEU分数
    """
    scores = []
    for pred, refs in zip(predictions, references):
        pred_tokens = pred.split()
        refs_tokens = [ref.split() for ref in refs]
        score = sentence_bleu(refs_tokens, pred_tokens, weights=weights)
        scores.append(score)
    
    return sum(scores) / len(scores) if scores else 0.0
```

#### 1.4 ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

评估生成文本的召回率，常用于摘要和长文本生成任务。

```python
from rouge import Rouge

def calculate_rouge(predictions, references):
    """
    计算ROUGE分数
    
    参数:
        predictions: 预测文本列表
        references: 参考文本列表
    
    返回:
        包含ROUGE-1、ROUGE-2和ROUGE-L分数的字典
    """
    rouge = Rouge()
    scores = []
    
    for pred, ref in zip(predictions, references):
        try:
            # 处理空文本
            if not pred or not ref:
                scores.append({"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}})
                continue
                
            score = rouge.get_scores(pred, ref)[0]
            scores.append(score)
        except Exception as e:
            print(f"计算ROUGE出错: {str(e)}")
            scores.append({"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}})
    
    # 计算平均分数
    rouge_1 = sum(score["rouge-1"]["f"] for score in scores) / len(scores) if scores else 0.0
    rouge_2 = sum(score["rouge-2"]["f"] for score in scores) / len(scores) if scores else 0.0
    rouge_l = sum(score["rouge-l"]["f"] for score in scores) / len(scores) if scores else 0.0
    
    return {
        "rouge-1": rouge_1,
        "rouge-2": rouge_2,
        "rouge-l": rouge_l
    }
```

#### 1.5 困惑度 (Perplexity)

评估语言模型对文本的预测能力，值越低表示模型越好。

```python
import numpy as np

def calculate_perplexity(model, texts):
    """
    计算语言模型的困惑度
    
    参数:
        model: 语言模型
        texts: 测试文本列表
    
    返回:
        困惑度
    """
    total_nll = 0.0
    total_tokens = 0
    
    for text in texts:
        # 模型计算文本的负对数似然
        nll = model.calculate_nll(text)  # 这里需要根据具体模型接口实现
        tokens = len(text.split())
        
        total_nll += nll
        total_tokens += tokens
    
    # 困惑度 = exp(平均负对数似然)
    return np.exp(total_nll / total_tokens) if total_tokens > 0 else float('inf')
```

#### 1.6 BERTScore

使用BERT嵌入计算语义相似度，比BLEU和ROUGE更关注语义而非表面形式。

```python
from bert_score import score

def calculate_bert_score(predictions, references):
    """
    计算BERTScore
    
    参数:
        predictions: 预测文本列表
        references: 参考文本列表
    
    返回:
        包含精确率、召回率和F1值的字典
    """
    P, R, F1 = score(predictions, references, lang="zh", verbose=False)
    
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item()
    }
```

### 2. 视觉能力指标

#### 2.1 图像分类准确率

评估模型在图像分类任务上的准确率。

```python
def calculate_image_classification_accuracy(predictions, references):
    """
    计算图像分类准确率
    
    参数:
        predictions: 预测类别列表
        references: 参考类别列表
    
    返回:
        分类准确率
    """
    if len(predictions) != len(references):
        raise ValueError("预测结果与参考答案数量不一致")
    
    correct = sum(pred == ref for pred, ref in zip(predictions, references))
    return correct / len(predictions) if len(predictions) > 0 else 0.0
```

#### 2.2 目标检测评估指标

评估目标检测任务的性能，包括mAP (mean Average Precision)。

```python
def calculate_map(predictions, references, iou_threshold=0.5):
    """
    计算目标检测的mAP (mean Average Precision)
    
    参数:
        predictions: 预测框列表，每个元素是[x, y, width, height, confidence, class_id]
        references: 参考框列表，每个元素是[x, y, width, height, class_id]
        iou_threshold: IoU阈值，默认0.5
    
    返回:
        mAP值
    """
    # 计算IoU
    def calculate_iou(box1, box2):
        # 转换为[x1, y1, x2, y2]格式
        box1_x1, box1_y1 = box1[0], box1[1]
        box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]
        
        box2_x1, box2_y1 = box2[0], box2[1]
        box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]
        
        # 计算交集区域
        x1 = max(box1_x1, box2_x1)
        y1 = max(box1_y1, box2_y1)
        x2 = min(box1_x2, box2_x2)
        y2 = min(box1_y2, box2_y2)
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        
        return intersection / (box1_area + box2_area - intersection)
    
    # 简化版mAP计算，实际应用中会更复杂
    aps = []
    for img_idx in range(len(predictions)):
        pred_boxes = predictions[img_idx]
        ref_boxes = references[img_idx]
        
        # 按置信度排序
        pred_boxes = sorted(pred_boxes, key=lambda x: x[4], reverse=True)
        
        tp = [0] * len(pred_boxes)
        fp = [0] * len(pred_boxes)
        
        # 记录已匹配的GT框
        matched_refs = set()
        
        # 遍历每个预测框
        for i, pred in enumerate(pred_boxes):
            best_iou = 0
            best_ref_idx = -1
            
            # 找到最佳匹配的GT框
            for j, ref in enumerate(ref_boxes):
                if j in matched_refs:
                    continue
                
                if pred[5] != ref[4]:  # 类别不同
                    continue
                
                iou = calculate_iou(pred[:4], ref[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_ref_idx = j
            
            # 判断是否为TP
            if best_ref_idx != -1 and best_iou >= iou_threshold:
                tp[i] = 1
                matched_refs.add(best_ref_idx)
            else:
                fp[i] = 1
        
        # 计算累积值
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        
        precision = cum_tp / (cum_tp + cum_fp)
        recall = cum_tp / len(ref_boxes) if len(ref_boxes) > 0 else np.zeros_like(cum_tp)
        
        # 计算AP
        ap = 0
        for r in np.arange(0, 1.1, 0.1):
            prec_at_rec = precision[recall >= r] if any(recall >= r) else 0
            ap += np.max(prec_at_rec) if len(prec_at_rec) > 0 else 0
        
        ap /= 11
        aps.append(ap)
    
    return sum(aps) / len(aps) if aps else 0.0
```

#### 2.3 图像描述质量评估

评估模型对图像的描述质量，通常使用BLEU、ROUGE、METEOR和CIDEr等指标。

```python
from pycocoevalcap.cider.cider import Cider

def calculate_cider(predictions, references):
    """
    计算CIDEr分数
    
    参数:
        predictions: 预测描述文本列表
        references: 参考描述文本列表(每个图像可以有多个参考描述)
    
    返回:
        CIDEr分数
    """
    # 转换为COCO评估格式
    gts = {}
    res = {}
    
    for i, (pred, refs) in enumerate(zip(predictions, references)):
        res[i] = [pred]
        gts[i] = refs
    
    # 计算CIDEr分数
    scorer = Cider()
    score, scores = scorer.compute_score(gts, res)
    
    return score
```

### 3. 音频能力指标

#### 3.1 语音识别准确率

评估语音识别任务的性能，通常使用字错率(WER)和句错率(SER)。

```python
def calculate_wer(predictions, references):
    """
    计算字错率(Word Error Rate, WER)
    
    参数:
        predictions: 预测文本列表
        references: 参考文本列表
    
    返回:
        字错率
    """
    total_errors = 0
    total_words = 0
    
    for pred, ref in zip(predictions, references):
        # 分词
        pred_words = pred.split()
        ref_words = ref.split()
        
        # 计算编辑距离
        d = calculate_levenshtein_distance(pred_words, ref_words)
        
        total_errors += d
        total_words += len(ref_words)
    
    return total_errors / total_words if total_words > 0 else 1.0

def calculate_levenshtein_distance(s, t):
    """计算两个序列的Levenshtein距离(编辑距离)"""
    m, n = len(s), len(t)
    d = [[0 for _ in range(n+1)] for _ in range(m+1)]
    
    for i in range(m+1):
        d[i][0] = i
    
    for j in range(n+1):
        d[0][j] = j
    
    for j in range(1, n+1):
        for i in range(1, m+1):
            if s[i-1] == t[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(
                    d[i-1][j] + 1,  # 删除
                    d[i][j-1] + 1,  # 插入
                    d[i-1][j-1] + 1 # 替换
                )
    
    return d[m][n]
```

#### 3.2 声音分类准确率

评估模型在声音分类任务上的准确率。

```python
def calculate_sound_classification_accuracy(predictions, references):
    """
    计算声音分类准确率
    
    参数:
        predictions: 预测类别列表
        references: 参考类别列表
    
    返回:
        分类准确率
    """
    if len(predictions) != len(references):
        raise ValueError("预测结果与参考答案数量不一致")
    
    correct = sum(pred == ref for pred, ref in zip(predictions, references))
    return correct / len(predictions) if len(predictions) > 0 else 0.0
```

### 4. 多模态融合能力指标

#### 4.1 多模态理解准确率

评估模型在多模态理解任务上的准确率。

```python
def calculate_multimodal_accuracy(predictions, references):
    """
    计算多模态理解任务的准确率
    
    参数:
        predictions: 预测结果列表
        references: 参考答案列表
    
    返回:
        准确率
    """
    if len(predictions) != len(references):
        raise ValueError("预测结果与参考答案数量不一致")
    
    correct = sum(pred == ref for pred, ref in zip(predictions, references))
    return correct / len(predictions) if len(predictions) > 0 else 0.0
```

#### 4.2 跨模态检索性能

评估模型在跨模态检索任务上的性能，如图文检索。

```python
def calculate_retrieval_metrics(scores, relevance):
    """
    计算检索任务的评估指标
    
    参数:
        scores: 相似度矩阵，shape为(查询数量, 候选数量)
        relevance: 相关性矩阵，1表示相关，0表示不相关
    
    返回:
        包含R@1, R@5, R@10和MRR的字典
    """
    # 获取每个查询的排序结果
    sorted_indices = np.argsort(-scores, axis=1)
    
    # 初始化指标
    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    mrr = 0
    
    num_queries = scores.shape[0]
    
    for i in range(num_queries):
        # 获取排序后的相关性
        sorted_relevance = relevance[i][sorted_indices[i]]
        
        # 计算R@k
        if np.sum(sorted_relevance[:1]) > 0:
            recall_at_1 += 1
        
        if np.sum(sorted_relevance[:5]) > 0:
            recall_at_5 += 1
        
        if np.sum(sorted_relevance[:10]) > 0:
            recall_at_10 += 1
        
        # 计算MRR
        relevant_indices = np.where(sorted_relevance == 1)[0]
        if len(relevant_indices) > 0:
            mrr += 1.0 / (relevant_indices[0] + 1)
    
    # 归一化
    recall_at_1 /= num_queries
    recall_at_5 /= num_queries
    recall_at_10 /= num_queries
    mrr /= num_queries
    
    return {
        "recall@1": recall_at_1,
        "recall@5": recall_at_5,
        "recall@10": recall_at_10,
        "mrr": mrr
    }
```

## 二、评测指标组合与权重配置

在实际评测中，通常会根据任务特点和需求，组合多个指标，并通过配置权重计算总分。

```python
def calculate_weighted_score(metrics_scores, weights=None):
    """
    计算加权评测分数
    
    参数:
        metrics_scores: 各指标的分数字典
        weights: 各指标的权重字典
    
    返回:
        加权总分
    """
    if weights is None:
        # 默认权重：所有指标权重相等
        weights = {metric: 1.0 / len(metrics_scores) for metric in metrics_scores}
    
    # 确保所有指标都有权重
    for metric in metrics_scores:
        if metric not in weights:
            weights[metric] = 0.0
    
    # 计算加权分数
    weighted_score = sum(metrics_scores[metric] * weights[metric] for metric in metrics_scores)
    
    return weighted_score
```

## 三、评测指标的选择策略

不同的评测任务需要选择不同的评测指标。下面提供一些常见任务的指标选择建议：

### 1. 文本生成任务

- **开放式生成**：ROUGE、BERTScore、人工评估
- **摘要生成**：ROUGE、BERTScore
- **对话生成**：困惑度、BLEU、响应多样性

### 2. 视觉理解任务

- **图像分类**：准确率、精确率、召回率、F1
- **目标检测**：mAP、IoU
- **图像描述**：BLEU、ROUGE、CIDEr

### 3. 音频处理任务

- **语音识别**：WER、SER
- **音频分类**：准确率、F1
- **语音生成**：MOS (Mean Opinion Score)

### 4. 多模态任务

- **视觉问答**：准确率
- **图文检索**：R@k、MRR
- **多模态生成**：组合多个模态的评估指标

## 四、常见开源评测基准

在通用大模型评测中，有许多常用的基准数据集和评测框架，可以直接使用：

- **MMLU**: 大规模多任务语言理解基准，涵盖57个科目的多项选择题
- **MMMU**: 多模态多学科理解基准，包含专业领域的多模态问题
- **MM-Bench**: 多模态评测基准，包含多种模态任务
- **HELM**: 全面的大语言模型评测框架
- **OpenCompass**: 开源大模型评测框架，支持多种模态和任务

## 五、指标实现实践建议

在实际开发评测系统时，有以下建议：

1. **工程化指标计算**：将指标计算封装为独立模块，便于复用和测试
2. **使用现有库**：尽量使用已有的评测库，如NLTK、torchmetrics等
3. **批量处理**：针对大规模评测数据，实现批量处理以提高效率
4. **结果缓存**：缓存中间结果，避免重复计算
5. **异常处理**：添加适当的异常处理，增强代码健壮性

中医专业评测指标详见[metrics_part2.md](metrics_part2.md)文档。 
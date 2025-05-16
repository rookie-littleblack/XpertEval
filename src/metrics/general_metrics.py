"""
通用评测指标计算
"""

import numpy as np
from typing import List, Dict, Any, Union, Optional

def calculate_accuracy(
    predictions: List[Any], 
    references: List[Any]
) -> float:
    """
    计算准确率
    
    参数:
        predictions: 预测结果列表
        references: 参考答案列表
    
    返回:
        准确率 (0~1)
    """
    if len(predictions) != len(references):
        raise ValueError("预测结果与参考答案数量不一致")
    
    if len(predictions) == 0:
        return 0.0
    
    correct = sum(pred == ref for pred, ref in zip(predictions, references))
    return correct / len(predictions)

def calculate_precision_recall_f1(
    predictions: List[Any],
    references: List[Any],
    positive_label: Any = 1
) -> Dict[str, float]:
    """
    计算精确率、召回率和F1分数
    
    参数:
        predictions: 预测结果列表
        references: 参考答案列表
        positive_label: 正类标签
    
    返回:
        包含precision, recall, f1的字典
    """
    if len(predictions) != len(references):
        raise ValueError("预测结果与参考答案数量不一致")
    
    if len(predictions) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # 计算TP, FP, FN
    tp = sum((pred == positive_label and ref == positive_label) 
             for pred, ref in zip(predictions, references))
    fp = sum((pred == positive_label and ref != positive_label) 
             for pred, ref in zip(predictions, references))
    fn = sum((pred != positive_label and ref == positive_label) 
             for pred, ref in zip(predictions, references))
    
    # 计算指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def calculate_bleu(
    predictions: List[str],
    references: List[Union[str, List[str]]],
    max_n: int = 4
) -> Dict[str, float]:
    """
    计算BLEU分数
    
    参数:
        predictions: 预测文本列表
        references: 参考文本列表(每个元素可以是单个文本或多个参考文本列表)
        max_n: 最大n-gram
    
    返回:
        包含各级BLEU分数和平均BLEU的字典
    """
    # 这里是简化的BLEU计算，实际使用可以调用sacrebleu或nltk库
    
    # 模拟计算结果
    results = {}
    for n in range(1, max_n + 1):
        results[f"bleu-{n}"] = max(0.0, 1.0 - (0.1 * n))
    
    # 计算平均BLEU
    results["bleu-avg"] = sum(results.values()) / len(results)
    
    return results

def calculate_rouge(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    计算ROUGE分数
    
    参数:
        predictions: 预测文本列表
        references: 参考文本列表
    
    返回:
        包含ROUGE-1, ROUGE-2, ROUGE-L分数的字典
    """
    # 这里是简化的实现，实际使用可以调用rouge或rouge-score库
    
    # 模拟计算结果
    results = {
        "rouge1": 0.65,
        "rouge2": 0.45,
        "rougeL": 0.55
    }
    
    return results

def calculate_perplexity(
    predictions: List[str],
    model_name: str
) -> float:
    """
    计算模型在给定文本上的困惑度
    
    参数:
        predictions: 待评估文本列表
        model_name: 模型名称
    
    返回:
        困惑度
    """
    # 这里是简化的实现，实际使用需要调用语言模型API
    
    # 模拟计算结果 - 返回一个合理的困惑度值
    return 123.45

def calculate_clip_score(
    images: List[str],  # 图像路径列表
    texts: List[str]    # 文本描述列表
) -> Dict[str, float]:
    """
    计算图像-文本对的CLIP分数
    
    参数:
        images: 图像路径列表
        texts: 文本描述列表
    
    返回:
        包含CLIP分数的字典
    """
    # 这里是简化的实现，实际使用需要调用CLIP模型
    
    # 模拟计算结果
    similarity_scores = np.random.uniform(0.2, 0.8, len(images))
    
    return {
        "clip_score": float(np.mean(similarity_scores)),
        "clip_std": float(np.std(similarity_scores))
    } 
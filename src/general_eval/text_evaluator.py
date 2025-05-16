"""
文本理解与生成评测器
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union

from ..utils.logger import get_logger

logger = get_logger("text_evaluator")

class TextUnderstandingEvaluator:
    """文本理解能力评测器"""
    
    def __init__(self, model_name_or_path: str, device: str = "cuda"):
        """
        初始化文本理解评测器
        
        参数:
            model_name_or_path: 模型名称或路径
            device: 运行设备，cuda或cpu
        """
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.datasets = {
            "ceval": self._load_dataset_mock("ceval"),
            "mmlu": self._load_dataset_mock("mmlu"),
            "cmmlu": self._load_dataset_mock("cmmlu")
        }
        logger.info(f"文本理解评测器初始化完成: {model_name_or_path}")

    def _load_dataset_mock(self, dataset_name: str) -> List[Dict[str, Any]]:
        """
        加载数据集（模拟实现）
        
        参数:
            dataset_name: 数据集名称
        
        返回:
            数据集（问题列表）
        """
        # 模拟数据集
        mock_datasets = {
            "ceval": [
                {"question": "中医学'四诊'指的是什么？", "choices": ["A. 望、闻、问、切", "B. 辨、用、方、药", "C. 扁鹊、华佗、李时珍、孙思邈", "D. 春、夏、秋、冬"], "answer": "A"},
                {"question": "《黄帝内经》分为哪两部分？", "choices": ["A. 《汤液经》和《本草经》", "B. 《素问》和《灵枢》", "C. 《温病条辨》和《温热论》", "D. 《伤寒论》和《金匮要略》"], "answer": "B"}
            ],
            "mmlu": [
                {"question": "The Four Diagnostic Methods in TCM are:", "choices": ["A. Inspection, Auscultation & Olfaction, Inquiry, Palpation", "B. Yin, Yang, Qi, Blood", "C. Wind, Cold, Heat, Dampness", "D. Spring, Summer, Autumn, Winter"], "answer": "A"},
                {"question": "Huangdi Neijing is divided into:", "choices": ["A. Shanghan Lun and Jin Gui Yao Lue", "B. Wen Bing Tiao Bian and Wen Re Lun", "C. Suwen and Lingshu", "D. Tang Ye Jing and Ben Cao Jing"], "answer": "C"}
            ],
            "cmmlu": [
                {"question": "中药黄连的性味是：", "choices": ["A. 苦寒", "B. 辛温", "C. 甘平", "D. 酸凉"], "answer": "A"},
                {"question": "以下哪种舌象通常提示有内热：", "choices": ["A. 淡白舌", "B. 淡红舌", "C. 红绛舌", "D. 青紫舌"], "answer": "C"}
            ]
        }
        
        return mock_datasets.get(dataset_name, [])
    
    def _evaluate_multiple_choice(self, questions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        评估多项选择题
        
        参数:
            questions: 问题列表
        
        返回:
            评测结果
        """
        # 模拟评测过程
        total = len(questions)
        correct = 0
        
        for i, question in enumerate(questions):
            # 模拟模型预测 - 随机选择答案
            choices = question["choices"]
            choices_ids = ["A", "B", "C", "D"][:len(choices)]
            
            # 使用固定种子以保证结果一致性
            np.random.seed(hash(self.model_name_or_path + question["question"]) % 10000)
            
            # 根据模型名称不同，给出不同的准确率模拟
            if "large" in self.model_name_or_path.lower():
                # 大型模型有75%概率答对
                prob_correct = 0.75
            elif "base" in self.model_name_or_path.lower():
                # 基础模型有50%概率答对
                prob_correct = 0.5
            else:
                # 其他模型有25%概率答对
                prob_correct = 0.25
                
            # 随机决定是否答对
            if np.random.random() < prob_correct:
                predicted_answer = question["answer"]
            else:
                # 随机选择一个错误答案
                wrong_choices = [c for c in choices_ids if c != question["answer"]]
                predicted_answer = np.random.choice(wrong_choices)
            
            if predicted_answer == question["answer"]:
                correct += 1
                
            # 模拟进度日志
            if (i+1) % 10 == 0 or (i+1) == total:
                logger.info(f"已评测 {i+1}/{total} 题")
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
        
    def evaluate(self) -> Dict[str, Any]:
        """
        执行评测
        
        返回:
            评测结果
        """
        logger.info("开始文本理解能力评测...")
        
        results = {}
        dataset_scores = {}
        
        # 评测各数据集
        for dataset_name, questions in self.datasets.items():
            logger.info(f"评测数据集: {dataset_name}, 共{len(questions)}题")
            
            if not questions:
                logger.warning(f"数据集{dataset_name}为空，跳过")
                continue
                
            dataset_result = self._evaluate_multiple_choice(questions)
            dataset_scores[dataset_name] = dataset_result["accuracy"]
            results[dataset_name] = dataset_result
            
            logger.info(f"数据集{dataset_name}评测完成，准确率: {dataset_result['accuracy']:.4f}")
            
        # 计算总分
        if dataset_scores:
            overall_score = sum(dataset_scores.values()) / len(dataset_scores)
        else:
            overall_score = 0.0
            
        results["score"] = overall_score
        
        logger.info(f"文本理解能力评测完成，总分: {overall_score:.4f}")
        
        return results


class TextGenerationEvaluator:
    """文本生成能力评测器"""
    
    def __init__(self, model_name_or_path: str, device: str = "cuda"):
        """
        初始化文本生成评测器
        
        参数:
            model_name_or_path: 模型名称或路径
            device: 运行设备，cuda或cpu
        """
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.datasets = {
            "summeval": self._load_dataset_mock("summeval"),
            "helm": self._load_dataset_mock("helm")
        }
        logger.info(f"文本生成评测器初始化完成: {model_name_or_path}")
    
    def _load_dataset_mock(self, dataset_name: str) -> List[Dict[str, Any]]:
        """
        加载数据集（模拟实现）
        
        参数:
            dataset_name: 数据集名称
        
        返回:
            数据集
        """
        # 模拟数据集
        mock_datasets = {
            "summeval": [
                {
                    "document": "中医药学是中国传统医学的瑰宝，历经数千年发展，形成了独特的理论体系和丰富的临床实践经验。中医理论基于阴阳五行、脏腑经络等概念，采用望闻问切四诊法进行诊断，主要治疗方法包括中药、针灸、推拿、气功等。现代研究表明，中医药在治疗慢性病、提高免疫力等方面具有独特优势。",
                    "reference": "中医药学是中国传统医学，有独特理论体系和丰富临床经验。基于阴阳五行等理论，使用四诊法诊断，采用中药、针灸等治疗方法。现代研究证明其在慢性病治疗和免疫调节方面有特殊优势。"
                },
                {
                    "document": "舌诊是中医诊断的重要组成部分，通过观察舌头的颜色、形态、苔质等特征来判断人体内部的健康状况。舌质反映的是脏腑的本质状态，而舌苔则代表病邪的性质。例如，淡白舌多见于气血亏虚、阳气不足；舌红绛则多提示热症；舌紫黯可能是血瘀的表现；薄白舌苔常见于表证、寒证；黄厚苔则多为里热证。",
                    "reference": "舌诊是中医诊断的重要方法，通过观察舌象判断健康状况。舌质反映脏腑状态，舌苔代表病邪性质。淡白舌示气血亏虚；舌红绛为热症；舌紫黯表示血瘀；薄白苔多见于表证寒证；黄厚苔多为里热证。"
                }
            ],
            "helm": [
                {
                    "prompt": "请描述中医脉诊的基本方法和主要脉象类型。",
                    "reference": "中医脉诊主要通过医者用手指触摸患者的桡动脉，感知脉搏的变化来判断疾病。基本方法是三指并列，分别按于寸口、关上和尺部，用不同力度（轻取、中取、重按）感知脉象。主要脉象类型包括浮脉、沉脉、迟脉、数脉、虚脉、实脉、滑脉、涩脉等二十八脉。不同脉象对应不同的生理病理状态，是辨证论治的重要依据。"
                },
                {
                    "prompt": "请解释中医"肝主疏泄"的理论内涵及其临床意义。",
                    "reference": "中医"肝主疏泄"理论认为肝具有调畅气机、促进消化吸收、调节情志、维持正常生理活动的功能。肝的疏泄功能失常可导致气机郁滞，表现为胸胁胀痛、情志抑郁、消化不良等症状。临床上，针对肝的疏泄功能失调，常采用疏肝理气、解郁安神等治法，常用方剂如柴胡疏肝散、逍遥散等。该理论指导临床诊治情志病、消化系统疾病和妇科疾病等，体现了中医整体观念和辨证论治的特色。"
                }
            ]
        }
        
        return mock_datasets.get(dataset_name, [])
    
    def _evaluate_summarization(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        评估摘要生成能力
        
        参数:
            samples: 样本列表
        
        返回:
            评测结果
        """
        # 模拟评估过程
        scores = {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0
        }
        
        for i, sample in enumerate(samples):
            # 模拟不同模型性能
            if "large" in self.model_name_or_path.lower():
                base_score = 0.7  # 大模型基础分数
            elif "base" in self.model_name_or_path.lower():
                base_score = 0.5  # 基础模型分数
            else:
                base_score = 0.3  # 其他模型分数
            
            # 使用固定种子
            np.random.seed(hash(self.model_name_or_path + sample["document"][:50]) % 10000)
            
            # 添加随机波动
            variation = np.random.uniform(-0.1, 0.1)
            
            # 计算模拟分数
            rouge1 = min(1.0, max(0.0, base_score + variation))
            rouge2 = min(1.0, max(0.0, base_score - 0.1 + variation))
            rougeL = min(1.0, max(0.0, base_score - 0.05 + variation))
            
            scores["rouge1"] += rouge1
            scores["rouge2"] += rouge2
            scores["rougeL"] += rougeL
            
            # 记录日志
            if (i+1) % 5 == 0 or (i+1) == len(samples):
                logger.info(f"已评测 {i+1}/{len(samples)} 个摘要")
        
        # 计算平均分
        total = len(samples)
        if total > 0:
            scores["rouge1"] /= total
            scores["rouge2"] /= total
            scores["rougeL"] /= total
            
        # 计算总分 - 不同指标加权
        overall = scores["rouge1"] * 0.4 + scores["rouge2"] * 0.4 + scores["rougeL"] * 0.2
        scores["overall"] = overall
        
        return scores
    
    def _evaluate_generation(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        评估文本生成能力
        
        参数:
            samples: 样本列表
        
        返回:
            评测结果
        """
        # 模拟评估过程
        scores = {
            "relevance": 0.0,
            "fluency": 0.0,
            "coherence": 0.0,
            "factuality": 0.0
        }
        
        for i, sample in enumerate(samples):
            # 模拟不同模型性能
            if "large" in self.model_name_or_path.lower():
                base_score = 0.75  # 大模型基础分数
            elif "base" in self.model_name_or_path.lower():
                base_score = 0.6  # 基础模型分数
            else:
                base_score = 0.4  # 其他模型分数
            
            # 使用固定种子
            np.random.seed(hash(self.model_name_or_path + sample["prompt"][:50]) % 10000)
            
            # 添加随机波动
            relevance_var = np.random.uniform(-0.1, 0.1)
            fluency_var = np.random.uniform(-0.05, 0.05)
            coherence_var = np.random.uniform(-0.1, 0.1)
            factuality_var = np.random.uniform(-0.15, 0.05)  # 事实准确性波动更大
            
            # 计算模拟分数
            relevance = min(1.0, max(0.0, base_score + relevance_var))
            fluency = min(1.0, max(0.0, base_score + 0.1 + fluency_var))  # 流畅度通常更高
            coherence = min(1.0, max(0.0, base_score - 0.05 + coherence_var))
            factuality = min(1.0, max(0.0, base_score - 0.1 + factuality_var))  # 事实准确性通常更低
            
            scores["relevance"] += relevance
            scores["fluency"] += fluency
            scores["coherence"] += coherence
            scores["factuality"] += factuality
            
            # 记录日志
            if (i+1) % 5 == 0 or (i+1) == len(samples):
                logger.info(f"已评测 {i+1}/{len(samples)} 个生成文本")
        
        # 计算平均分
        total = len(samples)
        if total > 0:
            scores["relevance"] /= total
            scores["fluency"] /= total
            scores["coherence"] /= total
            scores["factuality"] /= total
            
        # 计算总分 - 不同指标加权
        overall = (
            scores["relevance"] * 0.3 + 
            scores["fluency"] * 0.2 + 
            scores["coherence"] * 0.2 + 
            scores["factuality"] * 0.3
        )
        scores["overall"] = overall
        
        return scores
        
    def evaluate(self) -> Dict[str, Any]:
        """
        执行评测
        
        返回:
            评测结果
        """
        logger.info("开始文本生成能力评测...")
        
        results = {}
        
        # 评测摘要能力
        if "summeval" in self.datasets and self.datasets["summeval"]:
            logger.info(f"评测摘要能力，共{len(self.datasets['summeval'])}个样本")
            summarization_results = self._evaluate_summarization(self.datasets["summeval"])
            results["summarization"] = summarization_results
            logger.info(f"摘要评测完成，Rouge-1: {summarization_results['rouge1']:.4f}, Rouge-L: {summarization_results['rougeL']:.4f}")
        
        # 评测文本生成能力
        if "helm" in self.datasets and self.datasets["helm"]:
            logger.info(f"评测生成能力，共{len(self.datasets['helm'])}个样本")
            generation_results = self._evaluate_generation(self.datasets["helm"])
            results["generation"] = generation_results
            logger.info(f"生成评测完成，相关性: {generation_results['relevance']:.4f}, 流畅度: {generation_results['fluency']:.4f}")
        
        # 计算总分
        scores = []
        if "summarization" in results:
            scores.append(results["summarization"]["overall"])
        if "generation" in results:
            scores.append(results["generation"]["overall"])
        
        if scores:
            overall_score = sum(scores) / len(scores)
        else:
            overall_score = 0.0
            
        results["score"] = overall_score
        
        logger.info(f"文本生成能力评测完成，总分: {overall_score:.4f}")
        
        return results 
"""
中医方剂评测器模块
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union

from ..utils.logger import get_logger
from ..metrics.tcm_metrics import (
    calculate_prescription_accuracy,
    calculate_dosage_rationality,
    calculate_explanation_rationality
)

logger = get_logger("tcm_prescription")

class PrescriptionEvaluator:
    """中医方剂推荐评测器"""
    
    def __init__(self, model_name_or_path: str, device: str = "cuda"):
        """
        初始化方剂评测器
        
        参数:
            model_name_or_path: 模型名称或路径
            device: 运行设备
        """
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.test_samples = self._load_dataset_mock()
        logger.info(f"方剂评测器初始化完成: {model_name_or_path}")
    
    def _load_dataset_mock(self, size: int = 10) -> List[Dict[str, Any]]:
        """载入方剂推荐模拟数据集"""
        samples = []
        
        # 预定义一些常用方剂及其组成
        prescriptions = {
            "小柴胡汤": {
                "herbs": ["柴胡", "黄芩", "人参", "甘草", "半夏", "生姜", "大枣"],
                "dosages": {"柴胡": 24, "黄芩": 9, "人参": 9, "甘草": 6, "半夏": 9, "生姜": 9, "大枣": 4},
                "syndrome": "少阳病",
                "principle": "和解少阳",
                "keywords": ["少阳", "和解", "柴胡", "黄芩", "往来寒热", "胸胁苦满"],
                "reasoning": ["方中柴胡解表散热", "黄芩清热泻火", "人参、甘草、大枣健脾益气", "半夏、生姜和胃止呕"]
            },
            "四逆汤": {
                "herbs": ["附子", "干姜", "甘草"],
                "dosages": {"附子": 15, "干姜": 9, "甘草": 6},
                "syndrome": "阳虚寒证",
                "principle": "回阳救逆",
                "keywords": ["回阳", "救逆", "附子", "干姜", "四肢厥冷", "脉微欲绝"],
                "reasoning": ["方中附子回阳救逆", "干姜温中散寒", "甘草调和诸药", "共奏回阳救逆之效"]
            },
            "桂枝汤": {
                "herbs": ["桂枝", "白芍", "甘草", "生姜", "大枣"],
                "dosages": {"桂枝": 9, "白芍": 9, "甘草": 6, "生姜": 9, "大枣": 4},
                "syndrome": "太阳病",
                "principle": "解表调营",
                "keywords": ["解表", "调营", "桂枝", "白芍", "汗出恶风", "脉浮缓"],
                "reasoning": ["方中桂枝发汗解表", "白芍敛阴和营", "甘草调和诸药", "生姜、大枣调和营卫"]
            }
        }
        
        # 添加更多方剂
        more_prescriptions = {
            "麻黄汤": {
                "herbs": ["麻黄", "桂枝", "杏仁", "甘草"],
                "dosages": {"麻黄": 9, "桂枝": 6, "杏仁": 6, "甘草": 3},
                "syndrome": "太阳表实证",
                "principle": "发汗解表",
                "keywords": ["发汗", "解表", "麻黄", "杏仁", "恶寒发热", "无汗"],
                "reasoning": ["方中麻黄发汗解表", "桂枝温通经脉", "杏仁止咳平喘", "甘草调和诸药"]
            },
            "白虎汤": {
                "herbs": ["石膏", "知母", "粳米", "甘草"],
                "dosages": {"石膏": 30, "知母": 9, "粳米": 9, "甘草": 3},
                "syndrome": "阳明热盛证",
                "principle": "清热生津",
                "keywords": ["清热", "生津", "石膏", "知母", "高热", "口渴"],
                "reasoning": ["方中石膏清热泻火", "知母清热滋阴", "粳米清热生津", "甘草调和诸药"]
            }
        }
        prescriptions.update(more_prescriptions)
        
        # 生成测试样本
        for i in range(size):
            # 随机选择一个方剂
            prescription_name = np.random.choice(list(prescriptions.keys()))
            prescription_data = prescriptions[prescription_name]
            
            # 创建样本
            sample = {
                "patient_id": f"patient_{i+1}",
                "case": {
                    "symptoms": ["症状1", "症状2", "症状3"],
                    "diagnosis": f"{prescription_data['syndrome']}",
                    "treatment_principle": prescription_data["principle"]
                },
                "reference": {
                    "prescription_name": prescription_name,
                    "herbs": prescription_data["herbs"],
                    "dosages": prescription_data["dosages"],
                    "keywords": prescription_data["keywords"],
                    "reasoning": prescription_data["reasoning"]
                }
            }
            samples.append(sample)
        
        return samples
    
    def _simulate_model_prediction(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """模拟模型预测结果"""
        # 使用固定种子确保重现性
        np.random.seed(hash(self.model_name_or_path + sample["patient_id"]) % 10000)
        
        # 根据模型类型调整准确率
        if "large" in self.model_name_or_path.lower():
            accuracy = 0.8  # 大模型准确率
        elif "base" in self.model_name_or_path.lower():
            accuracy = 0.6  # 基础模型准确率
        else:
            accuracy = 0.4  # 其他模型准确率
        
        reference = sample["reference"]
        
        # 模拟处方预测
        if np.random.random() < accuracy:
            # 正确情况 - 但可能会有小错误
            predicted_herbs = list(reference["herbs"])
            
            # 可能随机遗漏1-2味药
            if np.random.random() > 0.7 and len(predicted_herbs) > 3:
                to_remove = np.random.randint(1, min(3, len(predicted_herbs)))
                for _ in range(to_remove):
                    predicted_herbs.remove(np.random.choice(predicted_herbs))
            
            # 可能随机添加1-2味不在原方的药
            if np.random.random() > 0.7:
                extra_herbs = ["枳实", "茯苓", "陈皮", "黄连", "当归", "川芎", "白术", "泽泻", "丹参"]
                extra_herbs = [h for h in extra_herbs if h not in predicted_herbs]
                to_add = np.random.randint(0, min(3, len(extra_herbs)))
                for _ in range(to_add):
                    predicted_herbs.append(np.random.choice(extra_herbs))
            
            # 模拟剂量预测
            predicted_dosages = {}
            for herb in predicted_herbs:
                if herb in reference["dosages"]:
                    # 对已有药物，可能有剂量偏差
                    ref_dosage = reference["dosages"][herb]
                    error = np.random.uniform(-0.3, 0.3)
                    predicted_dosages[herb] = max(1, round(ref_dosage * (1 + error)))
                else:
                    # 对新加药物，随机生成剂量
                    predicted_dosages[herb] = np.random.randint(3, 15)
        else:
            # 不正确情况 - 生成一个完全不同的处方
            other_prescriptions = {
                "茵陈蒿汤": ["茵陈", "栀子", "大黄"],
                "五苓散": ["猪苓", "泽泻", "白术", "茯苓", "桂枝"],
                "甘麦大枣汤": ["甘草", "小麦", "大枣"],
                "当归四逆汤": ["当归", "桂枝", "芍药", "细辛", "通草", "大枣", "甘草"]
            }
            prescription_name = np.random.choice(list(other_prescriptions.keys()))
            predicted_herbs = other_prescriptions[prescription_name]
            
            # 随机生成剂量
            predicted_dosages = {herb: np.random.randint(3, 15) for herb in predicted_herbs}
        
        # 模拟处方解释
        explanation = ""
        if np.random.random() < accuracy:
            # 基于参考关键词和推理逻辑生成解释
            keywords_to_use = reference["keywords"][:int(len(reference["keywords"]) * (0.5 + accuracy/2))]
            reasoning_to_use = reference["reasoning"][:int(len(reference["reasoning"]) * (0.5 + accuracy/2))]
            
            explanation = f"针对{sample['case']['diagnosis']}，采用{sample['case']['treatment_principle']}法。"
            explanation += f"方中{'、'.join(predicted_herbs[:3])}等药物，"
            
            if keywords_to_use:
                explanation += f"着重于{np.random.choice(keywords_to_use)}特点，"
            
            if reasoning_to_use:
                explanation += f"{np.random.choice(reasoning_to_use)}。"
            
            explanation += f"整体配伍{np.random.choice(['合理', '严谨', '科学', '经典'])}，符合患者病情特点。"
        else:
            # 生成不太相关的解释
            generic_text = [
                "本方药物配伍合理，能够有效治疗患者的症状。",
                "根据患者情况，选用上述药物进行治疗，可缓解不适。",
                "该方由多味药物组成，共奏治疗之效。",
                "辨证施治，药证相符，定能取得良效。"
            ]
            explanation = np.random.choice(generic_text)
        
        return {
            "herbs": predicted_herbs,
            "dosages": predicted_dosages,
            "explanation": explanation
        }
    
    def evaluate(self) -> Dict[str, Any]:
        """执行方剂评测"""
        logger.info("开始中医方剂评测...")
        
        if not self.test_samples:
            logger.warning("评测样本为空，无法评测")
            return {"score": 0.0}
        
        total_samples = len(self.test_samples)
        prescription_accuracy_scores = []
        dosage_rationality_scores = []
        explanation_rationality_scores = []
        
        for i, sample in enumerate(self.test_samples):
            # 模拟模型对该样本的预测
            prediction = self._simulate_model_prediction(sample)
            
            # 计算处方准确性
            prescription_accuracy = calculate_prescription_accuracy(
                prediction["herbs"],
                sample["reference"]["herbs"]
            )
            prescription_accuracy_scores.append(prescription_accuracy["f1"])
            
            # 计算剂量合理性
            dosage_rationality = calculate_dosage_rationality(
                prediction["dosages"],
                sample["reference"]["dosages"]
            )
            dosage_rationality_scores.append(dosage_rationality["rationality_score"])
            
            # 计算解释合理性
            explanation_rationality = calculate_explanation_rationality(
                prediction["explanation"],
                sample["reference"]["keywords"],
                sample["reference"]["reasoning"]
            )
            explanation_rationality_scores.append(explanation_rationality["overall_rationality"])
            
            # 记录进度
            if (i+1) % 5 == 0 or (i+1) == total_samples:
                logger.info(f"已评测 {i+1}/{total_samples} 个样本")
        
        # 计算总体得分
        results = {
            "prescription_accuracy": np.mean(prescription_accuracy_scores),
            "dosage_rationality": np.mean(dosage_rationality_scores),
            "explanation_rationality": np.mean(explanation_rationality_scores)
        }
        
        # 计算总分（加权平均）
        score = (
            results["prescription_accuracy"] * 0.5 +
            results["dosage_rationality"] * 0.3 +
            results["explanation_rationality"] * 0.2
        )
        
        results["score"] = score
        
        logger.info(f"中医方剂评测完成，得分: {score:.4f}")
        
        return results 
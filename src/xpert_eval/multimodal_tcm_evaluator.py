"""
中医多模态评测器模块
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union

from ..utils.logger import get_logger
from ..metrics.tcm_metrics import calculate_modal_consistency

logger = get_logger("tcm_multimodal")

class MultimodalTcmEvaluator:
    """中医多模态（四诊合参）评测器"""
    
    def __init__(self, model_name_or_path: str, device: str = "cuda"):
        """
        初始化多模态评测器
        
        参数:
            model_name_or_path: 模型名称或路径
            device: 运行设备
        """
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.test_samples = self._load_dataset_mock()
        logger.info(f"中医多模态评测器初始化完成: {model_name_or_path}")
    
    def _load_dataset_mock(self, size: int = 10) -> List[Dict[str, Any]]:
        """载入多模态诊断模拟数据集"""
        samples = []
        
        syndromes = [
            "肝郁气滞", "肝火上炎", "肝阳上亢",
            "心火亢盛", "心阴虚", "心脾两虚",
            "脾胃湿热", "脾虚湿盛", "脾肾阳虚",
            "肺热壅盛", "肺阴虚", "肺气虚",
            "肾阴虚", "肾阳虚", "肾精不足"
        ]
        
        for i in range(size):
            # 随机选择一个证型
            syndrome_idx = np.random.randint(0, len(syndromes))
            syndrome = syndromes[syndrome_idx]
            
            # 创建多模态样本
            sample = {
                "patient_id": f"patient_{i+1}",
                "visual_data": {
                    "face_image": f"mock_data/multimodal/patient_{i+1}_face.jpg",
                    "tongue_image": f"mock_data/multimodal/patient_{i+1}_tongue.jpg"
                },
                "audio_data": {
                    "voice_recording": f"mock_data/multimodal/patient_{i+1}_voice.wav",
                    "breathing_sound": f"mock_data/multimodal/patient_{i+1}_breathing.wav"
                },
                "text_data": {
                    "symptoms": ["症状1", "症状2", "症状3"],
                    "medical_history": "患者既往有...",
                    "dialogue": [
                        {"role": "doctor", "content": "请问有什么不舒服？"},
                        {"role": "patient", "content": "我感觉..."}
                    ]
                },
                "reference": {
                    "diagnosis": f"{syndrome}证",
                    "syndrome": syndrome,
                    "treatment_principle": self._get_treatment_principle(syndrome),
                    "explanation": f"根据四诊合参，患者表现为{syndrome}证..."
                }
            }
            samples.append(sample)
        
        return samples
    
    def _get_treatment_principle(self, syndrome: str) -> str:
        """根据证型获取治疗原则"""
        principles = {
            "肝郁气滞": "疏肝解郁",
            "肝火上炎": "清肝泻火",
            "肝阳上亢": "平肝潜阳",
            "心火亢盛": "清心泻火",
            "心阴虚": "养心滋阴",
            "心脾两虚": "健脾养心",
            "脾胃湿热": "清热化湿",
            "脾虚湿盛": "健脾祛湿",
            "脾肾阳虚": "温补脾肾",
            "肺热壅盛": "清肺泻热",
            "肺阴虚": "养阴润肺",
            "肺气虚": "补肺益气",
            "肾阴虚": "滋肾养阴",
            "肾阳虚": "温补肾阳",
            "肾精不足": "补肾填精"
        }
        return principles.get(syndrome, "辨证论治")
    
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
        
        # 模拟不同模态的预测结果
        reference = sample["reference"]
        
        # 模拟视觉模态预测
        visual_prediction = {
            "diagnosis": reference["diagnosis"] if np.random.random() < accuracy else self._get_random_diagnosis(reference["diagnosis"]),
            "syndrome": reference["syndrome"] if np.random.random() < accuracy else self._get_random_syndrome(reference["syndrome"]),
            "treatment_principle": reference["treatment_principle"] if np.random.random() < accuracy else self._get_random_treatment_principle()
        }
        
        # 模拟文本模态预测
        text_prediction = {
            "diagnosis": reference["diagnosis"] if np.random.random() < accuracy + 0.1 else self._get_random_diagnosis(reference["diagnosis"]),
            "syndrome": reference["syndrome"] if np.random.random() < accuracy + 0.1 else self._get_random_syndrome(reference["syndrome"]),
            "treatment_principle": reference["treatment_principle"] if np.random.random() < accuracy + 0.1 else self._get_random_treatment_principle()
        }
        
        # 模拟音频模态预测
        audio_prediction = {
            "diagnosis": reference["diagnosis"] if np.random.random() < accuracy - 0.2 else self._get_random_diagnosis(reference["diagnosis"]),
            "syndrome": reference["syndrome"] if np.random.random() < accuracy - 0.2 else self._get_random_syndrome(reference["syndrome"]),
            "treatment_principle": reference["treatment_principle"] if np.random.random() < accuracy - 0.1 else self._get_random_treatment_principle()
        }
        
        # 模拟多模态融合预测
        multimodal_prediction = {
            "diagnosis": reference["diagnosis"] if np.random.random() < accuracy + 0.15 else self._get_random_diagnosis(reference["diagnosis"]),
            "syndrome": reference["syndrome"] if np.random.random() < accuracy + 0.15 else self._get_random_syndrome(reference["syndrome"]),
            "treatment_principle": reference["treatment_principle"] if np.random.random() < accuracy + 0.15 else self._get_random_treatment_principle(),
            "explanation": f"根据四诊合参分析，患者表现为{reference['syndrome'] if np.random.random() < accuracy + 0.15 else self._get_random_syndrome(reference['syndrome'])}证..."
        }
        
        return {
            "visual_prediction": visual_prediction,
            "text_prediction": text_prediction,
            "audio_prediction": audio_prediction,
            "multimodal_prediction": multimodal_prediction
        }
    
    def _get_random_diagnosis(self, exclude: str) -> str:
        """获取随机诊断（排除正确答案）"""
        diagnoses = [
            "肝郁气滞证", "肝火上炎证", "肝阳上亢证", 
            "心火亢盛证", "心阴虚证", "心脾两虚证",
            "脾胃湿热证", "脾虚湿盛证", "脾肾阳虚证"
        ]
        candidates = [d for d in diagnoses if d != exclude]
        return np.random.choice(candidates)
    
    def _get_random_syndrome(self, exclude: str) -> str:
        """获取随机证型（排除正确答案）"""
        syndromes = [
            "肝郁气滞", "肝火上炎", "肝阳上亢",
            "心火亢盛", "心阴虚", "心脾两虚",
            "脾胃湿热", "脾虚湿盛", "脾肾阳虚"
        ]
        candidates = [s for s in syndromes if s != exclude]
        return np.random.choice(candidates)
    
    def _get_random_treatment_principle(self) -> str:
        """获取随机治疗原则"""
        principles = [
            "疏肝解郁", "清肝泻火", "平肝潜阳",
            "清心泻火", "养心滋阴", "健脾养心",
            "清热化湿", "健脾祛湿", "温补脾肾"
        ]
        return np.random.choice(principles)
    
    def evaluate(self) -> Dict[str, Any]:
        """执行多模态评测"""
        logger.info("开始中医多模态评测...")
        
        if not self.test_samples:
            logger.warning("评测样本为空，无法评测")
            return {"score": 0.0}
        
        total_samples = len(self.test_samples)
        modal_consistency_scores = []
        reference_consistency_scores = []
        syndrome_accuracy_scores = []
        
        for i, sample in enumerate(self.test_samples):
            # 模拟模型对该样本的预测
            predictions = self._simulate_model_prediction(sample)
            
            # 计算模态一致性
            consistency = calculate_modal_consistency(
                predictions["visual_prediction"],
                predictions["text_prediction"],
                predictions["audio_prediction"],
                sample["reference"]
            )
            
            modal_consistency_scores.append(consistency["overall_consistency"])
            reference_consistency_scores.append(consistency.get("reference_consistency", 0.0))
            
            # 计算证型准确性
            if predictions["multimodal_prediction"]["syndrome"] == sample["reference"]["syndrome"]:
                syndrome_accuracy_scores.append(1.0)
            else:
                syndrome_accuracy_scores.append(0.0)
            
            # 记录进度
            if (i+1) % 5 == 0 or (i+1) == total_samples:
                logger.info(f"已评测 {i+1}/{total_samples} 个样本")
        
        # 计算总体得分
        results = {
            "modal_consistency": np.mean(modal_consistency_scores),
            "reference_consistency": np.mean(reference_consistency_scores),
            "syndrome_accuracy": np.mean(syndrome_accuracy_scores)
        }
        
        # 计算总分（加权平均）
        score = (
            results["modal_consistency"] * 0.3 +
            results["reference_consistency"] * 0.4 +
            results["syndrome_accuracy"] * 0.3
        )
        
        results["score"] = score
        
        logger.info(f"中医多模态评测完成，得分: {score:.4f}")
        
        return results 
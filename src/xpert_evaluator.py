"""
XpertEval主评测器类
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Any

from .general_eval import (
    TextUnderstandingEvaluator,
    TextGenerationEvaluator,
    VisualEvaluator,
    AudioEvaluator,
    MultimodalEvaluator
)
from .xpert_eval import (
    MedicalDiagnosisEvaluator,
    MedicalImageEvaluator,
    LegalDocumentEvaluator,
    FinancialAnalysisEvaluator,
    ScientificResearchEvaluator,
    MultimodalExpertEvaluator
)
from .utils.logger import get_logger
from .utils.visualization import generate_radar_chart

logger = get_logger()

class XpertEvaluator:
    """多模态大模型评测器"""
    
    def __init__(
        self, 
        model_name_or_path: str,
        device: str = "cuda",
        output_dir: str = "results",
        config_path: Optional[str] = None
    ):
        """
        初始化评测器
        
        参数:
            model_name_or_path: 模型名称或路径
            device: 运行设备，"cuda"或"cpu"
            output_dir: 结果输出目录
            config_path: 配置文件路径，默认使用内置配置
        """
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.output_dir = output_dir
        
        # 加载配置
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            # 使用默认配置
            self.config = {
                "general_eval": {
                    "text_understanding": {"enabled": True, "weight": 0.1},
                    "text_generation": {"enabled": True, "weight": 0.1},
                    "visual": {"enabled": True, "weight": 0.1},
                    "audio": {"enabled": True, "weight": 0.05},
                    "multimodal": {"enabled": True, "weight": 0.15}
                },
                "xpert_eval": {
                    "medical_diagnosis": {"enabled": True, "weight": 0.1},
                    "medical_image": {"enabled": True, "weight": 0.1},
                    "legal_document": {"enabled": True, "weight": 0.1},
                    "financial_analysis": {"enabled": True, "weight": 0.1},
                    "scientific_research": {"enabled": True, "weight": 0.1},
                    "multimodal_expert": {"enabled": True, "weight": 0.1}
                }
            }
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化评测器
        logger.info(f"正在初始化评测器，模型: {model_name_or_path}, 设备: {device}")
        self._init_evaluators()
    
    def _init_evaluators(self):
        """初始化各个评测器"""
        self.evaluators = {}
        
        # 通用能力评测器
        if self.config["general_eval"]["text_understanding"]["enabled"]:
            self.evaluators["text_understanding"] = TextUnderstandingEvaluator(
                self.model_name_or_path, self.device
            )
        
        if self.config["general_eval"]["text_generation"]["enabled"]:
            self.evaluators["text_generation"] = TextGenerationEvaluator(
                self.model_name_or_path, self.device
            )
        
        if self.config["general_eval"]["visual"]["enabled"]:
            self.evaluators["visual"] = VisualEvaluator(
                self.model_name_or_path, self.device
            )
        
        if self.config["general_eval"]["audio"]["enabled"]:
            self.evaluators["audio"] = AudioEvaluator(
                self.model_name_or_path, self.device
            )
        
        if self.config["general_eval"]["multimodal"]["enabled"]:
            self.evaluators["multimodal"] = MultimodalEvaluator(
                self.model_name_or_path, self.device
            )
        
        # 专业能力评测器
        if self.config["xpert_eval"]["medical_diagnosis"]["enabled"]:
            self.evaluators["medical_diagnosis"] = MedicalDiagnosisEvaluator(
                self.model_name_or_path, self.device
            )
        
        if self.config["xpert_eval"]["medical_image"]["enabled"]:
            self.evaluators["medical_image"] = MedicalImageEvaluator(
                self.model_name_or_path, self.device
            )
        
        if self.config["xpert_eval"]["legal_document"]["enabled"]:
            self.evaluators["legal_document"] = LegalDocumentEvaluator(
                self.model_name_or_path, self.device
            )
        
        if self.config["xpert_eval"]["financial_analysis"]["enabled"]:
            self.evaluators["financial_analysis"] = FinancialAnalysisEvaluator(
                self.model_name_or_path, self.device
            )
        
        if self.config["xpert_eval"]["scientific_research"]["enabled"]:
            self.evaluators["scientific_research"] = ScientificResearchEvaluator(
                self.model_name_or_path, self.device
            )
        
        if self.config["xpert_eval"]["multimodal_expert"]["enabled"]:
            self.evaluators["multimodal_expert"] = MultimodalExpertEvaluator(
                self.model_name_or_path, self.device
            )
    
    def evaluate(self, tasks: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        执行评测任务
        
        参数:
            tasks: 需要评测的任务列表，如果为None则评测所有已启用的任务
        
        返回:
            评测结果字典
        """
        results = {}
        
        # 确定要评测的任务
        if tasks is None:
            tasks = list(self.evaluators.keys())
        else:
            # 过滤不存在的任务
            tasks = [task for task in tasks if task in self.evaluators]
        
        # 执行评测
        for task in tasks:
            logger.info(f"正在执行{task}评测...")
            try:
                if hasattr(self.evaluators[task], "evaluate") and callable(self.evaluators[task].evaluate):
                    results[task] = self.evaluators[task].evaluate()
                else:
                    logger.warning(f"{task}评测器没有实现evaluate方法")
            except Exception as e:
                logger.error(f"{task}评测出错: {str(e)}")
                results[task] = {"error": str(e)}
        
        # 计算总分
        overall_score = self._calculate_overall_score(results)
        results["overall"] = overall_score
        
        # 保存结果
        self._save_results(results)
        
        # 可视化结果
        self._visualize_results(results)
        
        return results
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> Dict[str, float]:
        """计算总分"""
        overall = {
            "general": 0.0,
            "expert": 0.0,
            "total": 0.0
        }
        
        total_general_weight = 0.0
        total_expert_weight = 0.0
        
        # 计算通用能力得分
        for task, config in self.config["general_eval"].items():
            if task in results and config["enabled"] and "score" in results[task]:
                weight = config["weight"]
                score = results[task]["score"]
                overall["general"] += score * weight
                total_general_weight += weight
        
        # 计算专业能力得分
        for task, config in self.config["xpert_eval"].items():
            if task in results and config["enabled"] and "score" in results[task]:
                weight = config["weight"]
                score = results[task]["score"]
                overall["expert"] += score * weight
                total_expert_weight += weight
        
        # 归一化处理
        if total_general_weight > 0:
            overall["general"] /= total_general_weight
        
        if total_expert_weight > 0:
            overall["expert"] /= total_expert_weight
        
        # 计算总分 (通用能力和专业能力各占50%)
        overall["total"] = 0.5 * overall["general"] + 0.5 * overall["expert"]
        
        return overall
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """保存评测结果到文件"""
        
        def make_serializable(obj):
            """将对象转换为可序列化的格式"""
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)
        
        # 转换为可序列化格式
        serializable_results = make_serializable(results)
        
        # 保存为JSON文件
        output_path = os.path.join(self.output_dir, f"{os.path.basename(self.model_name_or_path)}_results.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评测结果已保存到 {output_path}")
    
    def _visualize_results(self, results: Dict[str, Any]) -> None:
        """将评测结果可视化"""
        # 收集各任务分数
        tasks = []
        scores = []
        
        for task in results:
            if task != "overall" and isinstance(results[task], dict) and "score" in results[task]:
                tasks.append(task)
                scores.append(results[task]["score"])
        
        if tasks and scores:
            # 生成雷达图
            chart_path = os.path.join(self.output_dir, f"{os.path.basename(self.model_name_or_path)}_radar.png")
            generate_radar_chart(
                categories=tasks,
                values=scores,
                title=f"{os.path.basename(self.model_name_or_path)} 评测结果",
                output_path=chart_path
            )
            logger.info(f"评测结果可视化已保存到 {chart_path}")
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """打印评测结果"""
        print("\n" + "="*50)
        print(f"模型 {os.path.basename(self.model_name_or_path)} 评测结果")
        print("="*50)
        
        # 打印各任务结果
        for task in sorted(results.keys()):
            if task == "overall":
                continue
            
            if isinstance(results[task], dict) and "score" in results[task]:
                print(f"{task}: {results[task]['score']:.4f}")
            elif isinstance(results[task], dict) and "error" in results[task]:
                print(f"{task}: 出错 - {results[task]['error']}")
            else:
                print(f"{task}: 无评分数据")
        
        # 打印总分
        if "overall" in results:
            print("-"*50)
            print(f"通用能力: {results['overall']['general']:.4f}")
            print(f"专业能力: {results['overall']['expert']:.4f}")
            print(f"总分: {results['overall']['total']:.4f}")
        
        print("="*50 + "\n")


def evaluate(
    model_name_or_path: str,
    tasks: Optional[List[str]] = None,
    device: str = "cuda",
    output_dir: str = "results",
    config_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    便捷评测函数
    
    参数:
        model_name_or_path: 模型名称或路径
        tasks: 需要评测的任务列表
        device: 运行设备
        output_dir: 结果输出目录
        config_path: 配置文件路径
    
    返回:
        评测结果
    """
    evaluator = XpertEvaluator(
        model_name_or_path=model_name_or_path,
        device=device,
        output_dir=output_dir,
        config_path=config_path
    )
    
    results = evaluator.evaluate(tasks=tasks)
    evaluator.print_results(results)
    
    return results 
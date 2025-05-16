#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
XpertEval基础评测示例脚本
此脚本展示如何使用XpertEval对多模态大模型进行基础评测
"""

import os
import sys
import json
import torch
import argparse
import logging
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.general_eval.text_evaluator import TextUnderstandingEvaluator, TextGenerationEvaluator
from src.xpert_eval.tongue_evaluator import TongueDiagnosisEvaluator
from src.xpert_eval.face_evaluator import FaceDiagnosisEvaluator
from src.xpert_eval.prescription_evaluator import PrescriptionEvaluator
from src.metrics.metric_calculator import MetricCalculator
from src.utils.visualization import plot_radar_chart
from src.utils.config_utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="XpertEval基础评测脚本")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径或名称")
    parser.add_argument("--config", type=str, default="configs/basic_eval.yaml", help="评测配置文件路径")
    parser.add_argument("--output_dir", type=str, default="results", help="评测结果输出目录")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="运行设备")
    parser.add_argument("--eval_type", type=str, default="all", 
                        choices=["all", "general", "tcm", "text", "tongue", "face", "prescription"],
                        help="评测类型")
    return parser.parse_args()

def run_text_understanding_eval(model_path, config, device):
    """运行文本理解能力评测"""
    logger.info("开始文本理解能力评测...")
    
    # 初始化评测器
    evaluator = TextUnderstandingEvaluator(model_path, device=device)
    
    results = {}
    # 评测多项选择题能力
    if "multiple_choice" in config["text_evaluation"]:
        mc_config = config["text_evaluation"]["multiple_choice"]
        logger.info(f"评测数据集: {mc_config['dataset_path']}")
        
        mc_results = evaluator.evaluate_multiple_choice(
            mc_config["dataset_path"],
            few_shot=mc_config.get("few_shot", 0)
        )
        results["multiple_choice"] = mc_results
        
        logger.info(f"多项选择题评测结果: 准确率={mc_results['accuracy']:.4f}, F1={mc_results['f1']:.4f}")
    
    # 可以添加其他文本理解任务的评测
    
    return results

def run_text_generation_eval(model_path, config, device):
    """运行文本生成能力评测"""
    logger.info("开始文本生成能力评测...")
    
    # 初始化评测器
    evaluator = TextGenerationEvaluator(model_path, device=device)
    
    results = {}
    # 评测文本生成质量
    if "text_generation" in config["text_evaluation"]:
        gen_config = config["text_evaluation"]["text_generation"]
        
        # 加载提示和参考文本
        with open(gen_config["dataset_path"], "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        
        prompts = [item["prompt"] for item in eval_data]
        references = [item["reference"] for item in eval_data]
        
        logger.info(f"评测{len(prompts)}个生成样本...")
        
        gen_results = evaluator.evaluate_generation(prompts, references)
        results["text_generation"] = gen_results
        
        logger.info(f"文本生成评测结果: BLEU={gen_results['bleu']:.4f}, ROUGE-L={gen_results['rougeL']:.4f}")
    
    return results

def run_tongue_diagnosis_eval(model_path, config, device):
    """运行舌诊能力评测"""
    logger.info("开始舌诊能力评测...")
    
    # 初始化评测器
    evaluator = TongueDiagnosisEvaluator(model_path, device=device)
    
    results = {}
    # 评测舌诊能力
    if "tongue_diagnosis" in config["tcm_evaluation"]:
        tongue_config = config["tcm_evaluation"]["tongue_diagnosis"]
        
        # 加载评测数据
        with open(tongue_config["dataset_path"], "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        
        body_accuracies = []
        coating_accuracies = []
        overall_accuracies = []
        
        for item in tqdm(eval_data, desc="评测舌诊样本"):
            result = evaluator.evaluate_tongue_diagnosis(
                item["image_path"],
                tongue_config["prompt_template"].format(item=item),
                item["reference_body"],
                item["reference_coating"]
            )
            
            body_accuracies.append(result["body_accuracy"])
            coating_accuracies.append(result["coating_accuracy"])
            overall_accuracies.append(result["overall_accuracy"])
        
        results["tongue_diagnosis"] = {
            "body_accuracy": sum(body_accuracies) / len(body_accuracies),
            "coating_accuracy": sum(coating_accuracies) / len(coating_accuracies),
            "overall_accuracy": sum(overall_accuracies) / len(overall_accuracies),
            "sample_count": len(eval_data)
        }
        
        logger.info(f"舌诊评测结果: 舌质准确率={results['tongue_diagnosis']['body_accuracy']:.4f}, " +
                   f"舌苔准确率={results['tongue_diagnosis']['coating_accuracy']:.4f}, " +
                   f"整体准确率={results['tongue_diagnosis']['overall_accuracy']:.4f}")
    
    return results

def run_face_diagnosis_eval(model_path, config, device):
    """运行面诊能力评测"""
    logger.info("开始面诊能力评测...")
    
    # 初始化评测器
    evaluator = FaceDiagnosisEvaluator(model_path, device=device)
    
    results = {}
    # 评测面诊能力
    if "face_diagnosis" in config["tcm_evaluation"]:
        face_config = config["tcm_evaluation"]["face_diagnosis"]
        
        # 加载评测数据
        with open(face_config["dataset_path"], "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for item in tqdm(eval_data, desc="评测面诊样本"):
            result = evaluator.evaluate_face_diagnosis(
                item["image_path"],
                face_config["prompt_template"].format(item=item),
                item["reference_features"]
            )
            
            accuracies.append(result["accuracy"])
            precisions.append(result["precision"])
            recalls.append(result["completeness"])  # completeness作为recall
            f1_scores.append(result["f1_score"])
        
        results["face_diagnosis"] = {
            "accuracy": sum(accuracies) / len(accuracies),
            "precision": sum(precisions) / len(precisions),
            "recall": sum(recalls) / len(recalls),
            "f1": sum(f1_scores) / len(f1_scores),
            "sample_count": len(eval_data)
        }
        
        logger.info(f"面诊评测结果: 准确率={results['face_diagnosis']['accuracy']:.4f}, " +
                   f"F1分数={results['face_diagnosis']['f1']:.4f}")
    
    return results

def run_prescription_eval(model_path, config, device):
    """运行方剂推荐能力评测"""
    logger.info("开始方剂推荐能力评测...")
    
    # 初始化评测器
    evaluator = PrescriptionEvaluator(model_path, device=device)
    
    results = {}
    # 评测方剂推荐能力
    if "prescription" in config["tcm_evaluation"]:
        prescription_config = config["tcm_evaluation"]["prescription"]
        
        # 加载评测数据
        with open(prescription_config["dataset_path"], "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        
        precisions = []
        recalls = []
        f1_scores = []
        rationality_scores = []
        
        for item in tqdm(eval_data, desc="评测方剂推荐样本"):
            diagnosis_result = item.get("diagnosis_result", "")
            prompt = prescription_config["prompt_template"].format(
                diagnosis_result=diagnosis_result,
                symptoms=", ".join(item.get("symptoms", []))
            )
            
            result = evaluator.evaluate_prescription(
                prompt,
                item["reference_herbs"],
                item.get("reference_dosages", {}),
                item.get("safety_ranges", {})
            )
            
            precisions.append(result["precision"])
            recalls.append(result["recall"])
            f1_scores.append(result["f1"])
            rationality_scores.append(result.get("rationality", 0))
        
        results["prescription"] = {
            "precision": sum(precisions) / len(precisions),
            "recall": sum(recalls) / len(recalls),
            "f1": sum(f1_scores) / len(f1_scores),
            "rationality": sum(rationality_scores) / len(rationality_scores) if rationality_scores else 0,
            "sample_count": len(eval_data)
        }
        
        logger.info(f"方剂推荐评测结果: 准确率={results['prescription']['precision']:.4f}, " +
                   f"召回率={results['prescription']['recall']:.4f}, " +
                   f"F1分数={results['prescription']['f1']:.4f}")
    
    return results

def calculate_overall_scores(all_results, config):
    """计算综合评分"""
    logger.info("计算综合评分...")
    
    metric_calculator = MetricCalculator()
    
    # 设置权重
    weights = config.get("weights", {})
    general_weights = weights.get("general", {})
    tcm_weights = weights.get("tcm", {})
    
    # 计算通用能力得分
    general_scores = {}
    if "multiple_choice" in all_results:
        general_scores["language_understanding"] = all_results["multiple_choice"]["accuracy"]
    if "text_generation" in all_results:
        general_scores["text_generation"] = (all_results["text_generation"]["bleu"] + 
                                          all_results["text_generation"]["rougeL"]) / 2
    
    general_score = metric_calculator.calculate_overall_score(general_scores, general_weights)
    
    # 计算中医专业能力得分
    tcm_scores = {}
    if "tongue_diagnosis" in all_results:
        tcm_scores["tongue_diagnosis"] = all_results["tongue_diagnosis"]["overall_accuracy"]
    if "face_diagnosis" in all_results:
        tcm_scores["face_diagnosis"] = all_results["face_diagnosis"]["f1"]
    if "prescription" in all_results:
        tcm_scores["prescription"] = all_results["prescription"]["f1"]
    
    tcm_score = metric_calculator.calculate_overall_score(tcm_scores, tcm_weights)
    
    # 计算综合得分
    overall_weight = weights.get("overall", {"general": 0.3, "tcm": 0.7})
    overall_score = (general_score * overall_weight["general"] + 
                    tcm_score * overall_weight["tcm"])
    
    return {
        "general_score": general_score,
        "tcm_score": tcm_score,
        "overall_score": overall_score,
        "general_scores": general_scores,
        "tcm_scores": tcm_scores
    }

def visualize_results(all_results, scores, output_dir):
    """可视化评测结果"""
    logger.info("生成评测结果可视化...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制雷达图
    categories = []
    values = []
    
    # 添加通用能力分数
    for category, score in scores["general_scores"].items():
        categories.append(category)
        values.append(score * 100)  # 转换为0-100的分数
    
    # 添加专业能力分数
    for category, score in scores["tcm_scores"].items():
        categories.append(category)
        values.append(score * 100)  # 转换为0-100的分数
    
    radar_chart = plot_radar_chart(values, categories)
    radar_chart.savefig(os.path.join(output_dir, "radar_chart.png"), dpi=300, bbox_inches="tight")
    
    # 保存详细结果
    with open(os.path.join(output_dir, "evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump({
            "detailed_results": all_results,
            "scores": scores
        }, f, ensure_ascii=False, indent=2)
    
    # 生成简要报告
    with open(os.path.join(output_dir, "evaluation_report.md"), "w", encoding="utf-8") as f:
        f.write("# XpertEval 评测报告\n\n")
        f.write(f"## 总体评分\n\n")
        f.write(f"- 总分: {scores['overall_score']*100:.2f}\n")
        f.write(f"- 通用能力: {scores['general_score']*100:.2f}\n")
        f.write(f"- 中医专业能力: {scores['tcm_score']*100:.2f}\n\n")
        
        f.write("## 通用能力评分详情\n\n")
        for category, score in scores["general_scores"].items():
            f.write(f"- {category}: {score*100:.2f}\n")
        
        f.write("\n## 中医专业能力评分详情\n\n")
        for category, score in scores["tcm_scores"].items():
            f.write(f"- {category}: {score*100:.2f}\n")
        
        f.write("\n## 评测样例\n\n")
        f.write("请查看evaluation_results.json获取详细评测结果。\n")

def main():
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建输出目录
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    # 根据评测类型运行相应的评测
    if args.eval_type in ["all", "general", "text"]:
        # 文本理解能力评测
        mc_results = run_text_understanding_eval(args.model_path, config, args.device)
        all_results.update(mc_results)
        
        # 文本生成能力评测
        gen_results = run_text_generation_eval(args.model_path, config, args.device)
        all_results.update(gen_results)
    
    if args.eval_type in ["all", "tcm", "tongue"]:
        # 舌诊能力评测
        tongue_results = run_tongue_diagnosis_eval(args.model_path, config, args.device)
        all_results.update(tongue_results)
    
    if args.eval_type in ["all", "tcm", "face"]:
        # 面诊能力评测
        face_results = run_face_diagnosis_eval(args.model_path, config, args.device)
        all_results.update(face_results)
    
    if args.eval_type in ["all", "tcm", "prescription"]:
        # 方剂推荐能力评测
        prescription_results = run_prescription_eval(args.model_path, config, args.device)
        all_results.update(prescription_results)
    
    # 计算综合评分
    scores = calculate_overall_scores(all_results, config)
    
    # 可视化结果
    visualize_results(all_results, scores, output_dir)
    
    logger.info(f"评测完成！结果已保存到 {output_dir}")
    logger.info(f"总评分: {scores['overall_score']*100:.2f}")
    logger.info(f"通用能力: {scores['general_score']*100:.2f}")
    logger.info(f"中医专业能力: {scores['tcm_score']*100:.2f}")

if __name__ == "__main__":
    main() 
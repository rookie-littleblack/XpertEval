#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
示例代码：演示如何使用XpertEval对中医药多模态大模型进行评测
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.xpert_evaluator import XpertEvaluator
from src.general_eval.text_evaluator import TextEvaluator
from src.general_eval.visual_evaluator import VisualEvaluator
from src.general_eval.audio_evaluator import AudioEvaluator
from src.general_eval.multimodal_evaluator import MultimodalEvaluator
from src.xpert_eval.diagnosis_evaluator import DiagnosisEvaluator
from src.xpert_eval.prescription_evaluator import PrescriptionEvaluator
from src.metrics.tcm_metrics import (
    calculate_tongue_feature_recognition_accuracy,
    calculate_sound_classification_accuracy,
    calculate_symptom_recognition_rate,
    calculate_pulse_classification_accuracy,
    calculate_syndrome_differentiation_accuracy,
    calculate_prescription_accuracy,
    calculate_tcm_professionalism_score
)
from src.utils.visualization import plot_radar_chart, plot_comparison_bar_chart


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='XpertEval中医药模型评测示例')
    parser.add_argument('--model', type=str, required=True, help='模型名称或路径')
    parser.add_argument('--device', type=str, default='cuda', help='运行设备，可选cuda或cpu')
    parser.add_argument('--output_dir', type=str, default='results', help='结果保存目录')
    parser.add_argument('--tasks', nargs='+', default=['all'], 
                      help='要评测的任务，可选all，或指定具体任务如text_understanding, tongue_diagnosis等')
    parser.add_argument('--config', type=str, default=None, help='评测配置文件路径')
    return parser.parse_args()


def load_model(model_name, device='cuda'):
    """
    加载模型（实际场景中需要根据具体模型框架实现）
    
    参数:
        model_name: 模型名称或路径
        device: 运行设备
    
    返回:
        加载的模型
    """
    print(f"正在加载模型：{model_name}...")
    # 这里模拟模型加载，实际应用中需要替换为真实模型加载代码
    # 例如使用transformers库加载模型：
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    # 返回模拟模型对象
    return {"name": model_name, "device": device}


def evaluate_tcm_model_basic(model, data_dir="data/eval_samples", tasks=None):
    """
    基础用法：使用XpertEvaluator执行多任务评测
    
    参数:
        model: 待评测模型
        data_dir: 评测数据目录
        tasks: 要评测的任务列表，如None则评测所有任务
    
    返回:
        评测结果字典
    """
    print("开始基础评测...")
    
    # 初始化主评测器
    evaluator = XpertEvaluator(model)
    
    # 执行评测
    results = evaluator.evaluate(data_dir=data_dir, tasks=tasks)
    
    print("基础评测完成！")
    return results


def evaluate_tcm_model_advanced(model, config=None):
    """
    高级用法：分别评测各能力模块，并手动组合结果
    
    参数:
        model: 待评测模型
        config: 配置文件路径
    
    返回:
        评测结果字典
    """
    print("开始高级评测...")
    
    # 加载配置
    if config and os.path.exists(config):
        with open(config, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    else:
        config_data = {
            "general_eval": {
                "text_understanding": {"enabled": True, "weight": 0.1},
                "text_generation": {"enabled": True, "weight": 0.1},
                "visual_understanding": {"enabled": True, "weight": 0.1},
                "audio_recognition": {"enabled": True, "weight": 0.1},
                "multimodal_fusion": {"enabled": True, "weight": 0.1}
            },
            "xpert_eval": {
                "tongue_diagnosis": {"enabled": True, "weight": 0.1},
                "face_diagnosis": {"enabled": True, "weight": 0.05},
                "sound_diagnosis": {"enabled": True, "weight": 0.05},
                "inquiry_diagnosis": {"enabled": True, "weight": 0.1},
                "pulse_diagnosis": {"enabled": True, "weight": 0.05},
                "syndrome_differentiation": {"enabled": True, "weight": 0.1},
                "prescription": {"enabled": True, "weight": 0.15}
            }
        }
    
    results = {}
    
    # 评测通用能力
    if any(config_data["general_eval"][task]["enabled"] for task in config_data["general_eval"]):
        print("评测通用能力...")
        
        # 文本评测
        if config_data["general_eval"].get("text_understanding", {}).get("enabled", False):
            text_evaluator = TextEvaluator(model)
            results["text_understanding"] = text_evaluator.evaluate(task="understanding")
        
        if config_data["general_eval"].get("text_generation", {}).get("enabled", False):
            text_evaluator = TextEvaluator(model)
            results["text_generation"] = text_evaluator.evaluate(task="generation")
        
        # 视觉评测
        if config_data["general_eval"].get("visual_understanding", {}).get("enabled", False):
            visual_evaluator = VisualEvaluator(model)
            results["visual_understanding"] = visual_evaluator.evaluate()
        
        # 音频评测
        if config_data["general_eval"].get("audio_recognition", {}).get("enabled", False):
            audio_evaluator = AudioEvaluator(model)
            results["audio_recognition"] = audio_evaluator.evaluate()
        
        # 多模态融合评测
        if config_data["general_eval"].get("multimodal_fusion", {}).get("enabled", False):
            multimodal_evaluator = MultimodalEvaluator(model)
            results["multimodal_fusion"] = multimodal_evaluator.evaluate()
    
    # 评测中医专业能力
    if any(config_data["xpert_eval"][task]["enabled"] for task in config_data["xpert_eval"]):
        print("评测中医专业能力...")
        
        # 诊断能力评测
        diagnosis_evaluator = DiagnosisEvaluator(model)
        
        # 望诊评测
        if config_data["xpert_eval"].get("tongue_diagnosis", {}).get("enabled", False):
            results["tongue_diagnosis"] = diagnosis_evaluator.evaluate_tongue_diagnosis()
        
        if config_data["xpert_eval"].get("face_diagnosis", {}).get("enabled", False):
            results["face_diagnosis"] = diagnosis_evaluator.evaluate_face_diagnosis()
        
        # 闻诊评测
        if config_data["xpert_eval"].get("sound_diagnosis", {}).get("enabled", False):
            results["sound_diagnosis"] = diagnosis_evaluator.evaluate_sound_diagnosis()
        
        # 问诊评测
        if config_data["xpert_eval"].get("inquiry_diagnosis", {}).get("enabled", False):
            results["inquiry_diagnosis"] = diagnosis_evaluator.evaluate_inquiry()
        
        # 切诊评测
        if config_data["xpert_eval"].get("pulse_diagnosis", {}).get("enabled", False):
            results["pulse_diagnosis"] = diagnosis_evaluator.evaluate_pulse_diagnosis()
        
        # 辨证论治评测
        if config_data["xpert_eval"].get("syndrome_differentiation", {}).get("enabled", False):
            results["syndrome_differentiation"] = diagnosis_evaluator.evaluate_syndrome_differentiation()
        
        # 方剂推荐评测
        if config_data["xpert_eval"].get("prescription", {}).get("enabled", False):
            prescription_evaluator = PrescriptionEvaluator(model)
            results["prescription"] = prescription_evaluator.evaluate()
    
    # 计算总体评分
    general_scores = {k: v["score"] for k, v in results.items() 
                    if k in config_data["general_eval"] and "score" in v}
    
    xpert_scores = {k: v["score"] for k, v in results.items() 
                  if k in config_data["xpert_eval"] and "score" in v}
    
    # 计算通用能力总分
    if general_scores:
        general_weights = {k: config_data["general_eval"][k]["weight"] for k in general_scores}
        results["general_ability"] = {
            "score": sum(general_scores[k] * general_weights[k] for k in general_scores) / sum(general_weights.values()),
            "details": general_scores
        }
    
    # 计算中医专业能力总分
    if xpert_scores:
        xpert_weights = {k: config_data["xpert_eval"][k]["weight"] for k in xpert_scores}
        results["xpert_ability"] = {
            "score": sum(xpert_scores[k] * xpert_weights[k] for k in xpert_scores) / sum(xpert_weights.values()),
            "details": xpert_scores
        }
    
    # 计算综合评分
    if "general_ability" in results and "xpert_ability" in results:
        results["overall_score"] = 0.3 * results["general_ability"]["score"] + 0.7 * results["xpert_ability"]["score"]
    
    print("高级评测完成！")
    return results


def demonstrate_metrics_usage():
    """演示如何直接使用指标计算函数"""
    print("\n演示直接使用评测指标...")
    
    # 示例数据
    # 舌诊特征识别
    predictions = [
        {"tongue_color": "淡红", "tongue_shape": "胖", "tongue_coating": "白腻"},
        {"tongue_color": "淡白", "tongue_shape": "瘦", "tongue_coating": "少苔"}
    ]
    references = [
        {"tongue_color": "淡红", "tongue_shape": "胖", "tongue_coating": "白腻"},
        {"tongue_color": "淡白", "tongue_shape": "齿痕", "tongue_coating": "少苔"}
    ]
    
    # 计算舌象特征识别准确率
    tongue_accuracy = calculate_tongue_feature_recognition_accuracy(predictions, references)
    print("舌象特征识别准确率:", tongue_accuracy)
    
    # 声音分类
    sound_predictions = ["干咳", "湿咳", "喘息", "正常"]
    sound_references = ["干咳", "湿咳", "哮鸣", "正常"]
    
    # 计算声音分类准确率
    sound_accuracy = calculate_sound_classification_accuracy(sound_predictions, sound_references)
    print("声音分类准确率:", sound_accuracy)
    
    # 症状识别
    symptom_predictions = [["头痛", "发热", "乏力"], ["腹痛", "腹泻", "恶心"]]
    symptom_references = [["头痛", "发热", "咳嗽"], ["腹痛", "腹胀", "恶心"]]
    
    # 计算症状识别率
    symptom_metrics = calculate_symptom_recognition_rate(symptom_predictions, symptom_references)
    print("症状识别率:", symptom_metrics)
    
    # 脉象分类
    pulse_predictions = ["浮", "细", "滑", "沉弦"]
    pulse_references = ["浮", "细", "数", "沉弦"]
    
    # 计算脉象分类准确率
    pulse_accuracy = calculate_pulse_classification_accuracy(pulse_predictions, pulse_references)
    print("脉象分类准确率:", pulse_accuracy)
    
    # 辨证准确率
    syndrome_predictions = ["脾虚湿困", "肝郁气滞", "肾阳虚"]
    syndrome_references = ["脾虚湿困", "肝胆湿热", "肾阳虚"]
    
    # 计算辨证准确率
    syndrome_metrics = calculate_syndrome_differentiation_accuracy(syndrome_predictions, syndrome_references)
    print("辨证准确率:", syndrome_metrics)
    
    # 方剂推荐准确率
    prescription_predictions = ["黄芪", "党参", "白术", "茯苓", "甘草"]
    prescription_references = ["黄芪", "党参", "白术", "茯苓", "陈皮"]
    
    # 计算方剂推荐准确率
    prescription_metrics = calculate_prescription_accuracy(prescription_predictions, prescription_references)
    print("方剂推荐准确率:", prescription_metrics)
    
    # 计算中医专业性综合评分
    tcm_scores = {
        "望诊": 0.85,
        "闻诊": 0.75,
        "问诊": 0.82,
        "切诊": 0.78,
        "四诊合参": 0.80,
        "方剂推荐": 0.75
    }
    
    tcm_score = calculate_tcm_professionalism_score(tcm_scores)
    print("中医专业性综合评分:", tcm_score)


def visualize_results(results, output_dir="results"):
    """可视化评测结果"""
    print("\n生成可视化结果...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存原始结果
    with open(os.path.join(output_dir, "evaluation_results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 提取数据用于绘图
    if "general_ability" in results and "xpert_ability" in results:
        # 组合所有指标
        all_metrics = {}
        all_metrics.update(results["general_ability"]["details"])
        all_metrics.update(results["xpert_ability"]["details"])
        
        # 绘制雷达图
        categories = list(all_metrics.keys())
        values = [all_metrics[k] for k in categories]
        
        plt.figure(figsize=(10, 8))
        plot_radar_chart(categories, [values], ["模型评测结果"], title="XpertEval评测结果雷达图")
        plt.savefig(os.path.join(output_dir, "radar_chart.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 绘制能力条形图
        plt.figure(figsize=(12, 6))
        categories = ["通用能力", "中医专业能力", "综合评分"]
        values = [results["general_ability"]["score"], 
                results["xpert_ability"]["score"], 
                results["overall_score"]]
        
        plt.bar(categories, values, color=['blue', 'green', 'orange'])
        plt.ylim(0, 1.0)
        plt.title("模型能力评分")
        plt.ylabel("评分")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 在柱子上方显示具体分数
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        plt.savefig(os.path.join(output_dir, "ability_scores.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"可视化结果已保存到 {output_dir} 目录")


def compare_models(model_names, model_display_names=None, tasks=None, output_dir="results"):
    """
    比较多个模型的性能
    
    参数:
        model_names: 模型名称或路径列表
        model_display_names: 模型显示名称，用于图表展示
        tasks: 要评测的任务列表
        output_dir: 结果输出目录
    """
    print("\n开始多模型比较评测...")
    
    if model_display_names is None:
        model_display_names = model_names
    
    results = {}
    
    # 逐个评测模型
    for i, model_name in enumerate(model_names):
        print(f"\n正在评测模型 {i+1}/{len(model_names)}: {model_name}")
        
        # 加载模型
        model = load_model(model_name)
        
        # 评测模型
        model_results = evaluate_tcm_model_basic(model, tasks=tasks)
        
        # 存储结果
        results[model_display_names[i]] = model_results
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存比较结果
    with open(os.path.join(output_dir, "models_comparison.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 可视化比较结果
    visualize_comparison(results, output_dir)
    
    print(f"多模型比较评测完成，结果已保存到 {output_dir} 目录")
    
    return results


def visualize_comparison(results, output_dir):
    """可视化模型比较结果"""
    # 提取通用能力和专业能力评分
    model_names = list(results.keys())
    general_scores = []
    xpert_scores = []
    overall_scores = []
    
    for model in model_names:
        if "general_ability" in results[model] and "xpert_ability" in results[model]:
            general_scores.append(results[model]["general_ability"]["score"])
            xpert_scores.append(results[model]["xpert_ability"]["score"])
            overall_scores.append(results[model]["overall_score"])
    
    # 绘制比较柱状图
    plt.figure(figsize=(12, 6))
    plot_comparison_bar_chart(
        model_names, 
        [general_scores, xpert_scores, overall_scores],
        ["通用能力", "中医专业能力", "综合评分"],
        title="模型性能比较"
    )
    plt.savefig(os.path.join(output_dir, "models_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 为每个具体能力绘制雷达图比较
    if all("general_ability" in results[model] and "xpert_ability" in results[model] for model in model_names):
        # 组合所有指标
        all_metrics = {}
        for model in model_names:
            all_metrics[model] = {}
            all_metrics[model].update(results[model]["general_ability"]["details"])
            all_metrics[model].update(results[model]["xpert_ability"]["details"])
        
        # 统一所有模型的指标维度
        all_dimensions = set()
        for model in model_names:
            all_dimensions.update(all_metrics[model].keys())
        
        # 填充缺失维度
        for model in model_names:
            for dim in all_dimensions:
                if dim not in all_metrics[model]:
                    all_metrics[model][dim] = 0.0
        
        # 提取雷达图数据
        dimensions = sorted(list(all_dimensions))
        values = []
        for model in model_names:
            values.append([all_metrics[model][dim] for dim in dimensions])
        
        # 绘制雷达图
        plt.figure(figsize=(12, 10))
        plot_radar_chart(dimensions, values, model_names, title="模型能力雷达图比较")
        plt.savefig(os.path.join(output_dir, "models_radar_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """主程序"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"===== XpertEval中医药模型评测示例 =====")
    print(f"模型: {args.model}")
    print(f"设备: {args.device}")
    print(f"输出目录: {args.output_dir}")
    print(f"任务: {args.tasks}")
    print("===================================")
    
    # 加载模型
    model = load_model(args.model, args.device)
    
    # 基础用法
    if args.tasks == ['all'] or any(task in args.tasks for task in ["basic", "simple"]):
        results = evaluate_tcm_model_basic(model)
    else:
        # 高级用法：使用配置文件
        results = evaluate_tcm_model_advanced(model, args.config)
    
    # 演示直接使用指标
    demonstrate_metrics_usage()
    
    # 可视化结果
    visualize_results(results, args.output_dir)
    
    # 多模型比较示例
    if args.tasks == ['all'] or "comparison" in args.tasks:
        compare_models(
            [args.model, "ChatGLM3", "Qwen-VL-Plus"],
            ["评测模型", "ChatGLM3", "Qwen-VL-Plus"],
            tasks=["text_understanding", "tongue_diagnosis"],
            output_dir=os.path.join(args.output_dir, "comparison")
        )
    
    print(f"\n评测完成！结果已保存到 {args.output_dir} 目录")


if __name__ == "__main__":
    main() 
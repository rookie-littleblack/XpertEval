#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
XpertEval 系统演示脚本
这个脚本演示了如何使用XpertEval评测系统评测多个模型并生成对比报告
"""

import os
import sys
import argparse
import json
from datetime import datetime

# 添加项目根目录到sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.xpert_evaluator import XpertEvaluator, evaluate
from src.utils.visualization import generate_comparison_charts

def parse_args():
    parser = argparse.ArgumentParser(description="XpertEval系统演示")
    parser.add_argument("--models", nargs="+", default=["model_base", "model_large"], 
                        help="要评测的模型名称列表，可以是多个")
    parser.add_argument("--tasks", nargs="+", 
                        default=["text_understanding", "text_generation", "face_diagnosis", "tongue_diagnosis", "multimodal_tcm"],
                        help="要评测的任务列表")
    parser.add_argument("--output_dir", type=str, default="demo_results",
                        help="评测结果输出目录")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径，默认使用内置配置")
    return parser.parse_args()

def print_banner(text):
    """打印带有装饰的横幅"""
    width = len(text) + 6
    print("\n" + "=" * width)
    print(f"   {text}")
    print("=" * width + "\n")

def run_demo():
    """运行评测系统演示"""
    args = parse_args()
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"{args.output_dir}-{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print_banner("XpertEval 中医药多模态大模型评测系统演示")
    print(f"评测模型: {', '.join(args.models)}")
    print(f"评测任务: {', '.join(args.tasks)}")
    print(f"输出目录: {output_dir}")
    
    # 保存所有模型的评测结果
    all_results = {}
    
    # 对每个模型进行评测
    for model_name in args.models:
        print_banner(f"开始评测模型: {model_name}")
        
        # 为每个模型创建单独的结果目录
        model_output_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # 实例化评测器
        evaluator = XpertEvaluator(
            model_name_or_path=model_name,
            device="cuda",
            output_dir=model_output_dir,
            config_path=args.config
        )
        
        # 执行评测
        results = evaluator.evaluate(tasks=args.tasks)
        
        # 打印结果
        evaluator.print_results(results)
        
        # 保存结果用于后续比较
        all_results[model_name] = results
    
    # 生成模型比较图表
    print_banner("生成模型比较报告")
    generate_comparison_charts(
        model_results=all_results,
        output_dir=output_dir,
        prefix="comparison"
    )
    
    # 保存所有结果的JSON文件
    results_file = os.path.join(output_dir, "all_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n评测完成，结果已保存至: {output_dir}")
    print(f"所有模型的综合评测报告已保存至: {results_file}")
    print(f"比较图表已保存至: {output_dir}")

if __name__ == "__main__":
    run_demo() 
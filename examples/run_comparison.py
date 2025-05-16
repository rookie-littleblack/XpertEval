#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
XpertEval 多模型比较示例
"""

import os
import sys
import argparse
import json
from typing import List, Dict, Any

# 添加项目根目录到sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.xpert_evaluator import XpertEvaluator
from src.utils.visualization import generate_comparison_charts

def parse_args():
    parser = argparse.ArgumentParser(description="XpertEval 多模型比较示例")
    parser.add_argument("--models", type=str, nargs="+", required=True, help="模型名称或路径列表")
    parser.add_argument("--model_names", type=str, nargs="+", help="模型显示名称列表，与models顺序对应")
    parser.add_argument("--device", type=str, default="cuda", help="运行设备，cuda或cpu")
    parser.add_argument("--tasks", type=str, nargs="+", help="要评测的任务列表")
    parser.add_argument("--output_dir", type=str, default="comparison_results", help="结果输出目录")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    return parser.parse_args()

def evaluate_model(
    model_path: str, 
    tasks: List[str], 
    device: str,
    output_dir: str,
    config_path: str = None
) -> Dict[str, Any]:
    """评测单个模型"""
    # 创建模型特定的输出目录
    model_output_dir = os.path.join(output_dir, os.path.basename(model_path))
    os.makedirs(model_output_dir, exist_ok=True)
    
    # 初始化评测器
    evaluator = XpertEvaluator(
        model_name_or_path=model_path,
        device=device,
        output_dir=model_output_dir,
        config_path=config_path
    )
    
    # 执行评测
    results = evaluator.evaluate(tasks=tasks)
    
    return results

def main():
    args = parse_args()
    print(f"开始评测并比较模型: {args.models}")
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 如果未提供模型显示名称，使用模型路径的basename
    if args.model_names is None or len(args.model_names) != len(args.models):
        args.model_names = [os.path.basename(model_path) for model_path in args.models]
    
    # 存储所有模型的评测结果
    all_results = {}
    
    # 依次评测每个模型
    for model_path, model_name in zip(args.models, args.model_names):
        print(f"\n开始评测模型: {model_name} ({model_path})")
        
        # 执行评测
        results = evaluate_model(
            model_path=model_path,
            tasks=args.tasks,
            device=args.device,
            output_dir=args.output_dir,
            config_path=args.config
        )
        
        # 保存结果
        all_results[model_name] = results
        
        print(f"模型 {model_name} 评测完成")
    
    # 生成比较结果
    print("\n正在生成模型比较图表...")
    generate_comparison_charts(
        model_results=all_results,
        output_dir=args.output_dir
    )
    
    # 保存所有结果
    with open(os.path.join(args.output_dir, "all_results.json"), "w", encoding="utf-8") as f:
        # 将结果转换为可序列化格式
        serializable_results = {}
        for model_name, results in all_results.items():
            serializable_results[model_name] = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    serializable_results[model_name][key] = value
                else:
                    serializable_results[model_name][key] = str(value)
        
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n比较评测完成，结果已保存至: {args.output_dir}")
    print(f"比较图表已保存至: {args.output_dir}")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
XpertEval 简单使用示例
"""

import os
import sys
import argparse
from pprint import pprint

# 添加项目根目录到sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.xpert_evaluator import XpertEvaluator, evaluate

def parse_args():
    parser = argparse.ArgumentParser(description="XpertEval 简单使用示例")
    parser.add_argument("--model", type=str, required=True, help="模型名称或路径")
    parser.add_argument("--device", type=str, default="cuda", help="运行设备，cuda或cpu")
    parser.add_argument("--tasks", type=str, nargs="+", help="要评测的任务列表，例如text_understanding face_diagnosis")
    parser.add_argument("--output_dir", type=str, default="results", help="结果输出目录")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"开始评测模型: {args.model}")
    print(f"运行设备: {args.device}")
    print(f"评测任务: {args.tasks}")
    print(f"输出目录: {args.output_dir}")
    print(f"配置文件: {args.config}")
    
    # 方法1: 使用便捷函数
    if args.tasks:
        results = evaluate(
            model_name_or_path=args.model,
            tasks=args.tasks,
            device=args.device,
            output_dir=args.output_dir,
            config_path=args.config
        )
    else:
        # 方法2: 使用XpertEvaluator类
        evaluator = XpertEvaluator(
            model_name_or_path=args.model,
            device=args.device,
            output_dir=args.output_dir,
            config_path=args.config
        )
        
        # 执行评测
        results = evaluator.evaluate()
        
        # 打印结果
        evaluator.print_results(results)
    
    # 打印总体分数
    print("\n总体评测分数:")
    if "overall" in results:
        pprint(results["overall"])
    else:
        print("未获取到总体分数")
    
    print(f"\n评测完成，详细结果已保存至: {args.output_dir}")

if __name__ == "__main__":
    main() 
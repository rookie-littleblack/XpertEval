#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批量API评测脚本
"""

import os
import sys
import glob
import json
import argparse
import concurrent.futures
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到sys.path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from src.api_client import APIClient
from src.metrics.tcm_metrics import (
    calculate_syndrome_differentiation_accuracy,
    calculate_symptom_recognition_rate,
    calculate_prescription_accuracy
)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='XpertEval批量API评测')
    parser.add_argument('--task', type=str, required=True, 
                      choices=['text_understanding', 'tongue_diagnosis', 'face_diagnosis', 'prescription'],
                      help='要评测的任务类型')
    parser.add_argument('--model', type=str, default=None, 
                      help='要评测的模型名称，如不指定则使用环境变量中的默认值')
    parser.add_argument('--data_dir', type=str, default=None,
                      help='数据目录，包含要评测的图像或其他文件')
    parser.add_argument('--data_file', type=str, default=None,
                      help='数据文件，包含评测样本的JSON文件')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='结果输出目录')
    parser.add_argument('--max_workers', type=int, default=2,
                      help='最大并行工作线程数，避免API请求过快')
    parser.add_argument('--timeout', type=int, default=60,
                      help='API请求超时时间（秒）')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.task in ['tongue_diagnosis', 'face_diagnosis'] and not args.data_dir and not args.data_file:
        parser.error(f"任务 '{args.task}' 需要指定 --data_dir 或 --data_file")
    
    return args

def get_image_files(directory: str, extensions: List[str] = ['.jpg', '.jpeg', '.png']):
    """获取目录中的图像文件"""
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, f"*{ext}")))
    
    return sorted(files)

def process_tongue_diagnosis(client: APIClient, image_path: str, prompt: str = None):
    """处理单个舌诊图像"""
    if prompt is None:
        prompt = "作为中医师，请分析这张舌象图片，描述舌质、舌苔特征，并进行证型分析。请注明观察到的舌色、舌形态、舌苔颜色与厚薄等特征，以及对应的可能证型。"
    
    # 调用API
    response = client.vision_completion(prompt=prompt, image_paths=[image_path])
    
    if "error" in response:
        print(f"处理图像失败: {image_path}, 错误: {response['error']}")
        return None
    
    # 从响应中提取结果
    if "choices" in response and len(response["choices"]) > 0:
        result = response["choices"][0]["message"]["content"]
        return {
            "image_path": image_path,
            "prompt": prompt,
            "response": result
        }
    else:
        print(f"处理图像未返回有效结果: {image_path}")
        return None

def process_text_understanding(client: APIClient, question: str, reference_answer: str = None):
    """处理单个文本理解问题"""
    # 调用API
    response = client.text_completion(prompt=question)
    
    if "error" in response:
        print(f"处理问题失败: {question[:50]}..., 错误: {response['error']}")
        return None
    
    # 从响应中提取结果
    if "choices" in response and len(response["choices"]) > 0:
        result = response["choices"][0]["message"]["content"]
        return {
            "question": question,
            "reference_answer": reference_answer,
            "response": result
        }
    else:
        print(f"处理问题未返回有效结果: {question[:50]}...")
        return None

def batch_evaluate_tongue_diagnosis(args):
    """批量评测舌诊能力"""
    client = APIClient(default_model=args.model, timeout=args.timeout)
    
    # 准备数据
    samples = []
    
    if args.data_dir:
        # 从目录加载图像
        image_files = get_image_files(args.data_dir)
        samples = [{"image_path": img_path} for img_path in image_files]
        print(f"从目录 {args.data_dir} 中加载了 {len(samples)} 个舌诊图像")
    
    elif args.data_file:
        # 从JSON文件加载样本
        with open(args.data_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        print(f"从文件 {args.data_file} 中加载了 {len(samples)} 个舌诊样本")
    
    if not samples:
        print("未找到有效的评测样本")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 批量处理样本
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        
        for sample in samples:
            image_path = sample.get("image_path")
            prompt = sample.get("prompt")  # 可以为每个样本指定不同的提示
            
            future = executor.submit(process_tongue_diagnosis, client, image_path, prompt)
            futures.append(future)
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="处理舌诊图像"):
            result = future.result()
            if result:
                results.append(result)
    
    # 保存结果
    output_file = os.path.join(args.output_dir, "tongue_diagnosis_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"舌诊评测完成，结果已保存到 {output_file}")
    print(f"共评测 {len(samples)} 个样本，成功处理 {len(results)} 个")

def batch_evaluate_text_understanding(args):
    """批量评测文本理解能力"""
    client = APIClient(default_model=args.model, timeout=args.timeout)
    
    # 准备数据
    samples = []
    
    if args.data_file:
        # 从JSON文件加载样本
        with open(args.data_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        print(f"从文件 {args.data_file} 中加载了 {len(samples)} 个文本问题")
    else:
        print("未指定数据文件，使用内置示例问题")
        # 使用示例问题
        samples = [
            {
                "question": "解释中医的阴阳学说及其在诊断中的应用。",
                "reference_answer": "阴阳学说是中医理论基础，描述事物对立统一关系，用于解释生理病理现象和指导诊断治疗"
            },
            {
                "question": "简述中医'望闻问切'四诊法的内容和临床意义。",
                "reference_answer": "望闻问切是中医基本诊法，包括观察、听嗅、问诊和脉诊，全面收集患者信息，进行辨证论治"
            },
            {
                "question": "什么是'肝郁气滞'证？其临床表现和治疗原则是什么？",
                "reference_answer": "肝郁气滞是肝失疏泄导致的证候，表现为胸胁胀痛、情志抑郁、月经不调等，治疗以疏肝解郁为主"
            }
        ]
    
    if not samples:
        print("未找到有效的评测样本")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 批量处理样本
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        
        for sample in samples:
            question = sample.get("question")
            reference_answer = sample.get("reference_answer")
            
            future = executor.submit(process_text_understanding, client, question, reference_answer)
            futures.append(future)
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="处理文本问题"):
            result = future.result()
            if result:
                results.append(result)
    
    # 保存结果
    output_file = os.path.join(args.output_dir, "text_understanding_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"文本理解评测完成，结果已保存到 {output_file}")
    print(f"共评测 {len(samples)} 个样本，成功处理 {len(results)} 个")

def main():
    """主函数"""
    args = parse_args()
    
    print(f"===== XpertEval批量API评测 =====")
    print(f"任务: {args.task}")
    print(f"模型: {args.model or '从环境变量获取'}")
    if args.data_dir:
        print(f"数据目录: {args.data_dir}")
    if args.data_file:
        print(f"数据文件: {args.data_file}")
    print(f"输出目录: {args.output_dir}")
    print(f"最大线程数: {args.max_workers}")
    print("================================")
    
    # 根据任务类型执行相应的评测
    if args.task == "tongue_diagnosis":
        batch_evaluate_tongue_diagnosis(args)
    elif args.task == "text_understanding":
        batch_evaluate_text_understanding(args)
    elif args.task == "face_diagnosis":
        print("面诊评测功能尚未实现")
    elif args.task == "prescription":
        print("方剂评测功能尚未实现")
    else:
        print(f"不支持的任务类型: {args.task}")

if __name__ == "__main__":
    main() 
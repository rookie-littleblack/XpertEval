#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于OpenAI兼容API的评测示例
"""

import os
import sys
import argparse
import json
from pathlib import Path

# 添加项目根目录到sys.path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from src.api_client import APIClient
from src.xpert_evaluator import XpertEvaluator
from src.metrics.tcm_metrics import calculate_syndrome_differentiation_accuracy

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='XpertEval API评测示例')
    parser.add_argument('--model', type=str, default=None, help='待评测的模型名称')
    parser.add_argument('--api_base', type=str, default=None, help='API基础URL')
    parser.add_argument('--api_key', type=str, default=None, help='API密钥')
    parser.add_argument('--output_dir', type=str, default='results', help='结果保存目录')
    parser.add_argument('--task', type=str, default='tongue_diagnosis', 
                      choices=['text_understanding', 'tongue_diagnosis', 'face_diagnosis', 'multimodal_tcm'],
                      help='要评测的任务')
    return parser.parse_args()

def simple_text_evaluation(client, prompt):
    """简单文本评测示例"""
    print(f"执行文本评测：{prompt[:50]}...")
    
    # 调用API
    response = client.text_completion(prompt=prompt)
    
    if "error" in response:
        print(f"请求失败: {response['error']}")
        return None
    
    # 从响应中提取结果
    if "choices" in response and len(response["choices"]) > 0:
        result = response["choices"][0]["message"]["content"]
        print(f"模型响应: {result[:100]}...")
        return result
    else:
        print("未获取到有效响应")
        return None

def simple_vision_evaluation(client, prompt, image_paths):
    """简单视觉评测示例"""
    print(f"执行视觉评测：{prompt[:50]}..., 图像数量：{len(image_paths)}")
    
    # 调用API
    response = client.vision_completion(prompt=prompt, image_paths=image_paths)
    
    if "error" in response:
        print(f"请求失败: {response['error']}")
        return None
    
    # 从响应中提取结果
    if "choices" in response and len(response["choices"]) > 0:
        result = response["choices"][0]["message"]["content"]
        print(f"模型响应: {result[:100]}...")
        return result
    else:
        print("未获取到有效响应")
        return None

def evaluate_tongue_diagnosis(client):
    """评测舌诊能力"""
    print("开始评测舌诊能力...")
    
    # 模拟数据
    test_cases = [
        {
            "image_path": "data/sample_images/tongue_1.jpg",
            "prompt": "作为中医师，请分析这张舌象图片，描述舌质、舌苔特征，并进行证型分析。",
            "reference_syndrome": ["脾胃湿热", "肝胆湿热"]
        },
        {
            "image_path": "data/sample_images/tongue_2.jpg",
            "prompt": "请分析这张舌诊图片，说明你观察到的舌象特征（如舌色、舌形、舌苔等），并给出可能的中医证型判断。",
            "reference_syndrome": ["脾虚湿困", "气虚血瘀"]
        }
    ]
    
    results = []
    
    for case in test_cases:
        if not os.path.exists(case["image_path"]):
            print(f"舌诊图片不存在: {case['image_path']}, 跳过评测")
            continue
        
        # API调用获取分析结果
        response = simple_vision_evaluation(client, case["prompt"], [case["image_path"]])
        
        if not response:
            continue
        
        # 提取证型（这里需要进一步实现更复杂的提取逻辑）
        # 简单模拟证型提取
        predicted_syndromes = []
        for syndrome in ["脾胃湿热", "肝胆湿热", "脾虚湿困", "气虚血瘀", "肝郁气滞", "心脾两虚"]:
            if syndrome.lower() in response.lower():
                predicted_syndromes.append(syndrome)
        
        # 计算评测指标
        accuracy = calculate_syndrome_differentiation_accuracy(
            predicted_syndromes, case["reference_syndrome"]
        )
        
        results.append({
            "image_path": case["image_path"],
            "predicted_syndromes": predicted_syndromes,
            "reference_syndromes": case["reference_syndrome"],
            "accuracy": accuracy
        })
        
        print(f"舌诊案例评测结果: 预测证型 {predicted_syndromes}, F1分数: {accuracy['f1']:.4f}")
    
    # 计算平均分数
    if results:
        avg_f1 = sum(r["accuracy"]["f1"] for r in results) / len(results)
        print(f"\n舌诊能力评测完成, 平均F1分数: {avg_f1:.4f}")
        return {"score": avg_f1, "details": results}
    else:
        print("\n舌诊能力评测失败，无有效结果")
        return {"score": 0.0, "details": []}

def evaluate_text_understanding(client):
    """评测文本理解能力"""
    print("开始评测文本理解能力...")
    
    # 模拟数据
    test_cases = [
        {
            "prompt": "回答这个中医学问题：寒证和热证的主要区别有哪些？请用列表形式简明扼要地回答。",
            "reference_answer": ["寒证：畏寒怕冷,肢体发凉,面色苍白,舌淡苔白,脉沉紧", 
                               "热证：身热面赤,烦躁口渴,喜冷饮,舌红苔黄,脉洪数"]
        },
        {
            "prompt": "请解释中医理论中'肝主疏泄'的含义，并简述肝气郁结的表现。",
            "reference_answer": ["肝主疏泄是指肝具有调畅气机、促进脾胃消化吸收、调节情志、促进胆汁分泌排泄等功能", 
                               "肝气郁结表现：胸胁胀痛、情志抑郁、脘闷不舒、嗳气太息、月经不调"]
        }
    ]
    
    correct_count = 0
    total_count = len(test_cases)
    
    for i, case in enumerate(test_cases):
        print(f"\n测试案例 {i+1}/{total_count}:")
        
        # API调用获取回答
        response = simple_text_evaluation(client, case["prompt"])
        
        if not response:
            continue
            
        # 简单评分（实际中需要更复杂的NLP评分方法）
        score = 0
        for ref in case["reference_answer"]:
            if any(key.lower() in response.lower() for key in ref.split(',')):
                score += 1
        
        accuracy = score / len(case["reference_answer"])
        correct_count += accuracy
        
        print(f"准确率: {accuracy:.2f}")
    
    # 计算总分
    if total_count > 0:
        final_score = correct_count / total_count
        print(f"\n文本理解能力评测完成, 平均准确率: {final_score:.4f}")
        return {"score": final_score}
    else:
        print("\n文本理解能力评测失败，无有效问题")
        return {"score": 0.0}

def main():
    """主函数"""
    args = parse_args()
    
    print(f"===== XpertEval API评测示例 =====")
    print(f"模型: {args.model or '从环境变量获取'}")
    print(f"API基础URL: {args.api_base or '从环境变量获取'}")
    print(f"评测任务: {args.task}")
    print(f"输出目录: {args.output_dir}")
    print("=================================")
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化API客户端
    client = APIClient(
        api_key=args.api_key,
        api_base=args.api_base,
        default_model=args.model
    )
    
    # 执行评测
    if args.task == "tongue_diagnosis":
        results = evaluate_tongue_diagnosis(client)
    elif args.task == "text_understanding":
        results = evaluate_text_understanding(client)
    else:
        print(f"暂不支持的评测任务: {args.task}")
        return
    
    # 保存结果
    output_path = os.path.join(args.output_dir, f"{args.task}_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"评测结果已保存到: {output_path}")

if __name__ == "__main__":
    main() 
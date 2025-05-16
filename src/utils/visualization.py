"""
评测结果可视化工具
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Dict, Any, Optional

# 设置中文字体支持
try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
except:
    pass

def generate_radar_chart(
    scores: List[float], 
    categories: List[str], 
    output_path: str,
    title: str = "模型能力雷达图",
    figsize: tuple = (10, 8)
) -> None:
    """
    生成能力雷达图
    
    参数:
        scores: 各维度分数列表
        categories: 各维度名称列表
        output_path: 输出图像路径
        title: 图表标题
        figsize: 图像大小
    """
    # 转换中文类别名称
    category_names = {
        "text_understanding": "文本理解",
        "text_generation": "文本生成",
        "visual": "视觉能力",
        "audio": "音频能力",
        "multimodal": "多模态融合",
        "face_diagnosis": "面诊能力",
        "tongue_diagnosis": "舌诊能力",
        "breathing_sound": "闻诊能力",
        "symptom_understanding": "症状理解",
        "medical_history": "病史收集",
        "pulse_diagnosis": "脉诊能力",
        "multimodal_tcm": "四诊合参",
        "prescription": "方剂推荐"
    }
    
    # 应用中文名称转换
    display_categories = [category_names.get(cat, cat) for cat in categories]
    
    # 确保categories和scores长度相同
    if len(display_categories) != len(scores):
        raise ValueError("类别和分数列表长度必须相同")
    
    # 设置雷达图
    angles = np.linspace(0, 2*np.pi, len(display_categories), endpoint=False)
    
    # 闭合雷达图
    scores = np.concatenate((scores, [scores[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    display_categories = np.concatenate((display_categories, [display_categories[0]]))
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    # 绘制得分线
    ax.plot(angles, scores, 'o-', linewidth=2, color="#3370CC")
    ax.fill(angles, scores, alpha=0.25, color="#3370CC")
    
    # 设置每个轴的标签
    ax.set_thetagrids(angles[:-1] * 180/np.pi, display_categories[:-1])
    
    # 设置范围和网格
    ax.set_ylim(0, 100)
    ax.set_rlabel_position(0)
    ax.set_rticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"])
    ax.grid(True)
    
    # 设置标题
    plt.title(title, size=20, pad=20)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_bar_chart(
    scores: Dict[str, float],
    output_path: str,
    title: str = "评测结果对比",
    figsize: tuple = (12, 6)
) -> None:
    """
    生成柱状图比较不同模型或能力的分数
    
    参数:
        scores: 分数字典，格式为{名称: 分数}
        output_path: 输出图像路径
        title: 图表标题
        figsize: 图像大小
    """
    # 确保scores非空
    if not scores:
        return
    
    names = list(scores.keys())
    values = list(scores.values())
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 设置柱状图颜色
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(names)))
    
    # 绘制柱状图
    bars = ax.bar(names, values, color=colors)
    
    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3点垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # 设置标题和标签
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('能力维度', fontsize=12)
    ax.set_ylabel('分数', fontsize=12)
    
    # 设置Y轴范围
    ax.set_ylim(0, max(values) * 1.2)
    
    # 添加网格线
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 旋转X轴标签，避免重叠
    plt.xticks(rotation=45, ha='right')
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_comparison_charts(
    model_results: Dict[str, Dict[str, Any]],
    output_dir: str,
    prefix: str = "comparison"
) -> None:
    """
    生成多模型对比图表
    
    参数:
        model_results: 不同模型的评测结果字典，格式为{模型名称: 结果字典}
        output_dir: 输出目录
        prefix: 输出文件前缀
    """
    import os
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 总体分数对比
    overall_scores = {}
    for model_name, results in model_results.items():
        if "overall" in results and "overall_score" in results["overall"]:
            overall_scores[model_name] = results["overall"]["overall_score"] * 100
    
    if overall_scores:
        generate_bar_chart(
            overall_scores,
            os.path.join(output_dir, f"{prefix}_overall.png"),
            title="模型总体性能对比"
        )
    
    # 2. 通用能力对比
    general_scores = {}
    for model_name, results in model_results.items():
        if "overall" in results and "general_score" in results["overall"]:
            general_scores[model_name] = results["overall"]["general_score"] * 100
    
    if general_scores:
        generate_bar_chart(
            general_scores,
            os.path.join(output_dir, f"{prefix}_general.png"),
            title="模型通用能力对比"
        )
    
    # 3. 中医专业能力对比
    tcm_scores = {}
    for model_name, results in model_results.items():
        if "overall" in results and "tcm_score" in results["overall"]:
            tcm_scores[model_name] = results["overall"]["tcm_score"] * 100
    
    if tcm_scores:
        generate_bar_chart(
            tcm_scores,
            os.path.join(output_dir, f"{prefix}_tcm.png"),
            title="中医专业能力对比"
        )
    
    # 4. 各任务对比
    task_types = [
        # 通用能力
        "text_understanding", "text_generation", "visual", "audio", "multimodal",
        # 中医专业能力
        "face_diagnosis", "tongue_diagnosis", "breathing_sound", 
        "symptom_understanding", "medical_history", "pulse_diagnosis",
        "multimodal_tcm", "prescription"
    ]
    
    for task in task_types:
        task_scores = {}
        for model_name, results in model_results.items():
            if task in results and "score" in results[task]:
                task_scores[model_name] = results[task]["score"] * 100
        
        if task_scores:
            generate_bar_chart(
                task_scores,
                os.path.join(output_dir, f"{prefix}_{task}.png"),
                title=f"{task}能力对比"
            ) 
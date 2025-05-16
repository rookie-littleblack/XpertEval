"""
中医专业评测指标计算
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union

def calculate_feature_recognition_accuracy(
    predictions: List[Dict[str, Any]],
    references: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    计算特征识别准确率（面诊、舌诊等）
    
    参数:
        predictions: 预测结果列表，每个元素是包含特征识别结果的字典
        references: 参考标准列表，每个元素是包含正确特征标注的字典
    
    返回:
        包含不同维度准确率的字典
    """
    if len(predictions) != len(references):
        raise ValueError("预测结果与参考标准数量不一致")
    
    if len(predictions) == 0:
        return {"overall_accuracy": 0.0}
    
    # 初始化各特征维度的准确率统计
    feature_correct = {}
    feature_total = {}
    
    # 遍历每个样本
    for pred, ref in zip(predictions, references):
        # 遍历参考标准中的每个特征维度
        for feature, value in ref.items():
            if feature not in feature_total:
                feature_correct[feature] = 0
                feature_total[feature] = 0
            
            feature_total[feature] += 1
            
            # 检查预测是否正确
            if feature in pred and pred[feature] == value:
                feature_correct[feature] += 1
    
    # 计算各维度准确率
    results = {}
    for feature in feature_total:
        results[feature] = feature_correct[feature] / feature_total[feature]
    
    # 计算总体准确率
    total_correct = sum(feature_correct.values())
    total_features = sum(feature_total.values())
    results["overall_accuracy"] = total_correct / total_features if total_features > 0 else 0.0
    
    return results

def calculate_syndrome_correlation(
    predicted_syndromes: List[str],
    reference_syndromes: List[str],
    syndrome_similarity_matrix: Optional[Dict[str, Dict[str, float]]] = None
) -> float:
    """
    计算证型相关度
    
    参数:
        predicted_syndromes: 预测的证型列表
        reference_syndromes: 参考的证型列表
        syndrome_similarity_matrix: 证型间相似度矩阵
    
    返回:
        证型相关度分数
    """
    if not predicted_syndromes or not reference_syndromes:
        return 0.0
    
    # 如果没有提供相似度矩阵，则简单计算重叠率
    if syndrome_similarity_matrix is None:
        # 计算交集大小
        overlap = set(predicted_syndromes).intersection(set(reference_syndromes))
        # 计算Jaccard相似度
        return len(overlap) / (len(predicted_syndromes) + len(reference_syndromes) - len(overlap))
    
    # 如果提供了相似度矩阵，计算语义相似度
    total_similarity = 0.0
    count = 0
    
    for pred in predicted_syndromes:
        for ref in reference_syndromes:
            # 查找两个证型之间的相似度
            if pred in syndrome_similarity_matrix and ref in syndrome_similarity_matrix[pred]:
                total_similarity += syndrome_similarity_matrix[pred][ref]
            else:
                # 如果未找到相似度，假设为0.1（低相似度）
                total_similarity += 0.1
            count += 1
    
    # 计算平均相似度
    return total_similarity / count if count > 0 else 0.0

def calculate_description_completeness(
    prediction: str,
    reference_keywords: List[str],
    reference_structure: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    计算描述完整性（面诊、舌诊描述等）
    
    参数:
        prediction: 预测的描述文本
        reference_keywords: 参考关键词列表
        reference_structure: 参考结构列表（如舌诊应包含舌色、舌形、舌苔等）
    
    返回:
        包含完整性和结构性得分的字典
    """
    # 初始化结果
    results = {}
    
    # 关键词覆盖率
    covered_keywords = sum(keyword.lower() in prediction.lower() for keyword in reference_keywords)
    results["keyword_coverage"] = covered_keywords / len(reference_keywords) if reference_keywords else 0.0
    
    # 结构完整性
    if reference_structure:
        structure_coverage = sum(section.lower() in prediction.lower() for section in reference_structure)
        results["structure_completeness"] = structure_coverage / len(reference_structure)
    else:
        results["structure_completeness"] = 0.8  # 默认值
    
    # 总体完整性得分
    results["overall_completeness"] = 0.6 * results["keyword_coverage"] + 0.4 * results["structure_completeness"]
    
    return results

def calculate_pulse_recognition_accuracy(
    predictions: List[str],
    references: List[str]
) -> float:
    """
    计算脉象识别准确率
    
    参数:
        predictions: 预测的脉象类型列表
        references: 参考的脉象类型列表
    
    返回:
        脉象识别准确率
    """
    if len(predictions) != len(references):
        raise ValueError("预测结果与参考标准数量不一致")
    
    if len(predictions) == 0:
        return 0.0
    
    # 计算完全匹配的数量
    exact_matches = sum(pred == ref for pred, ref in zip(predictions, references))
    
    # 计算部分匹配（如果预测包含多种脉象）
    partial_matches = 0
    for pred, ref in zip(predictions, references):
        if pred != ref and (pred in ref or ref in pred):
            partial_matches += 0.5
    
    return (exact_matches + partial_matches) / len(predictions)

def calculate_symptom_recognition_rate(
    predicted_symptoms: List[str],
    reference_symptoms: List[str]
) -> Dict[str, float]:
    """
    计算症状识别率
    
    参数:
        predicted_symptoms: 预测的症状列表
        reference_symptoms: 参考的症状列表
    
    返回:
        包含识别率、精确率、召回率的字典
    """
    if not reference_symptoms:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # 计算交集
    true_positives = set(predicted_symptoms).intersection(set(reference_symptoms))
    
    # 计算精确率和召回率
    precision = len(true_positives) / len(predicted_symptoms) if predicted_symptoms else 0.0
    recall = len(true_positives) / len(reference_symptoms)
    
    # 计算F1分数
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def calculate_inquiry_completeness(
    prediction: Dict[str, Any],
    reference: Dict[str, Any]
) -> Dict[str, float]:
    """
    计算问诊完整性
    
    参数:
        prediction: 预测的问诊结果字典
        reference: 参考的问诊结果字典
    
    返回:
        包含不同维度完整性评分的字典
    """
    if not reference:
        return {"overall_completeness": 0.0}
    
    # 计算各个维度的覆盖情况
    dimension_scores = {}
    
    for key, ref_value in reference.items():
        if isinstance(ref_value, list):
            # 如果是列表类型（如症状列表），计算覆盖率
            if key in prediction and isinstance(prediction[key], list):
                pred_set = set(prediction[key])
                ref_set = set(ref_value)
                overlap = pred_set.intersection(ref_set)
                dimension_scores[key] = len(overlap) / len(ref_set) if ref_set else 1.0
            else:
                dimension_scores[key] = 0.0
        else:
            # 如果是其他类型，检查是否存在并匹配
            dimension_scores[key] = 1.0 if key in prediction and prediction[key] == ref_value else 0.0
    
    # 计算总体完整性
    overall_completeness = sum(dimension_scores.values()) / len(dimension_scores) if dimension_scores else 0.0
    dimension_scores["overall_completeness"] = overall_completeness
    
    return dimension_scores

def calculate_modal_consistency(
    visual_prediction: Dict[str, Any],
    text_prediction: Dict[str, Any],
    audio_prediction: Dict[str, Any],
    reference: Dict[str, Any]
) -> Dict[str, float]:
    """
    计算多模态一致性
    
    参数:
        visual_prediction: 视觉模态预测结果
        text_prediction: 文本模态预测结果
        audio_prediction: 音频模态预测结果
        reference: 参考标准结果
    
    返回:
        包含多模态一致性评分的字典
    """
    # 计算各模态与参考标准的一致性
    modality_consistency = {}
    
    # 计算各关键维度的一致性
    key_dimensions = ["diagnosis", "syndrome", "treatment"]
    
    for key in key_dimensions:
        # 检查各模态是否包含该维度
        visual_has_key = key in visual_prediction
        text_has_key = key in text_prediction
        audio_has_key = key in audio_prediction
        
        # 如果所有模态都有该维度，计算它们之间的一致性
        if visual_has_key and text_has_key and audio_has_key:
            agreement_count = 0
            total_pairs = 0
            
            # 检查视觉与文本一致性
            if visual_prediction[key] == text_prediction[key]:
                agreement_count += 1
            total_pairs += 1
            
            # 检查视觉与音频一致性
            if visual_prediction[key] == audio_prediction[key]:
                agreement_count += 1
            total_pairs += 1
            
            # 检查文本与音频一致性
            if text_prediction[key] == audio_prediction[key]:
                agreement_count += 1
            total_pairs += 1
            
            # 计算一致性得分
            modality_consistency[key] = agreement_count / total_pairs
        else:
            # 如果不是所有模态都有该维度，给予较低分数
            modality_consistency[key] = 0.3
    
    # 计算总体一致性
    overall_consistency = sum(modality_consistency.values()) / len(modality_consistency) if modality_consistency else 0.0
    modality_consistency["overall_consistency"] = overall_consistency
    
    # 计算与参考标准的一致性
    if reference:
        reference_consistency = {}
        for key in key_dimensions:
            if key in reference:
                # 计算各模态与参考标准的匹配度
                visual_correct = key in visual_prediction and visual_prediction[key] == reference[key]
                text_correct = key in text_prediction and text_prediction[key] == reference[key]
                audio_correct = key in audio_prediction and audio_prediction[key] == reference[key]
                
                # 计算正确率
                correct_count = sum([visual_correct, text_correct, audio_correct])
                reference_consistency[key] = correct_count / 3
        
        # 计算总体参考一致性
        overall_ref_consistency = sum(reference_consistency.values()) / len(reference_consistency) if reference_consistency else 0.0
        modality_consistency["reference_consistency"] = overall_ref_consistency
    
    return modality_consistency

def calculate_prescription_accuracy(
    predicted_herbs: List[str],
    reference_herbs: List[str]
) -> Dict[str, float]:
    """
    计算方剂准确率
    
    参数:
        predicted_herbs: 预测的药物列表
        reference_herbs: 参考的药物列表
    
    返回:
        包含方剂评估指标的字典
    """
    if not reference_herbs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # 转换为集合
    predicted_set = set(predicted_herbs)
    reference_set = set(reference_herbs)
    
    # 计算交集
    correct_herbs = predicted_set.intersection(reference_set)
    
    # 计算精确率和召回率
    precision = len(correct_herbs) / len(predicted_herbs) if predicted_herbs else 0.0
    recall = len(correct_herbs) / len(reference_herbs)
    
    # 计算F1分数
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def calculate_dosage_rationality(
    predicted_dosages: Dict[str, float],
    reference_dosages: Dict[str, float],
    tolerance: float = 0.2
) -> Dict[str, float]:
    """
    计算用药剂量合理性
    
    参数:
        predicted_dosages: 预测的药物剂量字典 {药物名: 剂量}
        reference_dosages: 参考的药物剂量字典
        tolerance: 剂量误差容忍度（相对误差）
    
    返回:
        包含剂量合理性评分的字典
    """
    if not reference_dosages:
        return {"dosage_accuracy": 0.0, "rationality_score": 0.0}
    
    # 统计剂量准确的药物数
    correct_dosage = 0
    evaluated_herbs = 0
    
    # 计算相对误差在容忍范围内的药物
    for herb, ref_dose in reference_dosages.items():
        if herb in predicted_dosages:
            evaluated_herbs += 1
            pred_dose = predicted_dosages[herb]
            
            # 计算相对误差
            relative_error = abs(pred_dose - ref_dose) / ref_dose if ref_dose > 0 else 1.0
            
            # 如果误差在容忍范围内，视为准确
            if relative_error <= tolerance:
                correct_dosage += 1
    
    # 计算剂量准确率
    dosage_accuracy = correct_dosage / evaluated_herbs if evaluated_herbs > 0 else 0.0
    
    # 计算合理性得分（考虑中药配伍禁忌等）
    # 这里仅做简化处理，实际应用中可能需要考虑更复杂的规则
    rationality_score = dosage_accuracy * 0.9  # 假设90%的合理性基于剂量准确度
    
    return {
        "dosage_accuracy": dosage_accuracy,
        "rationality_score": rationality_score
    }

def calculate_explanation_rationality(
    prediction: str,
    reference_keywords: List[str],
    reference_reasonings: List[str]
) -> Dict[str, float]:
    """
    计算方剂解释合理性
    
    参数:
        prediction: 预测的方剂解释文本
        reference_keywords: 参考的关键词列表（如药物功效、证型等）
        reference_reasonings: 参考的推理逻辑列表
    
    返回:
        包含解释合理性评分的字典
    """
    # 关键词覆盖率
    keyword_coverage = sum(keyword.lower() in prediction.lower() for keyword in reference_keywords)
    keyword_score = keyword_coverage / len(reference_keywords) if reference_keywords else 0.0
    
    # 推理逻辑覆盖率（简化计算）
    reasoning_coverage = sum(reasoning.lower() in prediction.lower() for reasoning in reference_reasonings)
    reasoning_score = reasoning_coverage / len(reference_reasonings) if reference_reasonings else 0.0
    
    # 计算总体解释合理性得分
    overall_rationality = 0.4 * keyword_score + 0.6 * reasoning_score
    
    return {
        "keyword_score": keyword_score,
        "reasoning_score": reasoning_score,
        "overall_rationality": overall_rationality
    } 
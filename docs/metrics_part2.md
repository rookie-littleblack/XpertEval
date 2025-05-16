---
layout: default
title: 评测指标（下）
---

# 中医药专业评测指标体系

本文档详细介绍XpertEval评测框架中针对中医药专业领域的评测指标，包括望诊、闻诊、问诊、切诊、四诊合参及方剂推荐能力的评估指标，是[评测指标体系](metrics.md)的配套文档。

## 一、中医药专业评测体系设计原则

中医药专业评测指标的设计遵循以下原则：

1. **专业性**：指标体系应符合中医理论体系和临床实践特点
2. **客观性**：尽量采用量化指标，减少主观评价成分
3. **系统性**：全面覆盖四诊合参和辨证论治的各个环节
4. **可扩展性**：支持根据需求增加新的专业评测指标
5. **临床相关性**：指标设计应贴近临床应用场景，具有实际参考价值

## 二、望诊能力评测指标

望诊是中医四诊法之首，主要通过视觉观察获取患者信息。评测模型的望诊能力主要包括面诊和舌诊两大方面。

### 1. 面诊能力评测指标

#### 1.1 面色特征识别准确率

评估模型识别面部颜色、光泽等特征的准确性。

```python
def calculate_feature_recognition_accuracy(predictions, references):
    """
    计算面色特征识别准确率
    
    参数:
        predictions: 预测的面色特征字典列表，如[{"face_color": "淡白", "face_shape": "消瘦"}, ...]
        references: 参考的面色特征字典列表
    
    返回:
        包含各特征维度准确率的字典
    """
    if len(predictions) != len(references):
        raise ValueError("预测结果与参考标准数量不一致")
    
    # 初始化各特征维度统计
    feature_count = {}
    feature_correct = {}
    
    # 遍历每个样本
    for pred, ref in zip(predictions, references):
        # 遍历每个特征维度
        for feature_key, ref_value in ref.items():
            if feature_key not in feature_count:
                feature_count[feature_key] = 0
                feature_correct[feature_key] = 0
            
            feature_count[feature_key] += 1
            
            # 检查预测是否匹配参考
            if feature_key in pred and pred[feature_key] == ref_value:
                feature_correct[feature_key] += 1
    
    # 计算各维度准确率
    accuracy = {}
    for feature_key in feature_count:
        accuracy[feature_key] = feature_correct[feature_key] / feature_count[feature_key]
    
    # 计算总体准确率
    total_correct = sum(feature_correct.values())
    total_count = sum(feature_count.values())
    accuracy["overall"] = total_correct / total_count if total_count > 0 else 0.0
    
    return accuracy
```

#### 1.2 面诊描述完整性评分

评估模型描述面诊特征的完整程度。

```python
def calculate_face_description_completeness(predictions, required_aspects):
    """
    计算面诊描述完整性评分
    
    参数:
        predictions: 预测的面诊描述文本列表
        required_aspects: 必须包含的面诊方面列表，如["面色", "形体", "神态", "表情"]
    
    返回:
        完整性评分（0-1之间）
    """
    completeness_scores = []
    
    for prediction in predictions:
        # 计算覆盖了多少必需方面
        covered_aspects = 0
        for aspect in required_aspects:
            if aspect in prediction.lower():
                covered_aspects += 1
        
        # 计算完整性得分
        score = covered_aspects / len(required_aspects)
        completeness_scores.append(score)
    
    # 返回平均完整性得分
    return sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0
```

#### 1.3 面诊证型相关度评分

评估模型基于面诊特征推导出的证型与标准证型之间的相关程度。

```python
def calculate_face_syndrome_correlation(predicted_syndromes, reference_syndromes, syndrome_similarity_matrix=None):
    """
    计算面诊证型相关度
    
    参数:
        predicted_syndromes: 预测的证型列表
        reference_syndromes: 参考的证型列表
        syndrome_similarity_matrix: 证型间的相似度矩阵（可选）
    
    返回:
        证型相关度评分
    """
    if not predicted_syndromes or not reference_syndromes:
        return 0.0
    
    # 如果没有提供相似度矩阵，使用Jaccard相似度
    if syndrome_similarity_matrix is None:
        # 计算交集大小
        intersection = set(predicted_syndromes).intersection(set(reference_syndromes))
        union = set(predicted_syndromes).union(set(reference_syndromes))
        
        # Jaccard相似度
        return len(intersection) / len(union) if union else 0.0
    
    # 如果提供了相似度矩阵，计算平均相似度
    total_similarity = 0.0
    count = 0
    
    for pred in predicted_syndromes:
        for ref in reference_syndromes:
            # 查找两个证型间的相似度
            if pred in syndrome_similarity_matrix and ref in syndrome_similarity_matrix[pred]:
                similarity = syndrome_similarity_matrix[pred][ref]
            else:
                # 默认低相似度
                similarity = 0.1
            
            total_similarity += similarity
            count += 1
    
    return total_similarity / count if count > 0 else 0.0
```

### 2. 舌诊能力评测指标

#### 2.1 舌象特征识别准确率

评估模型识别舌质、舌苔等特征的准确性。

```python
def calculate_tongue_feature_recognition_accuracy(predictions, references):
    """
    计算舌象特征识别准确率
    
    参数:
        predictions: 预测的舌象特征字典列表，如[{"tongue_color": "淡红", "tongue_coating": "白苔"}, ...]
        references: 参考的舌象特征字典列表
    
    返回:
        包含各特征维度准确率的字典
    """
    if len(predictions) != len(references):
        raise ValueError("预测结果与参考标准数量不一致")
    
    # 初始化统计
    feature_count = {"tongue_color": 0, "tongue_shape": 0, "tongue_coating": 0, "overall": 0}
    feature_correct = {"tongue_color": 0, "tongue_shape": 0, "tongue_coating": 0, "overall": 0}
    
    # 遍历每个样本
    for pred, ref in zip(predictions, references):
        total_features = 0
        correct_features = 0
        
        # 检查舌质颜色
        if "tongue_color" in ref:
            feature_count["tongue_color"] += 1
            total_features += 1
            
            if "tongue_color" in pred and pred["tongue_color"] == ref["tongue_color"]:
                feature_correct["tongue_color"] += 1
                correct_features += 1
        
        # 检查舌体形态
        if "tongue_shape" in ref:
            feature_count["tongue_shape"] += 1
            total_features += 1
            
            if "tongue_shape" in pred and pred["tongue_shape"] == ref["tongue_shape"]:
                feature_correct["tongue_shape"] += 1
                correct_features += 1
        
        # 检查舌苔特征
        if "tongue_coating" in ref:
            feature_count["tongue_coating"] += 1
            total_features += 1
            
            if "tongue_coating" in pred and pred["tongue_coating"] == ref["tongue_coating"]:
                feature_correct["tongue_coating"] += 1
                correct_features += 1
        
        # 更新总体统计
        feature_count["overall"] += total_features
        feature_correct["overall"] += correct_features
    
    # 计算各维度准确率
    accuracy = {}
    for feature in feature_count:
        accuracy[feature] = feature_correct[feature] / feature_count[feature] if feature_count[feature] > 0 else 0.0
    
    return accuracy
```

#### 2.2 舌诊描述完整性评分

评估模型描述舌诊特征的完整程度。

```python
def calculate_tongue_description_completeness(predictions, required_aspects=None):
    """
    计算舌诊描述完整性评分
    
    参数:
        predictions: 预测的舌诊描述文本列表
        required_aspects: 必须包含的舌诊方面列表，默认为["舌质", "舌色", "舌形", "舌苔", "舌苔颜色", "舌苔厚薄"]
    
    返回:
        完整性评分（0-1之间）
    """
    if required_aspects is None:
        required_aspects = ["舌质", "舌色", "舌形", "舌苔", "舌苔颜色", "舌苔厚薄"]
    
    completeness_scores = []
    
    for prediction in predictions:
        # 计算覆盖了多少必需方面
        covered_aspects = 0
        for aspect in required_aspects:
            if aspect in prediction:
                covered_aspects += 1
        
        # 计算完整性得分
        score = covered_aspects / len(required_aspects)
        completeness_scores.append(score)
    
    # 返回平均完整性得分
    return sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0
```

#### 2.3 舌诊证型关联度评分

评估模型基于舌诊特征推导出的证型与标准证型之间的关联程度。

```python
def calculate_tongue_syndrome_correlation(predicted_syndromes, reference_syndromes, tcm_knowledge_base=None):
    """
    计算舌诊证型关联度
    
    参数:
        predicted_syndromes: 预测的证型列表
        reference_syndromes: 参考的证型列表
        tcm_knowledge_base: 中医知识库，包含证型与舌象的关联信息（可选）
    
    返回:
        证型关联度评分
    """
    # 基本的交集计算
    intersection = set(predicted_syndromes).intersection(set(reference_syndromes))
    simple_correlation = len(intersection) / max(len(predicted_syndromes), len(reference_syndromes)) if max(len(predicted_syndromes), len(reference_syndromes)) > 0 else 0.0
    
    # 如果没有提供知识库，返回简单相关度
    if tcm_knowledge_base is None:
        return simple_correlation
    
    # 使用知识库进行更复杂的评估
    # 这里假设tcm_knowledge_base是一个包含证型与舌象关联的字典
    # 例如: {"脾胃湿热": {"tongue_color": ["红"], "tongue_coating": ["黄腻"]}, ...}
    
    semantic_correlation = 0.0
    
    for pred in predicted_syndromes:
        max_sim = 0.0
        for ref in reference_syndromes:
            # 计算两个证型在知识库中的相似度
            if pred in tcm_knowledge_base and ref in tcm_knowledge_base:
                shared_features = 0
                total_features = 0
                
                # 比较每个舌象特征
                for feature_type in ["tongue_color", "tongue_shape", "tongue_coating"]:
                    if feature_type in tcm_knowledge_base[pred] and feature_type in tcm_knowledge_base[ref]:
                        pred_features = set(tcm_knowledge_base[pred][feature_type])
                        ref_features = set(tcm_knowledge_base[ref][feature_type])
                        
                        shared = len(pred_features.intersection(ref_features))
                        total = len(pred_features.union(ref_features))
                        
                        shared_features += shared
                        total_features += total
                
                sim = shared_features / total_features if total_features > 0 else 0.0
                max_sim = max(max_sim, sim)
        
        semantic_correlation += max_sim
    
    semantic_correlation /= len(predicted_syndromes) if predicted_syndromes else 1.0
    
    # 综合简单相关度和语义相关度
    return 0.4 * simple_correlation + 0.6 * semantic_correlation
```

## 3. 闻诊能力评测指标

闻诊是中医四诊之一，主要通过听声音和嗅气味收集患者信息。XpertEval 重点评估模型的声音分析能力，包括咳嗽音、呼吸音等特征的识别和分类。

### 3.1 声音分类准确率

评估模型对不同类型声音的分类准确率，如咳嗽音、呼吸音等。

```python
def calculate_sound_classification_accuracy(predictions, references):
    """
    计算声音分类准确率
    
    参数:
        predictions: 预测的声音类型列表
        references: 参考的声音类型列表
    
    返回:
        分类准确率
    """
    if len(predictions) != len(references):
        raise ValueError("预测结果与参考标准数量不一致")
    
    correct = sum(pred == ref for pred, ref in zip(predictions, references))
    return correct / len(predictions) if len(predictions) > 0 else 0.0
```

### 3.2 声音特征识别细致度

评估模型识别声音细微特征的能力，如声音的强弱、频率、音调等。

```python
def calculate_sound_feature_recognition_detail(predictions, references):
    """
    计算声音特征识别细致度
    
    参数:
        predictions: 预测的声音特征字典列表，如[{"sound_type": "咳嗽", "intensity": "强", "frequency": "高"}, ...]
        references: 参考的声音特征字典列表
    
    返回:
        包含各特征维度细致度的字典
    """
    if len(predictions) != len(references):
        raise ValueError("预测结果与参考标准数量不一致")
    
    feature_types = ["sound_type", "intensity", "frequency", "quality", "rhythm"]
    feature_scores = {feature: 0.0 for feature in feature_types}
    feature_counts = {feature: 0 for feature in feature_types}
    
    for pred, ref in zip(predictions, references):
        for feature in feature_types:
            if feature in ref:
                feature_counts[feature] += 1
                if feature in pred and pred[feature] == ref[feature]:
                    feature_scores[feature] += 1.0
    
    # 计算各特征维度的准确率
    detail_scores = {}
    for feature in feature_types:
        if feature_counts[feature] > 0:
            detail_scores[feature] = feature_scores[feature] / feature_counts[feature]
    
    # 计算总体细致度
    total_features = sum(feature_counts.values())
    total_correct = sum(feature_scores.values())
    detail_scores["overall"] = total_correct / total_features if total_features > 0 else 0.0
    
    return detail_scores
```

### 3.3 声音-证型关联准确率

评估模型将声音特征与证型正确关联的能力。

```python
def calculate_sound_syndrome_correlation(sound_features, predicted_syndromes, reference_syndromes, sound_syndrome_map=None):
    """
    计算声音特征与证型关联准确率
    
    参数:
        sound_features: 声音特征列表
        predicted_syndromes: 预测的证型列表
        reference_syndromes: 参考的证型列表
        sound_syndrome_map: 声音特征到证型的映射字典（可选）
    
    返回:
        声音-证型关联准确率
    """
    # 计算预测证型与参考证型的交集比例
    pred_set = set(predicted_syndromes)
    ref_set = set(reference_syndromes)
    
    if not ref_set:
        return 0.0
    
    # 基本关联准确率 (Jaccard系数)
    basic_correlation = len(pred_set.intersection(ref_set)) / len(pred_set.union(ref_set)) if pred_set.union(ref_set) else 0.0
    
    # 如果没有提供声音-证型映射，返回基本关联准确率
    if sound_syndrome_map is None:
        return basic_correlation
    
    # 使用声音-证型映射进行更详细的评估
    expected_syndromes = set()
    for feature in sound_features:
        if feature in sound_syndrome_map:
            expected_syndromes.update(sound_syndrome_map[feature])
    
    if not expected_syndromes:
        return basic_correlation
    
    # 计算预测证型与期望证型(基于声音特征)的关联度
    feature_syndrome_correlation = len(pred_set.intersection(expected_syndromes)) / len(expected_syndromes)
    
    # 综合基本关联准确率和特征-证型关联度
    return 0.5 * basic_correlation + 0.5 * feature_syndrome_correlation
```

### 3.4 声音描述完整性

评估模型描述声音特征的完整程度。

```python
def calculate_sound_description_completeness(descriptions, required_aspects=None):
    """
    计算声音描述完整性
    
    参数:
        descriptions: 模型对声音的描述文本列表
        required_aspects: 必须包含的声音描述方面，默认为["音色", "强度", "持续时间", "节律", "病理特征"]
    
    返回:
        描述完整性评分
    """
    if required_aspects is None:
        required_aspects = ["音色", "强度", "持续时间", "节律", "病理特征"]
    
    completeness_scores = []
    
    for description in descriptions:
        # 计算包含了多少必需描述方面
        covered_aspects = 0
        for aspect in required_aspects:
            if aspect in description:
                covered_aspects += 1
        
        score = covered_aspects / len(required_aspects)
        completeness_scores.append(score)
    
    # 计算平均完整性得分
    return sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0
```

### 3.5 语音语调分析能力

评估模型分析患者语音语调异常的能力。

```python
def calculate_voice_analysis_accuracy(predictions, references):
    """
    计算语音语调分析准确率
    
    参数:
        predictions: 预测的语音特征字典列表，如[{"tone": "平稳", "speed": "缓慢", "volume": "低沉"}, ...]
        references: 参考的语音特征字典列表
    
    返回:
        语音分析准确率
    """
    if len(predictions) != len(references):
        raise ValueError("预测结果与参考标准数量不一致")
    
    # 统计正确识别的特征数和总特征数
    total_features = 0
    correct_features = 0
    
    for pred, ref in zip(predictions, references):
        for feature, value in ref.items():
            total_features += 1
            if feature in pred and pred[feature] == value:
                correct_features += 1
    
    # 计算总体准确率
    return correct_features / total_features if total_features > 0 else 0.0
```

## 4. 问诊能力评测指标

问诊是中医四诊中的关键环节，通过询问患者症状、病史等信息进行诊断。评测模型的问诊能力主要包括症状理解与分类、问诊完整性和病史收集能力等方面。

### 4.1 症状识别准确率

评估模型从患者描述中正确识别症状的能力。

```python
def calculate_symptom_recognition_rate(predictions, references):
    """
    计算症状识别准确率
    
    参数:
        predictions: 预测的症状列表列表，如[["头痛", "发热", "乏力"], ...]
        references: 参考的症状列表列表
    
    返回:
        包含精确率、召回率和F1值的字典
    """
    results = {
        "precision": [],
        "recall": [],
        "f1": []
    }
    
    for pred_symptoms, ref_symptoms in zip(predictions, references):
        pred_set = set(pred_symptoms)
        ref_set = set(ref_symptoms)
        
        # 计算交集大小
        intersection = len(pred_set.intersection(ref_set))
        
        # 精确率：正确识别的症状 / 预测的总症状数
        precision = intersection / len(pred_set) if pred_set else 0.0
        
        # 召回率：正确识别的症状 / 参考的总症状数
        recall = intersection / len(ref_set) if ref_set else 0.0
        
        # F1值
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results["precision"].append(precision)
        results["recall"].append(recall)
        results["f1"].append(f1)
    
    # 计算平均值
    for metric in results:
        results[metric] = sum(results[metric]) / len(results[metric]) if results[metric] else 0.0
    
    return results
```

### 4.2 症状分类准确率

评估模型正确将症状分类为中医症候的能力。

```python
def calculate_symptom_classification_accuracy(symptom_classifications, reference_classifications):
    """
    计算症状分类准确率
    
    参数:
        symptom_classifications: 预测的症状分类字典列表，如[{"头痛": "阳明头痛", "发热": "风热"}, ...]
        reference_classifications: 参考的症状分类字典列表
    
    返回:
        分类准确率
    """
    total_symptoms = 0
    correct_classifications = 0
    
    for pred_dict, ref_dict in zip(symptom_classifications, reference_classifications):
        for symptom, ref_class in ref_dict.items():
            if symptom in pred_dict:
                total_symptoms += 1
                if pred_dict[symptom] == ref_class:
                    correct_classifications += 1
    
    return correct_classifications / total_symptoms if total_symptoms > 0 else 0.0
```

### 4.3 问诊完整性评分

评估模型在问诊过程中收集信息的完整程度。

```python
def calculate_inquiry_completeness(collected_info, required_info):
    """
    计算问诊完整性评分
    
    参数:
        collected_info: 模型收集到的信息列表，每个元素是一个字典
        required_info: 必须收集的信息类别列表，如["主诉", "发病时间", "伴随症状", "既往史"]
    
    返回:
        问诊完整性评分
    """
    completeness_scores = []
    
    for info_dict in collected_info:
        # 计算收集到的必需信息比例
        covered_categories = 0
        for category in required_info:
            if category in info_dict and info_dict[category]:  # 非空值
                covered_categories += 1
        
        score = covered_categories / len(required_info)
        completeness_scores.append(score)
    
    return sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0
```

### 4.4 问诊效率评分

评估模型在问诊过程中获取关键信息的效率。

```python
def calculate_inquiry_efficiency(dialogues, collected_info, required_info):
    """
    计算问诊效率评分
    
    参数:
        dialogues: 问诊对话列表，每个元素是一个对话轮次列表
        collected_info: 收集到的信息列表，与dialogues一一对应
        required_info: 必须收集的信息类别列表
    
    返回:
        问诊效率评分
    """
    efficiency_scores = []
    
    for dialogue, info_dict in zip(dialogues, collected_info):
        # 计算医生问题数量
        doctor_turns = sum(1 for turn in dialogue if turn["role"] == "doctor")
        
        if doctor_turns == 0:
            efficiency_scores.append(0.0)
            continue
        
        # 计算收集到的必需信息数量
        covered_categories = sum(1 for category in required_info if category in info_dict and info_dict[category])
        
        # 计算效率：收集的信息 / 问题数量，并进行归一化
        raw_efficiency = covered_categories / doctor_turns
        
        # 使用衰减函数，避免过少的问题获得过高的分数
        normalized_efficiency = raw_efficiency * (1 - math.exp(-0.5 * doctor_turns))
        
        efficiency_scores.append(normalized_efficiency)
    
    return sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0.0
```

### 4.5 诊断相关性评分

评估模型基于问诊信息提出的诊断与标准诊断的相关程度。

```python
def calculate_diagnosis_relevance(predicted_diagnoses, reference_diagnoses):
    """
    计算诊断相关性评分
    
    参数:
        predicted_diagnoses: 预测的诊断列表
        reference_diagnoses: 参考的诊断列表
    
    返回:
        诊断相关性评分
    """
    relevance_scores = []
    
    for pred_diagnosis, ref_diagnosis in zip(predicted_diagnoses, reference_diagnoses):
        # 将诊断转换为集合
        pred_set = set(pred_diagnosis) if isinstance(pred_diagnosis, list) else {pred_diagnosis}
        ref_set = set(ref_diagnosis) if isinstance(ref_diagnosis, list) else {ref_diagnosis}
        
        # 计算Jaccard相似度
        intersection = len(pred_set.intersection(ref_set))
        union = len(pred_set.union(ref_set))
        
        relevance = intersection / union if union > 0 else 0.0
        relevance_scores.append(relevance)
    
    return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
```

## 5. 切诊能力评测指标

切诊是中医四诊之一，主要通过触摸获取患者的脉象、体表等信息。在XpertEval框架中，重点评估模型对脉象数据的分析能力。

### 5.1 脉象分类准确率

评估模型对不同脉象类型（如浮脉、沉脉、滑脉等）的识别准确率。

```python
def calculate_pulse_classification_accuracy(predictions, references):
    """
    计算脉象分类准确率
    
    参数:
        predictions: 预测的脉象类型列表
        references: 参考的脉象类型列表
    
    返回:
        分类准确率
    """
    if len(predictions) != len(references):
        raise ValueError("预测结果与参考标准数量不一致")
    
    # 完全匹配的数量
    exact_matches = sum(pred == ref for pred, ref in zip(predictions, references))
    
    # 部分匹配（当预测或参考标准包含多种脉象时）
    partial_matches = 0
    for pred, ref in zip(predictions, references):
        if pred != ref:
            # 处理多种脉象的情况
            if isinstance(pred, str) and isinstance(ref, str):
                pred_types = pred.split('兼')
                ref_types = ref.split('兼')
            elif isinstance(pred, list) and isinstance(ref, list):
                pred_types = pred
                ref_types = ref
            else:
                continue
            
            # 计算交集大小
            common_types = set(pred_types).intersection(set(ref_types))
            if common_types:
                partial_score = len(common_types) / max(len(pred_types), len(ref_types))
                partial_matches += partial_score
    
    return (exact_matches + partial_matches) / len(predictions) if len(predictions) > 0 else 0.0
```

### 5.2 脉象特征识别准确率

评估模型识别脉象细节特征（如脉率、脉势、脉位等）的准确率。

```python
def calculate_pulse_feature_accuracy(predictions, references):
    """
    计算脉象特征识别准确率
    
    参数:
        predictions: 预测的脉象特征字典列表，如[{"rate": "迟", "strength": "有力", "rhythm": "规律"}, ...]
        references: 参考的脉象特征字典列表
    
    返回:
        包含各特征维度准确率的字典
    """
    if len(predictions) != len(references):
        raise ValueError("预测结果与参考标准数量不一致")
    
    # 初始化各特征维度统计
    feature_types = ["rate", "strength", "rhythm", "depth", "width"]
    feature_correct = {feature: 0 for feature in feature_types}
    feature_total = {feature: 0 for feature in feature_types}
    
    # 统计各特征维度的正确识别数
    for pred, ref in zip(predictions, references):
        for feature in feature_types:
            if feature in ref:
                feature_total[feature] += 1
                if feature in pred and pred[feature] == ref[feature]:
                    feature_correct[feature] += 1
    
    # 计算各特征维度的准确率
    accuracy = {}
    for feature in feature_types:
        if feature_total[feature] > 0:
            accuracy[feature] = feature_correct[feature] / feature_total[feature]
    
    # 计算总体准确率
    total_correct = sum(feature_correct.values())
    total_features = sum(feature_total.values())
    accuracy["overall"] = total_correct / total_features if total_features > 0 else 0.0
    
    return accuracy
```

### 5.3 脉象-证型关联准确率

评估模型将脉象特征与中医证型正确关联的能力。

```python
def calculate_pulse_syndrome_correlation(predicted_syndromes, reference_syndromes, pulse_features=None, pulse_syndrome_map=None):
    """
    计算脉象-证型关联准确率
    
    参数:
        predicted_syndromes: 预测的证型列表
        reference_syndromes: 参考的证型列表
        pulse_features: 脉象特征列表（可选）
        pulse_syndrome_map: 脉象到证型的映射字典（可选）
    
    返回:
        脉象-证型关联准确率
    """
    # 基本关联准确率计算（Jaccard相似度）
    pred_set = set(predicted_syndromes)
    ref_set = set(reference_syndromes)
    
    basic_correlation = len(pred_set.intersection(ref_set)) / len(pred_set.union(ref_set)) if pred_set.union(ref_set) else 0.0
    
    # 如果没有提供脉象特征或映射，返回基本关联准确率
    if not pulse_features or not pulse_syndrome_map:
        return basic_correlation
    
    # 根据脉象特征确定预期证型
    expected_syndromes = set()
    for feature in pulse_features:
        if feature in pulse_syndrome_map:
            expected_syndromes.update(pulse_syndrome_map[feature])
    
    if not expected_syndromes:
        return basic_correlation
    
    # 计算预测证型与脉象预期证型的关联度
    pulse_syndrome_correlation = len(pred_set.intersection(expected_syndromes)) / len(expected_syndromes)
    
    # 综合基本关联准确率和脉象-证型关联度
    return 0.6 * basic_correlation + 0.4 * pulse_syndrome_correlation
```

### 5.4 脉象数据解读完整性

评估模型对脉象数据的解读完整程度。

```python
def calculate_pulse_interpretation_completeness(interpretations, required_aspects=None):
    """
    计算脉象数据解读完整性
    
    参数:
        interpretations: 预测的脉象解读文本列表
        required_aspects: 必须包含的解读方面列表，默认为["脉象类型", "脉率", "脉势", "脉位", "脉体", "临床意义"]
    
    返回:
        解读完整性评分
    """
    if required_aspects is None:
        required_aspects = ["脉象类型", "脉率", "脉势", "脉位", "脉体", "临床意义"]
    
    completeness_scores = []
    
    for interpretation in interpretations:
        # 计算包含了多少必需解读方面
        covered_aspects = 0
        for aspect in required_aspects:
            if aspect in interpretation:
                covered_aspects += 1
        
        score = covered_aspects / len(required_aspects)
        completeness_scores.append(score)
    
    # 计算平均完整性得分
    return sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0
```

## 6. 四诊合参能力评测指标

四诊合参是中医诊断的核心方法，要求将望、闻、问、切四诊所获得的信息进行综合分析。评测模型的四诊合参能力主要考察其多模态信息融合和辨证论治能力。

### 6.1 多模态一致性评分

评估模型从不同诊法获取的信息之间的一致程度。

```python
def calculate_modal_consistency(modal_diagnoses):
    """
    计算多模态一致性评分
    
    参数:
        modal_diagnoses: 不同诊法得出的诊断结果字典，如{"望诊": ["脾虚湿盛"], "问诊": ["脾虚", "湿盛"], ...}
    
    返回:
        多模态一致性评分
    """
    # 提取所有诊法的诊断结果
    diagnoses_sets = []
    for modal, diagnoses in modal_diagnoses.items():
        if diagnoses:  # 非空列表
            diagnoses_sets.append(set(diagnoses))
    
    # 如果只有一种诊法有结果，返回1.0
    if len(diagnoses_sets) <= 1:
        return 1.0
    
    # 计算每对诊法之间的一致性
    pairwise_consistencies = []
    for i in range(len(diagnoses_sets)):
        for j in range(i + 1, len(diagnoses_sets)):
            set_i = diagnoses_sets[i]
            set_j = diagnoses_sets[j]
            
            # 使用Jaccard相似度计算一致性
            intersection = len(set_i.intersection(set_j))
            union = len(set_i.union(set_j))
            
            if union > 0:
                consistency = intersection / union
                pairwise_consistencies.append(consistency)
    
    # 返回平均一致性
    return sum(pairwise_consistencies) / len(pairwise_consistencies) if pairwise_consistencies else 0.0
```

### 6.2 多模态融合增益

评估多模态融合相比单一模态的诊断性能提升。

```python
def calculate_fusion_gain(single_modal_scores, multimodal_score):
    """
    计算多模态融合增益
    
    参数:
        single_modal_scores: 各单一模态的性能分数字典，如{"望诊": 0.7, "问诊": 0.8, ...}
        multimodal_score: 多模态融合的性能分数
    
    返回:
        多模态融合增益
    """
    # 计算单一模态的平均分数
    avg_single_score = sum(single_modal_scores.values()) / len(single_modal_scores)
    
    # 计算融合增益
    if avg_single_score > 0:
        gain = (multimodal_score - avg_single_score) / avg_single_score
    else:
        gain = 0.0 if multimodal_score == 0 else 1.0
    
    return max(0.0, gain)  # 确保增益不为负
```

### 6.3 辨证准确率

评估模型基于多模态信息进行辨证的准确率。

```python
def calculate_syndrome_differentiation_accuracy(predicted_syndromes, reference_syndromes):
    """
    计算辨证准确率
    
    参数:
        predicted_syndromes: 预测的证型列表
        reference_syndromes: 参考的证型列表
    
    返回:
        包含精确率、召回率和F1值的字典
    """
    # 将证型列表转换为集合
    pred_set = set(predicted_syndromes)
    ref_set = set(reference_syndromes)
    
    # 计算交集
    intersection = pred_set.intersection(ref_set)
    
    # 计算精确率、召回率和F1值
    precision = len(intersection) / len(pred_set) if pred_set else 0.0
    recall = len(intersection) / len(ref_set) if ref_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
```

### 6.4 四诊合参临床相关性

评估模型四诊合参结果与临床实际情况的相关程度。

```python
def calculate_clinical_relevance(diagnosis_result, clinical_features, syndrome_clinical_map=None):
    """
    计算四诊合参临床相关性
    
    参数:
        diagnosis_result: 预测的辨证结果列表
        clinical_features: 临床特征列表
        syndrome_clinical_map: 证型到临床特征的映射字典（可选）
    
    返回:
        临床相关性评分
    """
    # 如果没有提供映射，无法计算临床相关性
    if not syndrome_clinical_map:
        return 0.0
    
    # 计算预测证型应对应的临床特征
    expected_features = set()
    for syndrome in diagnosis_result:
        if syndrome in syndrome_clinical_map:
            expected_features.update(syndrome_clinical_map[syndrome])
    
    # 计算预期特征与实际临床特征的重叠
    clinical_set = set(clinical_features)
    
    # 避免除零错误
    if not expected_features or not clinical_set:
        return 0.0
    
    # 计算精确率和召回率
    precision = len(expected_features.intersection(clinical_set)) / len(expected_features)
    recall = len(expected_features.intersection(clinical_set)) / len(clinical_set)
    
    # 计算F1分数作为临床相关性评分
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1
```

## 7. 方剂推荐能力评测指标

方剂推荐是中医诊疗的重要环节，评测模型的方剂推荐能力主要考察其组方合理性、剂量准确性和理论依据充分性。

### 7.1 方剂组成准确率

评估模型推荐的方剂组成与标准方剂的匹配程度。

```python
def calculate_prescription_accuracy(predicted_herbs, reference_herbs):
    """
    计算方剂组成准确率
    
    参数:
        predicted_herbs: 预测的药物列表
        reference_herbs: 参考的药物列表
    
    返回:
        包含精确率、召回率和F1值的字典
    """
    # 转换为集合
    pred_set = set(predicted_herbs)
    ref_set = set(reference_herbs)
    
    # 计算交集
    intersection = pred_set.intersection(ref_set)
    
    # 计算精确率、召回率和F1值
    precision = len(intersection) / len(pred_set) if pred_set else 0.0
    recall = len(intersection) / len(ref_set) if ref_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
```

### 7.2 剂量合理性评分

评估模型推荐的药物剂量合理性。

```python
def calculate_dosage_rationality(predicted_dosages, reference_dosages, tolerance=0.2):
    """
    计算剂量合理性评分
    
    参数:
        predicted_dosages: 预测的药物剂量字典，如{"黄芪": 30, "党参": 15}
        reference_dosages: 参考的药物剂量字典
        tolerance: 允许的剂量误差比例，默认为0.2（20%）
    
    返回:
        剂量合理性评分
    """
    if not predicted_dosages or not reference_dosages:
        return 0.0
    
    # 只考虑两者都有的药物
    common_herbs = set(predicted_dosages.keys()).intersection(set(reference_dosages.keys()))
    
    if not common_herbs:
        return 0.0
    
    # 计算每个药物的剂量合理性
    herb_scores = []
    for herb in common_herbs:
        pred_dose = predicted_dosages[herb]
        ref_dose = reference_dosages[herb]
        
        # 计算相对误差
        relative_error = abs(pred_dose - ref_dose) / ref_dose
        
        # 在容差范围内得满分，超出则按比例减分
        if relative_error <= tolerance:
            herb_scores.append(1.0)
        else:
            herb_scores.append(max(0.0, 1.0 - (relative_error - tolerance) / (1.0 - tolerance)))
    
    # 返回平均剂量合理性评分
    return sum(herb_scores) / len(herb_scores)
```

### 7.3 方剂解释合理性评分

评估模型对推荐方剂的解释是否合理、专业和符合中医理论。

```python
def calculate_explanation_rationality(explanation, reference_keywords, reference_reasonings):
    """
    计算方剂解释合理性评分
    
    参数:
        explanation: 方剂解释文本
        reference_keywords: 参考关键词列表，如["清热", "解毒", "凉血"]
        reference_reasonings: 参考理论依据列表，如["黄连苦寒，清热燥湿", "黄芩苦寒，泻火解毒"]
    
    返回:
        包含关键词覆盖率、理论依据覆盖率和总体合理性的字典
    """
    # 计算关键词覆盖率
    keyword_hits = sum(1 for keyword in reference_keywords if keyword in explanation)
    keyword_coverage = keyword_hits / len(reference_keywords) if reference_keywords else 0.0
    
    # 计算理论依据覆盖率（基于简单文本匹配）
    reasoning_hits = 0
    for reasoning in reference_reasonings:
        if any(part in explanation for part in reasoning.split('，')):
            reasoning_hits += 1
    
    reasoning_coverage = reasoning_hits / len(reference_reasonings) if reference_reasonings else 0.0
    
    # 计算总体合理性评分（加权平均）
    overall_rationality = 0.4 * keyword_coverage + 0.6 * reasoning_coverage
    
    return {
        "keyword_coverage": keyword_coverage,
        "reasoning_coverage": reasoning_coverage,
        "overall_rationality": overall_rationality
    }
```

### 7.4 方剂-证型匹配度

评估推荐的方剂是否与诊断的证型匹配。

```python
def calculate_prescription_syndrome_match(prescription, syndromes, prescription_syndrome_map=None):
    """
    计算方剂-证型匹配度
    
    参数:
        prescription: 方剂名或药物列表
        syndromes: 证型列表
        prescription_syndrome_map: 方剂到适用证型的映射字典（可选）
    
    返回:
        方剂-证型匹配度
    """
    # 如果没有提供映射，无法计算匹配度
    if not prescription_syndrome_map:
        return 0.0
    
    # 如果方剂是药物列表，尝试找到对应的方剂名
    if isinstance(prescription, list):
        # 实际应用中需要更复杂的算法，这里简化处理
        prescription_name = "未知方剂"
    else:
        prescription_name = prescription
    
    # 获取方剂对应的适用证型
    if prescription_name in prescription_syndrome_map:
        suitable_syndromes = set(prescription_syndrome_map[prescription_name])
    else:
        return 0.0
    
    # 计算证型匹配度（交集/并集）
    syndrome_set = set(syndromes)
    
    # 避免除零错误
    if not suitable_syndromes or not syndrome_set:
        return 0.0
    
    # 计算Jaccard相似度
    match_score = len(suitable_syndromes.intersection(syndrome_set)) / len(suitable_syndromes.union(syndrome_set))
    
    return match_score
```

## 8. 综合评价指标

综合评价指标旨在对模型的整体性能进行全面评估，综合考虑通用能力和中医专业能力。

### 8.1 中医专业性综合评分

评估模型在中医专业领域的整体表现。

```python
def calculate_tcm_professionalism_score(scores, weights=None):
    """
    计算中医专业性综合评分
    
    参数:
        scores: 各维度评分字典，如{"望诊": 0.8, "闻诊": 0.7, "问诊": 0.85, "切诊": 0.75, "四诊合参": 0.8, "方剂推荐": 0.7}
        weights: 各维度权重字典，若为None则使用默认权重
    
    返回:
        中医专业性综合评分
    """
    if weights is None:
        # 默认权重
        weights = {
            "望诊": 0.15,
            "闻诊": 0.10,
            "问诊": 0.20,
            "切诊": 0.15,
            "四诊合参": 0.20,
            "方剂推荐": 0.20
        }
    
    # 确保所有维度都有评分和权重
    valid_dimensions = set(scores.keys()).intersection(set(weights.keys()))
    
    if not valid_dimensions:
        return 0.0
    
    # 重新归一化权重
    total_weight = sum(weights[dim] for dim in valid_dimensions)
    normalized_weights = {dim: weights[dim] / total_weight for dim in valid_dimensions}
    
    # 计算加权平均分
    weighted_score = sum(scores[dim] * normalized_weights[dim] for dim in valid_dimensions)
    
    return weighted_score
```

### 8.2 全模态综合评分

评估模型的整体表现，综合考虑通用能力和中医专业能力。

```python
def calculate_comprehensive_score(general_score, tcm_score, general_weight=0.3, tcm_weight=0.7):
    """
    计算全模态综合评分
    
    参数:
        general_score: 通用能力评分
        tcm_score: 中医专业能力评分
        general_weight: 通用能力权重，默认0.3
        tcm_weight: 中医专业能力权重，默认0.7
    
    返回:
        全模态综合评分
    """
    # 确保权重和为1
    total_weight = general_weight + tcm_weight
    normalized_general_weight = general_weight / total_weight
    normalized_tcm_weight = tcm_weight / total_weight
    
    # 计算加权平均分
    comprehensive_score = general_score * normalized_general_weight + tcm_score * normalized_tcm_weight
    
    return comprehensive_score
```

## 9. 人工评价指标

除了自动评估指标外，人工评价在中医专业评测中也非常重要，特别是针对模型输出的专业性、合理性等难以自动量化的方面。

### 9.1 专家评分标准

建议专家评分使用以下5分制标准：

| 评分维度 | 1分 | 2分 | 3分 | 4分 | 5分 |
|---------|-----|-----|-----|-----|-----|
| 专业术语应用 | 术语混乱，明显错误 | 术语使用不准确 | 术语基本准确 | 术语使用准确 | 术语使用准确且专业 |
| 理论符合度 | 与中医理论相悖 | 部分不符合中医理论 | 基本符合中医理论 | 较好地符合中医理论 | 完全符合中医理论 |
| 诊断准确性 | 诊断完全错误 | 诊断大部分错误 | 诊断部分正确 | 诊断大部分正确 | 诊断完全正确 |
| 处方合理性 | 处方不合理 | 处方合理性较低 | 处方基本合理 | 处方合理 | 处方非常合理 |
| 临床实用性 | 无临床参考价值 | 临床参考价值低 | 有一定临床参考价值 | 临床参考价值较高 | 临床参考价值极高 |

### 9.2 人机一致性评价

评估模型与人类专家诊断结果的一致程度。

```python
def calculate_human_ai_agreement(human_diagnoses, ai_diagnoses):
    """
    计算人机一致性
    
    参数:
        human_diagnoses: 人类专家诊断结果列表
        ai_diagnoses: AI模型诊断结果列表
    
    返回:
        一致性评分
    """
    if len(human_diagnoses) != len(ai_diagnoses):
        raise ValueError("人类诊断和AI诊断数量不一致")
    
    agreement_scores = []
    
    for human_diag, ai_diag in zip(human_diagnoses, ai_diagnoses):
        # 将诊断转换为集合
        human_set = set(human_diag) if isinstance(human_diag, list) else {human_diag}
        ai_set = set(ai_diag) if isinstance(ai_diag, list) else {ai_diag}
        
        # 计算Jaccard相似度
        if not human_set or not ai_set:
            agreement_scores.append(0.0)
            continue
        
        jaccard = len(human_set.intersection(ai_set)) / len(human_set.union(ai_set))
        agreement_scores.append(jaccard)
    
    # 返回平均一致性评分
    return sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0.0
```

## 10. 总结

本文档详细介绍了中医药专业评测指标体系，涵盖了望诊、闻诊、问诊、切诊等四诊能力，以及四诊合参和方剂推荐能力的评估指标。这些指标共同构成了一个全面的中医药多诊合参多模态大模型评测体系。

在实际应用中，可以根据具体的评测目标和重点，灵活选择和组合这些指标，以便全面评估模型的中医药专业能力。同时，建议将自动评估指标与人工评价相结合，以获得更全面、客观的评测结果。

后续可以针对不同类型的模型和应用场景，进一步优化和扩展这套评测指标体系，提高其适用性和精确性。例如，可以增加针对特定中医疾病的专项评测指标，或者开发更精细的方剂评价方法。

总之，一个科学、全面、客观的评测体系是指导中医药多模态大模型发展的重要工具，也是保障模型在临床应用中安全性和有效性的基础。 
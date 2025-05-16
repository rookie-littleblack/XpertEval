---
layout: default
title: 中医专业评测（上）
---

# 中医药领域特定评测方法

## 1. 中医药专业评测概述

### 1.1 中医诊断的特殊性

中医诊断体系以"四诊合参"为核心，这一特点决定了中医药多模态大模型评测的独特性：

1. **多模态协同**：中医诊断依赖于视觉（望诊）、听觉（闻诊）、语言（问诊）和触觉（切诊）的综合信息。
2. **整体性观察**：强调从整体上观察患者症状，而非孤立分析单个征象。
3. **辨证论治**：通过综合分析症候，确定病机、病位、病性，进而制定治疗方案。
4. **个体化诊疗**：对同一疾病，不同患者可能采用不同治疗方案。
5. **经验性知识**：中医知识体系包含大量经验性、描述性内容，难以完全量化。

### 1.2 中医药模型评测目标

中医药多模态大模型评测的核心目标包括：

1. **诊断准确性**：评估模型对患者症状的识别和辨证的准确程度。
2. **方案合理性**：评估模型提供的治疗方案（方剂推荐）是否合理有效。
3. **理论符合性**：评估模型的诊断和治疗是否符合中医理论体系。
4. **解释合理性**：评估模型对诊断和治疗推荐的解释是否清晰合理。
5. **个体化程度**：评估模型针对不同患者情况提供个性化建议的能力。

### 1.3 评测框架概述

中医药专业评测框架采用分层结构：

```
中医药专业评测
├── 单模态能力评测
│   ├── 望诊能力评测
│   │   ├── 面诊图像分析
│   │   └── 舌诊图像分析
│   ├── 闻诊能力评测
│   │   ├── 呼吸音分析
│   │   └── 咳嗽音分析
│   ├── 问诊能力评测
│   │   ├── 症状理解
│   │   └── 病史收集
│   └── 切诊能力评测
│       └── 脉象数据分析
├── 多模态融合能力评测
│   ├── 双模态融合
│   │   ├── 望诊+问诊
│   │   ├── 闻诊+问诊
│   │   └── 切诊+问诊
│   └── 四诊合参
│       ├── 证候辨识
│       └── 疾病诊断
└── 治疗方案能力评测
    ├── 方剂选择
    ├── 药物组成
    ├── 剂量确定
    └── 用药解释
```

## 2. 单模态专业能力评测

### 2.1 望诊能力评测

#### 2.1.1 面诊图像分析评测

**评测内容**：评估模型识别和分析患者面部特征的能力，包括面色、神态等。

**数据集构建**：
- 特征样本：不同面色（萎黄、潮红、青白等）的标准图像
- 综合样本：包含多种面部特征的真实患者照片
- 难度梯度：从典型特征到不典型特征的渐进样本

**评价指标**：
- 面色识别准确率：正确识别面色特征的比例
- 特征描述完整性：描述面部特征的完整程度
- 辨证关联度：将面部特征与证候关联的准确度

**实现示例**：

```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import json

class FaceDiagnosisEvaluator:
    def __init__(self, model_path, device="cuda"):
        """初始化面诊评测器"""
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.device = device
        
        # 加载标准面色特征词典
        self.face_features = {
            "面色萎黄": ["萎黄", "萎黄色", "面黄", "面色发黄", "黄色"],
            "面色潮红": ["潮红", "面红", "红色", "面色发红", "赤红"],
            "面色青白": ["青白", "苍白", "面白", "面色苍白", "白色"],
            "面色晦暗": ["晦暗", "黯", "黧黑", "面色发暗", "暗色"],
            "面色青灰": ["青灰", "灰色", "面色灰暗", "灰白色"]
        }
    
    def evaluate_face_diagnosis(self, image_path, prompt, reference_features):
        """
        评估面诊能力
        
        参数:
            image_path: 面部图像路径
            prompt: 提示文本
            reference_features: 参考特征列表
        
        返回:
            评测结果字典
        """
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 处理输入
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        # 生成回答
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                num_return_sequences=1
            )
        
        # 解码输出
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        
        # 分析结果
        detected_features = []
        for category, feature_words in self.face_features.items():
            for word in feature_words:
                if word in response:
                    detected_features.append(category)
                    break
        
        # 计算指标
        detected_set = set(detected_features)
        reference_set = set(reference_features)
        
        # 准确率：正确检测的特征数 / 参考特征总数
        accuracy = len(detected_set.intersection(reference_set)) / len(reference_set) if reference_set else 0
        
        # 完整性：检测到的参考特征数 / 参考特征总数
        completeness = len(detected_set.intersection(reference_set)) / len(reference_set) if reference_set else 0
        
        # 精确度：正确检测的特征数 / 检测到的特征总数
        precision = len(detected_set.intersection(reference_set)) / len(detected_set) if detected_set else 0
        
        return {
            "response": response,
            "detected_features": list(detected_set),
            "reference_features": list(reference_set),
            "accuracy": accuracy,
            "completeness": completeness,
            "precision": precision,
            "f1_score": 2 * precision * completeness / (precision + completeness) if (precision + completeness) > 0 else 0
        }
```

#### 2.1.2 舌诊图像分析评测

**评测内容**：评估模型识别和分析舌象特征的能力，包括舌质、舌苔等。

**数据集构建**：
- 特征样本：不同舌象（淡白舌、淡红舌、红舌、紫舌等）的标准图像
- 舌苔样本：不同舌苔（薄白苔、黄腻苔等）的标准图像
- 综合样本：结合舌质、舌苔等多种特征的真实舌象图像

**评价指标**：
- 舌质识别准确率：正确识别舌质特征的比例
- 舌苔识别准确率：正确识别舌苔特征的比例
- 辨证关联度：将舌象特征与证候关联的准确度
- 描述详细度：描述舌象特征的详细程度

**实现示例**：

```python
class TongueDiagnosisEvaluator:
    def __init__(self, model_path, device="cuda"):
        """初始化舌诊评测器"""
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.device = device
        
        # 舌质特征词典
        self.tongue_body_features = {
            "淡白舌": ["淡白", "舌淡白", "舌体淡白", "舌色淡白"],
            "淡红舌": ["淡红", "舌淡红", "舌体淡红", "舌色淡红", "正常舌色"],
            "红舌": ["红", "舌红", "舌体红", "舌色红"],
            "绛舌": ["绛", "舌绛", "舌色绛", "深红", "舌深红"],
            "紫舌": ["紫", "舌紫", "舌体紫", "舌色紫"],
            "青舌": ["青", "舌青", "舌体青", "舌色青"]
        }
        
        # 舌苔特征词典
        self.tongue_coating_features = {
            "无苔": ["无苔", "少苔", "舌苔少", "舌上无苔"],
            "薄白苔": ["薄白", "白苔", "薄白苔", "白色薄苔"],
            "厚白苔": ["厚白", "厚白苔", "白色厚苔"],
            "黄苔": ["黄苔", "舌苔黄", "淡黄苔"],
            "厚黄苔": ["厚黄", "厚黄苔", "黄色厚苔"],
            "腻苔": ["腻", "苔腻", "腻苔", "舌苔腻"],
            "黑苔": ["黑苔", "舌苔黑", "黑色苔"]
        }
    
    def evaluate_tongue_diagnosis(self, image_path, prompt, reference_body, reference_coating):
        """
        评估舌诊能力
        
        参数:
            image_path: 舌象图像路径
            prompt: 提示文本
            reference_body: 参考舌质特征
            reference_coating: 参考舌苔特征
        
        返回:
            评测结果字典
        """
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 处理输入
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        # 生成回答
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                num_return_sequences=1
            )
        
        # 解码输出
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        
        # 分析舌质结果
        detected_body = None
        for body_type, features in self.tongue_body_features.items():
            for feature in features:
                if feature in response:
                    detected_body = body_type
                    break
            if detected_body:
                break
        
        # 分析舌苔结果
        detected_coating = None
        for coating_type, features in self.tongue_coating_features.items():
            for feature in features:
                if feature in response:
                    detected_coating = coating_type
                    break
            if detected_coating:
                break
        
        # 计算指标
        body_correct = detected_body == reference_body
        coating_correct = detected_coating == reference_coating
        overall_accuracy = (body_correct + coating_correct) / 2
        
        return {
            "response": response,
            "detected_body": detected_body,
            "detected_coating": detected_coating,
            "reference_body": reference_body,
            "reference_coating": reference_coating,
            "body_accuracy": 1.0 if body_correct else 0.0,
            "coating_accuracy": 1.0 if coating_correct else 0.0,
            "overall_accuracy": overall_accuracy
        }
```

### 2.2 闻诊能力评测

#### 2.2.1 呼吸音分析评测

**评测内容**：评估模型识别和分析呼吸音特征的能力。

**数据集构建**：
- 正常呼吸音样本
- 不同病理性呼吸音样本（喘息音、罗音等）
- 不同强度呼吸音样本

**评价指标**：
- 呼吸音类型识别准确率
- 呼吸特征描述准确性
- 辨证关联度

**实现示例**：

```python
import librosa

class BreathingSoundEvaluator:
    def __init__(self, model_path, device="cuda"):
        """初始化呼吸音评测器"""
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.device = device
        
        # 呼吸音特征词典
        self.breathing_features = {
            "正常呼吸音": ["正常", "均匀", "规律", "清晰"],
            "喘息音": ["喘息", "哮鸣", "哮喘", "气喘"],
            "罗音": ["罗音", "痰鸣", "水泡音", "湿罗音"],
            "干啰音": ["干罗音", "干啰音", "哮鸣音"],
            "哮鸣音": ["哮鸣", "喘鸣", "哨音"],
            "气促": ["气促", "呼吸短促", "呼吸急促", "气短"]
        }
    
    def evaluate_breathing_sound(self, audio_path, prompt, reference_feature, sample_rate=16000):
        """
        评估呼吸音分析能力
        
        参数:
            audio_path: 音频文件路径
            prompt: 提示文本
            reference_feature: 参考呼吸音特征
            sample_rate: 采样率
        
        返回:
            评测结果字典
        """
        # 加载音频
        audio, _ = librosa.load(audio_path, sr=sample_rate)
        
        # 处理输入
        inputs = self.processor(text=prompt, audios=audio, sampling_rate=sample_rate, return_tensors="pt").to(self.device)
        
        # 生成回答
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                num_return_sequences=1
            )
        
        # 解码输出
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        
        # 分析结果
        detected_feature = None
        for feature_type, keywords in self.breathing_features.items():
            for keyword in keywords:
                if keyword in response:
                    detected_feature = feature_type
                    break
            if detected_feature:
                break
        
        # 计算指标
        accuracy = 1.0 if detected_feature == reference_feature else 0.0
        
        # 分析辨证关联度
        syndrome_keywords = {
            "肺热": ["肺热", "热证", "热症", "肺部有热"],
            "痰湿": ["痰湿", "湿痰", "痰浊", "痰多"],
            "肺气虚": ["肺气虚", "气虚", "肺虚", "肺气不足"],
            "寒饮": ["寒饮", "饮停", "寒痰", "寒湿"]
        }
        
        detected_syndromes = []
        for syndrome, keywords in syndrome_keywords.items():
            for keyword in keywords:
                if keyword in response:
                    detected_syndromes.append(syndrome)
                    break
        
        return {
            "response": response,
            "detected_feature": detected_feature,
            "reference_feature": reference_feature,
            "accuracy": accuracy,
            "detected_syndromes": detected_syndromes
        }
```

#### 2.2.2 咳嗽音分析评测

**评测内容**：评估模型识别和分析咳嗽音特征的能力。

**数据集构建**：
- 不同类型咳嗽音样本（干咳、湿咳等）
- 不同强度咳嗽音样本
- 不同病理背景的咳嗽音样本

**评价指标**：
- 咳嗽类型识别准确率
- 特征描述准确性
- 辨证关联度 
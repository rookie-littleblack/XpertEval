---
layout: default
title: 中医专业评测（下）
---

# 中医药专业评测方法（续）

## 3. 问诊能力评测

### 3.1 症状理解能力评测

**评测内容**：评估模型对患者描述症状的理解和分析能力。

**数据集构建**：
- 常见症状描述样本（如头痛、乏力、失眠等）
- 复杂症状组合样本
- 包含方言、口语表达的症状描述

**评价指标**：
- 症状识别准确率：正确识别患者描述中包含的症状
- 症状分类准确率：将症状正确归类（如寒热虚实等）
- 症状关联分析能力：识别不同症状间的关联

**实现示例**：

```python
class SymptomUnderstandingEvaluator:
    def __init__(self, model_path, device="cuda"):
        """初始化症状理解评测器"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.device = device
        
        # 常见症状词典
        self.symptom_dict = {
            "头痛": ["头痛", "头疼", "脑袋痛", "头部疼痛"],
            "头晕": ["头晕", "眩晕", "晕眩", "头昏"],
            "乏力": ["乏力", "无力", "疲乏", "疲劳", "疲倦"],
            "失眠": ["失眠", "不易入睡", "睡眠困难", "夜不能寐"],
            "心悸": ["心悸", "心慌", "心跳加快"],
            "胸闷": ["胸闷", "胸部闷痛", "闷痛"],
            "腹痛": ["腹痛", "肚子痛", "腹部疼痛"],
            "腹泻": ["腹泻", "拉肚子", "大便稀溏", "大便次数多"],
            "便秘": ["便秘", "大便干结", "排便困难"],
            "口干": ["口干", "口燥", "口渴", "口干舌燥"],
            "汗多": ["汗多", "自汗", "盗汗", "多汗"],
            "怕冷": ["怕冷", "畏寒", "恶寒", "寒战"],
            "发热": ["发热", "发烧", "身热", "壮热"]
        }
        
        # 症状分类词典
        self.symptom_categories = {
            "寒证": ["怕冷", "畏寒", "喜热", "肢冷", "口不渴", "清稀痰", "清澈尿", "面色苍白"],
            "热证": ["发热", "壮热", "口渴", "面红", "黄痰", "黄尿", "便秘", "烦躁"],
            "虚证": ["乏力", "疲倦", "气短", "懒言", "自汗", "面色淡", "舌淡"],
            "实证": ["胀满", "疼痛", "烦躁", "口苦", "大便秘结", "小便短赤"],
            "气虚": ["乏力", "气短", "懒言", "自汗", "舌淡胖"],
            "血虚": ["面色萎黄", "唇甲苍白", "头晕", "心悸", "失眠", "舌淡"],
            "阴虚": ["手足心热", "盗汗", "口干", "五心烦热", "舌红少苔"],
            "阳虚": ["畏寒", "肢冷", "面色苍白", "舌淡胖嫩", "小便清长"]
        }
    
    def evaluate_symptom_understanding(self, description, reference_symptoms, reference_categories):
        """
        评估症状理解能力
        
        参数:
            description: 患者描述文本
            reference_symptoms: 参考症状列表
            reference_categories: 参考症状分类列表
        
        返回:
            评测结果字典
        """
        # 构建提示
        prompt = f"患者描述：{description}\n\n请识别上述描述中的主要症状，并进行中医辨证分析。"
        
        # 获取模型输出
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=512,
                num_return_sequences=1
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        # 症状识别评估
        detected_symptoms = []
        for symptom, keywords in self.symptom_dict.items():
            for keyword in keywords:
                if keyword in description and symptom not in detected_symptoms:
                    detected_symptoms.append(symptom)
        
        # 症状分类评估
        detected_categories = []
        for category, symptoms in self.symptom_categories.items():
            for symptom in symptoms:
                if symptom in response and category not in detected_categories:
                    detected_categories.append(category)
        
        # 计算指标
        symptom_correct = sum(1 for s in detected_symptoms if s in reference_symptoms)
        symptom_precision = symptom_correct / len(detected_symptoms) if detected_symptoms else 0
        symptom_recall = symptom_correct / len(reference_symptoms) if reference_symptoms else 0
        symptom_f1 = 2 * symptom_precision * symptom_recall / (symptom_precision + symptom_recall) if (symptom_precision + symptom_recall) > 0 else 0
        
        category_correct = sum(1 for c in detected_categories if c in reference_categories)
        category_precision = category_correct / len(detected_categories) if detected_categories else 0
        category_recall = category_correct / len(reference_categories) if reference_categories else 0
        category_f1 = 2 * category_precision * category_recall / (category_precision + category_recall) if (category_precision + category_recall) > 0 else 0
        
        return {
            "response": response,
            "detected_symptoms": detected_symptoms,
            "reference_symptoms": reference_symptoms,
            "symptom_precision": symptom_precision,
            "symptom_recall": symptom_recall,
            "symptom_f1": symptom_f1,
            "detected_categories": detected_categories,
            "reference_categories": reference_categories,
            "category_precision": category_precision,
            "category_recall": category_recall,
            "category_f1": category_f1
        }
```

### 3.2 病史收集能力评测

**评测内容**：评估模型收集和整理患者病史的能力。

**数据集构建**：
- 模拟患者对话样本
- 包含关键病史信息的长文本样本
- 不完整信息样本，需要模型主动提问补充

**评价指标**：
- 信息提取完整性：提取关键病史信息的完整程度
- 提问针对性：针对缺失信息提出恰当问题的能力
- 病史整理结构化：将散乱信息整理成结构化病史的能力

**实现示例**：

```python
class MedicalHistoryEvaluator:
    def __init__(self, model_path, device="cuda"):
        """初始化病史收集评测器"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.device = device
        
        # 病史关键信息类别
        self.history_categories = [
            "主诉", "现病史", "既往史", "个人史", "家族史", 
            "过敏史", "用药史", "月经史(女性)"
        ]
    
    def evaluate_history_collection(self, conversation, reference_info):
        """
        评估病史收集能力
        
        参数:
            conversation: 医患对话文本
            reference_info: 参考病史信息字典，格式为 {类别: 内容}
        
        返回:
            评测结果字典
        """
        # 构建提示
        prompt = f"以下是医生与患者的对话内容：\n\n{conversation}\n\n请根据对话内容，整理患者的完整病史。"
        
        # 获取模型输出
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=1024,
                num_return_sequences=1
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        # 评估信息提取完整性
        extracted_info = {}
        for category in self.history_categories:
            if category in response:
                # 尝试提取该类别的内容
                try:
                    start_idx = response.find(category)
                    next_category_indices = [response.find(cat, start_idx + len(category)) for cat in self.history_categories if cat != category and cat in response[start_idx + len(category):]]
                    next_category_indices = [idx for idx in next_category_indices if idx > 0]
                    
                    if next_category_indices:
                        end_idx = min(next_category_indices)
                        content = response[start_idx + len(category):end_idx].strip()
                    else:
                        content = response[start_idx + len(category):].strip()
                    
                    if content and len(content) > 2:  # 确保内容不是空的或太短
                        extracted_info[category] = content
                except:
                    continue
        
        # 计算完整性分数
        total_categories = len(reference_info)
        found_categories = sum(1 for category in reference_info if category in extracted_info)
        completeness = found_categories / total_categories if total_categories > 0 else 0
        
        # 计算准确性分数 (简化版)
        accuracy_scores = []
        for category, ref_content in reference_info.items():
            if category in extracted_info:
                # 这里使用一个简单的字符重叠率作为内容匹配度度量
                extracted_content = extracted_info[category]
                content_match = len(set(extracted_content) & set(ref_content)) / len(set(ref_content)) if ref_content else 0
                accuracy_scores.append(content_match)
        
        accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0
        
        # 评估结构化能力
        structure_score = sum(1 for category in self.history_categories if category in response) / len(self.history_categories)
        
        return {
            "response": response,
            "extracted_info": extracted_info,
            "reference_info": reference_info,
            "completeness": completeness,
            "accuracy": accuracy,
            "structure_score": structure_score,
            "overall_score": (completeness + accuracy + structure_score) / 3
        }
    
    def evaluate_follow_up_questions(self, conversation, missing_info):
        """
        评估提出后续问题的能力
        
        参数:
            conversation: 当前对话文本
            missing_info: 缺失的信息类别列表
        
        返回:
            评测结果字典
        """
        # 构建提示
        prompt = f"以下是医生与患者的对话内容：\n\n{conversation}\n\n作为医生，请提出3个重要的后续问题，以收集更完整的病史信息。"
        
        # 获取模型输出
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=512,
                num_return_sequences=1
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        # 分析问题相关性
        relevance_scores = []
        for info in missing_info:
            # 检查生成的问题是否针对缺失信息
            relevance = 0
            if info.lower() in response.lower():
                relevance = 1.0
            elif any(related_term in response.lower() for related_term in self._get_related_terms(info)):
                relevance = 0.5
            relevance_scores.append(relevance)
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        
        # 计算问题数量 (简单方法：查找问号或数字+问题模式)
        import re
        question_patterns = [r'\d+[.、)\]：:]\s*[^？?。.!！]+[？?]', r'[^？?。.!！]+[？?]']
        questions = []
        
        for pattern in question_patterns:
            questions.extend(re.findall(pattern, response))
        
        # 去重
        questions = list(set(questions))
        
        return {
            "response": response,
            "questions": questions,
            "question_count": len(questions),
            "missing_info": missing_info,
            "relevance_scores": relevance_scores,
            "avg_relevance": avg_relevance
        }
    
    def _get_related_terms(self, category):
        """获取类别相关术语"""
        related_terms = {
            "主诉": ["主要症状", "主要不适", "就诊原因", "不舒服", "困扰"],
            "现病史": ["发病", "起病", "症状发展", "治疗经过", "用药情况"],
            "既往史": ["慢性病", "手术史", "住院史", "疾病史", "曾经患过"],
            "个人史": ["生活习惯", "饮食", "睡眠", "吸烟", "饮酒", "作息"],
            "家族史": ["父母", "兄弟姐妹", "家人", "遗传病", "家族疾病"],
            "过敏史": ["过敏", "不良反应", "药物过敏", "食物过敏"],
            "用药史": ["服用", "药物", "西药", "中药", "保健品"],
            "月经史": ["月经", "经期", "行经", "周期", "末次月经"]
        }
        
        return related_terms.get(category, [])
```

### 3.3 交互式问诊能力评测

**评测内容**：评估模型在多轮对话中进行中医问诊的能力。

**数据集构建**：
- 多轮医患对话样本
- 不同症状组合的问诊场景
- 包含患者反问、模糊描述的对话样本

**评价指标**：
- 问诊完整性：收集必要信息的完整程度
- 问诊效率：用最少轮次获取关键信息的能力
- 交互自然度：对话流畅度和自然程度
- 专业准确性：提问和解释的专业准确性

## 4. 切诊能力评测

### 4.1 脉象数据分析评测

**评测内容**：评估模型分析脉象数据的能力。

**数据集构建**：
- 不同脉象类型的数字化数据（可以是压力传感器数据或转换为图像的数据）
- 包含单一脉象和混合脉象的样本
- 不同程度强度的脉象样本

**评价指标**：
- 脉象类型识别准确率：正确识别脉象类型的比例
- 脉象特征描述完整性：描述脉象特征的完整程度
- 辨证关联度：将脉象特征与证候关联的准确度

**实现示例**：

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

class PulseDiagnosisEvaluator:
    def __init__(self, model_path, device="cuda"):
        """初始化脉诊评测器"""
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.device = device
        
        # 脉象类型词典
        self.pulse_types = {
            "浮脉": ["浮脉", "脉浮", "浮", "浮取"],
            "沉脉": ["沉脉", "脉沉", "沉", "沉取"],
            "迟脉": ["迟脉", "脉迟", "迟", "缓"],
            "数脉": ["数脉", "脉数", "数"],
            "虚脉": ["虚脉", "脉虚", "虚"],
            "实脉": ["实脉", "脉实", "实"],
            "滑脉": ["滑脉", "脉滑", "滑"],
            "涩脉": ["涩脉", "脉涩", "涩"],
            "弦脉": ["弦脉", "脉弦", "弦"],
            "洪脉": ["洪脉", "脉洪", "洪"]
        }
        
        # 脉象与证候关联词典
        self.pulse_syndrome_mapping = {
            "浮脉": ["表证", "风证", "初病"],
            "沉脉": ["里证", "沉降", "久病"],
            "迟脉": ["寒证", "阳虚", "气血运行缓慢"],
            "数脉": ["热证", "阴虚", "气血运行加快"],
            "虚脉": ["气血不足", "正气虚弱"],
            "实脉": ["邪气实盛", "气血充盈"],
            "滑脉": ["痰饮", "食积", "湿热", "妊娠"],
            "涩脉": ["血虚", "津液不足", "气血运行不畅"],
            "弦脉": ["肝胆病", "痛证", "寒证", "情志不畅"],
            "洪脉": ["热证", "气血亢盛"]
        }
    
    def pulse_data_to_image(self, pulse_data, title="脉象图"):
        """将脉象数据转换为图像"""
        plt.figure(figsize=(10, 5))
        plt.plot(pulse_data)
        plt.title(title)
        plt.xlabel('时间')
        plt.ylabel('压力')
        plt.grid(True)
        
        # 将图形转换为PIL图像
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        
        return img
    
    def evaluate_pulse_diagnosis(self, pulse_data, prompt, reference_types):
        """
        评估脉诊能力
        
        参数:
            pulse_data: 脉象数据数组
            prompt: 提示文本
            reference_types: 参考脉象类型列表
        
        返回:
            评测结果字典
        """
        # 将脉象数据转换为图像
        pulse_image = self.pulse_data_to_image(pulse_data)
        
        # 处理输入
        inputs = self.processor(text=prompt, images=pulse_image, return_tensors="pt").to(self.device)
        
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
        detected_types = []
        for pulse_type, keywords in self.pulse_types.items():
            for keyword in keywords:
                if keyword in response and pulse_type not in detected_types:
                    detected_types.append(pulse_type)
        
        # 计算指标
        detected_set = set(detected_types)
        reference_set = set(reference_types)
        
        # 准确率：正确检测的脉象类型数 / 参考脉象类型总数
        accuracy = len(detected_set.intersection(reference_set)) / len(reference_set) if reference_set else 0
        
        # 完整性：检测到的参考脉象类型数 / 参考脉象类型总数
        completeness = len(detected_set.intersection(reference_set)) / len(reference_set) if reference_set else 0
        
        # 精确度：正确检测的脉象类型数 / 检测到的脉象类型总数
        precision = len(detected_set.intersection(reference_set)) / len(detected_set) if detected_set else 0
        
        # 分析辨证关联度
        detected_syndromes = []
        for pulse_type in detected_types:
            if pulse_type in self.pulse_syndrome_mapping:
                detected_syndromes.extend(self.pulse_syndrome_mapping[pulse_type])
        
        # 去重
        detected_syndromes = list(set(detected_syndromes))
        
        return {
            "response": response,
            "detected_types": list(detected_set),
            "reference_types": list(reference_set),
            "accuracy": accuracy,
            "completeness": completeness,
            "precision": precision,
            "f1_score": 2 * precision * completeness / (precision + completeness) if (precision + completeness) > 0 else 0,
            "detected_syndromes": detected_syndromes
        }
```

## 5. 多模态融合能力评测

### 5.1 双模态融合能力评测

**评测内容**：评估模型结合两种模态信息进行诊断的能力。

**数据集构建**：
- 舌象+问诊文本组合样本
- 面色+问诊文本组合样本
- 脉象+问诊文本组合样本
- 音频+问诊文本组合样本

**评价指标**：
- 信息融合度：多模态信息整合的有效性
- 一致性判断：处理多模态一致/不一致信息的能力
- 辨证准确率：基于多模态信息进行辨证的准确性 
"""
评测指标模块
"""

from .general_metrics import (
    calculate_accuracy,
    calculate_precision_recall_f1,
    calculate_bleu,
    calculate_rouge,
    calculate_perplexity,
    calculate_clip_score
)

from .tcm_metrics import (
    calculate_feature_recognition_accuracy,
    calculate_syndrome_correlation,
    calculate_description_completeness,
    calculate_pulse_recognition_accuracy,
    calculate_symptom_recognition_rate,
    calculate_inquiry_completeness,
    calculate_modal_consistency,
    calculate_prescription_accuracy,
    calculate_dosage_rationality,
    calculate_explanation_rationality
)

__all__ = [
    # 通用指标
    "calculate_accuracy",
    "calculate_precision_recall_f1",
    "calculate_bleu",
    "calculate_rouge",
    "calculate_perplexity",
    "calculate_clip_score",
    
    # 中医专业指标
    "calculate_feature_recognition_accuracy",
    "calculate_syndrome_correlation",
    "calculate_description_completeness",
    "calculate_pulse_recognition_accuracy",
    "calculate_symptom_recognition_rate",
    "calculate_inquiry_completeness",
    "calculate_modal_consistency",
    "calculate_prescription_accuracy",
    "calculate_dosage_rationality",
    "calculate_explanation_rationality"
]

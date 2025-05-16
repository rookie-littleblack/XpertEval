"""
XpertEval: 全模态大模型一站式评测框架
"""

__version__ = "0.1.0"

from .xpert_evaluator import XpertEvaluator, evaluate
from .utils.logger import get_logger

__all__ = ["XpertEvaluator", "evaluate", "get_logger"]

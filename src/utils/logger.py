"""
日志工具模块
"""

import os
import logging
import sys
from typing import Optional

def get_logger(name: Optional[str] = "xpert_eval", level: int = logging.INFO) -> logging.Logger:
    """
    获取配置好的日志记录器
    
    参数:
        name: 日志记录器名称
        level: 日志级别
    
    返回:
        配置好的日志记录器
    """
    # 检查是否已经存在相同名称的logger
    if name in logging.root.manager.loggerDict:
        return logging.getLogger(name)
    
    # 创建新的logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 防止日志重复输出
    if logger.handlers:
        return logger
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # 设置格式
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(console_handler)
    
    # 确保日志目录存在
    os.makedirs("logs", exist_ok=True)
    
    # 添加文件处理器
    file_handler = logging.FileHandler(f"logs/{name}.log", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger 
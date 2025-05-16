"""
与OpenAI兼容API通信的客户端模块
"""

import os
import json
import base64
import time
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import requests
import dotenv

from .utils.logger import get_logger

# 加载.env文件中的环境变量
dotenv.load_dotenv()

logger = get_logger("api_client")

class APIClient:
    """OpenAI兼容API客户端"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        default_model: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: int = 3
    ):
        """
        初始化API客户端
        
        参数:
            api_key: API密钥，如果为None则从环境变量获取
            api_base: API基础URL，如果为None则从环境变量获取
            api_version: API版本，如果为None则从环境变量获取
            default_model: 默认模型，如果为None则从环境变量获取
            timeout: 请求超时时间(秒)，如果为None则从环境变量获取
            max_retries: 最大重试次数
        """
        # 从环境变量或参数获取配置
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.api_version = api_version or os.getenv("OPENAI_API_VERSION", "2023-05-15")
        self.default_model = default_model or os.getenv("DEFAULT_MODEL", "gpt-4")
        self.default_vision_model = os.getenv("DEFAULT_VISION_MODEL", "gpt-4-vision-preview")
        self.default_embedding_model = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-ada-002")
        
        # 请求配置
        self.timeout = timeout or int(os.getenv("REQUEST_TIMEOUT", "120"))
        self.max_retries = max_retries or int(os.getenv("RETRY_ATTEMPTS", "3"))
        self.retry_delay = int(os.getenv("RETRY_DELAY", "2"))
        
        # 校验必要配置
        if not self.api_key:
            logger.warning("未提供API密钥，请设置OPENAI_API_KEY环境变量或在初始化时提供")
        
        # 日志配置
        self.save_responses = os.getenv("SAVE_RESPONSES", "false").lower() == "true"
        if self.save_responses:
            os.makedirs("logs/responses", exist_ok=True)
        
        logger.info(f"API客户端初始化完成: {self.api_base}, 默认模型: {self.default_model}")
    
    def text_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        文本补全请求
        
        参数:
            prompt: 提示文本
            model: 模型名称，如果为None则使用默认模型
            max_tokens: 最大生成令牌数
            temperature: 采样温度
            top_p: 核采样概率
            **kwargs: 其他参数
        
        返回:
            API响应字典
        """
        model = model or self.default_model
        max_tokens = max_tokens or int(os.getenv("MAX_TOKENS", "4096"))
        temperature = temperature or float(os.getenv("TEMPERATURE", "0.1"))
        top_p = top_p or float(os.getenv("TOP_P", "0.95"))
        
        # 构建请求数据
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            **kwargs
        }
        
        # 发送请求
        return self._send_request("chat/completions", data)
    
    def vision_completion(
        self,
        prompt: str,
        image_paths: List[str],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        detail: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        视觉任务请求
        
        参数:
            prompt: 提示文本
            image_paths: 图像路径列表
            model: 模型名称，如果为None则使用默认视觉模型
            max_tokens: 最大生成令牌数
            temperature: 采样温度
            detail: 图像分析详细程度，可选"high", "low", "auto"
            **kwargs: 其他参数
        
        返回:
            API响应字典
        """
        model = model or self.default_vision_model
        max_tokens = max_tokens or int(os.getenv("MAX_TOKENS", "4096"))
        temperature = temperature or float(os.getenv("TEMPERATURE", "0.1"))
        detail = detail or os.getenv("IMAGE_DETAIL", "high")
        
        # 构建包含图像的消息
        content = [{"type": "text", "text": prompt}]
        
        # 添加图像
        for img_path in image_paths:
            if not os.path.exists(img_path):
                logger.warning(f"图像文件不存在: {img_path}")
                continue
            
            # 读取图像并编码为base64
            with open(img_path, "rb") as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
            
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}",
                    "detail": detail
                }
            })
        
        # 构建请求数据
        data = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        # 发送请求
        return self._send_request("chat/completions", data)
    
    def audio_transcription(
        self,
        audio_path: str,
        model: str = "whisper-1",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        音频转录请求
        
        参数:
            audio_path: 音频文件路径
            model: 转录模型名称
            language: 音频语言，例如"zh"表示中文
            prompt: 提示文本，可以帮助改善转录准确性
            **kwargs: 其他参数
        
        返回:
            API响应字典
        """
        if not os.path.exists(audio_path):
            logger.error(f"音频文件不存在: {audio_path}")
            return {"error": f"音频文件不存在: {audio_path}"}
        
        # 准备文件和表单数据
        files = {
            'file': open(audio_path, 'rb')
        }
        
        data = {
            'model': model
        }
        
        if language:
            data['language'] = language
        
        if prompt:
            data['prompt'] = prompt
        
        # 添加其他参数
        for key, value in kwargs.items():
            data[key] = value
        
        # 发送请求
        response = self._send_request("audio/transcriptions", data, files=files, is_form=True)
        
        # 关闭文件
        files['file'].close()
        
        return response
    
    def _send_request(
        self,
        endpoint: str,
        data: Dict[str, Any],
        files: Optional[Dict[str, Any]] = None,
        is_form: bool = False
    ) -> Dict[str, Any]:
        """
        发送API请求
        
        参数:
            endpoint: API端点
            data: 请求数据
            files: 文件数据 (用于表单上传)
            is_form: 是否为表单请求
        
        返回:
            API响应
        """
        url = f"{self.api_base}/{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        
        if not is_form:
            headers["Content-Type"] = "application/json"
        
        if self.api_version:
            headers["api-version"] = self.api_version
        
        # 重试逻辑
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"发送请求到 {endpoint}, 模型: {data.get('model', 'N/A')}")
                
                if is_form:
                    response = requests.post(url, data=data, files=files, headers=headers, timeout=self.timeout)
                else:
                    response = requests.post(url, json=data, headers=headers, timeout=self.timeout)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # 保存响应
                    if self.save_responses:
                        self._save_response(endpoint, data, result)
                    
                    return result
                else:
                    error_detail = response.json() if response.text else {"status_code": response.status_code}
                    logger.error(f"API请求失败: {response.status_code}, {error_detail}")
                    
                    # 如果是速率限制错误，等待后重试
                    if response.status_code == 429:
                        retry_after = int(response.headers.get("Retry-After", self.retry_delay))
                        logger.warning(f"API速率限制，等待 {retry_after} 秒后重试")
                        time.sleep(retry_after)
                        continue
                    
                    return {"error": error_detail}
            
            except Exception as e:
                logger.error(f"请求异常: {str(e)}")
                if attempt < self.max_retries - 1:
                    logger.info(f"重试 {attempt + 1}/{self.max_retries}")
                    time.sleep(self.retry_delay)
                else:
                    return {"error": str(e)}
        
        return {"error": "达到最大重试次数"}
    
    def _save_response(self, endpoint: str, request: Dict[str, Any], response: Dict[str, Any]) -> None:
        """
        保存API请求和响应
        
        参数:
            endpoint: API端点
            request: 请求数据
            response: 响应数据
        """
        timestamp = int(time.time())
        model = request.get("model", "unknown")
        filename = f"logs/responses/{timestamp}_{endpoint.replace('/', '_')}_{model}.json"
        
        data = {
            "timestamp": timestamp,
            "endpoint": endpoint,
            "request": request,
            "response": response
        }
        
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存响应失败: {str(e)}") 
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 设置工作目录
WORKDIR /app

# 避免交互式前端
ENV DEBIAN_FRONTEND=noninteractive

# 安装基本依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    python3.10-venv \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    git \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 设置Python环境
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# 创建Python虚拟环境
RUN python3.10 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# 升级pip
RUN pip install --upgrade pip

# 复制requirements.txt并安装依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 如果使用PyTorch，可以添加特定版本
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# 复制项目代码
COPY . .

# 安装项目
RUN pip install -e .

# 创建结果目录
RUN mkdir -p results

# 设置容器启动命令
ENTRYPOINT ["python", "-m", "xpert_eval"] 
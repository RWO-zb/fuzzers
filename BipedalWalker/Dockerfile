# 1. 使用 NVIDIA CUDA 11.8 开发版镜像
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 2. 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/miniconda3/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# 3. 安装系统基础依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    swig \
    ffmpeg \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# 4. 安装 Miniconda (Python 3.10)
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && mkdir /root/.conda \
    && bash miniconda.sh -b -p /root/miniconda3 \
    && rm -f miniconda.sh \
    && conda create -n mdpfuzz python=3.10 -y \
    && conda clean -ya

# 5. 切换 shell 环境
SHELL ["conda", "run", "-n", "mdpfuzz", "/bin/bash", "-c"]

# 6. 安装 PyTorch (CUDA 11.8)
RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# 7. 预安装 setuptools (解决旧版 gym 安装报错的关键)
RUN pip install setuptools==65.5.0 wheel

# 8. 复制并安装 requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# --- 核心修改部分 START ---

# 9. 将当前目录下所有文件(gym, stable_baselines3, 代码等)复制到容器的 /app
COPY . /app

# 10. 设置工作目录
WORKDIR /app

# 11. 安装本地的 gym 和 stable_baselines3
# 注意：这里直接指向当前目录下的文件夹
RUN pip install -e ./gym
RUN pip install -e ./stable_baselines3

# --- 核心修改部分 END ---

# 12. 默认命令
CMD ["python"]
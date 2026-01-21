# 1. 使用 NVIDIA CUDA 11.8 开发版镜像 (为了支持 nvcc 和编译某些包)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 2. 设置环境变量，防止交互式安装卡住
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/miniconda3/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# 3. 安装系统基础依赖 (OpenCV, Gym 渲染, 编译工具等)
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

# 5. 激活环境并将 shell 切换为 conda 环境
SHELL ["conda", "run", "-n", "mdpfuzz", "/bin/bash", "-c"]

# 6. 安装 PyTorch (指定 CUDA 11.8 源)
# 注意：这里假设 2026 年 torch 2.7.1 依然兼容 cu118 的源，如果不兼容，可能需要调整 index-url
RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# 7. 复制并安装通用依赖
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# 8. 处理本地包 (Gym 0.19.0 和 SB3)
# 将本地文件夹复制进容器
COPY qd_based_testing_rl /app/qd_based_testing_rl

# 安装本地包
# 假设 gym 0.19.0 在 /app/qd_based_testing_rl 根目录或子目录
# ⚠️ 这里需要你根据实际文件夹结构调整
WORKDIR /app/qd_based_testing_rl
RUN pip install -e . 
# 如果 stable_baselines3 也在这个目录下，通常 pip install -e . 会自动读取 setup.py 安装
# 如果它们是分开的文件夹，请分别 pip install -e ./gym 和 pip install -e ./stable_baselines3

# 9. 设置工作目录
WORKDIR /workspace

# 10. 默认启动命令
CMD ["python"]
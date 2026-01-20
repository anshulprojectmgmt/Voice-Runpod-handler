FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Prevent tzdata prompt (CRITICAL)
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    sox \
    libgl1 \
    libglib2.0-0 \
    tzdata \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Install PyTorch (CUDA 11.8)
RUN pip3 install torch==2.0.1+cu118 torchaudio==2.0.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Python deps
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy handler code
COPY . .

CMD ["python3", "handler.py"]

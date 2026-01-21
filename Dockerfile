FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install Python and git (git is missing!)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    sox \
    libsndfile1 \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Upgrade pip first
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 FIRST (most important!)
RUN pip3 install --no-cache-dir \
    torch==2.1.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install other requirements one by one to avoid conflicts
COPY requirements.txt .
RUN pip3 install --no-cache-dir \
    numpy==1.24.3 \
    soundfile==0.12.1 \
    librosa==0.10.1 \
    requests==2.31.0 \
    python-dotenv==1.0.0 \
    huggingface-hub==0.20.3

# Install runpod separately
RUN pip3 install --no-cache-dir runpod==1.14.0

# Copy handler
COPY . .

CMD ["python3", "handler.py"]
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    sox \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Install PyTorch with CUDA 12.1 FIRST
RUN pip3 install --no-cache-dir \
    torch==2.1.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install runpod
RUN pip3 install --no-cache-dir runpod==1.14.0

# Install other requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

CMD ["python3", "handler.py"]
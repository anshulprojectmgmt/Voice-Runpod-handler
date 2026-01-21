 FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    sox \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install deps (🔥 legacy resolver avoids conflict)
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir \
    --use-deprecated=legacy-resolver \
    -r requirements.txt

# Copy handler
COPY handler.py .

CMD ["python3", "handler.py"]

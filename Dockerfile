FROM runpod/pytorch:2.0.1-cuda11.8-runtime-ubuntu20.04

# 🔥 Prevent tzdata interactive prompt
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

WORKDIR /app

# System dependencies for audio
RUN apt-get update && apt-get install -y \
    ffmpeg \
    sox \
    libgl1 \
    libglib2.0-0 \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler code
COPY . .

CMD ["python", "handler.py"]

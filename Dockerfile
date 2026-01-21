 
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    ffmpeg \
    sox \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 🔥 Python / pip consistency (CRITICAL)
RUN python -m pip install --upgrade pip

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "handler.py"]

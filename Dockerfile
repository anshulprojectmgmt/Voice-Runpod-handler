FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

WORKDIR /app

# ---------- SYSTEM ----------
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-distutils \
    python3-pip \
    ffmpeg \
    sox \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---------- FORCE PYTHON 3.10 ----------
RUN ln -sf /usr/bin/python3.10 /usr/bin/python
RUN python --version

# ---------- PIP ----------
RUN python -m pip install --upgrade pip

# ---------- DEPENDENCIES ----------
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# ---------- HANDLER ----------
COPY handler.py .

CMD ["python", "handler.py"]

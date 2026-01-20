 FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler and chatterbox
COPY handler.py .
COPY chatterbox ./chatterbox

# RunPod entry
CMD ["python", "-u", "handler.py"]

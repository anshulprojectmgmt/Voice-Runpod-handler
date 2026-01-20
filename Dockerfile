FROM runpod/pytorch:2.1.0-cuda12.1-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY handler.py .
COPY chatterbox ./chatterbox

ENV PYTHONPATH="/app"

CMD ["python", "handler.py"]

"""
RunPod Serverless Handler for Chatterbox
Supports:
1) Speaker embedding extraction
2) TTS generation using cached embeddings
"""

import base64
import io
import json
import os
import sys

# Try to import runpod, but handle the case where it might not be installed yet
try:
    import runpod
except ImportError:
    # If runpod is not installed, install it and restart
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "runpod"])
    # Restart the script with the new module
    os.execl(sys.executable, sys.executable, *sys.argv)

# Now we can safely import the rest
import torch
import torchaudio
from chatterbox.tts import ChatterboxTTS

MODEL = None


def load_model():
    global MODEL
    if MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        MODEL = ChatterboxTTS.from_pretrained(device=device)
    return MODEL


def decode_audio(b64_audio: str, path="/tmp/ref.wav"):
    audio_bytes = base64.b64decode(b64_audio)
    with open(path, "wb") as f:
        f.write(audio_bytes)
    return path


def handler(job):
    model = load_model()
    data = job["input"]
    task = data.get("task")

    # ----------------------------
    # 1️⃣ EXTRACT SPEAKER EMBEDDING
    # ----------------------------
    if task == "extract_embedding":
        audio_path = decode_audio(data["audio_b64"])

        # Prepare conditionals once
        model.prepare_conditionals(audio_path, exaggeration=0.3)

        # Convert tensors → JSON-safe
        speaker_embedding = {
            k: v.cpu().numpy().tolist()
            for k, v in model.conds.items()
        }

        return {
            "speaker_embedding": speaker_embedding
        }

    # ----------------------------
    # 2️⃣ TEXT TO SPEECH
    # ----------------------------
    if task == "tts":
        text = data["text"]
        conds = data["speaker_embedding"]

        # Restore tensors
        model.conds = {
            k: torch.tensor(v).to(model.device)
            for k, v in conds.items()
        }

        wav = model.generate(
            text,
            temperature=data.get("temperature", 0.6),
            cfg_weight=data.get("cfg_weight", 0.3),
        )

        buffer = io.BytesIO()
        torchaudio.save(buffer, wav.cpu(), model.sr, format="wav")
        buffer.seek(0)

        return {
            "audio_b64": base64.b64encode(buffer.read()).decode(),
            "sample_rate": model.sr,
        }

    return {"error": "Invalid task type"}


runpod.serverless.start({"handler": handler})
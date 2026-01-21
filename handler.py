import runpod
import torch
import torchaudio
import base64
import io
import os
import json
from chatterbox.tts import ChatterboxTTS

# Load model once per container
MODEL = None


def load_model():
    global MODEL
    if MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[RunPod] Loading Chatterbox on {device}")
        MODEL = ChatterboxTTS.from_pretrained(device=device)
        print("[RunPod] Model loaded")
    return MODEL


def decode_audio(b64_audio: str, path="/tmp/ref.wav"):
    audio_bytes = base64.b64decode(b64_audio)
    with open(path, "wb") as f:
        f.write(audio_bytes)
    return path


def handler(job):
    try:
        model = load_model()
        data = job["input"]
        task = data.get("task")

        # --------------------------------------------------
        # 1️⃣ EXTRACT SPEAKER EMBEDDING
        # --------------------------------------------------
        if task == "extract_embedding":
            ref_audio_b64 = data.get("ref_audio_b64")
            if not ref_audio_b64:
                return {"error": "ref_audio_b64 missing"}

            wav_path = decode_audio(ref_audio_b64)

            # Prepare conditionals
            model.prepare_conditionals(wav_path, exaggeration=0.3)

            # Serialize conditionals
            speaker_embedding = {
                k: v.cpu().numpy().tolist()
                for k, v in model.conds.items()
            }

            return {
                "speaker_embedding": speaker_embedding
            }

        # --------------------------------------------------
        # 2️⃣ TTS WITH SPEAKER EMBEDDING
        # --------------------------------------------------
        if task == "tts":
            text = data.get("text")
            speaker_embedding = data.get("speaker_embedding")

            if not text or not speaker_embedding:
                return {"error": "text or speaker_embedding missing"}

            # Restore tensors
            model.conds = {
                k: torch.tensor(v).to(model.device)
                for k, v in speaker_embedding.items()
            }

            wav = model.generate(
                text,
                temperature=data.get("temperature", 0.6),
                cfg_weight=data.get("cfg_weight", 0.3),
            )

            buf = io.BytesIO()
            torchaudio.save(buf, wav.cpu(), model.sr, format="wav")
            buf.seek(0)

            return {
                "audio_b64": base64.b64encode(buf.read()).decode(),
                "sample_rate": model.sr
            }

        return {"error": f"Invalid task: {task}"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})

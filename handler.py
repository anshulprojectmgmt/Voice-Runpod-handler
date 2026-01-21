import runpod
import torch
import torchaudio
import base64
import io
from chatterbox.tts import ChatterboxTTS

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
    data = job["input"]
    task = data.get("task")
    model = load_model()

    # 1️⃣ Extract speaker embedding
    if task == "extract_embedding":
        audio_path = decode_audio(data["audio_b64"])
        model.prepare_conditionals(audio_path, exaggeration=0.3)

        speaker_embedding = {
            k: v.cpu().numpy().tolist()
            for k, v in model.conds.items()
        }

        return {"speaker_embedding": speaker_embedding}

    # 2️⃣ TTS with embedding
    if task == "tts":
        text = data["text"]
        conds = data["speaker_embedding"]

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

    return {"error": "Invalid task"}


runpod.serverless.start({"handler": handler})

import runpod
import torch
import base64
import io
import numpy as np
import soundfile as sf

from chatterbox.tts import ChatterboxTTS, Conditionals, T3Cond

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

    # ================================
    # 1️⃣ EXTRACT SPEAKER EMBEDDING
    # ================================
    if task == "extract_embedding":
        audio_b64 = data.get("audio_b64") or data.get("ref_audio_b64")
        if not audio_b64:
            return {"error": "audio_b64 missing"}

        audio_path = decode_audio(audio_b64)

        # ✅ LOW exaggeration = clean base voice
        model.prepare_conditionals(audio_path, exaggeration=0.1)

        speaker_embedding = {
            "t3": {
                "speaker_emb": model.conds.t3.speaker_emb.detach().cpu().tolist(),
                "emotion_adv": model.conds.t3.emotion_adv.detach().cpu().tolist(),
            },
            "gen": {
                k: (v.detach().cpu().tolist() if torch.is_tensor(v) else v)
                for k, v in model.conds.gen.items()
            }
        }

        return {"speaker_embedding": speaker_embedding}

    # ================================
    # 2️⃣ TTS USING EMBEDDING
    # ================================
    elif task == "tts":
        text = data.get("text")
        embedding = data.get("speaker_embedding")

        if not text or not embedding:
            return {"error": "text or speaker_embedding missing"}

        # ✅ RECONSTRUCT CONDITIONING (CORRECT)
        model.conds = Conditionals(
            t3=T3Cond(
                speaker_emb=torch.tensor(
                    embedding["t3"]["speaker_emb"],
                    dtype=torch.float32,
                    device=model.device
                ),
                emotion_adv=torch.tensor(
                    embedding["t3"]["emotion_adv"],
                    dtype=torch.float32,
                    device=model.device
                ),
            ),
            gen={
                k: torch.tensor(v, device=model.device) if isinstance(v, list) else v
                for k, v in embedding["gen"].items()
            }
        )

        with torch.inference_mode():
            wav = model.generate(
                text=text,
                temperature=data.get("temperature", 0.35),
                cfg_weight=data.get("cfg_weight", 1.1),
            )

        # ================================
        # 🔥 AUDIO FIX (CRITICAL)
        # ================================

        # Ensure shape [T]
        if wav.ndim > 1:
            wav = wav.squeeze(0)

        wav = wav.detach().cpu().numpy()

        # Remove DC offset
        wav = wav - np.mean(wav)

        # RMS normalization (natural loudness)
        rms = np.sqrt(np.mean(wav ** 2))
        if rms > 0:
            wav = wav / rms * 0.1

        # Hard clip safety
        wav = np.clip(wav, -1.0, 1.0)

        # Encode WAV correctly
        buffer = io.BytesIO()
        sf.write(
            buffer,
            wav,
            model.sr,
            format="WAV",
            subtype="PCM_16"
        )
        buffer.seek(0)

        return {
            "audio_b64": base64.b64encode(buffer.read()).decode(),
            "sample_rate": model.sr,
        }

    else:
        return {"error": f"Invalid task: {task}"}


runpod.serverless.start({"handler": handler})

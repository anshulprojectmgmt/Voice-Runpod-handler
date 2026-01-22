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


def handler(job):
    data = job["input"]
    task = data.get("task")
    model = load_model()

    # ===============================
    # 1️⃣ EXTRACT SPEAKER EMBEDDING
    # ===============================
    if task == "extract_embedding":
        audio_b64 = data.get("audio_b64")
        audio_bytes = base64.b64decode(audio_b64)

        with open("/tmp/ref.wav", "wb") as f:
            f.write(audio_bytes)

        model.prepare_conditionals("/tmp/ref.wav", exaggeration=0.1)

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

    # ===============================
    # 2️⃣ TTS (CHUNKED – FIXED)
    # ===============================
    elif task == "tts":
        text_chunks = data.get("text_chunks")
        embedding = data.get("speaker_embedding")

        if not text_chunks or not embedding:
            return {"error": "text_chunks or speaker_embedding missing"}

        model.conds = Conditionals(
            t3=T3Cond(
                speaker_emb=torch.tensor(embedding["t3"]["speaker_emb"], device=model.device),
                emotion_adv=torch.tensor(embedding["t3"]["emotion_adv"], device=model.device),
            ),
            gen={
                k: torch.tensor(v, device=model.device) if isinstance(v, list) else v
                for k, v in embedding["gen"].items()
            }
        )

        wavs = []

        for text in text_chunks:
            text = text.strip()
            if not text:
                continue

            with torch.inference_mode():
                wav = model.generate(
                    text=text,
                    temperature=data.get("temperature", 0.25),
                    cfg_weight=data.get("cfg_weight", 1.15),
                   
                )

            wav = wav.squeeze(0)
            wav = wav / wav.abs().max().clamp(min=1e-6)
            wavs.append(wav)

            # 🔥 Natural pause
            silence = torch.zeros(int(0.35 * model.sr), device=wav.device)
            wavs.append(silence)

        full_wav = torch.cat(wavs).cpu().numpy()

        buffer = io.BytesIO()
        sf.write(buffer, full_wav, model.sr,format="WAV",subtype="PCM_16")
        buffer.seek(0)

        return {
            "audio_b64": base64.b64encode(buffer.read()).decode(),
            "sample_rate": model.sr,
        }

    else:
        return {"error": f"Invalid task: {task}"}


runpod.serverless.start({"handler": handler})
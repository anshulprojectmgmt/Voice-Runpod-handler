 import runpod
import torch
import torchaudio
import base64
import io
import os
import json
from chatterbox.tts import ChatterboxTTS

tts_model = None


def load_model():
    global tts_model
    if tts_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[RunPod] Loading Chatterbox TTS on {device}")
        tts_model = ChatterboxTTS.from_pretrained(device=device)
        print("[RunPod] Model loaded")
    return tts_model


def save_ref_audio(b64_audio: str) -> str:
    audio_bytes = base64.b64decode(b64_audio)
    path = "/tmp/ref.wav"
    with open(path, "wb") as f:
        f.write(audio_bytes)
    return path


def handler(job):
    try:
        inp = job["input"]
        task = inp.get("task")

        model = load_model()

        # -------------------------------------------------
        # 1️⃣ EXTRACT + CACHE CONDITIONING (VOICE UPLOAD)
        # -------------------------------------------------
        if task == "extract_embedding":
            ref_audio_b64 = inp.get("ref_audio_b64")
            exaggeration = inp.get("exaggeration", 0.5)

            if not ref_audio_b64:
                return {"error": "ref_audio_b64 missing"}

            wav_path = save_ref_audio(ref_audio_b64)

            # 🔥 Correct Chatterbox API
            model.prepare_conditionals(
                wav_path,
                exaggeration=exaggeration
            )

            conds = model.conds

            os.remove(wav_path)

            return {
                "conds": conds,   # FULL conditioning dict
                "cached": True
            }

        # -------------------------------------------------
        # 2️⃣ TTS USING CACHED CONDITIONING
        # -------------------------------------------------
        elif task == "tts_with_embedding":
            text = inp.get("text")
            conds = inp.get("conds")

            temperature = inp.get("temperature", 0.8)
            cfg_weight = inp.get("cfg_weight", 0.5)

            if not text or not conds:
                return {"error": "text or conds missing"}

            # 🔥 SAFE: restore conditionals
            model.conds = conds

            wav = model.generate(
                text,
                temperature=temperature,
                cfg_weight=cfg_weight
            )

            buf = io.BytesIO()
            torchaudio.save(buf, wav.cpu(), model.sr, format="wav")
            audio_b64 = base64.b64encode(buf.getvalue()).decode()

            return {
                "audio_b64": audio_b64,
                "sample_rate": model.sr
            }

        else:
            return {"error": f"Invalid task: {task}"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})

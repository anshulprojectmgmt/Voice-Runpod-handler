"""
RunPod Serverless Handler
Embedding-based Chatterbox TTS inference
"""

import runpod
import torch
import torchaudio
import base64
import io
from chatterbox.tts import ChatterboxTTS, Conditionals, T3Cond

# Global model (loaded once per container)
tts_model = None


def load_model():
    global tts_model
    if tts_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Chatterbox TTS on {device}")
        tts_model = ChatterboxTTS.from_pretrained(device=device)
        print("Model loaded")
    return tts_model


def handler(job):
    try:
        job_input = job["input"]

        if job_input.get("task") != "tts":
            return {"error": "Invalid task"}

        text = job_input.get("text")
        conds_b64 = job_input.get("speaker_embedding_b64")
        temperature = job_input.get("temperature", 0.6)
        cfg_weight = job_input.get("cfg_weight", 0.3)

        if not text or not conds_b64:
            return {"error": "Missing text or speaker embedding"}

        model = load_model()

        # -------------------------------
        # Decode speaker embedding
        # -------------------------------
        buffer = io.BytesIO(base64.b64decode(conds_b64))
        payload = torch.load(buffer, map_location=model.device)

        t3_cond = T3Cond(**payload["t3"]).to(model.device)
        gen_dict = {
            k: v.to(model.device) if torch.is_tensor(v) else v
            for k, v in payload["gen"].items()
        }

        model.conds = Conditionals(
            t3=t3_cond,
            gen=gen_dict
        )

        # -------------------------------
        # Generate audio
        # -------------------------------
        with torch.inference_mode():
            wav = model.generate(
                text=text,
                temperature=temperature,
                cfg_weight=cfg_weight,
            )

        # Convert to WAV bytes
        out_buf = io.BytesIO()
        torchaudio.save(out_buf, wav.cpu(), model.sr, format="wav")
        out_buf.seek(0)

        audio_b64 = base64.b64encode(out_buf.read()).decode("utf-8")

        return {
            "audio_b64": audio_b64,
            "sample_rate": model.sr,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})

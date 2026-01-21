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
        print(f"Loading Chatterbox on {device}")
        MODEL = ChatterboxTTS.from_pretrained(device=device)
        print("Model loaded")
    return MODEL


def handler(job):
    try:
        model = load_model()
        data = job["input"]
        task = data.get("task")

        # 1️⃣ Extract embedding
        if task == "extract_embedding":
            audio_b64 = data["audio_b64"]
            audio_bytes = base64.b64decode(audio_b64)

            path = "/tmp/ref.wav"
            with open(path, "wb") as f:
                f.write(audio_bytes)

            model.prepare_conditionals(path, exaggeration=0.3)

            embedding = {
                k: v.cpu().numpy().tolist()
                for k, v in model.conds.items()
            }

            return {"speaker_embedding": embedding}

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

            buf = io.BytesIO()
            torchaudio.save(buf, wav.cpu(), model.sr, format="wav")
            buf.seek(0)

            return {
                "audio_b64": base64.b64encode(buf.read()).decode(),
                "sample_rate": model.sr
            }

        return {"error": "Invalid task"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})

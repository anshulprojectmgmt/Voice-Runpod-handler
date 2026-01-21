import runpod
import torch
import torchaudio
import base64
import io
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

        # Prepare conditionals
        model.prepare_conditionals(audio_path, exaggeration=0.3)

        # ✅ Serialize correctly
        speaker_embedding = {
            "t3": {
                "speaker_emb": model.conds.t3.speaker_emb.detach().cpu().numpy().tolist(),
                "emotion_adv": model.conds.t3.emotion_adv.detach().cpu().numpy().tolist(),
            },
            "gen": {
                k: (v.detach().cpu().numpy().tolist() if torch.is_tensor(v) else v)
                for k, v in model.conds.gen.items()
            }
        }

        return {
            "speaker_embedding": speaker_embedding
        }

    # ================================
    # 2️⃣ TTS USING EMBEDDING
    # ================================
    elif task == "tts":
        text = data.get("text")
        embedding = data.get("speaker_embedding")

        if not text or not embedding:
            return {"error": "text or speaker_embedding missing"}

        # Restore conditionals
        model.conds = Conditionals(
            t3=T3Cond(
                speaker_emb=torch.tensor(
                    embedding["t3"]["speaker_emb"], dtype=torch.float32
                ).to(model.device),
                emotion_adv=torch.tensor(
                    embedding["t3"]["emotion_adv"], dtype=torch.float32
                ).to(model.device),
            ),
            gen={
                k: torch.tensor(v).to(model.device) if isinstance(v, list) else v
                for k, v in embedding["gen"].items()
            }
        )

        with torch.inference_mode():
            wav = model.generate(
                text=text,
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

    # ================================
    # INVALID TASK
    # ================================
    else:
        return {"error": f"Invalid task: {task}"}


runpod.serverless.start({"handler": handler})


import runpod
import torch
import torchaudio
import base64
import io
import os
import numpy as np
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


def decode_audio(b64_audio: str) -> str:
    audio_bytes = base64.b64decode(b64_audio)
    temp_path = "/tmp/ref.wav"
    with open(temp_path, "wb") as f:
        f.write(audio_bytes)
    return temp_path


def handler(job):
    try:
        inp = job["input"]
        task = inp.get("task")

        model = load_model()

        # ----------------------------------
        # TASK 1: EXTRACT SPEAKER EMBEDDING
        # ----------------------------------
        if task == "extract_embedding":
            ref_audio_b64 = inp.get("ref_audio_b64")
            if not ref_audio_b64:
                return {"error": "ref_audio_b64 missing"}

            wav_path = decode_audio(ref_audio_b64)

            embedding = model.extract_speaker_embedding(wav_path)

            os.remove(wav_path)

            return {
                "speaker_embedding": embedding.tolist(),
                "embedding_dim": len(embedding)
            }

        # ----------------------------------
        # TASK 2: TTS WITH EMBEDDING
        # ----------------------------------
        elif task == "tts_with_embedding":
            text = inp.get("text")
            embedding = inp.get("speaker_embedding")

            if not text or embedding is None:
                return {"error": "text or speaker_embedding missing"}

            exaggeration = inp.get("exaggeration", 0.5)
            temperature = inp.get("temperature", 0.8)
            cfg_weight = inp.get("cfg_weight", 0.5)

            # Rebuild conditionals using embedding
            embedding_tensor = torch.tensor(
                embedding, dtype=torch.float32
            ).unsqueeze(0).to(model.device)

            model.conds.t3.speaker_emb = embedding_tensor
            model.conds.t3.emotion_adv = exaggeration * torch.ones(
                1, 1, 1, device=model.device
            )

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

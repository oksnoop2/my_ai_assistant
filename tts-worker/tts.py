from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel
from TTS.api import TTS
import torch
import numpy as np
import io
import os
import gc
import sys

# Import the required configuration classes from the TTS library
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

# Handle compatibility with different PyTorch versions
try:
    # This is available in newer PyTorch versions
    torch.serialization.add_safe_globals([
        XttsConfig,
        XttsAudioConfig,
        XttsArgs,
        BaseDatasetConfig
    ])
except AttributeError:
    # Older PyTorch versions don't have this method
    print("⚠️  Using older PyTorch version without add_safe_globals", file=sys.stderr)
    pass

# --- Global variable to hold the model ---
tts_model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

class TTSRequest(BaseModel):
    text: str
    speaker_wav: str = None

app = FastAPI()

@app.get("/health")
def health_check():
    status = "loaded" if tts_model is not None else "unloaded"
    return {"status": "ok", "model_status": status}

@app.post("/load")
def load_model_endpoint():
    global tts_model
    if tts_model is None:
        print(f"🚀 Loading TTS model onto device: {device}...")
        try:
            tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())
            print("✅ TTS model loaded.")
        except Exception as e:
            print(f"❌ Failed to load TTS model: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
    return {"status": "model_loaded"}

@app.post("/unload")
def unload_model_endpoint():
    global tts_model
    if tts_model is not None:
        print("卸载 Unloading TTS model from VRAM...")
        del tts_model
        tts_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✅ TTS model unloaded.")
    return {"status": "model_unloaded"}

@app.post("/api/tts")
def text_to_speech(request: TTSRequest):
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please call /load first.")
    
    print(f"🗣️  Synthesizing speech: '{request.text}'")
    try:
        speaker_wav_path = os.path.normpath(os.path.join("/", request.speaker_wav.strip("/"))) if request.speaker_wav else None
        wav_data = tts_model.tts(text=request.text, speaker_wav=speaker_wav_path, language="en")
        
        if isinstance(wav_data, list):
            wav_data = np.array(wav_data, dtype=np.float32)
        
        if isinstance(wav_data, np.ndarray):
            if wav_data.ndim > 1:
                wav_data = wav_data.mean(axis=1)
            wav_data = (wav_data * 32767).astype(np.int16)

        from scipy.io.wavfile import write
        wav_io = io.BytesIO()
        write(wav_io, 24000, wav_data)
        wav_io.seek(0)

        print("✅ Speech synthesis complete.")
        return Response(content=wav_io.read(), media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

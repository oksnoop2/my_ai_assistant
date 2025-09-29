# tts-service/service.py
import os
import requests
from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel
from TTS.api import TTS
import torch
import numpy as np
import io
import gc
from scipy.io.wavfile import write
import sys

# Import the required configuration classes from the TTS library
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

# Handle compatibility with PyTorch 2.6+
try:
    torch.serialization.add_safe_globals([
        XttsConfig,
        XttsAudioConfig,
        XttsArgs,
        BaseDatasetConfig
    ])
except AttributeError:
    print("⚠️  Using older PyTorch version without add_safe_globals", file=sys.stderr)
    pass

# --- Globals ---
tts_model = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# This is no longer needed for the TTS service itself, but keep it for consistency
RESOURCE_MANAGER_URL = os.environ["RESOURCE_MANAGER_URL"]

class TTSRequest(BaseModel):
    text: str
    speaker_wav: str

app = FastAPI()

@app.get("/health")
def health_check():
    status = "loaded" if tts_model is not None else "unloaded"
    return {"status": "ok", "model_status": status}

@app.post("/load")
def load_model_endpoint():
    """Loads the TTS model directly onto the primary device (GPU if available)."""
    global tts_model
    if tts_model is None:
        print(f"🚀 Loading TTS model directly onto device: {DEVICE}...")
        try:
            # --- REVERTED TO YOUR OLD, CORRECT LOGIC ---
            tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())
            print("✅ TTS model loaded.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
    return {"status": "model_loaded"}

@app.post("/unload")
def unload_model_endpoint():
    """Unloads the model from memory."""
    global tts_model
    if tts_model is not None:
        print("Unloading TTS model from memory...")
        del tts_model
        tts_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✅ TTS model unloaded.")
    return {"status": "model_unloaded"}

@app.post("/api/tts")
def text_to_speech(request: TTSRequest):
    """Performs inference on the model which is already on the correct device."""
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Resource Manager should have called /load.")
    
    print(f"🗣️  Synthesizing speech: '{request.text}'")
    
    try:
        # The model generates audio as a list of floats in the range [-1.0, 1.0]
        wav_data_float = tts_model.tts(text=request.text, speaker_wav=request.speaker_wav, language="en")
        
        # --- FIX: Convert float audio to 16-bit integer format ---
        # Scale the float data to the range of a 16-bit integer (-32768 to 32767)
        wav_data_int = (np.array(wav_data_float) * 32767).astype(np.int16)
        
        # Now write the correctly formatted integer data to the WAV file
        wav_io = io.BytesIO()
        write(wav_io, tts_model.synthesizer.output_sample_rate, wav_data_int)
        wav_io.seek(0)

        print("✅ Speech synthesis complete.")
        return Response(content=wav_io.read(), media_type="audio/wav")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

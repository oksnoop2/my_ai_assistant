# FILE: my_ai_assistant/asr/asr.py
import os
import whisper
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
import tempfile
import requests

# --- Configuration ---
# The model will now load to the CPU first to conserve GPU memory.
CPU_DEVICE = "cpu"
GPU_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SIZE = "base"
RESOURCE_MANAGER_URL = os.environ.get("RESOURCE_MANAGER_URL")

# --- Model Loading ---
print(f"🚀 ASR service starting...")
print(f"📦 Loading Whisper model '{MODEL_SIZE}' into system RAM...")
# Load to CPU initially
model = whisper.load_model(MODEL_SIZE, device=CPU_DEVICE)
print("✅ Whisper model loaded to RAM.")

app = FastAPI()

@app.get("/health")
def health_check():
    """A simple health check endpoint for the orchestrator."""
    return Response(status_code=200, content='{"status":"ok"}', media_type="application/json")

@app.post("/asr")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """
    Accepts an audio file, acquires a GPU lock, moves the model to the GPU for transcription,
    and then moves it back to the CPU.
    """
    if not audio_file:
        raise HTTPException(status_code=400, detail="No audio file provided.")

    if not RESOURCE_MANAGER_URL:
        raise HTTPException(status_code=500, detail="RESOURCE_MANAGER_URL is not configured.")

    print(f"📝 Transcribing file: {audio_file.filename}")
    try:
        # Step 1: Acquire GPU lock
        print("⏳ Requesting GPU lock from Resource Manager...")
        requests.post(f"{RESOURCE_MANAGER_URL}/acquire_gpu", timeout=300).raise_for_status()
        print("✅ GPU lock acquired by ASR Service.")

        # Step 2: Move model to GPU
        model.to(GPU_DEVICE)
        print("➡️  Moved ASR model to GPU.")

        # Step 3: Perform transcription
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await audio_file.read())
            tmp_path = tmp.name

        result = model.transcribe(tmp_path, fp16=torch.cuda.is_available())
        transcription = result.get("text", "").strip()
        print(f"🗣️  Transcription result: '{transcription}'")
        
        # Clean up the temp file
        os.unlink(tmp_path)
        
        return {"text": transcription}

    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Step 4: CRITICAL - Move model back to CPU and release the lock
        model.to(CPU_DEVICE)
        print("⬅️  Moved ASR model back to CPU.")
        
        requests.post(f"{RESOURCE_MANAGER_URL}/release_gpu", timeout=60)
        print("✅ GPU lock released by ASR Service.")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

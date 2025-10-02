# FILE: my_ai_assistant/asr/asr.py

import os
import whisper
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
import tempfile

# --- Model Loading ---
# Check for GPU and load the Whisper model.
# "base" is a good starting point. You can use "small" or "medium" for more
# accuracy if you have enough VRAM and memory.
device = "cuda" if torch.cuda.is_available() else "cpu"
model_size = "base"

print(f"🚀 ASR service starting on device: {device}")
print(f"📦 Loading Whisper model '{model_size}'... (This may take a moment on first run)")
model = whisper.load_model(model_size, device=device)
print("✅ Whisper model loaded.")

app = FastAPI()

@app.on_event("startup")
def startup_log():
    """Log essential info when the server starts."""
    print("✅ ASR service initialized.")
    print(f"💻 Device in use: {device}")
    print(f"🧠 Model size: {model_size}")

@app.get("/health")
def health_check():
    """A simple health check endpoint for the orchestrator."""
    return Response(status_code=200, content='{"status":"ok"}', media_type="application/json")

@app.post("/asr")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """
    Accepts an audio file upload and returns the transcribed text.
    """
    if not audio_file:
        raise HTTPException(status_code=400, detail="No audio file provided.")

    print(f"📝 Transcribing file: {audio_file.filename}")
    # Whisper works best with file paths, so we save the upload to a temporary file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            # Write the uploaded file content to the temporary file
            tmp.write(await audio_file.read())
            tmp_path = tmp.name

        # Perform the transcription
        result = model.transcribe(tmp_path, fp16=torch.cuda.is_available())
        transcription = result.get("text", "").strip()

        print(f"🗣️ Transcription result: '{transcription}'")
        return {"text": transcription}

    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Ensure the temporary file is cleaned up
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)

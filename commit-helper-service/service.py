# commit-helper-service/service.py (Final Corrected Version)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
import os

app = FastAPI()

class CompletionRequest(BaseModel):
    prompt: str

# --- Globals ---
llm = None
MODEL_PATH = "/models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"

# MODIFIED: The health check now just confirms the server is running.
# The model might not be loaded yet.
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": llm is not None}

# NEW: Added an explicit /load endpoint.
@app.post("/load")
def load_model():
    """Explicitly loads the model into memory."""
    global llm
    if llm is None:
        print("🚀 Loading standalone LLM model into memory...")
        try:
            llm = Llama(
                model_path=MODEL_PATH,
                n_gpu_layers=-1,
                n_ctx=4096,
                verbose=True,
                n_threads=16
            )
            print("✅ Standalone LLM model loaded.")
            return {"status": "model_loaded"}
        except Exception as e:
            print(f"🔥 Failed to load model: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    return {"status": "model_already_loaded"}


@app.post("/completion")
def completion(payload: CompletionRequest):
    if llm is None:
        # This error is now the definitive sign that /load wasn't called or failed.
        raise HTTPException(status_code=503, detail="Model is not loaded. Call /load first.")
    
    try:
        output = llm(payload.prompt, max_tokens=256, stop=["User:", "\n"], echo=False)
        content = output["choices"][0]["text"].strip()
        return {"content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

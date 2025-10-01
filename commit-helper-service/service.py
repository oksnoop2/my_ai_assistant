# commit-helper-service/service.py (Simplified)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
import os

app = FastAPI()

class CompletionRequest(BaseModel):
    prompt: str

# --- Globals ---
llm = None # Still useful to have a global placeholder
MODEL_PATH = "/models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"

@app.on_event("startup")
def load_model_on_startup():
    """Load the model as soon as the service starts."""
    global llm
    # The 'if llm is None:' check is removed as this runs only once.
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
    except Exception as e:
        print(f"🔥 Failed to load model on startup: {e}")
        # llm will remain None if the above fails, which is the desired outcome.

@app.get("/health")
def health():
    if llm is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {"status": "ok", "model_status": "loaded"}

@app.post("/completion")
def completion(payload: CompletionRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    try:
        output = llm(payload.prompt, max_tokens=256, stop=["User:", "\n"], echo=False)
        content = output["choices"][0]["text"].strip()
        return {"content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

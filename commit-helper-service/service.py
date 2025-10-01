# commit-helper-service/service.py
# A simplified, standalone LLM service for internal tooling.

from fastapi import FastAPI, HTTPException
from llama_cpp import Llama
import os

app = FastAPI()

# --- Globals ---
llm = None
MODEL_PATH = "/models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"

@app.on_event("startup")
def load_model_on_startup():
    """Load the model as soon as the service starts."""
    global llm
    if llm is None:
        print("🚀 Loading standalone LLM model into memory...")
        try:
            llm = Llama(
                model_path=MODEL_PATH,
                n_gpu_layers=-1, # Use GPU if available
                n_ctx=4096,
                verbose=False, # Keep logs clean
                n_threads=16
            )
            print("✅ Standalone LLM model loaded.")
        except Exception as e:
            print(f"🔥 Failed to load model on startup: {e}")
            llm = None

@app.get("/health")
def health():
    if llm is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {"status": "ok", "model_status": "loaded"}

@app.post("/completion")
def completion(request: dict):
    """Performs inference directly without any external dependencies."""
    if llm is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    prompt = request.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt not provided.")

    try:
        output = llm(prompt, max_tokens=256, stop=["User:", "\n"], echo=False)
        content = output["choices"][0]["text"].strip()
        return {"content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

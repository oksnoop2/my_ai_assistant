# commit-helper-service/service.py
# A simplified, standalone LLM service for internal tooling.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel  # Import BaseModel from pydantic
from llama_cpp import Llama
import os

app = FastAPI()

# --- Define a Pydantic model for the request body ---
class CompletionRequest(BaseModel):
    prompt: str

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
def completion(payload: CompletionRequest): # Use the Pydantic model for validation
    """Performs inference directly without any external dependencies."""
    if llm is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    # Access the prompt directly from the validated payload object
    prompt = payload.prompt
    # The 'if not prompt:' check is no longer needed as Pydantic handles it

    try:
        output = llm(prompt, max_tokens=256, stop=["User:", "\n"], echo=False)
        content = output["choices"][0]["text"].strip()
        return {"content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

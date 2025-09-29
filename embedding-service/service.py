# embedding-service/service.py
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
import torch
import gc
import os        
import requests  

app = FastAPI()

# --- Globals ---
embedding_model = None
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
CPU_DEVICE = "cpu"
GPU_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESOURCE_MANAGER_URL = os.environ["RESOURCE_MANAGER_URL"] # <-- ADD THIS

@app.get("/health")
def health():
    return {"status": "ok", "model_status": "loaded" if embedding_model else "unloaded"}

@app.post("/load")
def load_model():
    """Loads the model into system RAM (CPU), controlled by the Resource Manager."""
    global embedding_model
    if embedding_model is None:
        print(f"🚀 Loading embedding model '{MODEL_NAME}' into SYSTEM RAM...")
        embedding_model = SentenceTransformer(MODEL_NAME, device=CPU_DEVICE, trust_remote_code=True)
        print("✅ Embedding model loaded to RAM.")
    return {"status": "model_loaded"}

@app.post("/unload")
def unload_model():
    """Unloads the model from memory."""
    global embedding_model
    if embedding_model is not None:
        print("Unloading embedding model from memory...")
        del embedding_model
        embedding_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✅ Embedding model unloaded.")
    return {"status": "model_unloaded"}

@app.post("/embed")
def get_embedding(payload: dict):
    """Performs inference, moving the model to/from the GPU."""
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Resource Manager should have called /load.")
    
    text = payload.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="No text provided.")
        
    try:
        # Step 1: Acquire the GPU lock from the central manager
        print("⏳ Requesting GPU lock from Resource Manager...")
        requests.post(f"{RESOURCE_MANAGER_URL}/acquire_gpu", timeout=300).raise_for_status()
        print("✅ GPU lock acquired by Embedding Service.") # CORRECTED: Log message
        
        embedding_model.to(GPU_DEVICE)
        print("➡️  Moved embedding model to GPU.")
        
        embedding = embedding_model.encode(text).tolist()
        
        return {"embedding": embedding} # CORRECTED: Return from inside the try block

    except Exception as e:
        # Re-raise the exception so the client knows something went wrong
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Step 3: CRITICAL - Always release the lock and move model back to CPU
        embedding_model.to(CPU_DEVICE) # CORRECTED: Use the correct variable name
        print("⬅️  Moved Embedding model back to CPU.")
        
        # Release the GPU lock via the central manager
        requests.post(f"{RESOURCE_MANAGER_URL}/release_gpu", timeout=60)
        print("✅ GPU lock released by Embedding Service.")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

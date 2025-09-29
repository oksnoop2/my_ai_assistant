# resource-manager-service/service.py (Corrected Version)
import os
import sys
import time
import threading
from fastapi import FastAPI, HTTPException
import requests
from pydantic import BaseModel

# --- Pydantic Model for Request Body ---
class ModelRequest(BaseModel):
    model_name: str

# --- App Initialization ---
app = FastAPI(
    title="Resource Manager",
    description="Manages loading/unloading of AI models and GPU access.",
)

# --- NEW: GPU State Management ---
GPU_IN_USE = False
gpu_lock = threading.Lock() # A lock to protect the GPU_IN_USE flag

# --- Configuration ---
try:
    TOTAL_SYSTEM_RAM_MB = int(os.getenv("TOTAL_SYSTEM_RAM_MB", 32 * 1024))
    MODEL_METADATA = {
        "llm": { "ram_mb": int(os.getenv("LLM_RAM_MB", 8 * 1024)), "service_url": os.environ["LLM_SERVICE_URL"] },
        "tts": { "ram_mb": int(os.getenv("TTS_RAM_MB", 6 * 1024)), "service_url": os.environ["TTS_SERVICE_URL"] },
        "embedding": { "ram_mb": int(os.getenv("EMBEDDING_RAM_MB", 4 * 1024)), "service_url": os.environ["EMBEDDING_SERVICE_URL"] }
    }
except KeyError as e:
    print(f"🔥 Critical environment variable missing: {e}", file=sys.stderr)
    sys.exit(1)

# --- State Management ---
LOADED_MODELS = {}
state_lock = threading.Lock()

def get_current_ram_usage():
    return sum(MODEL_METADATA[name]['ram_mb'] for name in LOADED_MODELS)

# --- API Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/state")
def get_state():
    with state_lock, gpu_lock:
        return {
            "total_ram_mb": TOTAL_SYSTEM_RAM_MB,
            "current_ram_usage_mb": get_current_ram_usage(),
            "gpu_in_use": GPU_IN_USE,
            "loaded_models": dict(sorted(LOADED_MODELS.items(), key=lambda item: item[1], reverse=True))
        }

# --- NEW: API Endpoints for GPU Locking ---

@app.post("/acquire_gpu")
def acquire_gpu():
    global GPU_IN_USE
    while True:
        with gpu_lock:
            if not GPU_IN_USE:
                GPU_IN_USE = True
                print("✅ GPU lock acquired.")
                return {"status": "gpu lock acquired"}
        time.sleep(0.1)

@app.post("/release_gpu")
def release_gpu():
    global GPU_IN_USE
    with gpu_lock:
        GPU_IN_USE = False
    print("✅ GPU lock released.")
    return {"status": "gpu lock released"}

# --- Existing /request_model endpoint (no changes needed) ---
@app.post("/request_model")
def request_model(payload: ModelRequest):
    model_name = payload.model_name
    if model_name not in MODEL_METADATA:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found in registry.")

    with state_lock:
        if model_name in LOADED_MODELS:
            LOADED_MODELS[model_name] = time.time()
            return {"status": f"model '{model_name}' is already loaded"}

        required_ram = MODEL_METADATA[model_name]['ram_mb']
        
        while (TOTAL_SYSTEM_RAM_MB - get_current_ram_usage()) < required_ram:
            if not LOADED_MODELS:
                raise HTTPException(status_code=507, detail="Not enough RAM to load model.")
            lru_model_name = min(LOADED_MODELS, key=LOADED_MODELS.get)
            unload_url = f"{MODEL_METADATA[lru_model_name]['service_url']}/unload"
            try:
                requests.post(unload_url, timeout=60)
                del LOADED_MODELS[lru_model_name]
            except requests.RequestException as e:
                print(f"🔥 Failed to unload model '{lru_model_name}': {e}.")


        load_url = f"{MODEL_METADATA[model_name]['service_url']}/load"
        try:
            response = requests.post(load_url, timeout=180)
            response.raise_for_status()
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model at {load_url}: {e}")
        
        LOADED_MODELS[model_name] = time.time()

    return {"status": f"model '{model_name}' was loaded successfully"}

# llm-service/service.py (Corrected Version)
from fastapi import FastAPI, HTTPException
from llama_cpp import Llama
import gc
import os
import requests
import torch 

app = FastAPI()

# --- Globals ---
llm = None
MODEL_PATH = "/models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
# ADD THIS: Get the resource manager URL from environment variables
RESOURCE_MANAGER_URL = os.environ["RESOURCE_MANAGER_URL"]

@app.get("/health")
def health():
    status = "loaded" if llm is not None else "unloaded"
    return {"status": "ok", "model_status": status}

@app.post("/load")
def load_model():
    """Loads the model into memory, controlled by the Resource Manager."""
    global llm
    if llm is None:
        print("🚀 Loading LLM model into memory (GPU)...")
        try:
            llm = Llama(
                model_path=MODEL_PATH,
                n_gpu_layers=-1,
                n_ctx=4096,
                verbose=True,
                n_threads=16
            )
            print("✅ LLM model loaded.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
    return {"status": "model_already_loaded" if llm else "model_loaded"}

@app.post("/unload")
def unload_model():
    """Unloads the model from memory."""
    global llm
    if llm is not None:
        print("Unloading LLM model from memory...")
        del llm
        llm = None
        gc.collect()
        # llama-cpp handles its own VRAM release, but an explicit cache clear is good practice
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✅ LLM model unloaded.")
    return {"status": "model_unloaded"}

@app.post("/completion")
def completion(request: dict):
    """Performs inference. Uses the central resource manager for GPU locking."""
    if llm is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. The Resource Manager should have called /load first.")
    
    prompt = request.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt not provided.")

    try:
        # Step 1: Acquire the GPU lock from the central manager
        print("⏳ Requesting GPU lock from Resource Manager...")
        requests.post(f"{RESOURCE_MANAGER_URL}/acquire_gpu", timeout=300).raise_for_status()
        print("✅ GPU lock acquired by LLM Service.")

        # Step 2: Perform inference (model is already on GPU)
        output = llm(prompt, max_tokens=256, stop=["User:", "\n"], echo=False)
        content = output["choices"][0]["text"].strip()
        return {"content": content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Step 3: CRITICAL - Always release the lock
        # Note: We do not move the model back to CPU here. It stays on the GPU.
        requests.post(f"{RESOURCE_MANAGER_URL}/release_gpu", timeout=60)
        print("✅ GPU lock released by LLM Service.")

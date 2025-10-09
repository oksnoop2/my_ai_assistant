# in ./llm-service/service.py

import os
import sys
import logging
import requests
import gc
import torch
import base64
import threading
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# --- Llama CPP ---
from llama_cpp import Llama
from llama_cpp.llama_chat_format import LlamaChatCompletionRequestMessage

# --- Logging and Configuration ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(processName)s] %(message)s")
logger = logging.getLogger(__name__)

# --- NEW: Model Registry ---
# Map logical model names to their file paths. This is the single source of truth.
MODEL_REGISTRY = {
    "text": "/models/Hermes-3-Llama-3.1-8B-Q4_K_M.gguf",
    "vision": "/models/Qwen2.5-Omni-3B-Q6_K.gguf"
}

try:
    RESOURCE_MANAGER_URL = os.environ["RESOURCE_MANAGER_URL"]
except KeyError as e:
    logger.error(f"🔥 Critical environment variable missing: {e}")
    sys.exit(1)

# --- API Models ---
class CompletionRequest(BaseModel):
    model_name: str  # Client MUST specify which model to use
    prompt: str
    image_base64: Optional[str] = None

# --- Global State ---
llm: Optional[Llama] = None
current_model_name: Optional[str] = None
# An internal lock to prevent race conditions when two requests want to swap the model simultaneously
model_lock = threading.Lock()

# --- FastAPI App ---
app = FastAPI(title="Dynamic Multi-Model LLM Service")

# --- Helper Functions for Model Management ---
def _unload_model():
    """Safely unload the current model from memory."""
    global llm, current_model_name
    if llm is not None:
        logger.info(f"Unloading model: {current_model_name}")
        del llm
        llm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("✅ Model unloaded and VRAM cleared.")
    current_model_name = None

def _load_model(model_name_to_load: str):
    """Loads the specified model into memory, handling swapping if necessary."""
    global llm, current_model_name
    
    if current_model_name == model_name_to_load:
        logger.info(f"✅ Model '{model_name_to_load}' is already loaded.")
        return

    # A different model is loaded or no model is loaded, so we need to act.
    _unload_model()

    model_path = MODEL_REGISTRY.get(model_name_to_load)
    if not model_path or not Path(model_path).exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_name_to_load}' not found in registry or file system.")
    
    logger.info(f"🚀 Loading model '{model_name_to_load}' from {model_path}...")
    try:
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=4096,
            verbose=True,
            chat_format="chatml" # Common for modern models like Qwen and Hermes
        )
        current_model_name = model_name_to_load
        logger.info(f"✅ Model '{current_model_name}' loaded successfully.")
    except Exception as e:
        logger.error(f"🔥 Failed to load model '{model_name_to_load}': {e}", exc_info=True)
        current_model_name = None
        raise HTTPException(status_code=500, detail=f"Failed to load model {model_name_to_load}")

# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.get("/health")
def health():
    # Health now just means the service is running.
    return {"status": "ok", "current_model": current_model_name}

@app.post("/completion")
def completion(payload: CompletionRequest):
    """
    Handles inference by dynamically loading/swapping the required model.
    """
    if payload.model_name not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Invalid model_name: '{payload.model_name}'. Available models: {list(MODEL_REGISTRY.keys())}")

    try:
        logger.info("⏳ Requesting main GPU lock from Resource Manager...")
        requests.post(f"{RESOURCE_MANAGER_URL}/acquire_gpu", timeout=300).raise_for_status()
        logger.info("✅ GPU lock acquired.")

        # Acquire the internal lock to safely check and swap the model
        with model_lock:
            _load_model(payload.model_name)
            
            # Ensure the model is actually loaded before proceeding
            if llm is None:
                raise HTTPException(status_code=500, detail="Model could not be loaded for inference.")
            
            # --- Perform Inference ---
            if payload.image_base64 and payload.model_name == "vision":
                logger.info("🖼️  Processing vision-language request...")
                image_uri = f"data:image/jpeg;base64,{payload.image_base64}"
                msg = LlamaChatCompletionRequestMessage(role="user", content=[{"type": "image_url", "image_url": {"url": image_uri}}, {"type": "text", "text": payload.prompt}])
                response = llm.create_chat_completion(messages=[msg.model_dump()])
                content = response['choices'][0]['message']['content']
            elif payload.image_base64 and payload.model_name != "vision":
                 raise HTTPException(status_code=400, detail=f"Received an image but the requested model '{payload.model_name}' does not support vision.")
            else:
                logger.info("📝 Processing text-only request...")
                output = llm(payload.prompt, max_tokens=1024, stop=["<|im_end|>", "User:"], echo=False)
                content = output["choices"][0]["text"].strip()
        
        return {"content": content.strip()}

    except Exception as e:
        # Re-raise HTTPExceptions, log others
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"❌ Error during LLM inference: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        requests.post(f"{RESOURCE_MANAGER_URL}/release_gpu", timeout=60)
        logger.info("✅ GPU lock released.")

from fastapi import FastAPI, HTTPException
from llama_cpp import Llama
import gc
import os # <--- THIS IS THE FIX. Add this line.

app = FastAPI()

# Global variable to hold the model object. Starts as None.
llm = None
# CHANGE THIS LINE:
# FROM: MODEL_PATH = "/models/gguf-models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
# TO this simpler, configurable version:
MODEL_FILE_NAME = os.getenv("MODEL_FILE_NAME", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf")
MODEL_PATH = f"/models/{MODEL_FILE_NAME}"

@app.get("/health")
def health():
    status = "loaded" if llm is not None else "unloaded"
    return {"status": "ok", "model_status": status}

@app.post("/load")
def load_model():
    global llm
    if llm is None:
        print("🚀 Loading LLM model into VRAM...")
        try:
            llm = Llama(
                model_path=MODEL_PATH,
                n_gpu_layers=-1,  # Offload all possible layers to GPU
                n_ctx=4096,
                verbose=True
            )
            print("✅ LLM model loaded.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
    return {"status": "model_already_loaded" if llm else "model_loaded"}

@app.post("/unload")
def unload_model():
    global llm
    if llm is not None:
        print("卸载 Unloading LLM model from VRAM...")
        del llm
        llm = None
        gc.collect() # Trigger garbage collection
        print("✅ LLM model unloaded.")
    return {"status": "model_unloaded"}

@app.post("/completion")
def completion(request: dict):
    if llm is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please call /load first.")
    
    prompt = request.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt not provided.")

    try:
        output = llm(prompt, max_tokens=256, stop=["User:", "\n"], echo=False)
        content = output["choices"][0]["text"].strip()
        return {"content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

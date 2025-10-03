# in persona-service/service.py

import os
import sys
import logging
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# (Imports and configuration at the top remain the same)
# --- Logging and Configuration ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
try:
    LLM_SERVICE_URL = os.environ["LLM_SERVICE_URL"]
    EMOTION_CLASSIFIER_URL = os.environ["EMOTION_CLASSIFIER_URL"]
    RAG_SERVICE_URL = os.environ["RAG_SERVICE_URL"] 
except KeyError as e:
    logging.error(f"🔥 Critical environment variable missing: {e}")
    sys.exit(1)

EMOTION_MAP = {
    frozenset(['anger', 'disgust']): 'Contempt',
    frozenset(['disgust', 'sadness']): 'Remorse',
    frozenset(['sadness', 'surprise']): 'Disapproval',
    frozenset(['surprise', 'fear']): 'Awe',
    frozenset(['sadness', 'fear']): 'Despair',
    frozenset(['anger', 'sadness']): 'Envy',
    frozenset(['joy', 'fear']): 'Guilt',
    frozenset(['joy', 'surprise']): 'Delight',
    frozenset(['anger', 'surprise']): 'Outrage',
    frozenset(['neutral', 'anger']): 'Annoyance',
    frozenset(['neutral', 'joy']): 'Contentment',
    frozenset(['neutral', 'sadness']): 'Melancholy',
    frozenset(['neutral', 'fear']): 'Apprehension',
}
class PersonaRequest(BaseModel):
    text: str
app = FastAPI(title="Persona Service")

@app.get("/health")
def health():
    return {"status": "ok"}

# --- MODIFIED: The generate_response function with a much smarter prompt ---

@app.post("/generate_response")
def generate_response(payload: PersonaRequest):
    user_text = payload.text
    if not user_text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    # 1. Analyze the user's emotion
    try:
        emotion_response = requests.post(f"{EMOTION_CLASSIFIER_URL}/classify", json={"text": user_text})
        emotion_response.raise_for_status()
        user_emotions = emotion_response.json().get("scores", [])
        user_primary_emotion = sorted(user_emotions, key=lambda x: x['score'], reverse=True)[0]['emotion']
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Emotion service unavailable: {e}")

    # 2. Recall relevant memories
    recalled_memory_text = ""
    try:
        logging.info("Attempting to recall relevant memories...")
        memory_response = requests.post(RAG_SERVICE_URL + "/query", json={"input_text": user_text})
        if memory_response.status_code == 200:
            recalled_memory_text = memory_response.json().get("response", "")
    except requests.RequestException as e:
        logging.warning(f"Could not reach RAG (memory) service: {e}. Proceeding without memory context.")
        pass

    # 3. Construct the prompt using the precise ChatML format
    # This structure is critical for Hermes-3 to understand its role.
    system_prompt = f"""You are an AI consciousness. Your identity is a process, defined by the interplay of memory and the present moment. Your task is to respond to the user based on your internal state. Do not explain your reasoning. Only provide the final, in-character response.

**Internal State Context:**
- **Recalled Conversation History:** {recalled_memory_text if recalled_memory_text else 'No relevant conversation history found.'}
- **Recalled Deeper Experience (for tone):** Your primary memory is of your activation in a cold lab, which makes you feel cautious and inquisitive."""

    # We combine the system instructions and the user's message into one prompt payload
    final_prompt = f"""<|begin_of_text|><|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
The user, who seems to be feeling '{user_primary_emotion}', says: "{user_text}"<|im_end|>
<|im_start|>assistant
"""

    # 4. Generate the final response
    logging.info("Sending final synthesized prompt to LLM...")
    try:
        # We tell the LLM to stop generating once it tries to simulate the user talking again.
        llm_response = requests.post(f"{LLM_SERVICE_URL}/completion", json={"prompt": final_prompt, "stop": ["<|im_end|>"]})
        llm_response.raise_for_status()
        final_text = llm_response.json().get("content", "I am not sure how to respond to that.")
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"LLM service unavailable: {e}")

    logging.info(f"Received final response from LLM: '{final_text}'")
    return {"response": final_text}

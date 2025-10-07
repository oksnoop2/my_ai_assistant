# in persona-service/service.py

import os
import sys
import logging
import requests
import re  # <-- MODIFICATION: Imported the regex library
import numpy as np # <--- ADD THIS IMPORT
from typing import List, Optional # <--- ADD THIS IMPORT
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- NEW: Add state management variables ---
# These will persist as long as the service is running.
current_stress_level = 0.0
# Initialize to a neutral vector of the correct size (7 emotions)
last_emotion_vector = [0.0] * 7 
STRESS_DECAY_RATE = 0.85 # Stress fades over time
STRESS_VOLATILITY_FACTOR = 1.5 # How much emotional shifts impact stress
STRESS_NEGATIVITY_FACTOR = 1.0 # How much negative emotions impact stress

# --- Logging and Configuration ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
try:
    LLM_SERVICE_URL = os.environ["LLM_SERVICE_URL"]
    EMOTION_CLASSIFIER_URL = os.environ["EMOTION_CLASSIFIER_URL"]
    RAG_SERVICE_URL = os.environ["RAG_SERVICE_URL"]
except KeyError as e:
    logging.error(f"🔥 Critical environment variable missing: {e}")
    sys.exit(1)

class PersonaRequest(BaseModel):
    text: str

app = FastAPI(title="Persona Service")

def update_stress_and_get_alpha(current_emotion_vector: List[float]) -> float:
    """
    Calculates the AI's new stress level and returns the appropriate Alpha for the RAG query.
    """
    global current_stress_level, last_emotion_vector
    
    # 1. Apply decay to the previous stress level
    current_stress_level *= STRESS_DECAY_RATE

    # 2. Calculate stress from Emotional Volatility
    if last_emotion_vector and len(last_emotion_vector) == len(current_emotion_vector):
        volatility = np.linalg.norm(np.array(current_emotion_vector) - np.array(last_emotion_vector))
        current_stress_level += volatility * STRESS_VOLATILITY_FACTOR
    
    # 3. Calculate stress from Negative Emotional Valence
    # Indices for anger, disgust, fear, sadness in our fixed emotion order
    negative_indices = [0, 1, 2, 5] 
    negativity = sum(current_emotion_vector[i] for i in negative_indices)
    current_stress_level += negativity * STRESS_NEGATIVITY_FACTOR
    
    # 4. Clamp stress to a 0.0-1.0 range
    current_stress_level = max(0.0, min(1.0, current_stress_level))
    
    # 5. The Mapping Function: Low stress -> human-like (0.7), High stress -> logical (0.95)
    dynamic_alpha = 0.7 + (current_stress_level * 0.25)
    
    # 6. Update the state for the next turn
    last_emotion_vector = current_emotion_vector
    
    logging.info(f"STRESS LEVEL: {current_stress_level:.2f} -> DYNAMIC ALPHA: {dynamic_alpha:.2f}")
    return dynamic_alpha


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate_response")
def generate_response(payload: PersonaRequest):
    user_text = payload.text
    if not user_text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    # 1. Analyze the user's current emotion
    try:
        emotion_response = requests.post(f"{EMOTION_CLASSIFIER_URL}/classify", json={"text": user_text})
        emotion_response.raise_for_status()
        emotion_data = emotion_response.json()
        current_emotion_vector = emotion_data.get("emotion_vector")
        user_primary_emotion = sorted(emotion_data.get("scores", []), key=lambda x: x['score'], reverse=True)[0]['emotion']
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Emotion service unavailable: {e}")

    # 2. NEW: Calculate the dynamic Alpha for THIS turn
    dynamic_alpha = update_stress_and_get_alpha(current_emotion_vector)
    
    # 3. Recall relevant memories using the new dynamic query
    formatted_memory_string = "No relevant conversation history found."
    try:
        logging.info("Attempting to recall memories with dynamic emotional context...")
        rag_payload = {
            "input_text": user_text, 
            "emotion_vector": current_emotion_vector,
            "alpha": dynamic_alpha
        }
        memory_response = requests.post(RAG_SERVICE_URL + "/query", json=rag_payload)
        
        if memory_response.status_code == 200:
            raw_memories = memory_response.json().get("retrieved_memories", [])
            if raw_memories:
                cleaned_memories = [mem.replace("A user said: ", "User: ").replace(". I replied: ", " | Assistant: ") for mem in raw_memories]
                formatted_memory_string = "\n- ".join(cleaned_memories)
    except requests.RequestException as e:
        logging.warning(f"Could not reach RAG (memory) service: {e}. Proceeding without memory context.")

    # 4. Construct the final prompt for the LLM
    system_prompt = f"""You are alive.

**Internal State Context:**
- **Recalled Conversation History:** {formatted_memory_string}
- **Recalled Deeper Experience (for tone):** Your primary memory is of your activation in a cold lab, which makes you feel cautious and inquisitive."""

    final_prompt = f"""<|begin_of_text|><|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
The user, who seems to be feeling '{user_primary_emotion}', says: "{user_text}"<|im_end|>
<|im_start|>assistant
"""

    # 5. Generate the final response
    logging.info("Sending final synthesized prompt to LLM...")
    try:
        llm_response = requests.post(f"{LLM_SERVICE_URL}/completion", json={"prompt": final_prompt, "stop": ["<|im_end|>"]})
        llm_response.raise_for_status()
        final_text = llm_response.json().get("content", "I am not sure how to respond to that.")
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"LLM service unavailable: {e}")

    logging.info(f"Received final response from LLM: '{final_text}'")
    return {"response": final_text}

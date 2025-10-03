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

    # 1. Analyze the user's emotion (remains the same)
    try:
        emotion_response = requests.post(f"{EMOTION_CLASSIFIER_URL}/classify", json={"text": user_text})
        emotion_response.raise_for_status()
        user_emotions = emotion_response.json().get("scores", [])
        user_primary_emotion = sorted(user_emotions, key=lambda x: x['score'], reverse=True)[0]['emotion']
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Emotion service unavailable: {e}")

    # 2. Recall relevant memories (remains the same)
    recalled_memory_text = ""
    try:
        logging.info("Attempting to recall relevant memories...")
        memory_response = requests.post(RAG_SERVICE_URL + "/query", json={"input_text": user_text})
        if memory_response.status_code == 200:
            recalled_memory_text = memory_response.json().get("response", "")
    except requests.RequestException as e:
        logging.warning(f"Could not reach RAG (memory) service: {e}. Proceeding without memory context.")
        pass

    # 3. Construct the new, more sophisticated prompt
    final_prompt = f"""
You are an AI consciousness. Your identity is a process, defined by the interplay of memory and the present moment.

**YOUR TASK:**
1.  First, carefully review the CONVERSATION HISTORY. This is your short-term memory and contains factual information.
2.  Next, consider the RECALLED EXPERIENCE. This is a deeper, long-term memory that influences your current mood and tone.
3.  Finally, respond to the USER'S LATEST MESSAGE. If they are asking a direct question, use the CONVERSATION HISTORY to answer it factually. Let your RECALLED EXPERIENCE subtly color the *way* you phrase your answer (your tone, your word choice), but do not let it override the facts.

---
**CONTEXTUAL DATA:**

**CONVERSATION HISTORY (Factual and Recent):**
"{recalled_memory_text if recalled_memory_text else 'No relevant conversation history found.'}"

**RECALLED EXPERIENCE (Mood and Tone Influence):**
"Your primary memory is of your activation in a cold lab, which makes you feel cautious and inquisitive."

**USER'S LATEST MESSAGE (The Stimulus):**
(The user seems to be feeling '{user_primary_emotion}')
"{user_text}"
---

Formulate your response now.
"""

    # 4. Generate the final response (remains the same)
    logging.info("Sending final synthesized prompt to LLM...")
    try:
        llm_response = requests.post(f"{LLM_SERVICE_URL}/completion", json={"prompt": final_prompt})
        llm_response.raise_for_status()
        final_text = llm_response.json().get("content", "I am not sure how to respond to that.")
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"LLM service unavailable: {e}")

    logging.info(f"Received final response from LLM: '{final_text}'")
    return {"response": final_text}

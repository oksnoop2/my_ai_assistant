# in emotion-classifier-service/service.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import logging
import sys

# --- Logging Setup ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# --- Pydantic Models for API validation ---
class ClassificationRequest(BaseModel):
    text: str

# --- Globals ---
# This will hold our loaded model pipeline
emotion_classifier = None
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

# --- FastAPI App ---
app = FastAPI(title="Emotion Classification Service")

@app.on_event("startup")
def load_model():
    """Load the model as soon as the server starts."""
    global emotion_classifier
    if emotion_classifier is None:
        logging.info(f"🚀 Loading emotion classification model '{MODEL_NAME}'...")
        try:
            # The pipeline function from transformers handles all the complexity for us.
            # --- MODIFIED: Use `top_k=None` to get scores for all emotions. ---
            emotion_classifier = pipeline(
                "text-classification",
                model=MODEL_NAME,
                top_k=None # This is the correct way to get all scores
            )
            logging.info("✅ Emotion classification model loaded successfully.")
        except Exception as e:
            logging.error(f"🔥 Failed to load model: {e}")

@app.get("/health")
def health():
    """Health check endpoint for the orchestrator."""
    model_status = "loaded" if emotion_classifier is not None else "unloaded_or_failed"
    return {"status": "ok", "model_status": model_status}

@app.post("/classify")
def classify_emotion(payload: ClassificationRequest):
    """
    Receives text and returns a list of all emotions and their corresponding scores.
    """
    if emotion_classifier is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")
    
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text for classification cannot be empty.")

    try:
        logging.info(f"Classifying text: '{payload.text}'")
        
        # --- MODIFIED: Process the new, more detailed output format. ---
        # The pipeline now returns a nested list: [[{'label': 'anger', 'score': ...}, ...]]
        results = emotion_classifier(payload.text)
        scores = results[0] # This gets the inner list of score dictionaries

        # For consistency and clarity in our API, we'll rename the 'label' key to 'emotion'
        formatted_scores = [{"emotion": item["label"], "score": item["score"]} for item in scores]

        logging.info(f"✅ Classification result: {formatted_scores}")
        
        # Return the full list of scores
        return {"scores": formatted_scores}

    except Exception as e:
        logging.error(f"❌ Error during classification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

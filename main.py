import io
import json
import logging
import pickle
import re
import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import Dict

# --- Pydantic Schemas for API Data Structure ---

class SentimentRequest(BaseModel):
    """Schema for the input text."""
    text: str = Field(..., min_length=1, description="The text to be analyzed.")

class PredictionResponse(BaseModel):
    """Schema for the prediction response."""
    sentiment: str = Field(..., description="The predicted sentiment (e.g., 'Positive', 'Negative').")
    confidence: float = Field(..., ge=0.0, le=1.0, description="The model's confidence score for the prediction.")
    all_scores: Dict[str, float] = Field(..., description="Confidence scores for all sentiment categories.")

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Twitter Sentiment Analysis API",
    description="An API to predict the sentiment of a given text using a trained LSTM model.",
    version="1.0.0"
)

# --- Globals for Models and Preprocessors ---
model = None
tokenizer = None
label_encoder = None
MAX_SEQUENCE_LENGTH = 50  # This must match the value used during training

# --- Startup Event to Load Models ---
@app.on_event("startup")
async def startup_event():
    """
    Load all necessary models and preprocessors on application startup.
    """
    global model, tokenizer, label_encoder
    try:
        # Load the trained Keras model
        model = load_model('models/best_model.h5')
        logger.info("✅ Keras sentiment model loaded successfully.")

        # Load the fitted tokenizer
        with open('models/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        logger.info("✅ Tokenizer loaded successfully.")

        # Load the fitted label encoder
        with open('models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        logger.info("✅ Label encoder loaded successfully.")

    except Exception as e:
        logger.error(f"❌ Error during model loading: {e}")
        # Prevent the app from running if essential assets fail to load
        model = None
        tokenizer = None
        label_encoder = None

# --- Helper Function for Text Preprocessing ---
def preprocess_text(text: str) -> np.ndarray:
    """
    Cleans, tokenizes, and pads the input text to prepare it for the model.
    This logic must exactly match the preprocessing from the training notebook.
    """
    # 1. Clean the text
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s!?.,]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # 2. Tokenize the text
    sequence = tokenizer.texts_to_sequences([text])

    # 3. Pad the sequence
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    
    return padded_sequence

# --- API Endpoints ---
@app.get("/")
async def root():
    """Root endpoint with a welcome message."""
    return {"message": "Welcome to the Sentiment Analysis API! Visit /docs for more info."}

@app.post("/predict/", response_model=PredictionResponse)
async def predict(request: SentimentRequest):
    """
    Accepts text input and returns the predicted sentiment and confidence scores.
    """
    if not all([model, tokenizer, label_encoder]):
        raise HTTPException(status_code=503, detail="One or more models are not loaded. Please check server logs.")

    try:
        # Preprocess the input text
        processed_text = preprocess_text(request.text)
        
        # Make prediction
        prediction_probs = model.predict(processed_text)[0]
        
        # Get the top prediction
        predicted_index = np.argmax(prediction_probs)
        confidence = prediction_probs[predicted_index]
        predicted_sentiment = label_encoder.inverse_transform([predicted_index])[0]

        # Create a dictionary of all scores with their labels
        all_scores = {label_encoder.inverse_transform([i])[0]: float(prob) for i, prob in enumerate(prediction_probs)}
        
        return {
            "sentiment": predicted_sentiment,
            "confidence": float(confidence),
            "all_scores": all_scores
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")

# --- Uvicorn Runner ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

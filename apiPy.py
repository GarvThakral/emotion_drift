from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import mlflow.pyfunc
from transformers import AutoTokenizer
from datasets import Dataset
import logging

# ---- Logger Setup ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Global Variables ----
app = FastAPI()
model = None
tokenizer = None
model_name = "distilbert-base-uncased"


# ---- Data Model for Incoming Payload ----
class InputPayload(BaseModel):
    comments: List[str]


# ---- Preprocessing Functions ----
def tokenize_input(text: str) -> Dict[str, List[int]]:
    return tokenizer(text, padding="max_length", truncation=True, max_length=100)


def preprocessing_data(comments: List[str]) -> List[Dict[str, List[int]]]:
    return [tokenize_input(comment) for comment in comments]


# ---- App Startup Event ----
@app.on_event("startup")
async def load_resources():
    global model, tokenizer
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    model = mlflow.pyfunc.load_model("models:/emotion_classifier/Production")
    logger.info("Resources loaded successfully.")


# ---- Root Health Check ----
@app.get("/")
def root():
    return {"message": "Emotion classifier is running."}


# ---- Prediction Endpoint ----
@app.post("/predict")
async def predict(payload: InputPayload):
    preprocessed = preprocessing_data(payload.comments)

    # Convert to Hugging Face dataset
    ds = Dataset.from_list(preprocessed)

    # Convert to TF dataset
    tf_ds = ds.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        shuffle=False,
        batch_size=2,
    )

    # Make predictions
    preds = model.predict(tf_ds)

    # Return in JSON serializable format
    return {"predictions": preds.tolist()}
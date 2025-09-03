from fastapi import FastAPI
from itertools import islice
from youtube_comment_downloader import *
from transformers import AutoTokenizer, TFDistilBertForSequenceClassification
from typing import Tuple, List, Dict
from pydantic import BaseModel
from datasets import Dataset
import tensorflow as tf
import requests


app = FastAPI()

# Load model + tokenizer
model = TFDistilBertForSequenceClassification.from_pretrained("GarvThakral/EDM")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

compile_config = {
    "optimizer": "adam",
    "loss": tf.keras.losses.BinaryCrossentropy(from_logits=True),
    "metrics": ["binary_accuracy", "AUC", "Precision", "Recall"],
}
model.compile(**compile_config)

labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral"
]

@app.get("/working")
def working():
    return {"status": "Healthy"}

def fetch_comments(video_url: str, num_comments: int):
    lambda_url = "https://YOUR-LAMBDA-URL.lambda-url.us-east-1.on.aws/"  
    
    payload = {
        "video_url": video_url,
        "num_comments": num_comments
    }
    
    try:
        response = requests.post(lambda_url, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        comments = data.get('comments', [])
        
        if comments:
            print("Sample Comment : " + comments[0])
        
        return comments
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling Lambda: {e}")

def tokenize_input(example: dict) -> dict:
    tokenized = tokenizer(
        example, padding="max_length", truncation=True, max_length=100
    )
    return dict(tokenized)

def preprocessing_data(comments: list) -> Tuple[List[Dict[str, list]], List[str]]:
    comments_updated = [tokenize_input(comment) for comment in comments]
    print("Original comment : " + comments[0])
    print("Preprocessed comment : ", comments_updated[0])
    return comments_updated, comments

def predict(preprocessed_comments, comments_orig):
    ds = Dataset.from_list(preprocessed_comments)
    tf_ds = ds.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        batch_size=8,
    )
    result = model.predict(tf_ds)
    logits_tensor = tf.convert_to_tensor(result.logits)
    probs = tf.sigmoid(logits_tensor)
    return probs, comments_orig

def check_result(probs, comments_orig):
    conv_probs = tf.cast(probs > 0.2, dtype=tf.float32).numpy()
    pred_probs = []
    for i, x in enumerate(comments_orig):
        print(f"{i+1} : {x}")
        emotion_string = "Emotions found : " + " ".join(
            [labels[j] for j, pred in enumerate(conv_probs[i]) if pred == 1]
        )
        pred_probs.append(emotion_string)
        print(emotion_string)
    return {"comments": comments_orig, "predictions": pred_probs}

class PredictionRequest(BaseModel):
    video_url: str
    num_comments: int

@app.post("/makePred")
def make_prediction(req: PredictionRequest):
    print("Starting Pipeline")
    print("Fetching comments")
    comments = fetch_comments(req.video_url, req.num_comments)
    print("Preprocessing Comments")
    processed_comments, comments_orig = preprocessing_data(comments)
    result, comments_orig = predict(processed_comments, comments_orig)
    return check_result(result, comments_orig)

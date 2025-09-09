from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import mlflow.pyfunc
from transformers import AutoTokenizer
from datasets import Dataset
from typing import Tuple,Dict,List
import json
from transformers import TFDistilBertForSequenceClassification

model_name = "distilbert-base-uncased"


# ---- Data Model for Incoming Payload ----
class InputPayload(BaseModel):
    comments: List[str]


# ---- Preprocessing Functions ----
def tokenize_input(text: str) -> Dict[str, List[int]]:
    return tokenizer(text, padding="max_length", truncation=True, max_length=100)


def preprocessing_data(comments:list)->List[Dict[str, list]]:
    # Right now its a datasets , dataset . We tokenize it and then convert it into a tf dataset
    comments_updated = [tokenize_input(comment) for comment in comments]
    print(type(comments_updated[0]))
    return comments_updated 

just_test = ["whats happening in this series","Wassup my boi"]


app = FastAPI()
print("Listening")
# Load model once when the app starts
mlflow.set_tracking_uri("http://127.0.0.1:5000")
        
compile_config = {
    "optimizer": "adam", 
    "loss": tf.keras.losses.BinaryCrossentropy(from_logits=True),
    "metrics": ["binary_accuracy", "AUC", "Precision", "Recall"]
}
model = TFDistilBertForSequenceClassification.from_pretrained("./saved_models/trained_model")
model.compile(**compile_config)

@app.get("/predict")
async def predict():
    preprocessed = preprocessing_data(just_test)
    ds = Dataset.from_list(preprocessed)

    # Convert to TF dataset
    tf_ds = ds.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        shuffle=False,
        batch_size=2,
    )

    # Make predictions
    preds = model.predict(tf_ds)
    logits_tensor = tf.convert_to_tensor(preds.logits)
    probs = tf.sigmoid(logits_tensor)
    print(probs)
    return probs

    # Return in JSON serializable format
    return {"predictions": preds.tolist()}
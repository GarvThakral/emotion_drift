from fastapi import FastAPI, Request
import mlflow.pyfunc
import tensorflow as tf
from transformers import AutoTokenizer
from datasets import Dataset
from typing import Tuple,Dict,List
import json
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Preprocessing our data and making it training ready
def tokenize_input(example:dict)->dict:
    tokenized = tokenizer(example,padding = "max_length",truncation=True,max_length = 100)
    return dict(tokenized)

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
model = mlflow.pyfunc.load_model("models:/emotion_classifier/Production")

@app.get("/predict")
async def predict():
    preprocessed = preprocessing_data(just_test)
    ds = Dataset.from_list(preprocessed)
    tf_ds = ds.to_tf_dataset(
        columns = ['input_ids',"attention_mask"],
        shuffle = True,
        batch_size = 2
    )
    preds = model.predict(tf_ds)
    return preds

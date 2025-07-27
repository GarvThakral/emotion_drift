from zenml import step
import tensorflow as tf
from transformers import AutoTokenizer
import datasets
from typing import Tuple
import json
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Preprocessing our data and making it training ready
def tokenize_input(example:dict)->dict:
    tokenized = tokenizer(example,padding = "max_length",truncation=True,max_length = 100)
    return dict(tokenized)

@step
def preprocessing_data(comments:list):
    # Right now its a datasets , dataset . We tokenize it and then convert it into a tf dataset
    comments_updated = [tokenize_input(comment) for comment in comments]
    print(type(comments_updated[0]))
    return comments_updated
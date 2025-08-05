import streamlit as st
from transformers import TFDistilBertForSequenceClassification
from transformers import AutoTokenizer
import tensorflow as tf
import datasets as ds

def load_model():
    model:TFDistilBertForSequenceClassification = TFDistilBertForSequenceClassification.from_pretrained(
        "./saved_models/trained_model",
        num_labels=28,
        problem_type="multi_label_classification"
    )
    return model

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return tokenizer
    
# labels = [
#     'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
#     'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
#     'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
#     'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
#     'remorse', 'sadness', 'surprise', 'neutral'
# ]
# model:TFDistilBertForSequenceClassification = TFDistilBertForSequenceClassification.from_pretrained(
#     "./saved_models/trained_model",
#     num_labels=28,
#     problem_type="multi_label_classification"
# )
# model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def tokenize_data(example):
    example = tokenizer(example,padding = "max_length",truncation=True,max_length = 100)
    return dict(example)
def preprocess_data(exampleList):
    tokenized_example = [tokenize_data(example) for example in exampleList] 
    return tokenized_example

# model.compile(
#     optimizer="adam",
#     loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#     metrics=["binary_accuracy", "AUC", "Precision", "Recall"]
# )



# text = ['I love you' , 'you are the best' , 'ew bro get off of me']
# text = preprocess_data(text)
# datasets_ds = ds.Dataset.from_list(text)
# tf_ds = datasets_ds.to_tf_dataset(
#     columns = ['input_ids','attention_mask'],
#     batch_size=8,
#     shuffle=True
# )
# pred = model.predict(tf_ds)
# threshold = 0.6
# logits_tensor = tf.convert_to_tensor(pred.logits)
# probs = tf.sigmoid(logits_tensor)
# updatedProbs = (probs > threshold).numpy().astype(int)

# for indx,i in enumerate(text):
#     print(i)
#     for idx,y in enumerate(updatedProbs[indx]):
#         if(y == 1):
#             print(labels[idx])
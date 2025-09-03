from fastapi import FastAPI
from itertools import islice
from youtube_comment_downloader import *
from transformers import AutoTokenizer
app = FastAPI()
from typing import Tuple , List , Dict
from transformers import TFDistilBertForSequenceClassification
import tensorflow as tf
from datasets import Dataset

model = TFDistilBertForSequenceClassification.from_pretrained("GarvThakral/EDM")

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
compile_config = {
    "optimizer": 'adam',
    "loss": tf.keras.losses.BinaryCrossentropy(from_logits=True),
    "metrics": ["binary_accuracy", "AUC", "Precision", "Recall"]
}
model.compile(**compile_config)
labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
    'remorse', 'sadness', 'surprise', 'neutral'
]

@app.get('/working')
def working():
    return {"status":"Healthy"}

def fetch_comments(video_url:str , num_comments:int):
    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments_from_url(video_url, sort_by=SORT_BY_POPULAR)
    comments = [comment['text'] for comment in islice(comments, num_comments)]
    print(comments[0])
    return comments

# Preprocessing our data and making it training ready
def tokenize_input(example:dict)->dict:
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized = tokenizer(example,padding = "max_length",truncation=True,max_length = 100)
    return dict(tokenized)

def preprocessing_data(comments:list)->Tuple[List[Dict[str, list]], List[str]]:
    # Right now its a datasets , dataset . We tokenize it and then convert it into a tf dataset
    comments_updated = [tokenize_input(comment) for comment in comments]
    print(type(comments_updated[0]))
    return (comments_updated , comments)

def predict(preprocessed_comments,comments_orig):
    ds = Dataset.from_list(preprocessed_comments)

    tf_ds = ds.to_tf_dataset(
        columns=['input_ids', 'attention_mask'],
        batch_size=8,
        shuffle=True
    )

    result = model.predict(tf_ds)
    logits_tensor = tf.convert_to_tensor(result.logits)
    probs = tf.sigmoid(logits_tensor)
    return (probs , comments_orig)

def check_result(probs,comments_orig):
    conv_probs = tf.cast(probs>0.2,dtype = tf.float32).numpy()
    for i,x in enumerate(comments_orig):
        print(str(i+1)+" : ",end = "")
        print(x['text'])
        emotion_string = "Emotions found : " + " ".join([labels[j] for j,pred in enumerate(conv_probs[i]) if pred == 1])
        print(emotion_string)

# @app.post('/makePred')
def make_prediction(video_url:str , num_comments:int):
    print("Starting Pipeline")
    comments = fetch_comments(video_url,num_comments)
    (processed_comments,comments_orig) = preprocessing_data(comments)
    (result,comments_orig )= predict(processed_comments,comments_orig)
    check_result(result,comments_orig)

make_prediction('https://www.youtube.com/watch?v=aFicyfZu6pk',10)
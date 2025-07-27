from zenml import step
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification
from itertools import islice
from youtube_comment_downloader import *
from datasets import Dataset

@step
def predict(model_path,comments,num):
    model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
    model = TFDistilBertForSequenceClassification.from_pretrained(
        model_path,
        num_labels=28,
        problem_type="multi_label_classification"
    )
    model.compile(  # Must recompile before prediction
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["binary_accuracy", "AUC", "Precision", "Recall"]
    )
    ds = Dataset.from_list(comments)

    # Now convert to tf.data.Dataset
    tf_ds = ds.to_tf_dataset(
        columns=['input_ids', 'attention_mask'],
        batch_size=8,
        shuffle=True
    )
    result = model.predict(tf_ds)
    print(result)
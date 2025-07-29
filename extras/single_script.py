from zenml import step
import tensorflow as tf
from transformers import AutoTokenizer
import datasets
from typing import Tuple

ds = datasets.load_dataset("google-research-datasets/go_emotions")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Preprocessing our data and making it training ready
def tokenize_input(example):
    tokenized = tokenizer(example["text"],padding = "max_length",truncation=True,max_length = 100)
    updated_example = {"input_ids":tokenized["input_ids"],"attention_mask":tokenized["attention_mask"],"labels":example["labels"],"id":example["id"]}
    return updated_example

def one_hot_labels(example):
    one_hot_arr = [0]*28
    for i in example['labels']:
        one_hot_arr[i] = 1
    example['labels'] = one_hot_arr
    return example

ds['train'] = ds['train'].map(one_hot_labels,num_proc=16)
ds['train'] = ds['train'].map(tokenize_input,num_proc=16)
ds['test'] = ds['test'].map(tokenize_input)
ds['test'] = ds['test'].map(one_hot_labels)
ds['validation'] = ds['validation'].map(tokenize_input)
ds['validation'] = ds['validation'].map(one_hot_labels)  
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification

strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")  # or "/gpu:0" if available

# 2. Create model and optimizer inside the strategy scope
with strategy.scope():
    # Load your model (example with DistilBERT)
    model = TFDistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=28,  # Adjust based on your use case
        problem_type="multi_label_classification"
    )
    
    # Define the optimizer instance (instead of "adam")
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    
    # Define loss and metrics
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="bacc"),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="prec"),
        tf.keras.metrics.Recall(name="rec"),
    ]
    
    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        run_eagerly=False  # Set to True for debugging if needed
    )

# 3. Prepare your datasets (outside the scope)
tf_ds_train = ds['train'].to_tf_dataset(
    columns=['input_ids', 'attention_mask'],
    label_cols='labels',
    batch_size=8,
    shuffle=True
)
tf_ds_valid = ds['validation'].to_tf_dataset(
    columns=['input_ids', 'attention_mask'],
    label_cols='labels',
    batch_size=8
)

# 4. Train the model
model.fit(
    tf_ds_train,
    validation_data=tf_ds_valid,
    epochs=5,
    verbose=1
)

# 5. Save the model (optional)
model.save_pretrained("./saved_model")

# Example usage (adjust based on your setup)
# ds = {"train": train_dataset, "validation": valid_dataset}
# train_model(ds)
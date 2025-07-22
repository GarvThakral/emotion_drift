from zenml import step
import tensorflow as tf
from transformers import AutoTokenizer

# Preprocessing our data and making it training ready
def tokenize_input(example):
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized = tokenizer(example["text"],padding = "max_length",truncation=True,max_length = 180)
    updated_example = {"input_ids":tokenized["input_ids"],"attention_mask":tokenized["attention_mask"],"labels":example["labels"],"id":example["id"]}
    return updated_example

def one_hot_labels(example):
    one_hot_arr = [0]*28
    for i in example['labels']:
    one_hot_arr[i] = 1
    example['labels'] = one_hot_arr
    return example


@step
def preprocessing_data(ds):
    # Right now its a datasets , dataset . We tokenize it and then convert it into a tf dataset
    ds['train'] = ds['train'].map(one_hot_labels)
    ds['train'] = ds['train'].map(tokenize_input)
    ds['test'] = ds['test'].map(tokenize_input)
    ds['validation'] = ds['validation'].map(tokenize_input)

    #   Converting tokenized outputs into tf dataset
    ds['train'] = ds['train'].to_tf_dataset(
        columns = ['input_ids',"attention_mask"],
        label_cols = 'labels',
        output_type=tf.int64
    ).shuffle(buffer_size = 200).batch(32)

    ds['test'] = ds['test'].to_tf_dataset(
        columns = ['input_ids',"attention_mask"],
        label_cols = 'labels',
    ).shuffle(buffer_size = 200).batch(32)

    ds['validation'] = ds['validation'].to_tf_dataset(
        columns = ['input_ids',"attention_mask"],
        label_cols = 'labels',
    ).shuffle(buffer_size = 200).batch(32)
    
    return ds
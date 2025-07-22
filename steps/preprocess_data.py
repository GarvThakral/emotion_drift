from zenml import step
import tensorflow as tf
from transformers import AutoTokenizer
import datasets
from typing import Tuple

# Preprocessing our data and making it training ready
def tokenize_input(example:dict)->dict:
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized = tokenizer(example["text"],padding = "max_length",truncation=True,max_length = 180)
    updated_example = {"input_ids":tokenized["input_ids"],"attention_mask":tokenized["attention_mask"],"labels":example["labels"],"id":example["id"]}
    return updated_example

def one_hot_labels(example:dict)->dict:
    one_hot_arr = [0]*28
    for i in example['labels']:
        one_hot_arr[i] = 1
    example['labels'] = one_hot_arr
    return example


@step
def preprocessing_data(ds : datasets.dataset_dict.DatasetDict)-> Tuple[tf.data.Dataset,tf.data.Dataset,tf.data.Dataset]:
    # Right now its a datasets , dataset . We tokenize it and then convert it into a tf dataset
    ds['train'] = ds['train'].map(one_hot_labels)
    ds['train'] = ds['train'].map(tokenize_input)
    ds['test'] = ds['test'].map(tokenize_input)
    ds['validation'] = ds['validation'].map(tokenize_input)

    #   Converting tokenized outputs into tf dataset
    tf_ds_train = ds['train'].to_tf_dataset(
        columns = ['input_ids',"attention_mask"],
        label_cols = 'labels',
    ).shuffle(buffer_size = 200).batch(32)

    tf_ds_test = ds['test'].to_tf_dataset(
        columns = ['input_ids',"attention_mask"],
        label_cols = 'labels',
    ).shuffle(buffer_size = 200).batch(32)

    tf_ds_valid = ds['validation'].to_tf_dataset(
        columns = ['input_ids',"attention_mask"],
        label_cols = 'labels',
    ).shuffle(buffer_size = 200).batch(32)
    
    return (tf_ds_train,tf_ds_test,tf_ds_valid)
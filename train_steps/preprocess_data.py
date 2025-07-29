from zenml import step
import tensorflow as tf
from transformers import AutoTokenizer
import datasets
from typing import Tuple

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Preprocessing our data and making it training ready
def tokenize_input(example:dict)->dict:
    tokenized = tokenizer(example["text"],padding = "max_length",truncation=True,max_length = 100)
    updated_example = {"input_ids":tokenized["input_ids"],"attention_mask":tokenized["attention_mask"],"labels":example["labels"],"id":example["id"]}
    return updated_example

def one_hot_labels(example:dict)->dict:
    one_hot_arr = [0]*28
    for i in example['labels']:
        one_hot_arr[i] = 1
    example['labels'] = one_hot_arr
    return example


@step
def preprocessing_data(ds : datasets.dataset_dict.DatasetDict)-> datasets.dataset_dict.DatasetDict:
    # Right now its a datasets , dataset . We tokenize it and then convert it into a tf dataset
    ds['train'] = ds['train'].map(one_hot_labels,num_proc=16)
    ds['train'] = ds['train'].map(tokenize_input,num_proc=16)
    ds['test'] = ds['test'].map(tokenize_input)
    ds['test'] = ds['test'].map(one_hot_labels)
    ds['validation'] = ds['validation'].map(tokenize_input)
    ds['validation'] = ds['validation'].map(one_hot_labels)  
    tokenizer.save_pretrained("./saved_models/trained_model")
    return ds
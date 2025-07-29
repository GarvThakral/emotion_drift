from zenml import pipeline
from train_steps.ingest_data import ingest_data
from train_steps.preprocess_data import preprocessing_data
from train_steps.model_dev import compile_model
from train_steps.train_model import train_model
from train_steps.train_model_test import train_model_fixed
from train_steps.eval_model import eval_model

@pipeline
def train_pipeline():
    ds = ingest_data()
    processed_ds = preprocessing_data(ds)
    trained_path,run_id = train_model_fixed(processed_ds)
    result = eval_model(trained_path,processed_ds,run_id)
    return result
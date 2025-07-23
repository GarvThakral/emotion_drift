from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.preprocess_data import preprocessing_data
from steps.model_dev import compile_model
from steps.train_model import train_model
from steps.eval_mode import eval_model
@pipeline
def train_pipeline():
    ds = ingest_data()
    processed_ds = preprocessing_data(ds)
    trained_path = train_model(processed_ds)
    result = eval_model(trained_path,processed_ds)
    return result
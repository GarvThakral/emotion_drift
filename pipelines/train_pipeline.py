from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.preprocess_data import preprocessing_data
from steps.model_dev import compile_model
@pipeline
def train_pipeline():
    ds = ingest_data()
    (tf_ds_train,tf_ds_test,tf_ds_valid) = preprocessing_data(ds)
    compiled_model = compile_model()
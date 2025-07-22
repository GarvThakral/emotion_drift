from zenml import step

@step
def ingest_data():
  ds = datasets.load_dataset("google-research-datasets/go_emotions")
  return ds
from zenml import step
from datasets import load_dataset
import datasets
@step
def ingest_data():
  ds = load_dataset("google-research-datasets/go_emotions")
  return ds
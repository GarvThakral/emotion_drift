from transformers import TFDistilBertForSequenceClassification
import tensorflow as tf
import transformers
import tensorflow.keras as keras
from zenml import step

@step
def compile_model(
    optimizer = "adam",
    metrics = ['accuracy']
):
    model_name = "distilbert-base-uncased"
    model = TFDistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels = 28,
        problem_type = "multi_label_classification"
    )
    compile_config = {
        "optimizer": optimizer,
        "loss": keras.losses.CategoricalCrossentropy(),
        "metrics": metrics
    }
    model.compile(**compile_config)
    model.distilbert.trainable = False
    # model_path = "./saved_models/compiled_model"
    # model.save_pretrained(model_path)
    return model

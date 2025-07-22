from transformers import TFDistilBertForSequenceClassification
import tensorflow as tf
import transformers



def compile_model(
    optimizer = "adam",
    loss_function = tf.keras.losses.CategoricalCrossentropy(),
    metrics = ['accuracy']
)->TFDistilBertForSequenceClassification:
    model_name = "distilbert-base-uncased"
    model = TFDistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels = 28,
        problem_type = "multi_label_classification"
    )
    compile_config = {
        "optimizer": optimizer,
        "loss": loss_function,
        "metrics": metrics
    }
    model.compile(**compile_config)
    return model

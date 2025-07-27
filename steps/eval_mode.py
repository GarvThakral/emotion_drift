from zenml import step
from transformers import TFDistilBertForSequenceClassification
import tensorflow as tf
import mlflow
@step
def eval_model(model_path,ds,run_id):
    model = TFDistilBertForSequenceClassification.from_pretrained(
        model_path,
        num_labels=28,
        problem_type="multi_label_classification"
    )
    model.compile(  # Must recompile before evaluation
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["binary_accuracy", "AUC", "Precision", "Recall"]
    )
    tf_ds_test = ds['test'].to_tf_dataset(
        columns=['input_ids',"attention_mask"],
        label_cols="labels",
        batch_size = 8,
        shuffle = True
    )
    result = model.evaluate(tf_ds_test,return_dict = True)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    with mlflow.start_run(run_id = run_id):
        mlflow.log_metrics(result)
    return result
from zenml import step
from transformers import TFDistilBertForSequenceClassification
import tensorflow as tf

@step
def eval_model(model_path,ds):
    model = TFDistilBertForSequenceClassification.from_pretrained(
        model_path,
        num_labels=28,
        problem_type="multi_label_classification"
    )
    model.compile(  # Must recompile before evaluation
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    tf_ds_test = ds['test'].to_tf_dataset(
        columns=['input_ids',"attention_mask"],
        label_cols="labels",
        batch_size = 8,
        shuffle = True
    )
    result = model.evaluate(tf_ds_test)
    return result
from zenml import step
from transformers import TFDistilBertForSequenceClassification
import transformers
import tensorflow as tf
import datasets

@step
def train_model(ds: datasets.dataset_dict.DatasetDict)->str:
    model_name = "distilbert-base-uncased"
    model = TFDistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels = 28,
        problem_type = "multi_label_classification"
    )

    compile_config = {
        "optimizer": "adam",
        "loss": tf.keras.losses.BinaryCrossentropy(from_logits=True),
        "metrics":  ["binary_accuracy", "AUC", "Precision", "Recall"]
    }

    model.compile(**compile_config)
    model.distilbert.trainable = True
    # Converting tokenized outputs into tf dataset
    tf_ds_train = ds['train'].to_tf_dataset(
        columns=['input_ids',"attention_mask"],
        label_cols="labels",
        batch_size = 2,
        shuffle = True
    )

    tf_ds_valid = ds['validation'].to_tf_dataset(
        columns=['input_ids',"attention_mask"],
        label_cols="labels",
        batch_size = 2,
        shuffle = True
    )
    # model = TFDistilBertForSequenceClassification.from_pretrained(compiled_path)
    model.fit(
        x=tf_ds_train,
        batch_size=2,
        validation_data = tf_ds_valid,
        epochs=20
    ) # type: ignore
    model_path = "./saved_models/trained_model"
    model.save_pretrained(model_path)
    return model_path
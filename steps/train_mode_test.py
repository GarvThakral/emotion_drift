from zenml import step
from transformers import TFDistilBertForSequenceClassification
import tensorflow as tf
import datasets
import os
import glob

@step
def train_model_fixed(ds: datasets.dataset_dict.DatasetDict) -> str:
    model_name = "distilbert-base-uncased"
    model = TFDistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=28,
        problem_type="multi_label_classification"
    )
    compile_config = {
        "optimizer": "adam",
        "loss": tf.keras.losses.BinaryCrossentropy(from_logits=True),
        "metrics": ["binary_accuracy", "AUC", "Precision", "Recall"]
    }
    model.compile(**compile_config)
    model.distilbert.trainable = True

    tf_ds_train = ds['train'].to_tf_dataset(
        columns=['input_ids', "attention_mask"],
        label_cols="labels",
        batch_size=8,
        shuffle=True
    )

    tf_ds_valid = ds['validation'].to_tf_dataset(
        columns=['input_ids', "attention_mask"],
        label_cols="labels",
        batch_size=8,
        shuffle=True
    )

    # Checkpoint callback for every 3 epochs
    checkpoint_dir = "./saved_models/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "ckpt_epoch_{epoch}.weights.h5"),
        save_weights_only=True,
    )

    # Find latest checkpoint if exists
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "ckpt_epoch_*"))
    if checkpoint_files:
        # Find the latest epoch
        latest_ckpt = max(
            checkpoint_files,
            key=lambda x: int(x.split("_")[-1])
        )
        print(f"Resuming from checkpoint: {latest_ckpt}")
        model.load_weights(latest_ckpt)
    else:
        print("No checkpoint found, starting fresh.")

    model.fit(
        x=tf_ds_train,
        batch_size=8,
        validation_data=tf_ds_valid,
        epochs=20,
        callbacks=[checkpoint_callback]
    )
    model_path = "./saved_models/trained_model"
    model.save_pretrained(model_path)
    return model_path

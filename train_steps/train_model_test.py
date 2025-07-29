from zenml import step
from transformers import TFDistilBertForSequenceClassification
import tensorflow as tf
import datasets
import numpy as np
import mlflow
from typing import Tuple
@step
def train_model_fixed(ds: datasets.dataset_dict.DatasetDict) -> Tuple[str,str]:
    # Explicitly set the distribution strategy
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")  # Use "/gpu:0" if GPU is available

    # STEP 1: Calculate class weights to fix imbalance
    print("Calculating class weights...")
    train_labels = np.array(list(ds['train']['labels']))
    pos_counts = np.sum(train_labels, axis=0)
    total_samples = len(train_labels)
    
    class_weights = {}
    for i in range(28):
        if pos_counts[i] > 0:
            # Weight = total_samples / (2 * positive_samples)
            weight = total_samples / (2.0 * pos_counts[i])
        else:
            weight = 1.0
        class_weights[i] = weight
        if i < 5:  # Print first 5 for debugging
            print(f"Class {i}: {pos_counts[i]} samples, weight: {weight:.2f}")

    with strategy.scope():
        params = {
            "model_name":"distilbert-base-uncased",
            "epochs":1,
            "num_labels":28,
            "problem_type":"multi_label_classification",
            "optimizer":"adam",
            "trainable":True,
            "learning_rate":2e-5
        }
        model = TFDistilBertForSequenceClassification.from_pretrained(
            params["model_name"],
            num_labels=params["num_labels"],
            problem_type=params["problem_type"]
        )
        
        compile_config = {
            "optimizer": params["optimizer"], 
            "loss": tf.keras.losses.BinaryCrossentropy(from_logits=True),
            "metrics": ["binary_accuracy", "AUC", "Precision", "Recall"]
        }
        model.distilbert.trainable = params['trainable']
        model.compile(**compile_config)
        
        model.optimizer.learning_rate = params["learning_rate"]

    # Convert tokenized outputs into tf dataset
    tf_ds_train = ds['train'].to_tf_dataset(
        columns=['input_ids', "attention_mask"],
        label_cols="labels",
        batch_size=2,
        shuffle=True
    )

    tf_ds_valid = ds['validation'].to_tf_dataset(
        columns=['input_ids', "attention_mask"],
        label_cols="labels",
        batch_size=2,
        shuffle=True
    )

    # Train the model WITH CLASS WEIGHTS (no callbacks for now)
    model.fit(
        x=tf_ds_train,
        validation_data=tf_ds_valid,
        epochs=params['epochs'],  # Keep it short for testing
        class_weight=class_weights
    )
    
    # Save the model
    model_path = "./saved_models/trained_model"
    model.save_pretrained(model_path)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("emotion_drift_experiment_v2")

    with mlflow.start_run(run_name=params["model_name"]) as run:
        mlflow.log_params(params)
        run_id = run.info.run_id 
        mlflow.tensorflow.log_model(
            model=model,
            artifact_path="emotion_classifier_model",
        )
        mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/emotion_classifier_model",
            name="emotion_classifier"
        )

    return (model_path,run_id)
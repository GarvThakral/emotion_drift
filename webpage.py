import streamlit as st
from index import load_model, preprocess_data, tokenize_data
import tensorflow as tf
import datasets as ds

# Load and compile model
model = load_model()
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["binary_accuracy", "AUC", "Precision", "Recall"]
)

# Define emotion labels
labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
    'remorse', 'sadness', 'surprise', 'neutral'
]

# Prediction function
def make_pred(data):
    processed = preprocess_data(data)  
    dataset = ds.Dataset.from_list(processed)
    tf_dataset = dataset.to_tf_dataset(
        columns=['input_ids', 'attention_mask'],
        batch_size=1,
        shuffle=False
    )
    predictions = model.predict(tf_dataset)
    probs = tf.sigmoid(tf.convert_to_tensor(predictions.logits))
    labels_pred = (probs > 0.6).numpy().astype(int)

    emotion_output = []
    for idx, val in enumerate(labels_pred[0]):
        if val == 1:
            emotion_output.append(labels[idx])

    return emotion_output

# Streamlit UI
st.title("Emotion Detection App")

data = st.text_input(label="Enter your text here")

if st.button("Predict"):
    if data:
        emotions = make_pred(data)
        st.write("Predicted Emotions:", emotions)
    else:
        st.warning("Please enter some text to predict emotions.")

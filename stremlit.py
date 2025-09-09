import streamlit as st
from itertools import islice
from youtube_comment_downloader import *
from transformers import AutoTokenizer, TFDistilBertForSequenceClassification
from datasets import Dataset
import tensorflow as tf

# -------------------------------
# Load model + tokenizer
# -------------------------------
@st.cache_resource
def load_model():
    model = TFDistilBertForSequenceClassification.from_pretrained("GarvThakral/EDM")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    compile_config = {
        "optimizer": "adam",
        "loss": tf.keras.losses.BinaryCrossentropy(from_logits=True),
        "metrics": ["binary_accuracy", "AUC", "Precision", "Recall"],
    }
    model.compile(**compile_config)
    return model, tokenizer

model, tokenizer = load_model()

# -------------------------------
# Labels
# -------------------------------
labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral"
]

# -------------------------------
# Helpers
# -------------------------------
def fetch_comments(video_url: str, num_comments: int):
    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments_from_url(video_url, sort_by=SORT_BY_POPULAR)
    comments = [comment["text"] for comment in islice(comments, num_comments)]
    return comments

def tokenize_input(comment: str) -> dict:
    return tokenizer(comment, padding="max_length", truncation=True, max_length=100)

def predict(comments: list):
    processed = [tokenize_input(c) for c in comments]
    ds = Dataset.from_list(processed)
    tf_ds = ds.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        batch_size=8,
    )
    result = model.predict(tf_ds)
    probs = tf.sigmoid(tf.convert_to_tensor(result.logits))
    return probs.numpy()

def format_predictions(comments, probs):
    conv_probs = (probs > 0.2).astype(int)
    results = []
    for i, comment in enumerate(comments):
        detected = [labels[j] for j, pred in enumerate(conv_probs[i]) if pred == 1]
        results.append((comment, detected))
    return results

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🎭 Emotion Detection on YouTube Comments")
st.write("Enter a YouTube video URL and number of comments to analyze.")

video_url = st.text_input("YouTube Video URL", "")
num_comments = st.slider("Number of Comments", min_value=1, max_value=50, value=5)

if st.button("Analyze Comments"):
    if video_url.strip() == "":
        st.error("Please enter a valid YouTube URL")
    else:
        with st.spinner("Fetching and analyzing comments..."):
            comments = fetch_comments(video_url, num_comments)
            probs = predict(comments)
            results = format_predictions(comments, probs)

        st.success("Analysis Complete ✅")
        for i, (comment, detected) in enumerate(results, 1):
            st.write(f"**{i}. Comment:** {comment}")
            if detected:
                st.write("Emotions found: " + ", ".join(detected))
            else:
                st.write("No emotions detected (neutral).")
            st.markdown("---")

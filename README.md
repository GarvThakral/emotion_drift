# 🧠 Emotion Drift Monitor (EDM)

A real-time system to monitor emotional trends in social media comments (e.g., YouTube, Reddit), track shifts in public sentiment, and detect potential mental health signals over time.

---

## 🚀 Project Goals

- Classify emotions from text using a fine-tuned DistilBERT model.
- Analyze trends in emotional tone across time, topics, or users.
- Visualize changes in sentiment using dashboards.
- Detect and report possible emotion drift (e.g., joy → sadness).
- Use MLOps best practices (ZenML, MLflow) to manage training and inference pipelines.

---

## 💡 Features (Planned)

- ✅ Fine-tuned emotion classifier (based on GoEmotions dataset).
- ✅ Input preprocessing using `transformers` tokenizer.
- 🔲 Real-world comment ingestion (YouTube/Reddit).
- 🔲 Emotion drift detection over time (with thresholds).
- 🔲 Visualization dashboard (Streamlit/Plotly).
- 🔲 ZenML pipeline for reproducibility.
- 🔲 MLflow for experiment tracking.

---

## 🛠 Tech Stack

- **Model:** DistilBERT (`distilbert-base-uncased`)
- **Training:** TensorFlow + Keras
- **NLP:** Hugging Face Transformers, Datasets
- **Data:** GoEmotions, YouTube Comments (optional)
- **Visualization:** Streamlit, Plotly
- **MLOps:** ZenML, MLflow
- **Drift Detection:** Evidently AI (optional)

---

## 📂 Project Structure (Planned)

📈 Project Roadmap
 Load and preprocess dataset with tokenizer

 Build DistilBERT emotion classifier

 Evaluate and log runs with MLflow

 Build inference script for real-time predictions

 Add drift detection (e.g. emotion frequency shift)

 Ingest real-world comments from YouTube

 Launch interactive dashboard


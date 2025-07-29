from fastapi import FastAPI, Request
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

# Load model once when the app starts
model = mlflow.pyfunc.load_model("models:/emotion_classifier/Production")

@app.post("/predict")
async def predict(request: Request):
    # data = await request.json()
    # comments = data.get("comments", [])
    preds = model.predict(pd.Series(["whats happening in this series","Wassup my boi"]))
    return {"predictions": preds.tolist()}

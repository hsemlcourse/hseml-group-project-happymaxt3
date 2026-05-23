from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Fake News API")

model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


class NewsRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: NewsRequest):

    X = vectorizer.transform([request.text])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]

    return {
        "prediction": int(pred),
        "label": "FAKE" if pred == 1 else "REAL",
        "fake_probability": float(prob[1]),
        "real_probability": float(prob[0]),
    }
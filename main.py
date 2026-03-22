"""
Legal Clause Risk Classifier — FastAPI Server
Trained model: Logistic Regression + TF-IDF bigrams
Test F1 (weighted): 0.9066  |  Accuracy: 0.9063
"""
import re, os, joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

STOPWORDS = frozenset({
    "a","an","the","is","it","in","of","to","and","or","for","on","with","at","by","from","that",
    "this","be","are","was","were","has","have","had","but","as","if","do","did","so","its",
    "their","they","any","all","no","each","both","such","than","more","also",
    "been","between","within","upon","after","before","during","under","over","per",
    "whether","only","either","other","same","due","above","herein","hereof","hereby",
    "hereunder","thereof","thereto","whereas","provided","however","written",
})

def clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(w for w in text.split() if w not in STOPWORDS and len(w) > 2)

store = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    store["model"] = joblib.load("best_model.pkl")
    store["le"]    = joblib.load("label_encoder.pkl")
    print("Model loaded OK")
    yield
    store.clear()

app = FastAPI(
    title="Legal Risk Classifier",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SingleRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: list[str]

def _predict_one(text: str) -> dict:
    cleaned = clean(text)
    model, le = store["model"], store["le"]
    pred_id = model.predict([cleaned])[0]
    label   = le.inverse_transform([pred_id])[0]
    proba   = model.predict_proba([cleaned])[0]
    return {
        "risk":       label,
        "confidence": round(float(proba[pred_id]), 4),
        "scores": {
            cls: round(float(proba[i]), 4)
            for i, cls in enumerate(le.classes_)
        }
    }

@app.get("/health")
def health():
    return {"status": "ok", "model": "LR+TFIDF", "accuracy": 0.9063}

@app.post("/predict")
def predict(req: SingleRequest):
    if not req.text.strip():
        raise HTTPException(400, "text is empty")
    return {"input": req.text[:120], **_predict_one(req.text)}

@app.post("/predict_batch")
def predict_batch(req: BatchRequest):
    if not req.texts:
        raise HTTPException(400, "texts list is empty")
    return {
        "results": [_predict_one(t) for t in req.texts],
        "count":   len(req.texts)
    }

# ⚖️ Legal Risk Classifier API

A **FastAPI-based REST API** that classifies legal contract clauses by risk level using a trained Machine Learning model (Logistic Regression + TF-IDF bigrams).

> **Model Performance:** Test F1 (weighted): `0.9066` | Accuracy: `90.63%`

---

## 📁 Project Structure

```
legal-risk-api/
├── main.py               # FastAPI application & prediction logic
├── best_model.pkl        # Trained Logistic Regression + TF-IDF model
├── label_encoder.pkl     # Label encoder for risk categories
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker container configuration
└── railway.toml          # Railway deployment config
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### Local Setup

```bash
# Clone the repository
git clone https://github.com/Ishan-debugg/legal-risk-api.git
cd legal-risk-api

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

---

## 🐳 Docker

```bash
# Build the image
docker build -t legal-risk-api .

# Run the container
docker run -p 8000:8000 legal-risk-api
```

---

## 📡 API Endpoints

### `GET /health`

Returns the model status and accuracy.

**Response:**
```json
{
  "status": "ok",
  "model": "LR+TFIDF",
  "accuracy": 0.9063
}
```

---

### `POST /predict`

Classifies a single legal clause by risk level.

**Request Body:**
```json
{
  "text": "The party shall not be liable for any indirect or consequential damages."
}
```

**Response:**
```json
{
  "input": "The party shall not be liable for any indirect or consequential damages.",
  "risk": "High",
  "confidence": 0.9234,
  "scores": {
    "Low": 0.0312,
    "Medium": 0.0454,
    "High": 0.9234
  }
}
```

---

### `POST /predict_batch`

Classifies multiple legal clauses in a single request.

**Request Body:**
```json
{
  "texts": [
    "Either party may terminate this agreement with 30 days written notice.",
    "The licensor grants a non-exclusive, royalty-free license."
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "risk": "Medium",
      "confidence": 0.8721,
      "scores": { "Low": 0.05, "Medium": 0.8721, "High": 0.0779 }
    },
    {
      "risk": "Low",
      "confidence": 0.9102,
      "scores": { "Low": 0.9102, "Medium": 0.0654, "High": 0.0244 }
    }
  ],
  "count": 2
}
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| API Framework | FastAPI |
| ML Model | Logistic Regression + TF-IDF (scikit-learn) |
| Server | Uvicorn |
| Deployment | Docker / Railway |
| Data Validation | Pydantic v2 |

---

## 📦 Dependencies

```
fastapi==0.111.0
uvicorn==0.30.1
scikit-learn==1.5.0
joblib==1.4.2
numpy==1.26.4
pydantic==2.7.1
```

---

## ☁️ Deployment

This project is configured for deployment on [Railway](https://railway.app) via `railway.toml`. It also includes a `Dockerfile` for any container-based hosting platform (Render, Fly.io, AWS, GCP, etc.).

---

## 🔍 How It Works

1. **Text Cleaning** — Input clauses are lowercased, stripped of punctuation, and filtered through a legal-domain stopword list.
2. **Feature Extraction** — The cleaned text is transformed using TF-IDF with bigrams.
3. **Classification** — A Logistic Regression model predicts the risk category and outputs per-class confidence scores.

---

## 📄 License

This project is open source. Feel free to use, modify, and distribute.

---

## 👤 Author

**Ishan** — [@Ishan-debugg](https://github.com/Ishan-debugg)

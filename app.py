from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pathlib import Path
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import numpy as np
import re
import os

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

DEFAULT_MODEL = BASE_DIR / "models" / "model_v1.h5"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL)))

_model = None
def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        _model = load_model(str(MODEL_PATH))
    return _model

VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 20

def preprocess_text(text: str) -> np.ndarray:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    review = re.sub("[^a-zA-Z]", " ", text)
    review = review.lower().split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in stop_words]
    clean_text = " ".join(review)

    onehot_repr = [one_hot(clean_text, VOCAB_SIZE)]
    embedded_docs = pad_sequences(onehot_repr, padding="pre", maxlen=MAX_SEQUENCE_LENGTH)
    return np.array(embedded_docs)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...)):
    X_final = preprocess_text(text)
    model = get_model()
    y_pred = model.predict(X_final)

    score = float(y_pred[0][0])
    if score > 0.5:
        prediction = "Real"
        probability = round(score * 100, 2)
    else:
        prediction = "Fake"
        probability = round((1 - score) * 100, 2)

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction": prediction, "probability": probability},
    )

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    try:
        _ = get_model()
        return {"ready": True}
    except Exception:
        return {"ready": False}, 503
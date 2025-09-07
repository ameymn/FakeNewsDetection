from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
import re
import uvicorn

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

MODEL_PATH = "models/model_v1.h5"
model = load_model(MODEL_PATH)

VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 20 


def preprocess_text(text: str):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    # clean text
    review = re.sub("[^a-zA-Z]", " ", text)
    review = review.lower().split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in stop_words]
    clean_text = " ".join(review)

    # one_hot + pad_sequences (same as notebook)
    onehot_repr = [one_hot(clean_text, VOCAB_SIZE)]
    embedded_docs = pad_sequences(onehot_repr, padding="pre", maxlen=MAX_SEQUENCE_LENGTH)
    return np.array(embedded_docs)



@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...)):
    X_final = preprocess_text(text)
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



if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=True)

"""API для определения тональности русского текста."""

from typing import Annotated, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field, StringConstraints

from app.model import MODEL_NAME, predict, predict_batch

# Пустая строка или строка из одних пробелов — ошибка клиента (422).
# Длинные тексты модель всё равно обрезает до 512 токенов, лимит защищает
# сервис от многомегабайтных запросов.
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=5000)]

app = FastAPI(
    title="Sentiment API",
    description="Тональность русского текста: positive / neutral / negative. "
    f"Модель {MODEL_NAME}.",
    version="1.0.0",
)


class TextIn(BaseModel):
    text: Text

    model_config = {"json_schema_extra": {"examples": [{"text": "Отличный сервис, всем советую!"}]}}


class TextsIn(BaseModel):
    texts: list[Text] = Field(min_length=1, max_length=100)

    model_config = {
        "json_schema_extra": {"examples": [{"texts": ["Всё понравилось", "Ужасная поддержка"]}]}
    }


class Prediction(BaseModel):
    label: Literal["positive", "neutral", "negative"]
    score: float = Field(ge=0, le=1)


@app.get("/")
def root():
    return {"message": "Sentiment API. Документация: /docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/predict", response_model=Prediction)
def predict_one(item: TextIn):
    return predict(item.text)


@app.post("/predict/batch", response_model=list[Prediction])
def predict_many(items: TextsIn):
    return predict_batch(items.texts)

"""Модель тональности из с1сз1 (задача 1): positive / neutral / negative."""

from transformers import pipeline

MODEL_NAME = "seara/rubert-tiny2-russian-sentiment"

# Модель грузится один раз при импорте модуля: загрузка занимает секунды,
# а одно предсказание — миллисекунды.
clf = pipeline("text-classification", model=MODEL_NAME)


def predict(text: str) -> dict:
    """Вернуть самую вероятную метку и её вероятность."""
    result = clf(text, truncation=True)[0]
    return {"label": result["label"], "score": round(result["score"], 4)}


def predict_batch(texts: list[str], batch_size: int = 32) -> list[dict]:
    results = clf(texts, batch_size=batch_size, truncation=True)
    return [{"label": r["label"], "score": round(r["score"], 4)} for r in results]

# Sentiment API

![tests](https://github.com/sstegss/sentiment-api/actions/workflows/tests.yml/badge.svg)

API на FastAPI, который определяет тональность русского текста: `positive`, `neutral` или `negative`.

ПрИнж, семестр 1. Модель та же, что в задаче 1 прошлого задания ([ml-applied-tasks](https://github.com/sstegss/ml-applied-tasks/tree/main/01_text_sentiment)): [seara/rubert-tiny2-russian-sentiment](https://huggingface.co/seara/rubert-tiny2-russian-sentiment) с Hugging Face. Она маленькая, на CPU отвечает за несколько миллисекунд, видеокарта не нужна.

Задание делал один, API, тесты и CI мои.

## Запуск

Нужен Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

При первом запуске модель скачается с Hugging Face. Документация со всеми методами откроется на http://127.0.0.1:8000/docs, там же их можно вызвать из браузера.

## Методы

| Метод | Путь | Что делает |
|-------|------|------------|
| GET | `/` | ссылка на документацию |
| GET | `/health` | проверка, что сервис жив, и имя модели |
| POST | `/predict` | тональность одного текста |
| POST | `/predict/batch` | тональность списка текстов, до 100 за запрос |

Пример:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Отличный сервис, всем советую!"}'
```

```json
{"label": "positive", "score": 0.9951}
```

В `score` лежит вероятность выбранной метки. Если текст пустой, состоит из одних пробелов, длиннее 5000 символов или запрос не в том формате, API вернёт `422` и опишет, что не так. Слишком длинные тексты модель обрезает до 512 токенов.

## Тесты

```bash
pytest -v
```

Тесты лежат в [tests/test_api.py](tests/test_api.py). Отдельный сервер для них не нужен, API вызывается через `TestClient` из FastAPI. Что проверяется:

- формат ответа: метка одна из трёх, вероятность от 0 до 1
- правильные метки на фразах, в которых модель уверена
- ошибки на плохих запросах: пустой текст, не те поля, не JSON, слишком длинный текст, GET вместо POST
- пакетный метод возвращает ответы в том же порядке и с теми же метками, что поштучный
- документация OpenAPI открывается

После каждого push и pull request тесты запускает GitHub Actions, конфиг в [.github/workflows/tests.yml](.github/workflows/tests.yml). Модель там кэшируется, чтобы не качать её каждый раз.

## Структура

```
app/
  model.py   # загрузка модели, predict и predict_batch
  main.py    # эндпоинты FastAPI и схемы запросов на Pydantic
tests/
  conftest.py
  test_api.py
requirements.txt
```

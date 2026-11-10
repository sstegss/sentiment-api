import pytest

# Фразы, на которых модель уверена (вероятность > 0.9), чтобы тесты
# проверяли API, а не пограничные случаи модели.
POSITIVE = ["Отличный сервис, всем советую!", "Это лучший фильм, который я видел"]
NEGATIVE = ["Ужасная поддержка, больше никогда не закажу", "Отвратительное качество, деньги на ветер"]
NEUTRAL = ["Завтра в Москве ожидается дождь"]

LABELS = {"positive", "neutral", "negative"}


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "/docs" in response.json()["message"]


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "seara/rubert-tiny2-russian-sentiment"}


def test_predict_response_format(client):
    response = client.post("/predict", json={"text": "Обычный день"})
    assert response.status_code == 200
    data = response.json()
    assert set(data) == {"label", "score"}
    assert data["label"] in LABELS
    assert 0 <= data["score"] <= 1


@pytest.mark.parametrize(
    "text, label",
    [(t, "positive") for t in POSITIVE]
    + [(t, "negative") for t in NEGATIVE]
    + [(t, "neutral") for t in NEUTRAL],
)
def test_predict_label(client, text, label):
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 200
    assert response.json()["label"] == label


def test_predict_long_text(client):
    # Текст длиннее 512 токенов модель обрезает, а не падает.
    response = client.post("/predict", json={"text": "Отличный сервис! " * 250})
    assert response.status_code == 200
    assert response.json()["label"] == "positive"


@pytest.mark.parametrize(
    "payload",
    [
        {},  # нет поля text
        {"text": ""},
        {"text": "   "},
        {"text": 123},
        {"text": None},
        {"text": "а" * 5001},
        {"message": "Привет"},
    ],
)
def test_predict_invalid_input(client, payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_not_json(client):
    response = client.post("/predict", content="просто текст", headers={"Content-Type": "text/plain"})
    assert response.status_code == 422


def test_predict_wrong_method(client):
    assert client.get("/predict").status_code == 405


def test_batch(client):
    texts = [POSITIVE[0], NEGATIVE[0], NEUTRAL[0]]
    response = client.post("/predict/batch", json={"texts": texts})
    assert response.status_code == 200
    data = response.json()
    # Ответов столько же, сколько текстов, и в том же порядке.
    assert [d["label"] for d in data] == ["positive", "negative", "neutral"]


def test_batch_matches_single(client):
    texts = POSITIVE + NEGATIVE
    batch = client.post("/predict/batch", json={"texts": texts}).json()
    single = [client.post("/predict", json={"text": t}).json() for t in texts]
    assert [b["label"] for b in batch] == [s["label"] for s in single]
    for b, s in zip(batch, single):
        assert b["score"] == pytest.approx(s["score"], abs=1e-3)


@pytest.mark.parametrize(
    "payload",
    [
        {"texts": []},
        {"texts": "не список"},
        {"texts": ["нормально", ""]},
        {"texts": ["текст"] * 101},
    ],
)
def test_batch_invalid_input(client, payload):
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422


def test_openapi_docs(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/predict", "/predict/batch", "/health"} <= set(paths)

import pytest
from fastapi.testclient import TestClient

from app.main import app


# Один клиент на все тесты: модель грузится один раз при импорте app.model.
@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c

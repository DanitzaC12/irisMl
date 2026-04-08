import pytest
from fastapi.testclient import TestClient
from app import app, model

client = TestClient(app)


def test_model_is_loaded():
    assert model is not None


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_predict_endpoint():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert isinstance(data["prediction"], int)


def test_predict_invalid_input():
    payload = {
        "sepal_length": "abc",  
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
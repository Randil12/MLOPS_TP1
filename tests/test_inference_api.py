from fastapi.testclient import TestClient
import numpy as np

import app as api


client = TestClient(api.app)


class DummyModel:
    feature_names_in_ = [
        "age",
        "tenure_months",
        "monthly_charges",
        "support_calls",
        "contract_type_One year",
        "contract_type_Two year",
        "payment_method_Credit card",
        "payment_method_Electronic check",
        "payment_method_Mailed check",
        "internet_service_Fiber",
    ]

    def predict(self, X):
        return [1]

    def predict_proba(self, X):
        return np.array([[0.2, 0.8]])


def test_health_endpoint(monkeypatch):
    monkeypatch.setattr(api, "model", DummyModel())
    monkeypatch.setattr(api, "model_load_error", None)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_valid_input(monkeypatch):
    monkeypatch.setattr(api, "model", DummyModel())
    monkeypatch.setattr(api, "model_load_error", None)
    api.drift_buffer.clear()

    response = client.post(
        "/predict",
        json={
            "age": 35,
            "tenure_months": 12,
            "monthly_charges": 75.5,
            "support_calls": 2,
            "contract_type": "Monthly",
            "payment_method": "Electronic check",
            "internet_service": "Fiber",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert body["prediction"] in [0, 1, None]


def test_predict_returns_prediction(monkeypatch):
    monkeypatch.setattr(api, "model", DummyModel())
    monkeypatch.setattr(api, "model_load_error", None)
    api.drift_buffer.clear()

    response = client.post(
        "/predict",
        json={
            "age": 35,
            "tenure_months": 12,
            "monthly_charges": 75.5,
            "support_calls": 2,
            "contract_type": "Monthly",
            "payment_method": "Electronic check",
            "internet_service": "Fiber",
        },
    )

    assert response.status_code == 200
    assert response.json()["prediction"] == 1
    assert response.json()["label"] == "churn"
    assert response.json()["confidence"] == 0.8


def test_predict_rejects_missing_field():
    response = client.post("/predict", json={"tenure_months": 12})

    assert response.status_code == 422


def test_predict_rejects_invalid_type():
    response = client.post(
        "/predict",
        json={
            "age": 35,
            "tenure_months": "douze",
            "monthly_charges": 75.5,
            "support_calls": 2,
            "contract_type": "Monthly",
            "payment_method": "Electronic check",
            "internet_service": "Fiber",
        },
    )

    assert response.status_code == 422


def test_predict_detects_business_anomaly():
    response = client.post(
        "/predict",
        json={
            "age": 35,
            "tenure_months": 100,
            "monthly_charges": 75.5,
            "support_calls": 2,
            "contract_type": "Monthly",
            "payment_method": "Electronic check",
            "internet_service": "Fiber",
        },
    )

    assert response.status_code == 200
    assert response.json()["prediction"] is None
    assert response.json()["warning"] == "anomalie detectee"

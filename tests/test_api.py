from fastapi.testclient import TestClient

from src.api.main import app


def test_health():
    """Health endpoint should return API and model status."""

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_model_info():
    """Model info endpoint should return model metadata."""

    with TestClient(app) as client:
        response = client.get("/model/info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_type"] == "RandomForestRegressor"
    assert data["model_loaded"] is True
    assert data["target"] == "price"

    assert data["features"] == [
        "bed",
        "bath",
        "house_size",
        "acre_lot",
        "state",
        "city",
        "zip_code",
    ]


def test_predict():
    """Predict endpoint should return a valid property price."""

    property_data = {
        "bed": 3,
        "bath": 2,
        "house_size": 1800,
        "acre_lot": 0.25,
        "state": "California",
        "city": "Los Angeles",
        "zip_code": "90001",
    }

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=property_data
        )

    assert response.status_code == 200

    data = response.json()

    assert "predicted_price" in data
    assert isinstance(
        data["predicted_price"],
        (int, float)
    )
    assert data["predicted_price"] > 0


def test_predict_batch():
    """Batch endpoint should return one prediction per property."""

    batch_data = {
        "properties": [
            {
                "bed": 3,
                "bath": 2,
                "house_size": 1800,
                "acre_lot": 0.25,
                "state": "California",
                "city": "Los Angeles",
                "zip_code": "90001",
            },
            {
                "bed": 4,
                "bath": 3,
                "house_size": 2500,
                "acre_lot": 0.30,
                "state": "Florida",
                "city": "Miami",
                "zip_code": "33101",
            },
            {
                "bed": 2,
                "bath": 1,
                "house_size": 1200,
                "acre_lot": 0.15,
                "state": "New York",
                "city": "New York",
                "zip_code": "10001",
            },
        ]
    }

    with TestClient(app) as client:
        response = client.post(
            "/predict/batch",
            json=batch_data
        )

    assert response.status_code == 200

    data = response.json()

    assert "predictions" in data
    assert len(data["predictions"]) == 3

    for prediction in data["predictions"]:
        assert isinstance(
            prediction,
            (int, float)
        )
        assert prediction > 0


def test_invalid_property_data():
    """Invalid property data should be rejected."""

    invalid_data = {
        "bed": -3,
        "bath": 2,
        "house_size": 100,
        "acre_lot": 0.25,
        "state": "California",
        "city": "Los Angeles",
        "zip_code": "90001",
    }

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=invalid_data
        )

    assert response.status_code == 422
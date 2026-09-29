from src.ml.model_service import ModelService


def get_test_property():
    """Return valid property data for model tests."""

    return {
        "bed": 3,
        "bath": 2,
        "house_size": 1800,
        "acre_lot": 0.25,
        "state": "California",
        "city": "Los Angeles",
        "zip_code": "90001",
    }


def test_model_loading():
    """Saved ML model should load successfully."""

    service = ModelService()

    service.load_model()

    assert service.is_model_loaded() is True
    assert service.model is not None


def test_single_prediction():
    """Model should return one valid numeric prediction."""

    service = ModelService()

    service.load_model()

    prediction = service.predict(
        get_test_property()
    )

    assert isinstance(
        prediction,
        (int, float)
    )

    assert prediction > 0


def test_batch_prediction():
    """Model should return one prediction for each property."""

    service = ModelService()

    service.load_model()

    properties = [
        get_test_property(),
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

    predictions = service.predict_batch(
        properties
    )

    assert len(predictions) == 3

    for prediction in predictions:
        assert isinstance(
            prediction,
            (int, float)
        )

        assert prediction > 0
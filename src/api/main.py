from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    PropertyInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)

from src.ml.config import FEATURES, TARGET
from src.ml.model_service import ModelService
from src.ml.training_service import TrainingService


# Сервіс для прогнозування
model_service = ModelService()

# Сервіс для навчання
training_service = TrainingService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model when the API starts."""

    model_service.load_model()

    yield


app = FastAPI(
    title="Real Estate Price Prediction API",
    description="REST API for predicting and training real estate price models.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse
)
def health():
    """Check whether the API and model are available."""

    return {
        "status": "ok",
        "model_loaded": model_service.is_model_loaded(),
    }


@app.get(
    "/model/info",
    response_model=ModelInfoResponse
)
def model_info():
    """Return information about the loaded ML model."""

    model_type = (
        type(
            model_service.model.named_steps["model"]
        ).__name__
        if model_service.is_model_loaded()
        else "Not loaded"
    )

    return {
        "model_type": model_type,
        "model_loaded": model_service.is_model_loaded(),
        "features": FEATURES,
        "target": TARGET,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(data: PropertyInput):
    """Predict the price of one property."""

    try:
        prediction = model_service.predict(
            data.model_dump()
        )

        return {
            "predicted_price": prediction
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse
)
def predict_batch(data: BatchPredictionRequest):
    """Predict prices for multiple properties."""

    try:
        properties = [
            property_data.model_dump()
            for property_data in data.properties
        ]

        predictions = model_service.predict_batch(
            properties
        )

        return {
            "predictions": predictions
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@app.post("/train")
def train_model():
    """Train and save a new Random Forest model."""

    try:
        X_train, X_test, y_train, y_test = (
            training_service.load_training_data()
        )

        trained_model = training_service.train(
            X_train,
            y_train
        )

        model_path = training_service.save_model(
            trained_model
        )

        # Завантажуємо нову модель після навчання
        model_service.load_model()

        return {
            "status": "success",
            "message": "Model trained and saved successfully.",
            "model_path": str(model_path),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
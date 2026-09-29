from pydantic import BaseModel, Field


class PropertyInput(BaseModel):
    """Input data for one real estate property."""

    bed: float = Field(gt=0, le=30)
    bath: float = Field(gt=0, le=30)
    house_size: float = Field(ge=300, le=50_000)
    acre_lot: float | None = Field(default=None, ge=0, le=1_000)

    state: str = Field(min_length=1)
    city: str = Field(min_length=1)
    zip_code: str = Field(min_length=1)


class PredictionResponse(BaseModel):
    """Prediction returned for one property."""

    predicted_price: float


class BatchPredictionRequest(BaseModel):
    """Input data for batch prediction."""

    properties: list[PropertyInput]


class BatchPredictionResponse(BaseModel):
    """Predictions returned for multiple properties."""

    predictions: list[float]


class HealthResponse(BaseModel):
    """Current API and model status."""

    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Information about the currently loaded ML model."""

    model_type: str
    model_loaded: bool
    features: list[str]
    target: str
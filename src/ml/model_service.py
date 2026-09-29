import joblib
import pandas as pd

from .config import FEATURES, MODEL_PATH


class ModelService:
    """Service responsible for loading and using the ML model."""

    def __init__(self):
        self.model = None

    def load_model(self) -> None:
        """Load the trained model pipeline from disk."""

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model was not found: {MODEL_PATH}"
            )

        self.model = joblib.load(MODEL_PATH)

    def is_model_loaded(self) -> bool:
        """Return whether a trained model is currently loaded."""

        return self.model is not None

    def predict(self, property_data: dict) -> float:
        """Predict the price of one property."""

        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        input_df = pd.DataFrame(
            [property_data],
            columns=FEATURES
        )

        prediction = self.model.predict(input_df)[0]

        return float(prediction)

    def predict_batch(self, properties: list[dict]) -> list[float]:
        """Predict prices for multiple properties."""

        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        input_df = pd.DataFrame(
            properties,
            columns=FEATURES
        )

        predictions = self.model.predict(input_df)

        return [float(value) for value in predictions]
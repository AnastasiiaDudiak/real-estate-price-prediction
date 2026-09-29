from pathlib import Path

import joblib
import pandas as pd

from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.ml.config import MODEL_PATH


class TrainingService:
    """Service responsible for creating, training and saving the ML model."""

    def __init__(self):
        self.model_path = MODEL_PATH
        self.data_path = Path(
            "data/processed/real_estate_clean.csv"
        )

    def load_training_data(self):
        """Load cleaned data and create train/test datasets."""

        df = pd.read_csv(self.data_path)

        features = [
            "bed",
            "bath",
            "house_size",
            "acre_lot",
            "state",
            "city",
            "zip_code",
        ]

        target = "price"

        X = df[features].copy()
        y = df[target].copy()

        # ZIP code is a categorical feature
        X["zip_code"] = (
            X["zip_code"]
            .astype("string")
            .str.replace(r"\.0$", "", regex=True)
            .fillna("Unknown")
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        return X_train, X_test, y_train, y_test

    def build_pipeline(self) -> Pipeline:
        """Create the preprocessing and Random Forest pipeline."""

        numeric_features = [
            "bed",
            "bath",
            "house_size",
            "acre_lot",
        ]

        state_features = [
            "state",
        ]

        high_cardinality_features = [
            "city",
            "zip_code",
        ]

        numeric_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
        ])

        state_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True
                )
            ),
        ])

        high_cardinality_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                TargetEncoder()
            ),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numeric_pipeline,
                    numeric_features
                ),
                (
                    "state",
                    state_pipeline,
                    state_features
                ),
                (
                    "high_cardinality",
                    high_cardinality_pipeline,
                    high_cardinality_features
                ),
            ],
            remainder="drop"
        )

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=42,
            n_jobs=-1
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        return pipeline

    def train(self, X, y) -> Pipeline:
        """Train a new model pipeline."""

        model = self.build_pipeline()

        model.fit(X, y)

        return model

    def save_model(self, model) -> Path:
        """Save the trained model pipeline to disk."""

        self.model_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(
            model,
            self.model_path
        )

        return self.model_path
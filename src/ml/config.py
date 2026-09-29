from pathlib import Path


# Коренева папка проєкту
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Основні директорії
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Збережена модель
MODEL_PATH = MODELS_DIR / "real_estate_rf.joblib"


# Ознаки, які очікує модель
FEATURES = [
    "bed",
    "bath",
    "house_size",
    "acre_lot",
    "state",
    "city",
    "zip_code",
]

TARGET = "price"
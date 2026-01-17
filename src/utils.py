from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "products.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

import sys
from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.utils import DATA_PATH, MODEL_PATH


@dataclass
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = {"title", "category"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must have columns: {required}")

    df["title"] = df["title"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()

    df = df[(df["title"] != "") & (df["category"] != "")]
    if len(df) < 10:
        raise ValueError("Dataset too small after cleaning. Add more rows.")

    return df


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]
    )


def main() -> int:
    cfg = TrainConfig()
    df = load_dataset()

    X = df["title"].values
    y = df["category"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    print("\n=== Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1 : {f1:.4f}")
    print("\nClassification report:\n")
    print(classification_report(y_test, preds, zero_division=0))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n✅ Model saved to: {MODEL_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# src/train_model.py
# ------------------------------------------------------------
# Train product category classifier from product titles.
# - Loads data/products.csv
# - Cleans text + fixes column name spacing issues
# - Drops invalid/NaN labels (fix for "Input contains NaN")
# - Feature engineering from title
# - Compares 2 models (LogReg vs LinearSVC)
# - Selects best by Macro F1
# - Retrains on full data and saves models/model.pkl
# ------------------------------------------------------------

import os
import re
import argparse
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


# -----------------------------
# 1) Text cleaning
# -----------------------------
def clean_title(text: str) -> str:
    """Basic normalization for product titles."""
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)  # multiple spaces -> single
    return text


# -----------------------------
# 2) Simple engineered features
# -----------------------------
def add_title_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds small numeric features derived from title.
    These often help text classifiers (numbers, lengths, codes).
    """
    df = df.copy()
    titles = df["title"].fillna("").astype(str)

    df["title_len_chars"] = titles.str.len().astype(int)
    df["title_len_words"] = titles.apply(lambda x: len(str(x).split())).astype(int)
    df["has_digits"] = titles.str.contains(r"\d", regex=True).astype(int)
    df["special_char_count"] = titles.apply(
        lambda x: len(re.findall(r"[^a-zA-Z0-9\s]", str(x)))
    ).astype(int)
    df["max_word_len"] = titles.apply(
        lambda x: max([len(w) for w in str(x).split()] + [0])
    ).astype(int)

    return df


# -----------------------------
# 3) Data loader (robust)
# -----------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    """
    Loads CSV and returns a cleaned dataframe with columns:
      - title
      - label
      - engineered numeric features
    Fixes common issues:
      - leading/trailing spaces in column names
      - missing/blank labels (NaN) that break stratify split
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Nu găsesc fișierul: {csv_path}")

    df = pd.read_csv(csv_path)

    # FIX 1: dataset has spaces in column names (e.g. " Category Label")
    df.columns = df.columns.str.strip()

    # Expect these after stripping:
    required_cols = ["Product Title", "Category Label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Lipsesc coloane necesare: {missing}. Coloane existente: {df.columns.tolist()}"
        )

    # Rename to simple names
    df = df.rename(columns={"Product Title": "title", "Category Label": "label"}).copy()

    # Clean title
    df["title"] = df["title"].apply(clean_title)

    # FIX 2: remove NaN/blank labels BEFORE converting to string
    # - Treat empty/space-only labels as NA
    df["label"] = df["label"].replace(r"^\s*$", pd.NA, regex=True)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(str).str.strip()

    # Drop invalid rows
    df = df[(df["title"] != "") & (df["label"] != "")]
    df = df.drop_duplicates(subset=["title", "label"]).reset_index(drop=True)

    # Add engineered features
    df = add_title_features(df)

    return df


# -----------------------------
# 4) Build candidate models
# -----------------------------
def build_models():
    """
    Pipeline = TF-IDF (1-2 grams) + numeric title features + classifier
    Compare at least 2 models (required by assignment).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9), "title"),
            ("num", "passthrough", ["title_len_chars", "title_len_words", "has_digits",
                                    "special_char_count", "max_word_len"]),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    models = {
        "LogisticRegression": Pipeline(
            [("prep", preprocessor),
             ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]
        ),
        "LinearSVC": Pipeline(
            [("prep", preprocessor),
             ("clf", LinearSVC())]
        ),
    }
    return models


# -----------------------------
# 5) Main training flow
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train product category classifier from product titles.")
    parser.add_argument("--data", type=str, default="data/products.csv", help="Path to products.csv")
    parser.add_argument("--out", type=str, default="models/model.pkl", help="Output path for model.pkl")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size fraction")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    df = load_data(args.data)

    print("After cleaning:", df.shape)
    print("NaN in label:", df["label"].isna().sum())
    print("Unique labels:", df["label"].nunique())

    feature_cols = [
        "title",
        "title_len_chars",
        "title_len_words",
        "has_digits",
        "special_char_count",
        "max_word_len",
    ]

    X = df[feature_cols].copy()
    y = df["label"].copy()

    # Stratified split (keeps class proportions)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    models = build_models()

    best_name, best_f1, best_model = None, -1.0, None
    results = []

    for name, model in models.items():
        print("\n==============================")
        print("Training model:", name)
        print("==============================")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1m = f1_score(y_test, preds, average="macro")
        results.append((name, acc, f1m))

        print("Accuracy:", round(acc, 4))
        print("Macro F1:", round(f1m, 4))
        print("\nClassification report:\n")
        print(classification_report(y_test, preds, zero_division=0))

        if f1m > best_f1:
            best_name, best_f1, best_model = name, f1m, model

    print("\n===== Summary (sorted by Macro F1) =====")
    for n, a, f in sorted(results, key=lambda x: x[2], reverse=True):
        print(f"{n:18s}  acc={a:.4f}  macro_f1={f:.4f}")

    print("\nBest model:", best_name, "Macro F1:", round(best_f1, 4))

    # Retrain best model on full dataset for final artifact
    best_model.fit(X, y)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(best_model, args.out)
    print("\nSaved model to:", args.out)


if __name__ == "__main__":
    main()

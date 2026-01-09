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


def clean_title(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def add_title_features(df: pd.DataFrame) -> pd.DataFrame:
    titles = df["title"].fillna("").astype(str)

    df["title_len_chars"] = titles.str.len().astype(int)
    df["title_len_words"] = titles.apply(lambda x: len(str(x).split())).astype(int)
    df["has_digits"] = titles.str.contains(r"\d", regex=True).astype(int)
    df["special_char_count"] = titles.apply(lambda x: len(re.findall(r"[^a-zA-Z0-9\s]", str(x)))).astype(int)
    df["max_word_len"] = titles.apply(lambda x: max([len(w) for w in str(x).split()] + [0])).astype(int)
    return df


def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Nu găsesc fișierul: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # FIX: datasetul are spații în numele coloanelor

    df = df.rename(columns={
        "Product Title": "title",
        "Category Label": "label"
    }).copy()

    df["title"] = df["title"].apply(clean_title)
    df["label"] = df["label"].astype(str).str.strip()

    df = df[(df["title"] != "") & (df["label"] != "")]
    df = df.drop_duplicates(subset=["title", "label"]).reset_index(drop=True)

    df = add_title_features(df)
    return df


def build_models():
    preprocessor = ColumnTransformer(
        transformers=[
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9), "title"),
            ("num", "passthrough", ["title_len_chars","title_len_words","has_digits","special_char_count","max_word_len"]),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )

    models = {
        "LogisticRegression": Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
        ]),
        "LinearSVC": Pipeline([
            ("prep", preprocessor),
            ("clf", LinearSVC())
        ])
    }
    return models


def main():
    parser = argparse.ArgumentParser(description="Train product category classifier from titles.")
    parser.add_argument("--data", type=str, default="products.csv", help="Path to products.csv")
    parser.add_argument("--out", type=str, default="model.pkl", help="Output path for model.pkl")
    args = parser.parse_args()

    df = load_data(args.data)

    feature_cols = ["title","title_len_chars","title_len_words","has_digits","special_char_count","max_word_len"]
    X = df[feature_cols].copy()
    y = df["label"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = build_models()

    best_name, best_f1, best_model = None, -1.0, None
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1m = f1_score(y_test, preds, average="macro")

        print("\n==============================")
        print("Model:", name)
        print("Accuracy:", round(acc, 4))
        print("Macro F1:", round(f1m, 4))
        print(classification_report(y_test, preds, zero_division=0))

        if f1m > best_f1:
            best_name, best_f1, best_model = name, f1m, model

    print("\nBest model:", best_name, "Macro F1:", round(best_f1, 4))

    # Refit on all data
    best_model.fit(X, y)
    joblib.dump(best_model, args.out)
    print("Saved model to:", args.out)


if __name__ == "__main__":
    main()

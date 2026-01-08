# train_model.py
# Scop: antrenez un model ML care prezice categoria produsului pe baza titlului (Product Title)
# și salvez modelul final în format .pkl pentru utilizare ulterioară.

import os
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = os.path.join("data", "products.csv")
MODEL_PATH = os.path.join("models", "product_category_model.pkl")


def load_and_clean_data(path: str) -> pd.DataFrame:
    """
    Încarc datele și fac curățare minimă necesară:
    - elimin rândurile fără titlu sau fără categorie
    - standardizez textul (string + strip)
    """
    df = pd.read_csv(path)

    # În dataset, coloanele așteptate sunt: Product Title și Category Label
    # (dacă la tine au alt nume, schimbă aici EXACT după cum apar în CSV)
    df = df.rename(columns=lambda c: c.strip())

    required_cols = ["Product Title", "Category Label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Lipsesc coloane în CSV: {missing}. Verifică numele coloanelor din products.csv.")

    df = df.dropna(subset=["Product Title", "Category Label"]).copy()

    df["Product Title"] = df["Product Title"].astype(str).str.strip()
    df["Category Label"] = df["Category Label"].astype(str).str.strip()

    # elimin titluri goale
    df = df[df["Product Title"].str.len() > 0]
    df = df[df["Category Label"].str.len() > 0]

    return df


def main():
    print("Loading data...")
    df = load_and_clean_data(DATA_PATH)

    X = df["Product Title"]
    y = df["Category Label"]

    # Split 80/20 cu stratify ca să păstrez distribuția categoriilor în train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline = vectorizare TF-IDF + algoritm de clasificare
    # Aleg LinearSVC pentru că merge foarte bine pe text (multi-class) și e rapid.
    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),     # unigram + bigram (ajută la expresii ca "mark iii", "washing machine")
                min_df=2,               # ignor cuvinte care apar extrem de rar
                max_df=0.95             # ignor cuvinte care apar aproape peste tot
            )),
            ("clf", LinearSVC())
        ]
    )

    print("Training model...")
    model.fit(X_train, y_train)

    print("Evaluating on test set...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}\n")

    # Raport pe clase (precision/recall/F1)
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    # Salvez modelul final (pipeline complet) într-un .pkl
    os.makedirs("models", exist_ok=True)
    dump(model, MODEL_PATH)

    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()

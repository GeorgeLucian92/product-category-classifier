import argparse
import joblib
import pandas as pd
import re


def clean_title(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def make_features(title: str) -> pd.DataFrame:
    t = clean_title(title)
    s = pd.Series([t])

    df = pd.DataFrame({"title": s})
    df["title_len_chars"] = s.str.len().astype(int)
    df["title_len_words"] = s.apply(lambda x: len(str(x).split())).astype(int)
    df["has_digits"] = s.str.contains(r"\d", regex=True).astype(int)
    df["special_char_count"] = s.apply(lambda x: len(re.findall(r"[^a-zA-Z0-9\s]", str(x)))).astype(int)
    df["max_word_len"] = s.apply(lambda x: max([len(w) for w in str(x).split()] + [0])).astype(int)
    return df


def main():
    parser = argparse.ArgumentParser(description="Interactive category predictor")
    parser.add_argument("--model", type=str, default="model.pkl")
    args = parser.parse_args()

    model = joblib.load(args.model)

    print("Model loaded. Introdu titlul produsului (exit pentru ieșire).")
    while True:
        title = input("Titlu produs: ").strip()
        if title.lower() in {"exit", "quit", "q"}:
            print("Bye!")
            break

        X = make_features(title)
        pred = model.predict(X)[0]
        print("Categoria prezisă:", pred)


if __name__ == "__main__":
    main()

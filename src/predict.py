import sys
import joblib

from src.utils import MODEL_PATH


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python -m src.predict "product title here"')
        return 1

    title = " ".join(sys.argv[1:]).strip()
    if not title:
        print("Error: title must be non-empty.")
        return 1

    if not MODEL_PATH.exists():
        print("Model not found. Train first:")
        print("  python -m src.train")
        return 1

    model = joblib.load(MODEL_PATH)
    pred = model.predict([title])[0]
    print(pred)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

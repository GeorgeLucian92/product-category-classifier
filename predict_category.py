# predict_category.py
# Scop: încarc modelul antrenat (.pkl) și permit utilizatorului să introducă un titlu,
# apoi afișez categoria prezisă.

import os
from joblib import load

MODEL_PATH = os.path.join("models", "product_category_model.pkl")


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Nu găsesc modelul la {MODEL_PATH}. Rulează mai întâi: python train_model.py"
        )

    model = load(MODEL_PATH)

    print("\nProduct Category Predictor (type 'exit' to quit)\n")

    while True:
        title = input("Enter product title: ").strip()
        if title.lower() == "exit":
            print("Bye.")
            break

        if not title:
            print("Please enter a non-empty title.\n")
            continue

        pred = model.predict([title])[0]
        print(f"Predicted category: {pred}\n")


if __name__ == "__main__":
    main()

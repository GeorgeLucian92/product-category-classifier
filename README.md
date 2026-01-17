# 🧠 Product Category Classifier (ML + NLP)

A small, end-to-end Machine Learning project that predicts a product category based on its title.
Built with a scikit-learn pipeline (TF-IDF + Logistic Regression) and a CLI prediction script.

This repository is designed to be runnable immediately after cloning (includes a demo dataset).

---

## ✨ Features
- Train a text classification model from `data/raw/products.csv`
- Evaluate the model (Accuracy + Macro F1 + classification report)
- Save the trained model to `models/model.joblib`
- Predict a category from the command line

---

## 🛠 Tech Stack
- Python
- pandas
- scikit-learn (TF-IDF + Logistic Regression)
- joblib

---

## 📁 Project Structure
```bash
product-category-classifier/
│
├── src/
│ ├── train.py
│ ├── predict.py
│ └── utils.py
│
├── data/
│ └── raw/
│ └── products.csv
│
├── models/
│ └── model.joblib (generated after training)
│
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

### 1) Create and activate a virtual environment (Windows)
```bash
py -m venv .venv
.\.venv\Scripts\activate
```
### 2) Install dependencies
```bash
pip install -r requirements.txt
```
### 3) Train the model
```bash
python -m src.train
```
### 4) Predict from CLI
```bash
python -m src.predict "non-stick frying pan 24cm"
python -m src.predict "men cotton hoodie black"
python -m src.predict "car phone holder dashboard"
```

## ✅ Notes

The included dataset is intentionally small (demo). For higher accuracy, add more labeled product titles to:
data/raw/products.csv.

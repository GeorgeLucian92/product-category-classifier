# Product Category Classifier (ML)

Proiect ML care prezice categoria produsului folosind doar titlul (Product Title), pe baza datasetului products.csv.

## Structura proiectului
- data/products.csv – dataset
- notebooks/01_eda_training.ipynb – analiză completă + antrenare + evaluare
- src/train_model.py – antrenează, compară 2 modele, salvează modelul
- src/predict_category.py – predicție interactivă (introdu titlu → primești categoria)
- models/model.pkl – modelul antrenat salvat

## Instalare (local)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt

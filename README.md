# Credit Card Fraud Detection

A complete end-to-end machine learning project to detect fraudulent credit card transactions using various ML algorithms, with full data preprocessing, feature engineering, model training, evaluation, and deployment-ready code.

---

##  Project Structure

```
credit_card_fraud_detection/
│
├── data/
│   ├── raw/                   # Raw dataset (place creditcard.csv here)
│   └── processed/             # Cleaned & processed data
│
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Data loading & generation utilities
│   ├── preprocessing.py       # Preprocessing pipeline
│   ├── feature_engineering.py # Feature creation & selection
│   ├── train.py               # Model training script
│   ├── evaluate.py            # Evaluation metrics & plots
│   └── predict.py             # Inference / prediction script
│
├── models/                    # Saved trained models (.pkl files)
│
├── reports/
│   └── figures/               # Confusion matrix, ROC curves, etc.
│
│
├── requirements.txt           # Python dependencies
├── config.py                  # Central configuration
├── main.py                    # Run full pipeline end-to-end
└── README.md
```

---

##  Quick Start

### 1. Clone / Download & Navigate

```bash
cd credit_card_fraud_detection
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate          # Linux/Mac
venv\Scripts\activate             # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Dataset

- **Option A (Recommended):** Download the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` inside `data/raw/`.
- **Option B:** The project will **auto-generate synthetic data** if no CSV is found — great for testing without Kaggle.

### 5. Run the Full Pipeline

```bash
python main.py
```

### 6. Predict on New Transactions

```bash
python src/predict.py
```

### 7. (Optional) Launch Jupyter Notebook

```bash
jupyter notebook notebooks/01_EDA_and_Modeling.ipynb
```

---

##  ML Pipeline Overview

```
Raw Data → EDA → Preprocessing → Feature Engineering
    → Train/Test Split → Handle Class Imbalance (SMOTE)
        → Model Training (LR, RF, XGBoost, etc.)
            → Evaluation (AUC-ROC, F1, Precision, Recall)
                → Best Model Saved → Predict New Transactions
```

---

## Models Implemented

| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline linear model |
| Random Forest | Ensemble of decision trees |
| XGBoost | Gradient boosted trees |
| LightGBM | Fast gradient boosting |
| Decision Tree | Simple interpretable model |

---

## Evaluation Metrics

Since the dataset is highly imbalanced (~0.17% fraud), we focus on:

- **AUC-ROC Score** — Overall discriminative ability
- **Precision-Recall AUC** — Performance on minority class
- **F1-Score** — Harmonic mean of precision & recall
- **Confusion Matrix** — TP, TN, FP, FN breakdown
- **Matthews Correlation Coefficient (MCC)** — Balanced metric

---

## Configuration

Edit `config.py` to control:

- Dataset path
- Test size & random seed
- SMOTE oversampling ratio
- Model hyperparameters
- Output paths

---

## Run Tests

```bash
python -m pytest tests/ -v
```

---

## Dataset Info

The [Kaggle dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) contains:
- **284,807 transactions** over 2 days
- **492 fraudulent** (0.172%)
- Features **V1–V28**: PCA-transformed for privacy
- **Time**, **Amount**, **Class** (0 = legitimate, 1 = fraud)

---

## Tech Stack

- Python 3.8+
- Scikit-learn
- XGBoost / LightGBM
- Imbalanced-learn (SMOTE)
- Pandas / NumPy
- Matplotlib / Seaborn
- Joblib (model persistence)

---

## Results (Typical on Kaggle Dataset)

| Model | AUC-ROC | F1-Score |
|-------|---------|----------|
| Logistic Regression | ~0.97 | ~0.72 |
| Random Forest | ~0.98 | ~0.85 |
| **XGBoost** | **~0.99** | **~0.88** |
| LightGBM | ~0.99 | ~0.87 |

---

## Author

Built from scratch as a complete ML project for fraud detection.

---


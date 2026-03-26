"""
predict.py — Load the best trained model and predict on new transactions.

Usage (CLI):
    python src/predict.py                    # runs demo on random samples
    python src/predict.py --csv path/to.csv  # predict on a CSV file

Usage (module):
    from src.predict import predict_transactions
    results = predict_transactions(X_new, feature_names)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import BEST_MODEL_PATH, SCALER_PATH, THRESHOLD


# ─── Public API ───────────────────────────────────────────────────────────────

def predict_transactions(X: np.ndarray,
                         feature_names: list = None,
                         threshold: float = THRESHOLD) -> pd.DataFrame:
    """
    Run the saved best model on an array of transactions.

    Parameters
    ----------
    X             : 2-D array, shape (n_transactions, n_features)
    feature_names : optional column names for the output dataframe
    threshold     : decision threshold (default 0.5)

    Returns
    -------
    pd.DataFrame with columns: fraud_probability, prediction, label
    """
    model = _load_model()

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    labels = ["FRAUD 🚨" if p == 1 else "Legit ✅" for p in y_pred]

    result_df = pd.DataFrame({
        "fraud_probability": np.round(y_prob, 4),
        "prediction":        y_pred,
        "label":             labels,
    })

    if feature_names is not None:
        feat_df  = pd.DataFrame(X, columns=feature_names)
        result_df = pd.concat([feat_df, result_df], axis=1)

    return result_df


def predict_csv(csv_path: str, threshold: float = THRESHOLD) -> pd.DataFrame:
    """
    Load a CSV of transactions (without 'Class') and return predictions.
    Applies the same scaling used during training.
    """
    df = pd.read_csv(csv_path)
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])

    # Scale Time / Amount if scaler exists
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        cols   = [c for c in ["Time", "Amount"] if c in df.columns]
        if cols:
            df[cols] = scaler.transform(df[cols])

    return predict_transactions(df.values, list(df.columns), threshold)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_model():
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{BEST_MODEL_PATH}'.\n"
            "Run  python main.py  (or python src/train.py) first."
        )
    return joblib.load(BEST_MODEL_PATH)


# ─── CLI demo ─────────────────────────────────────────────────────────────────

def _demo():
    """Predict on 10 random synthetic transactions."""
    print("\n[Predict] Demo — generating 10 random transactions …\n")

    rng = np.random.RandomState(99)
    n_features = 30  # Time + V1-V28 + Amount
    X_demo = rng.randn(10, n_features)

    feat_cols = (["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"])
    results = predict_transactions(X_demo, feat_cols)

    pd.set_option("display.max_columns", 5)
    pd.set_option("display.width", 120)
    print(results[["fraud_probability", "prediction", "label"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict credit card fraud.")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to a CSV file of transactions to predict.")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"Decision threshold (default {THRESHOLD}).")
    args = parser.parse_args()

    if args.csv:
        results = predict_csv(args.csv, args.threshold)
        print(results.to_string(index=False))
    else:
        _demo()

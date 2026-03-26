"""
train.py — Train multiple classifiers and persist the best one.

Supported models (configured in config.py):
  - logistic_regression
  - decision_tree
  - random_forest
  - xgboost
  - lightgbm

Usage (standalone):
    python src/train.py

Usage (from main pipeline):
    from src.train import train_all_models
    results, best_name, best_model = train_all_models(X_train, y_train, X_test, y_test)
"""

import os
import sys
import time
import joblib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MODELS_TO_TRAIN, MODEL_PARAMS, MODELS_DIR, BEST_MODEL_PATH

from sklearn.linear_model   import LogisticRegression
from sklearn.tree           import DecisionTreeClassifier
from sklearn.ensemble       import RandomForestClassifier
from sklearn.metrics        import roc_auc_score, f1_score, classification_report

try:
    from xgboost  import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False


# ─── Model factory ────────────────────────────────────────────────────────────

def _build_model(name: str):
    """Instantiate a model from its name using config parameters."""
    params = MODEL_PARAMS.get(name, {})

    if name == "logistic_regression":
        return LogisticRegression(**params)

    if name == "decision_tree":
        return DecisionTreeClassifier(**params)

    if name == "random_forest":
        return RandomForestClassifier(**params)

    if name == "xgboost":
        if not _HAS_XGB:
            raise ImportError("xgboost not installed. Run: pip install xgboost")
        # Remove unsupported kwarg in newer XGBoost versions
        params = {k: v for k, v in params.items() if k != "use_label_encoder"}
        return XGBClassifier(**params)

    if name == "lightgbm":
        if not _HAS_LGB:
            raise ImportError("lightgbm not installed. Run: pip install lightgbm")
        return LGBMClassifier(**params)

    raise ValueError(f"Unknown model name: '{name}'")


# ─── Public API ───────────────────────────────────────────────────────────────

def train_all_models(X_train, y_train, X_test, y_test):
    """
    Train every model listed in MODELS_TO_TRAIN, evaluate on the test set,
    and save the best model to disk.

    Returns
    -------
    results    : dict  {model_name: {"auc": float, "f1": float, "model": estimator}}
    best_name  : str
    best_model : fitted estimator
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    results = {}

    print("\n" + "=" * 60)
    print(" MODEL TRAINING")
    print("=" * 60)

    for name in MODELS_TO_TRAIN:
        print(f"\n[Train] ── {name.upper()} ──")
        try:
            model = _build_model(name)
        except (ImportError, ValueError) as exc:
            print(f"[Train] Skipping {name}: {exc}")
            continue

        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        # Evaluate
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_prob)
        f1  = f1_score(y_test, y_pred, zero_division=0)

        print(f"[Train]   AUC-ROC : {auc:.4f}")
        print(f"[Train]   F1-Score: {f1:.4f}")
        print(f"[Train]   Time    : {elapsed:.1f}s")
        print(f"[Train]   Report  :\n"
              f"{classification_report(y_test, y_pred, target_names=['Legit','Fraud'], zero_division=0)}")

        # Save individual model
        model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(model, model_path)
        print(f"[Train]   Saved → {model_path}")

        results[name] = {"auc": auc, "f1": f1, "model": model}

    if not results:
        raise RuntimeError("[Train] No models were trained successfully.")

    # ── Pick best model by AUC ────────────────────────────────────────────────
    best_name  = max(results, key=lambda n: results[n]["auc"])
    best_model = results[best_name]["model"]

    joblib.dump(best_model, BEST_MODEL_PATH)
    print(f"\n[Train] ✓ Best model : {best_name} "
          f"(AUC={results[best_name]['auc']:.4f})")
    print(f"[Train] ✓ Saved best → {BEST_MODEL_PATH}")

    return results, best_name, best_model


def load_best_model(path: str = BEST_MODEL_PATH):
    """Load the previously saved best model."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model found at '{path}'. Run training first.")
    return joblib.load(path)


# ─── CLI entry ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.data_loader        import load_data
    from src.preprocessing      import preprocess
    from src.feature_engineering import engineer_features, apply_smote

    df = load_data()
    X_train, X_test, y_train, y_test, scaler, feat_names = preprocess(df)
    X_train, feat_names = engineer_features(X_train, feat_names)
    X_test,  _          = engineer_features(X_test,  feat_names)
    X_train, y_train    = apply_smote(X_train, y_train)
    train_all_models(X_train, y_train, X_test, y_test)

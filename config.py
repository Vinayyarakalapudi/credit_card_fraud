"""
config.py — Central configuration for the Credit Card Fraud Detection project.
Edit values here to control dataset paths, model settings, and output locations.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
REPORTS_DIR     = os.path.join(BASE_DIR, "reports")
FIGURES_DIR     = os.path.join(REPORTS_DIR, "figures")

RAW_CSV         = os.path.join(DATA_RAW_DIR, "creditcard.csv")
PROCESSED_CSV   = os.path.join(DATA_PROC_DIR, "processed_data.csv")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
SCALER_PATH     = os.path.join(MODELS_DIR, "scaler.pkl")

# ─── Data ─────────────────────────────────────────────────────────────────────
TEST_SIZE       = 0.20      # fraction held out for testing
RANDOM_STATE    = 42        # reproducibility seed
SYNTHETIC_ROWS  = 10_000    # rows to generate if no real CSV found

# ─── Class Imbalance ──────────────────────────────────────────────────────────
USE_SMOTE               = True
SMOTE_SAMPLING_STRATEGY = 0.3   # minority:majority ratio after resampling

# ─── Feature Engineering ──────────────────────────────────────────────────────
LOG_TRANSFORM_AMOUNT    = True   # apply log1p to Amount
AMOUNT_BINS             = 5      # bins for Amount_bin feature

# ─── Models to train ──────────────────────────────────────────────────────────
MODELS_TO_TRAIN = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
    "lightgbm",
]

# ─── Hyperparameters ──────────────────────────────────────────────────────────
MODEL_PARAMS = {
    "logistic_regression": {
        "C": 0.1,
        "max_iter": 1000,
        "solver": "lbfgs",
        "class_weight": "balanced",
    },
    "decision_tree": {
        "max_depth": 8,
        "min_samples_split": 10,
        "class_weight": "balanced",
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 10,
        "n_jobs": -1,
        "class_weight": "balanced",
    },
    "xgboost": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 10,
        "use_label_encoder": False,
        "eval_metric": "logloss",
    },
    "lightgbm": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "class_weight": "balanced",
        "verbose": -1,
    },
}


# ─── Evaluation ───────────────────────────────────────────────────────────────
THRESHOLD = 0.5     # default decision threshold for binary classification

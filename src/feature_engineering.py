"""
feature_engineering.py — Create additional features and optionally apply SMOTE.

Features added (when columns exist)
-------------------------------------
- Amount_log      : log1p(Amount) to reduce skew
- Amount_bin      : quantile-binned Amount category
- Hour            : hour of day from Time (seconds)
- Is_Night        : binary flag for night-time transactions

SMOTE resampling (optional)
----------------------------
Balances the training set by oversampling the minority (fraud) class.

Usage:
    from src.feature_engineering import engineer_features, apply_smote
    X_train_e = engineer_features(X_train, feature_names)
    X_res, y_res = apply_smote(X_train_e, y_train)
"""

import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (USE_SMOTE, SMOTE_SAMPLING_STRATEGY, RANDOM_STATE,
                    LOG_TRANSFORM_AMOUNT, AMOUNT_BINS)


# ─── Public API ───────────────────────────────────────────────────────────────

def engineer_features(X: np.ndarray, feature_names: list) -> tuple:
    """
    Add derived features to X.

    Parameters
    ----------
    X             : 2-D feature array
    feature_names : column names matching X

    Returns
    -------
    X_new          : 2-D numpy array with extra columns
    new_feat_names : updated list of column names
    """
    df = pd.DataFrame(X, columns=feature_names)
    new_cols = []

    # Amount-based features
    if "Amount" in df.columns:
        if LOG_TRANSFORM_AMOUNT:
            df["Amount_log"] = np.log1p(df["Amount"])
            new_cols.append("Amount_log")

        df["Amount_bin"] = pd.qcut(df["Amount"], q=AMOUNT_BINS,
                                   labels=False, duplicates="drop")
        df["Amount_bin"] = df["Amount_bin"].fillna(0).astype(float)
        new_cols.append("Amount_bin")

    # Time-based features
    if "Time" in df.columns:
        # Time in Kaggle CSV is seconds from first transaction
        # After StandardScaler it's normalised — use raw if available
        raw_time = df["Time"]
        # Try to recover approximate second-of-day
        seconds_in_day = 86_400
        df["Hour"] = (raw_time % seconds_in_day / 3600).abs() % 24
        df["Is_Night"] = ((df["Hour"] >= 22) | (df["Hour"] < 6)).astype(float)
        new_cols += ["Hour", "Is_Night"]

    if new_cols:
        print(f"[FeatureEng] Added features: {new_cols}")
    else:
        print("[FeatureEng] No new features added (columns Time/Amount not found).")

    new_feat_names = list(df.columns)
    return df.values, new_feat_names


def apply_smote(X_train: np.ndarray, y_train: np.ndarray):
    """
    Apply SMOTE to balance the training set.

    Returns
    -------
    X_resampled, y_resampled
    """
    if not USE_SMOTE:
        print("[FeatureEng] SMOTE disabled (USE_SMOTE=False).")
        return X_train, y_train

    before_fraud = y_train.sum()
    smote = SMOTE(
        sampling_strategy=SMOTE_SAMPLING_STRATEGY,
        random_state=RANDOM_STATE,
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)
    after_fraud = y_res.sum()

    print(f"[FeatureEng] SMOTE applied — fraud before: {before_fraud:,}  "
          f"after: {after_fraud:,}  total samples: {len(X_res):,}")
    return X_res, y_res

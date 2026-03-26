"""
preprocessing.py — Clean and scale the raw dataframe.

Steps
-----
1. Drop duplicates & handle missing values
2. Scale 'Time' and 'Amount' columns (StandardScaler)
3. Split into train / test sets

Usage:
    from src.preprocessing import preprocess
    X_train, X_test, y_train, y_test, scaler = preprocess(df)
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (TEST_SIZE, RANDOM_STATE, SCALER_PATH,
                    PROCESSED_CSV, DATA_PROC_DIR)


# ─── Public API ───────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, save_scaler: bool = True):
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    df          : raw dataframe (must contain 'Class' column)
    save_scaler : persist the fitted scaler to SCALER_PATH

    Returns
    -------
    X_train, X_test, y_train, y_test  (numpy arrays)
    scaler                             (fitted StandardScaler)
    feature_names                      (list[str])
    """
    print("\n[Preprocessing] Starting …")

    # 1. Duplicates & nulls
    before = len(df)
    df = df.drop_duplicates()
    df = df.dropna()
    print(f"[Preprocessing] Removed {before - len(df):,} duplicate/null rows. "
          f"Remaining: {len(df):,}")

    # 2. Separate features and target
    target = df["Class"].values
    features = df.drop(columns=["Class"])

    # 3. Scale Time & Amount (V1-V28 are already PCA-scaled in Kaggle dataset)
    scaler = StandardScaler()
    cols_to_scale = [c for c in ["Time", "Amount"] if c in features.columns]
    if cols_to_scale:
        features[cols_to_scale] = scaler.fit_transform(features[cols_to_scale])
        print(f"[Preprocessing] Scaled columns: {cols_to_scale}")
    else:
        scaler.fit(features)  # fit on all features for synthetic data

    feature_names = list(features.columns)
    X = features.values

    # 4. Train / test split (stratified to preserve class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    print(f"[Preprocessing] Train size : {len(X_train):,}  |  "
          f"Test size : {len(X_test):,}")
    print(f"[Preprocessing] Train fraud: {y_train.sum():,}  |  "
          f"Test fraud: {y_test.sum():,}")

    # 5. Persist scaler
    if save_scaler:
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        print(f"[Preprocessing] Scaler saved → {SCALER_PATH}")

    # 6. Optionally save processed data
    os.makedirs(DATA_PROC_DIR, exist_ok=True)
    processed_df = pd.DataFrame(X, columns=feature_names)
    processed_df["Class"] = target
    processed_df.to_csv(PROCESSED_CSV, index=False)
    print(f"[Preprocessing] Processed data saved → {PROCESSED_CSV}")

    return X_train, X_test, y_train, y_test, scaler, feature_names


def load_scaler(path: str = SCALER_PATH) -> StandardScaler:
    """Load a previously saved scaler."""
    return joblib.load(path)

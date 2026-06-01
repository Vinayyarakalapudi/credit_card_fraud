"""
data_loader.py — Load the credit card dataset or generate synthetic data.

Usage:
    from src.data_loader import load_data
    df = load_data()
"""

import os
import numpy as np
import pandas as pd

# ─── Public API ───────────────────────────────────────────────────────────────

def load_data(csv_path: str = RAW_CSV) -> pd.DataFrame:
    """
    Load the dataset from *csv_path*.

    If the file is not found, synthetic data is generated automatically so the
    rest of the pipeline can be exercised without the Kaggle download.

    Returns
    -------
    pd.DataFrame  with columns: Time, V1–V28, Amount, Class
    """
    if os.path.exists(csv_path):
        print(f"[DataLoader] Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"[DataLoader] Loaded {len(df):,} rows, {df.shape[1]} columns.")
    else:
        print(f"[DataLoader] '{csv_path}' not found — generating synthetic data.")
        df = _generate_synthetic(n_rows=SYNTHETIC_ROWS)

    _validate(df)
    return df




def _validate(df: pd.DataFrame) -> None:
    """Raise if the dataframe is missing expected columns."""
    required = {"Class"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"[DataLoader] Missing required columns: {missing}")

    n_fraud = df["Class"].sum()
    print(f"[DataLoader] Class distribution — Fraud: {n_fraud:,} "
          f"({n_fraud / len(df) * 100:.3f}%)  "
          f"Legit: {(len(df) - n_fraud):,}")

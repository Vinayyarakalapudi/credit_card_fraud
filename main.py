"""
main.py — Run the complete Credit Card Fraud Detection pipeline end-to-end.

Steps
-----
1. Load data  (real CSV or synthetic)
2. Preprocess (scale, split)
3. Feature engineering + SMOTE resampling
4. Train all configured models
5. Evaluate best model & save plots
6. Print final summary

Usage:
    python main.py
"""

import os
import sys
import time

# ── Ensure project root is on sys.path ────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data_loader          import load_data
from src.preprocessing        import preprocess
from src.feature_engineering  import engineer_features, apply_smote
from src.train                import train_all_models
from src.evaluate             import evaluate_model, plot_model_comparison


def main():
    t_start = time.time()
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║   CREDIT CARD FRAUD DETECTION — Full Pipeline            ║")
    print("╚" + "═" * 58 + "╝\n")

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    print("━" * 60)
    print(" STEP 1 — Load Data")
    print("━" * 60)
    df = load_data()

    # ── Step 2: Preprocess ────────────────────────────────────────────────────
    print("\n" + "━" * 60)
    print(" STEP 2 — Preprocess")
    print("━" * 60)
    X_train, X_test, y_train, y_test, scaler, feat_names = preprocess(df)

    # ── Step 3: Feature engineering + SMOTE ──────────────────────────────────
    print("\n" + "━" * 60)
    print(" STEP 3 — Feature Engineering & Class Balancing")
    print("━" * 60)
    X_train_e, feat_names_e = engineer_features(X_train, feat_names)
    X_test_e,  _            = engineer_features(X_test,  feat_names)
    X_train_r, y_train_r    = apply_smote(X_train_e, y_train)

    # ── Step 4: Train models ──────────────────────────────────────────────────
    print("\n" + "━" * 60)
    print(" STEP 4 — Train Models")
    print("━" * 60)
    results, best_name, best_model = train_all_models(
        X_train_r, y_train_r, X_test_e, y_test
    )

    # ── Step 5: Evaluate best model ───────────────────────────────────────────
    print("\n" + "━" * 60)
    print(f" STEP 5 — Evaluate Best Model ({best_name})")
    print("━" * 60)
    metrics = evaluate_model(best_model, X_test_e, y_test, model_name=best_name)
    plot_model_comparison(results)

    # ── Step 6: Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║   PIPELINE COMPLETE                                       ║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  Best model  : {best_name:<42} ║")
    print(f"║  AUC-ROC     : {metrics['auc']:<42.4f} ║")
    print(f"║  PR-AUC      : {metrics['pr_auc']:<42.4f} ║")
    print(f"║  F1-Score    : {metrics['f1']:<42.4f} ║")
    print(f"║  MCC         : {metrics['mcc']:<42.4f} ║")
    print(f"║  Total time  : {elapsed:<39.1f}s  ║")
    print("╠" + "═" * 58 + "╣")
    print("║  Artefacts saved:                                         ║")
    print("║    models/best_model.pkl                                  ║")
    print("║    models/scaler.pkl                                      ║")
    print("║    reports/figures/*.png                                  ║")
    print("╚" + "═" * 58 + "╝\n")

    return metrics


if __name__ == "__main__":
    main()

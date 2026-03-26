"""
evaluate.py — Comprehensive model evaluation with plots.

Metrics
-------
- Confusion Matrix
- ROC Curve + AUC
- Precision-Recall Curve + AUC
- Classification Report
- Matthews Correlation Coefficient

Usage:
    from src.evaluate import evaluate_model, plot_model_comparison
    evaluate_model(model, X_test, y_test, model_name="XGBoost")
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving files
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    classification_report, matthews_corrcoef, f1_score,
    roc_auc_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import FIGURES_DIR, THRESHOLD


# ─── Colour palette ───────────────────────────────────────────────────────────
_PALETTE = {"fraud": "#e74c3c", "legit": "#2ecc71", "main": "#3498db"}


def _ensure_figures_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


# ─── Public API ───────────────────────────────────────────────────────────────

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   model_name: str = "Model",
                   threshold: float = THRESHOLD) -> dict:
    """
    Full evaluation of a single model.  Saves plots to FIGURES_DIR.

    Returns
    -------
    dict with keys: auc, pr_auc, f1, mcc
    """
    _ensure_figures_dir()

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # ── Metrics ───────────────────────────────────────────────────────────────
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc  = average_precision_score(y_test, y_prob)
    f1      = f1_score(y_test, y_pred, zero_division=0)
    mcc     = matthews_corrcoef(y_test, y_pred)

    print(f"\n{'─'*50}")
    print(f" Evaluation — {model_name}")
    print(f"{'─'*50}")
    print(f"  AUC-ROC  : {roc_auc:.4f}")
    print(f"  PR-AUC   : {pr_auc:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  MCC      : {mcc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legit','Fraud'], zero_division=0)}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    tag = model_name.lower().replace(" ", "_")
    _plot_confusion_matrix(y_test, y_pred, model_name, tag)
    _plot_roc_curve(y_test, y_prob, roc_auc, model_name, tag)
    _plot_pr_curve(y_test, y_prob, pr_auc, model_name, tag)

    return {"auc": roc_auc, "pr_auc": pr_auc, "f1": f1, "mcc": mcc}


def plot_model_comparison(results: dict) -> None:
    """
    Bar chart comparing AUC-ROC and F1 across all trained models.

    Parameters
    ----------
    results : dict  {model_name: {"auc": float, "f1": float, ...}}
    """
    _ensure_figures_dir()

    names = list(results.keys())
    aucs  = [results[n]["auc"] for n in names]
    f1s   = [results[n]["f1"]  for n in names]

    x     = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, aucs, width, label="AUC-ROC",  color=_PALETTE["main"],  alpha=0.85)
    ax.bar(x + width / 2, f1s,  width, label="F1-Score", color=_PALETTE["fraud"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — AUC-ROC vs F1-Score", fontsize=13, fontweight="bold")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    for i, (a, f) in enumerate(zip(aucs, f1s)):
        ax.text(i - width / 2, a + 0.01, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i + width / 2, f + 0.01, f"{f:.3f}", ha="center", fontsize=8)

    path = os.path.join(FIGURES_DIR, "model_comparison.png")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Evaluate] Saved model comparison → {path}")


# ─── Private plot helpers ─────────────────────────────────────────────────────

def _plot_confusion_matrix(y_test, y_pred, model_name, tag):
    cm   = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit", "Fraud"],
                yticklabels=["Legit", "Fraud"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
    path = os.path.join(FIGURES_DIR, f"{tag}_confusion_matrix.png")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Evaluate] Saved confusion matrix → {path}")


def _plot_roc_curve(y_test, y_prob, roc_auc_val, model_name, tag):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=_PALETTE["main"], lw=2,
            label=f"AUC = {roc_auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}", fontweight="bold")
    ax.legend(loc="lower right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    path = os.path.join(FIGURES_DIR, f"{tag}_roc_curve.png")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Evaluate] Saved ROC curve → {path}")


def _plot_pr_curve(y_test, y_prob, pr_auc_val, model_name, tag):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color=_PALETTE["fraud"], lw=2,
            label=f"PR-AUC = {pr_auc_val:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}", fontweight="bold")
    ax.legend(loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    path = os.path.join(FIGURES_DIR, f"{tag}_pr_curve.png")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Evaluate] Saved PR curve → {path}")

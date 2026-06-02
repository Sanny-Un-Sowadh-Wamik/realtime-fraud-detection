"""Evaluation for imbalanced classification.

Headline metric is **PR-AUC** (average precision) — accuracy and even ROC-AUC are
misleading at 0.17% prevalence. Precision/recall/F1 are reported at a chosen
operating threshold, with the full confusion matrix.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)


def metrics_at_threshold(y_true, y_proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    # ROC/PR-AUC need both classes present and a non-constant score.
    auc = float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(auc, 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def best_f1_threshold(y_true, y_proba) -> float:
    """Threshold that maximises F1 on the precision-recall curve."""
    prec, rec, thr = precision_recall_curve(np.asarray(y_true), np.asarray(y_proba))
    if len(thr) == 0:
        return 0.5
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thr[int(np.nanargmax(f1))])

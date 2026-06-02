"""SHAP explainability for the XGBoost component — why a transaction was flagged.

SHAP is imported lazily so the base package stays light.
"""

from __future__ import annotations

import numpy as np


def make_explainer(xgb_model):
    import shap

    return shap.TreeExplainer(xgb_model)


def top_contributions(explainer, x_row, feature_names: list[str], k: int = 6) -> list[tuple[str, float]]:
    """Return the top-k (feature, signed SHAP value) for a single transaction row."""
    sv = explainer.shap_values(x_row)
    arr = np.asarray(sv)
    if arr.ndim == 3:  # (n, features, classes)
        arr = arr[..., -1]
    row = arr.reshape(-1)
    order = np.argsort(np.abs(row))[::-1][:k]
    return [(feature_names[i], float(row[i])) for i in order]

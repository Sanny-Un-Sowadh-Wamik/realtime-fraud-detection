"""Classifiers: XGBoost (supervised), Isolation Forest (unsupervised), and a
weighted ensemble of the two. Isolation-Forest scores are mapped to [0, 1] using
a range fixed on the training data (so serving is batch-independent).
"""

from __future__ import annotations

import numpy as np
import xgboost as xgb
from sklearn.ensemble import IsolationForest


def train_xgb(X, y, scale_pos_weight: float | None = None, random_state: int = 42) -> xgb.XGBClassifier:
    """XGBoost classifier. ``scale_pos_weight`` defaults to neg/pos (class weighting)."""
    y_arr = np.asarray(y)
    if scale_pos_weight is None:
        neg, pos = int((y_arr == 0).sum()), max(1, int((y_arr == 1).sum()))
        scale_pos_weight = neg / pos
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X, y)
    return model


def train_isoforest(X, contamination: float = 0.0017, random_state: int = 42) -> IsolationForest:
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=random_state, n_jobs=-1)
    iso.fit(X)
    return iso


def isoforest_anomaly(iso: IsolationForest, X) -> np.ndarray:
    """Higher = more anomalous."""
    return -iso.score_samples(X)


def isoforest_proba(iso: IsolationForest, X, lo: float, hi: float) -> np.ndarray:
    """Map anomaly scores into [0, 1] using a fixed (train) range."""
    return np.clip((isoforest_anomaly(iso, X) - lo) / (hi - lo + 1e-12), 0.0, 1.0)


def ensemble_proba(xgb_p, iso_p, w_xgb: float = 0.7) -> np.ndarray:
    return w_xgb * np.asarray(xgb_p) + (1.0 - w_xgb) * np.asarray(iso_p)

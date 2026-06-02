"""Preprocessing: stratified split + robust scaling of Amount/Time.

V1–V28 are already PCA components (centred, comparable scale), so only the raw
``Amount`` and ``Time`` columns need scaling. The scaler is fit on the **training
fold only** — the test fold never informs preprocessing.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from frauddet.config import AppConfig, load_config

V_COLUMNS = [f"V{i}" for i in range(1, 29)]
FEATURES = V_COLUMNS + ["Amount_scaled", "Time_scaled"]
TARGET = "Class"


def _apply_scaler(X: pd.DataFrame, scaler: RobustScaler) -> pd.DataFrame:
    out = X.copy()
    out[["Amount_scaled", "Time_scaled"]] = scaler.transform(X[["Amount", "Time"]])
    return out[FEATURES]


def prepare(df: pd.DataFrame, config: AppConfig | None = None):
    """Return ``(X_train, X_test, y_train, y_test, scaler, feature_cols)`` — leakage-free."""
    config = config or load_config()
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.data.test_size, stratify=y, random_state=config.data.random_state
    )
    scaler = RobustScaler().fit(X_tr[["Amount", "Time"]])
    X_tr = _apply_scaler(X_tr, scaler).reset_index(drop=True)
    X_te = _apply_scaler(X_te, scaler).reset_index(drop=True)
    return X_tr, X_te, y_tr.reset_index(drop=True), y_te.reset_index(drop=True), scaler, list(FEATURES)

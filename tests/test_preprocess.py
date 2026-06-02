"""Preprocessing tests (hermetic — synthetic data)."""

import numpy as np
import pandas as pd

from frauddet.config import load_config
from frauddet.features import FEATURES, prepare


def _synthetic(n: int = 2000, n_fraud: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(0, 1, (n, 28)), columns=[f"V{i}" for i in range(1, 29)])
    df["Time"] = rng.uniform(0, 1e5, n)
    df["Amount"] = rng.lognormal(3, 1, n)
    y = np.zeros(n, dtype=int)
    y[:n_fraud] = 1
    rng.shuffle(y)
    df["Class"] = y
    return df


def test_prepare_shapes_and_features():
    cfg = load_config()
    X_tr, X_te, y_tr, y_te, scaler, cols = prepare(_synthetic(), cfg)
    assert list(X_tr.columns) == FEATURES
    assert cols == FEATURES
    assert len(X_tr) + len(X_te) == 2000
    assert {"Amount_scaled", "Time_scaled"} <= set(X_tr.columns)


def test_prepare_preserves_fraud_count_and_stratifies():
    cfg = load_config()
    _, _, y_tr, y_te, _, _ = prepare(_synthetic(n_fraud=40), cfg)
    assert int(y_tr.sum()) + int(y_te.sum()) == 40
    assert y_te.sum() > 0  # stratified split keeps fraud in the test fold

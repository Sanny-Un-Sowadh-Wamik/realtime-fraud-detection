"""Model + metric tests (hermetic)."""

import numpy as np
import pandas as pd

from frauddet.models.classifiers import ensemble_proba, train_xgb
from frauddet.models.evaluate import best_f1_threshold, metrics_at_threshold


def test_xgb_trains_and_metrics_in_range():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(0, 1, (600, 5)), columns=[f"f{i}" for i in range(5)])
    y = (X["f0"] + rng.normal(0, 0.3, 600) > 1).astype(int)
    model = train_xgb(X, y)
    proba = model.predict_proba(X)[:, 1]
    m = metrics_at_threshold(y, proba, best_f1_threshold(y, proba))
    assert 0.0 <= m["pr_auc"] <= 1.0
    assert 0.0 <= m["precision"] <= 1.0
    assert m["tp"] + m["fn"] == int(y.sum())


def test_ensemble_is_weighted_average():
    e = ensemble_proba(np.array([0.8, 0.2]), np.array([0.4, 0.6]), w_xgb=0.7)
    assert abs(float(e[0]) - (0.7 * 0.8 + 0.3 * 0.4)) < 1e-9


def test_baseline_metrics_are_degenerate():
    y = np.array([0, 0, 1, 0, 1])
    m = metrics_at_threshold(y, np.zeros(5), 0.5)
    assert m["precision"] == 0.0 and m["recall"] == 0.0

"""Inference: load the serving bundle and score a transaction with the ensemble.

A "transaction" is a dict with V1..V28, Amount and Time. Returns the fraud
probability, the per-model breakdown, an ALERT/REVIEW/OK decision, and (optionally)
the top SHAP factors explaining the score.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from frauddet.config import REPO_ROOT
from frauddet.models.classifiers import isoforest_proba

MODELS_DIR = Path(os.getenv("FRAUDDET_MODELS_DIR", str(REPO_ROOT / "models")))

_EXPLAINER = None


@lru_cache(maxsize=1)
def load_bundle() -> tuple[dict, dict]:
    bundle = joblib.load(MODELS_DIR / "fraud_bundle.joblib")
    meta = json.loads((MODELS_DIR / "metadata.json").read_text())
    return bundle, meta


def model_metadata() -> dict:
    return load_bundle()[1]


def _get_explainer():
    global _EXPLAINER
    if _EXPLAINER is None:
        from frauddet.models.explain import make_explainer

        _EXPLAINER = make_explainer(load_bundle()[0]["xgb"])
    return _EXPLAINER


def _featurize(tx: dict, bundle: dict) -> pd.DataFrame:
    row = {f"V{i}": float(tx.get(f"V{i}", 0.0)) for i in range(1, 29)}
    amt_time = pd.DataFrame([[float(tx.get("Amount", 0.0)), float(tx.get("Time", 0.0))]], columns=["Amount", "Time"])
    scaled = bundle["scaler"].transform(amt_time)[0]
    row["Amount_scaled"], row["Time_scaled"] = float(scaled[0]), float(scaled[1])
    return pd.DataFrame([row])[bundle["features"]]


def score_transaction(tx: dict, explain: bool = False) -> dict:
    bundle, _ = load_bundle()
    X = _featurize(tx, bundle)

    p_xgb = float(bundle["xgb"].predict_proba(X)[:, 1][0])
    p_iso = float(isoforest_proba(bundle["iso"], X, bundle["iso_lo"], bundle["iso_hi"])[0])
    w = bundle["w_xgb"]
    proba = w * p_xgb + (1.0 - w) * p_iso

    if proba >= bundle["alert_threshold"]:
        decision = "ALERT"
    elif proba >= bundle["review_threshold"]:
        decision = "REVIEW"
    else:
        decision = "OK"

    out = {
        "fraud_probability": round(proba, 4),
        "p_xgb": round(p_xgb, 4),
        "p_isolation_forest": round(p_iso, 4),
        "decision": decision,
    }
    if explain:
        from frauddet.models.explain import top_contributions

        out["top_factors"] = [
            {"feature": f, "shap": round(v, 4)} for f, v in top_contributions(_get_explainer(), X, bundle["features"])
        ]
    return out

"""Inference tests — skipped if the model bundle hasn't been trained."""

import pytest

from frauddet.predict import MODELS_DIR, model_metadata, score_transaction

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "fraud_bundle.joblib").exists(),
    reason="model not trained — run scripts/train.py",
)

_TX = {f"V{i}": 0.0 for i in range(1, 29)} | {"Amount": 100.0, "Time": 0.0}


def test_score_structure():
    r = score_transaction(_TX)
    assert 0.0 <= r["fraud_probability"] <= 1.0
    assert r["decision"] in {"ALERT", "REVIEW", "OK"}
    assert 0.0 <= r["p_xgb"] <= 1.0


def test_metadata_complete():
    meta = model_metadata()
    assert "metrics" in meta and "features" in meta


def test_explain_when_shap_available():
    try:
        import shap  # noqa: F401
    except ImportError:
        pytest.skip("shap not installed")
    r = score_transaction(_TX, explain=True)
    assert len(r["top_factors"]) >= 3

"""FastAPI endpoint tests — skipped if the model bundle hasn't been trained."""

import pytest
from fastapi.testclient import TestClient

from frauddet.predict import MODELS_DIR

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "fraud_bundle.joblib").exists(),
    reason="model not trained — run scripts/train.py",
)

from api.main import app  # noqa: E402

client = TestClient(app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_sample_then_score():
    tx = client.get("/sample").json()
    resp = client.post("/score", json=tx)
    assert resp.status_code == 200
    assert resp.json()["decision"] in {"ALERT", "REVIEW", "OK"}


def test_models_has_metrics():
    assert "metrics" in client.get("/models").json()

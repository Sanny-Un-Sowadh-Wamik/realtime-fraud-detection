"""FastAPI service for real-time fraud scoring.

    GET  /health             — liveness
    GET  /models             — model metadata + held-out metrics
    GET  /sample             — a random example transaction (to try /score)
    POST /score?explain=true — score one transaction (optional SHAP factors)
    WS   /ws/stream          — live stream of scored transactions (demo)

Interactive docs at /docs.
"""

from __future__ import annotations

import asyncio

from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from frauddet import __version__
from frauddet.config import load_config
from frauddet.predict import model_metadata, score_transaction
from frauddet.stream import transaction_stream

app = FastAPI(
    title="Real-Time Fraud Detection API",
    version=__version__,
    description="Score card transactions with an XGBoost + Isolation Forest ensemble. Educational demo.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_EXAMPLE = {f"V{i}": 0.0 for i in range(1, 29)} | {"Amount": 149.62, "Time": 0.0}


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/models", tags=["meta"])
def models() -> dict:
    meta = model_metadata()
    keys = ("serving_model", "registry_name", "data_is_real", "fraud_rate", "threshold", "metrics", "trained_at")
    return {k: meta[k] for k in keys if k in meta}


@app.get("/sample", tags=["score"])
def sample() -> dict:
    tx = next(transaction_stream(load_config(), fraud_boost=0.3, seed=None))
    tx.pop("_true_label", None)
    return tx


@app.post("/score", tags=["score"])
def score(tx: dict = Body(..., examples=[_EXAMPLE]), explain: bool = False) -> dict:
    clean = {k: v for k, v in tx.items() if k != "_true_label"}
    return score_transaction(clean, explain=explain)


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket) -> None:
    await ws.accept()
    cfg = load_config()
    delay = 1.0 / max(cfg.serving.stream_rate_hz, 0.1)
    gen = transaction_stream(cfg, fraud_boost=0.05)
    try:
        while True:
            tx = next(gen)
            true_label = tx.pop("_true_label", None)
            scored = score_transaction(tx)
            await ws.send_json({"amount": round(tx["Amount"], 2), **scored, "true_label": true_label})
            await asyncio.sleep(delay)
    except WebSocketDisconnect:
        return

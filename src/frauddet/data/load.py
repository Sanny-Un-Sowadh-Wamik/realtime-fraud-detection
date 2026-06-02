"""Resilient, keyless dataset loading for the ULB Credit Card Fraud dataset.

Primary source: OpenML (data_id 1597) — the same 284,807-transaction dataset as the
Kaggle mirror, with **no API key**. Pulled once and cached to Parquet.

Resilience: OpenML occasionally 503s. We retry with backoff and, as a last resort,
fall back to a *learnable synthetic* dataset so the pipeline always runs. Synthetic
data is **never cached over real data**, so a later run auto-upgrades to the real set.
A small stratified sample (all fraud + N legit) is committed for offline demo/CI.
"""

from __future__ import annotations

import logging
import os
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd

from frauddet.config import AppConfig, load_config

logger = logging.getLogger(__name__)

RAW_NAME = "creditcard.parquet"
SAMPLE_NAME = "creditcard_sample.parquet"
V_COLUMNS = [f"V{i}" for i in range(1, 29)]


def _fetch_openml(config: AppConfig, retries: int = 4, backoff: float = 4.0) -> pd.DataFrame | None:
    """Fetch from OpenML with retry/backoff; return None if it stays unavailable."""
    from sklearn.datasets import fetch_openml

    socket.setdefaulttimeout(60)  # abort a stalled OpenML download instead of hanging forever
    for attempt in range(1, retries + 1):
        try:
            logger.info("OpenML fetch (data_id=%s) attempt %d/%d…", config.data.openml_data_id, attempt, retries)
            ds = fetch_openml(data_id=config.data.openml_data_id, as_frame=True, parser="auto")
            df = ds.frame.copy()
            if "Class" not in df.columns:
                df = df.rename(columns={ds.target.name: "Class"})
            df["Class"] = pd.to_numeric(df["Class"]).astype(int)
            return df
        except Exception as exc:  # noqa: BLE001 — network / service errors vary
            logger.warning("OpenML attempt %d failed: %s", attempt, exc)
            if attempt < retries:
                time.sleep(backoff * attempt)
    return None


def _synthetic_creditcard(seed: int = 42, n: int = 60_000, fraud_rate: float = 0.0030) -> pd.DataFrame:
    """Learnable synthetic stand-in (DEMO ONLY): PCA-like features with a few shifted
    for fraud separability, lognormal amounts, ~0.3% fraud."""
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    legit = rng.normal(0, 1, size=(n_legit, 28))
    fraud = rng.normal(0, 1, size=(n_fraud, 28))
    for j, shift in [(3, 2.5), (9, 1.6), (11, -2.2), (13, -3.0), (16, 2.0)]:
        fraud[:, j] += shift  # make a handful of components discriminative

    x = np.vstack([legit, fraud])
    y = np.r_[np.zeros(n_legit), np.ones(n_fraud)].astype(int)
    amount = np.r_[rng.lognormal(3.0, 1.0, n_legit), rng.lognormal(4.0, 1.2, n_fraud)]

    df = pd.DataFrame(x, columns=V_COLUMNS)
    df.insert(0, "Time", rng.uniform(0, 172_800, n))
    df["Amount"] = amount
    df["Class"] = y
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def load_creditcard(config: AppConfig | None = None, force: bool = False) -> pd.DataFrame:
    """Return the full dataset (cache-first → OpenML with retry → synthetic fallback)."""
    config = config or load_config()
    cache = config.data.raw_dir / RAW_NAME
    if cache.exists() and not force:
        logger.info("cache hit: %s", cache)
        return pd.read_parquet(cache)

    if os.getenv("FRAUDDET_FORCE_SYNTHETIC") == "1":
        logger.warning("FRAUDDET_FORCE_SYNTHETIC=1 → synthetic data (skipping OpenML)")
        return _synthetic_creditcard(config.data.random_state)

    df = _fetch_openml(config)
    if df is not None:
        config.data.raw_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
        logger.info("cached REAL data: %d rows (%d fraud) → %s", len(df), int(df["Class"].sum()), cache)
        return df

    logger.warning("OpenML unavailable — using SYNTHETIC data (not cached). Re-run later for the real set.")
    return _synthetic_creditcard(config.data.random_state)


def build_sample(config: AppConfig | None = None) -> Path:
    """Persist a stratified sample (ALL fraud + N legit) to commit as an offline fixture."""
    config = config or load_config()
    df = load_creditcard(config)
    fraud = df[df["Class"] == 1]
    n_legit = min(config.data.sample_legit, int((df["Class"] == 0).sum()))
    legit = df[df["Class"] == 0].sample(n=n_legit, random_state=config.data.random_state)
    sample = pd.concat([fraud, legit]).sample(frac=1, random_state=config.data.random_state).reset_index(drop=True)
    config.data.sample_dir.mkdir(parents=True, exist_ok=True)
    out = config.data.sample_dir / SAMPLE_NAME
    sample.to_parquet(out)
    logger.info("sample: %d rows (%d fraud) → %s", len(sample), int(sample["Class"].sum()), out)
    return out


def load_sample(config: AppConfig | None = None) -> pd.DataFrame:
    """Serving/test loader: the committed sample, or the full dataset if absent."""
    config = config or load_config()
    path = config.data.sample_dir / SAMPLE_NAME
    return pd.read_parquet(path) if path.exists() else load_creditcard(config)

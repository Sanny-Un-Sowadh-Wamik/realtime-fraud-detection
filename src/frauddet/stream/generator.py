"""Transaction stream for the live demo.

Samples real rows from the committed dataset sample. Fraud is *boosted* above its
true 0.17% rate so the live dashboard actually shows alerts — clearly a demo choice,
not a claim about base rates. Swap this generator for a Redis/Kafka consumer in prod.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from frauddet.config import AppConfig, load_config
from frauddet.data import load_sample


def transaction_stream(
    config: AppConfig | None = None, fraud_boost: float = 0.05, seed: int | None = None
) -> Iterator[dict]:
    """Yield transaction dicts (V1..V28, Amount, Time) + a hidden ``_true_label``."""
    config = config or load_config()
    df = load_sample(config)
    rng = random.Random(seed)
    fraud_idx = df.index[df["Class"] == 1].tolist()
    legit_idx = df.index[df["Class"] == 0].tolist()
    feature_cols = [c for c in df.columns if c != "Class"]

    while True:
        pool = fraud_idx if (fraud_boost and fraud_idx and rng.random() < fraud_boost) else legit_idx
        row = df.loc[rng.choice(pool)]
        tx = {c: float(row[c]) for c in feature_cols}
        tx["_true_label"] = int(row["Class"])
        yield tx

"""Streamlit live fraud-detection dashboard.

Self-contained (loads the committed model bundle directly), so it deploys cleanly to
a Hugging Face Docker Space. An auto-refreshing fragment streams + scores synthetic
transactions; alerts get a SHAP explanation.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frauddet.config import load_config
from frauddet.predict import load_bundle, score_transaction
from frauddet.stream import transaction_stream

st.set_page_config(page_title="Fraud Detection — Live", page_icon="💳", layout="wide")
CFG = load_config()

try:
    _, META = load_bundle()
except Exception:
    st.error("Model bundle not found — run `python scripts/train.py` to create it, then reload.")
    st.stop()

st.title("💳 Real-Time Fraud Detection")
st.caption(
    "XGBoost + Isolation Forest ensemble scoring a live transaction stream, with SHAP "
    "explanations for every alert. **Demo stream — not financial advice.**"
)

ss = st.session_state
if "stream" not in ss:
    ss.stream = transaction_stream(CFG, fraud_boost=0.06, seed=7)
    ss.feed = deque(maxlen=40)
    ss.n = ss.alerts = ss.reviews = 0
    ss.last_alert = None

with st.sidebar:
    st.header("Controls")
    running = st.toggle("▶ Live stream", value=True)
    batch = st.slider("Transactions per tick", 1, 10, 4)
    st.markdown("---")
    st.caption(f"Serving: **{META.get('serving_model', 'ensemble')}**")
    st.caption(f"Alert ≥ {META.get('alert_threshold', 0.85):.2f} · Review ≥ {META.get('review_threshold', 0.6):.2f}")
    if META.get("data_is_real") is False:
        st.warning(
            "Running on **synthetic** data (OpenML was down at build time). Re-run training for real ULB metrics."
        )

kpi = st.container()
gauge_box, feed_box = st.columns([1, 2])
shap_box = st.container()


def _render() -> None:
    c1, c2, c3, c4 = kpi.columns(4)
    c1.metric("Processed", f"{ss.n:,}")
    c2.metric("🚨 Alerts", ss.alerts)
    c3.metric("⚠️ Review", ss.reviews)
    c4.metric("Alert rate", f"{(ss.alerts / ss.n * 100) if ss.n else 0:.1f}%")

    g = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=(ss.alerts / ss.n * 100) if ss.n else 0,
            number={"suffix": "%"},
            title={"text": "Live alert rate"},
            gauge={"axis": {"range": [0, 15]}, "bar": {"color": "#ef5350"}},
        )
    )
    g.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
    gauge_box.plotly_chart(g, width="stretch")

    if ss.feed:
        df = pd.DataFrame(list(ss.feed))
        styles = {"ALERT": "background-color:#ffcdd2", "REVIEW": "background-color:#fff3cd", "OK": ""}
        styled = df.style.map(lambda v: styles.get(v, ""), subset=["decision"]).format(
            {"amount": "${:,.2f}", "fraud_probability": "{:.3f}"}
        )
        feed_box.dataframe(styled, height=300, width="stretch")

    if ss.last_alert and ss.last_alert.get("top_factors"):
        factors = ss.last_alert["top_factors"][::-1]
        bar = go.Figure(
            go.Bar(
                x=[f["shap"] for f in factors],
                y=[f["feature"] for f in factors],
                orientation="h",
                marker_color=["#ef5350" if f["shap"] > 0 else "#42a5f5" for f in factors],
            )
        )
        bar.update_layout(
            height=300,
            title=f"Why the last alert fired (SHAP) · P(fraud)={ss.last_alert['fraud_probability']:.2f}",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        shap_box.plotly_chart(bar, width="stretch")


@st.fragment(run_every=1.0 if running else None)
def _tick() -> None:
    if running:
        for _ in range(batch):
            tx = next(ss.stream)
            tx.pop("_true_label", None)
            res = score_transaction(tx)
            ss.n += 1
            ss.feed.appendleft(
                {"amount": tx["Amount"], "fraud_probability": res["fraud_probability"], "decision": res["decision"]}
            )
            if res["decision"] == "ALERT":
                ss.alerts += 1
                ss.last_alert = {**res, **score_transaction(tx, explain=True)}
            elif res["decision"] == "REVIEW":
                ss.reviews += 1
    _render()


_tick()

with st.expander("📊 Held-out model comparison (PR-AUC is the headline metric)"):
    metrics = META.get("metrics", {})
    if metrics:
        cols = ["pr_auc", "roc_auc", "precision", "recall", "f1", "fp", "fn"]
        table = pd.DataFrame(metrics).T.reindex(columns=cols)
        styled = table.style.format({c: "{:.4f}" for c in cols[:5]}).highlight_max(subset=["pr_auc"], color="#c8e6c9")
        st.dataframe(styled, width="stretch")
        st.caption("SMOTE lifts recall but adds false positives — the classic imbalanced-learning trade-off.")

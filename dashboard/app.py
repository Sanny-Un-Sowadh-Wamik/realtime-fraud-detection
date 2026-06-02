"""Premium live fraud-detection dashboard.

Streaming + scoring + ALL rendering happen INSIDE the auto-refreshing fragment, so the
panel updates in place each tick (fixed heights, no downward page growth). Alerts get a
SHAP explanation.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
for _p in (_HERE, _SRC):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import theme
from frauddet.config import load_config
from frauddet.predict import load_bundle, score_transaction
from frauddet.stream import transaction_stream

st.set_page_config(page_title="Fraud Detection — Live", page_icon="💳", layout="wide")
theme.inject()
CFG = load_config()

try:
    _, META = load_bundle()
except Exception:
    st.error("Model bundle not found — run `python scripts/train.py`, then reload.")
    st.stop()

theme.hero(
    "💳 Real-Time Fraud Detection",
    "XGBoost + Isolation Forest ensemble scoring a live transaction stream · SHAP explanation for every alert.",
    tag="REAL-TIME · SMOTE · SHAP · WEBSOCKET",
)
_demo = " · ⚙️ synthetic demo stream" if META.get("data_is_real") is False else ""
st.markdown(
    f'<span class="live"><span class="livedot"></span>LIVE</span>'
    f'<span style="color:#7a8699"> · scoring transactions in real time{_demo} · not financial advice</span>',
    unsafe_allow_html=True,
)

ss = st.session_state
if "stream" not in ss:
    ss.stream = transaction_stream(CFG, fraud_boost=0.06, seed=7)
    ss.feed = deque(maxlen=25)
    ss.n = ss.alerts = ss.reviews = 0
    ss.last_alert = None

with st.sidebar:
    st.markdown("### ⚙️ Controls")
    running = st.toggle("▶ Live stream", value=True)
    batch = st.slider("Transactions per tick", 1, 10, 4)
    st.markdown("---")
    st.caption(f"Serving: **{META.get('serving_model', 'ensemble')}**")
    st.caption(
        f"🚨 Alert ≥ {META.get('alert_threshold', 0.85):.2f} · ⚠️ Review ≥ {META.get('review_threshold', 0.6):.2f}"
    )


def _gauge() -> go.Figure:
    rate = (ss.alerts / ss.n * 100) if ss.n else 0
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=rate,
            number={"suffix": "%", "font": {"size": 28}},
            title={"text": "Live alert rate"},
            gauge={
                "axis": {"range": [0, 15]},
                "bar": {"color": "#ef4444"},
                "steps": [
                    {"range": [0, 5], "color": "rgba(148,163,184,.18)"},
                    {"range": [5, 10], "color": "rgba(239,68,68,.18)"},
                    {"range": [10, 15], "color": "rgba(239,68,68,.32)"},
                ],
            },
        )
    )
    fig.update_layout(height=260, **theme.PLOTLY_LAYOUT)
    return fig


@st.fragment(run_every=1.0 if running else None)
def live_panel() -> None:
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

    # Everything below renders inside the fragment → replaced in place each tick.
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Processed", f"{ss.n:,}")
    c2.metric("🚨 Alerts", ss.alerts)
    c3.metric("⚠️ Review", ss.reviews)
    c4.metric("Alert rate", f"{(ss.alerts / ss.n * 100) if ss.n else 0:.1f}%")

    gcol, fcol = st.columns([1, 2])
    gcol.plotly_chart(_gauge(), width="stretch")
    if ss.feed:
        df = pd.DataFrame(list(ss.feed))
        styles = {
            "ALERT": "background-color:rgba(239,68,68,.18);color:#ef4444;font-weight:700",
            "REVIEW": "background-color:rgba(245,158,11,.18);color:#d97706",
            "OK": "color:#16a34a",
        }
        styled = df.style.map(lambda v: styles.get(v, ""), subset=["decision"]).format(
            {"amount": "${:,.2f}", "fraud_probability": "{:.3f}"}
        )
        fcol.dataframe(styled, height=270, width="stretch", hide_index=True)
    else:
        fcol.info("Stream starting…")

    if ss.last_alert and ss.last_alert.get("top_factors"):
        factors = ss.last_alert["top_factors"][::-1]
        bar = go.Figure(
            go.Bar(
                x=[f["shap"] for f in factors],
                y=[f["feature"] for f in factors],
                orientation="h",
                marker_color=["#ef4444" if f["shap"] > 0 else "#3b82f6" for f in factors],
            )
        )
        bar.update_layout(
            height=280,
            title=f"🔎 Why the last alert fired (SHAP) · P(fraud)={ss.last_alert['fraud_probability']:.2f}",
            **theme.PLOTLY_LAYOUT,
        )
        st.plotly_chart(bar, width="stretch")
    else:
        st.info("🔎 The SHAP panel explains the most recent **ALERT** — waiting for the first one to fire…")


live_panel()

with st.expander("📊 Held-out model comparison · why we report PR-AUC (and the synthetic-data caveat)"):
    if META.get("data_is_real") is False:
        st.caption(
            "⚠️ Current metrics are on **synthetic** data (OpenML was unreachable at build time). "
            "Re-run `scripts/build_dataset.py && scripts/train.py` on an open network for real 284k-row ULB metrics."
        )
    metrics = META.get("metrics", {})
    if metrics:
        cols = ["pr_auc", "roc_auc", "precision", "recall", "f1", "fp", "fn"]
        table = pd.DataFrame(metrics).T.reindex(columns=cols)
        styled = table.style.format({c: "{:.4f}" for c in cols[:5]}).highlight_max(subset=["pr_auc"], color="#0f9d58")
        st.dataframe(styled, width="stretch")
        st.caption("SMOTE lifts recall but adds false positives — the classic imbalanced-learning trade-off.")

"""Premium UI theme — CSS injection + components. Adapts cleanly to light AND dark."""

from __future__ import annotations

import streamlit as st

_FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"


def inject(grad=("#141828", "#3a1020", "#ef4444"), accent="#ef4444") -> None:
    st.markdown(
        f"""
        <style>
          #MainMenu {{visibility: hidden;}}
          footer {{visibility: hidden;}}
          [data-testid="stToolbar"], [data-testid="stDecoration"], .stDeployButton {{display: none !important;}}
          [data-testid="stHeader"] {{background: transparent;}}
          [data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {{visibility: visible !important;}}
          html, body, [class*="css"] {{font-family: {_FONT};}}
          .block-container {{padding-top: 1.2rem; animation: fadeUp .6s ease;}}
          @keyframes fadeUp {{from {{opacity:0; transform:translateY(10px);}} to {{opacity:1; transform:none;}}}}
          .hero {{background: linear-gradient(120deg, {grad[0]}, {grad[1]}, {grad[2]}); background-size:220% 220%;
                  animation: grad 14s ease infinite; border-radius:18px; padding:26px 30px; color:#fff !important;
                  margin-bottom:16px; box-shadow:0 12px 34px rgba(0,0,0,.26);}}
          @keyframes grad {{0%{{background-position:0% 50%}}50%{{background-position:100% 50%}}100%{{background-position:0% 50%}}}}
          .hero h1 {{margin:0; font-size:2.05rem; font-weight:800; letter-spacing:-0.6px; color:#fff;}}
          .hero p {{margin:.45rem 0 0; opacity:.95; font-size:1.02rem; max-width:60rem; color:#fff;}}
          .badge {{display:inline-block; padding:3px 11px; border-radius:999px; font-size:.74rem; font-weight:700;
                   letter-spacing:.3px; margin-bottom:10px; color:#fff; background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.28);}}
          .live {{display:inline-flex; align-items:center; gap:7px; font-weight:700; font-size:.8rem; color:#16a34a;}}
          .livedot {{height:9px;width:9px;background:#22c55e;border-radius:50%;display:inline-block; animation:pulse 1.4s infinite;}}
          @keyframes pulse {{0%{{box-shadow:0 0 0 0 rgba(34,197,94,.6)}}70%{{box-shadow:0 0 0 9px rgba(34,197,94,0)}}100%{{box-shadow:0 0 0 0 rgba(34,197,94,0)}}}}
          div[data-testid="stMetric"] {{background: rgba(128,128,128,.08); border:1px solid rgba(128,128,128,.18);
                  border-radius:14px; padding:14px 16px; box-shadow:0 4px 14px rgba(0,0,0,.06);
                  transition:transform .15s ease, box-shadow .15s ease;}}
          div[data-testid="stMetric"]:hover {{transform:translateY(-3px); box-shadow:0 12px 26px rgba(0,0,0,.13);}}
          button[data-baseweb="tab"] {{font-weight:600;}}
          [data-baseweb="tab-highlight"] {{background:{accent} !important;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, tag: str | None = None) -> None:
    tag_html = f'<span class="badge">{tag}</span><br>' if tag else ""
    st.markdown(f'<div class="hero">{tag_html}<h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=_FONT, size=13, color="#7a8699"),
    margin=dict(l=10, r=10, t=44, b=10),
    title_font=dict(size=15),
    hoverlabel=dict(font_size=12),
    xaxis=dict(gridcolor="rgba(128,128,128,.15)"),
    yaxis=dict(gridcolor="rgba(128,128,128,.15)"),
)

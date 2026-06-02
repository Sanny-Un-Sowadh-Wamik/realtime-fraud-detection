"""frauddet — real-time credit-card fraud detection on imbalanced data.

Sub-packages:
    data      — keyless dataset loading (OpenML) + stratified sampling
    features  — preprocessing / scaling
    models    — SMOTE + XGBoost + Isolation Forest ensemble, evaluation, SHAP
    stream    — real-time transaction stream (in-memory or Redis) + scoring
"""

__version__ = "0.1.0"

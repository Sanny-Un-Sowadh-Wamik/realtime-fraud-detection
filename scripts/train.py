"""Train + evaluate the fraud models, log to MLflow, and persist the serving bundle.

Compares: a trivial baseline, XGBoost with class weighting, XGBoost with SMOTE
(applied to the TRAIN fold only), an unsupervised Isolation Forest, and a weighted
ensemble. The ensemble (XGBoost class-weight + Isolation Forest) is what we serve.

Usage:  python scripts/train.py
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import joblib
import numpy as np

from frauddet.config import REPO_ROOT, load_config
from frauddet.data import load_creditcard
from frauddet.features import prepare
from frauddet.models import registry
from frauddet.models.classifiers import (
    ensemble_proba,
    isoforest_anomaly,
    isoforest_proba,
    train_isoforest,
    train_xgb,
)
from frauddet.models.evaluate import best_f1_threshold, metrics_at_threshold

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("train")
MODELS_DIR = REPO_ROOT / "models"


def main() -> dict:
    cfg = load_config()
    df = load_creditcard(cfg)
    is_real = len(df) > 200_000
    X_tr, X_te, y_tr, y_te, scaler, cols = prepare(df, cfg)
    log.info(
        "train=%d (%d fraud) | test=%d (%d fraud) | data=%s",
        len(y_tr),
        int(y_tr.sum()),
        len(y_te),
        int(y_te.sum()),
        "REAL" if is_real else "SYNTHETIC",
    )

    results: dict[str, dict] = {}

    # Trivial baseline (predict no fraud) — anchors why accuracy is useless here.
    results["baseline"] = metrics_at_threshold(y_te, np.zeros(len(y_te)), 0.5)

    # XGBoost with class weighting (scale_pos_weight = neg/pos).
    xgb_cw = train_xgb(X_tr, y_tr, random_state=cfg.model.random_state)
    p_cw = xgb_cw.predict_proba(X_te)[:, 1]
    results["xgb_classweight"] = metrics_at_threshold(y_te, p_cw, best_f1_threshold(y_te, p_cw))
    log.info("xgb_classweight → %s", results["xgb_classweight"])

    # XGBoost with SMOTE — resample the TRAIN fold only (never the test fold).
    if cfg.model.use_smote:
        try:
            from imblearn.over_sampling import SMOTE

            X_sm, y_sm = SMOTE(random_state=cfg.model.random_state).fit_resample(X_tr, y_tr)
            xgb_sm = train_xgb(X_sm, y_sm, scale_pos_weight=1.0, random_state=cfg.model.random_state)
            p_sm = xgb_sm.predict_proba(X_te)[:, 1]
            results["xgb_smote"] = metrics_at_threshold(y_te, p_sm, best_f1_threshold(y_te, p_sm))
            log.info("xgb_smote       → %s", results["xgb_smote"])
        except Exception as exc:  # noqa: BLE001
            log.warning("SMOTE skipped: %s", exc)

    # Isolation Forest (unsupervised) — fixed [0,1] mapping from train scores.
    iso = train_isoforest(X_tr, cfg.model.isolation_forest_contamination, cfg.model.random_state)
    train_scores = isoforest_anomaly(iso, X_tr)
    lo, hi = float(train_scores.min()), float(train_scores.max())
    p_iso = isoforest_proba(iso, X_te, lo, hi)
    results["isolation_forest"] = metrics_at_threshold(y_te, p_iso, best_f1_threshold(y_te, p_iso))
    log.info("isolation_forest → %s", results["isolation_forest"])

    # Weighted ensemble — the served model.
    p_ens = ensemble_proba(p_cw, p_iso, cfg.model.ensemble_weight_xgb)
    thr_ens = best_f1_threshold(y_te, p_ens)
    results["ensemble"] = metrics_at_threshold(y_te, p_ens, thr_ens)
    log.info("ensemble         → %s", results["ensemble"])

    _print_table(results)

    # MLflow tracking + registry.
    try:
        registry.init_mlflow()
        for name in ("baseline", "xgb_smote", "isolation_forest", "ensemble"):
            if name in results:
                registry.log_run(name, {}, results[name])
        registry.log_run(
            "xgb_classweight",
            {"scale_pos_weight": "auto"},
            results["xgb_classweight"],
            model=xgb_cw,
            flavor="xgboost",
            registered_name=cfg.model.registry_name,
        )
        log.info("logged runs to MLflow")
    except Exception as exc:  # noqa: BLE001
        log.warning("MLflow logging skipped: %s", exc)

    # Persist serving bundle (the whole ensemble + scaler + thresholds).
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "xgb": xgb_cw,
            "iso": iso,
            "iso_lo": lo,
            "iso_hi": hi,
            "scaler": scaler,
            "w_xgb": cfg.model.ensemble_weight_xgb,
            "threshold": thr_ens,
            "alert_threshold": cfg.serving.alert_threshold,
            "review_threshold": cfg.serving.review_threshold,
            "features": cols,
        },
        MODELS_DIR / "fraud_bundle.joblib",
    )
    metadata = {
        "registry_name": cfg.model.registry_name,
        "serving_model": "ensemble (xgb_classweight + isolation_forest)",
        "data_is_real": is_real,
        "trained_at": datetime.now(UTC).isoformat(),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "fraud_rate": round(float(df["Class"].mean()), 5),
        "threshold": round(thr_ens, 4),
        "alert_threshold": cfg.serving.alert_threshold,
        "review_threshold": cfg.serving.review_threshold,
        "features": cols,
        "metrics": results,
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    log.info("saved bundle + metadata → %s", MODELS_DIR)
    return results


def _print_table(results: dict[str, dict]) -> None:
    print("\n" + "=" * 88)
    print(f"{'model':<20}{'PR-AUC':>9}{'ROC-AUC':>9}{'precision':>11}{'recall':>9}{'F1':>8}{'FP':>7}{'FN':>6}")
    print("-" * 88)
    for name, m in results.items():
        print(
            f"{name:<20}{m['pr_auc']:>9.4f}{m['roc_auc']:>9.4f}{m['precision']:>11.4f}"
            f"{m['recall']:>9.4f}{m['f1']:>8.4f}{m['fp']:>7}{m['fn']:>6}"
        )
    print("=" * 88 + "\n")


if __name__ == "__main__":
    main()

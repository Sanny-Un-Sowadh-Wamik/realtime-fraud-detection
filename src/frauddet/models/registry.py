"""Thin MLflow helpers (submodules imported at top to avoid the local-shadowing trap)."""

from __future__ import annotations

import logging

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from frauddet.config import get_settings

logger = logging.getLogger(__name__)


def init_mlflow(experiment: str = "fraud-detection") -> None:
    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(experiment)


def log_run(
    name: str, params: dict, metrics: dict, model=None, flavor: str = "xgboost", registered_name: str | None = None
) -> None:
    with mlflow.start_run(run_name=name):
        if params:
            mlflow.log_params(params)
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        if model is not None:
            mod = mlflow.xgboost if flavor == "xgboost" else mlflow.sklearn
            try:
                mod.log_model(model, name="model", registered_model_name=registered_name)
            except TypeError:
                mod.log_model(model, "model", registered_model_name=registered_name)

"""Typed configuration for the fraud-detection pipeline.

Plain parameters from ``config/config.yaml`` validated into pydantic models;
secrets from the environment / ``.env``. Paths resolve relative to the repo root.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


class DataConfig(BaseModel):
    source: str = "openml"
    openml_data_id: int = 1597
    test_size: float = 0.2
    random_state: int = 42
    sample_legit: int = 12000
    raw_dir: Path = REPO_ROOT / "data" / "raw"
    processed_dir: Path = REPO_ROOT / "data" / "processed"
    sample_dir: Path = REPO_ROOT / "data" / "sample"


class ModelConfig(BaseModel):
    use_smote: bool = True
    isolation_forest_contamination: float = 0.0017
    ensemble_weight_xgb: float = 0.7
    random_state: int = 42
    registry_name: str = "fraud-detector"


class ServingConfig(BaseModel):
    alert_threshold: float = 0.85
    review_threshold: float = 0.6
    stream_rate_hz: float = 10.0


class AppConfig(BaseModel):
    project_name: str = "fraud-detection"
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    serving: ServingConfig = ServingConfig()


class Settings(BaseSettings):
    """Secrets / runtime settings (never committed)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    redis_url: str | None = None
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"


@lru_cache
def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return AppConfig(**raw)


@lru_cache
def get_settings() -> Settings:
    return Settings()

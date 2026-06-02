"""Deploy the FastAPI fraud-scoring service to a public HF **Docker** Space.

Auth: HF token from the standard cache or HF_TOKEN env var — no secret in this file.

Usage:
    python scripts/deploy_hf_api.py --user Sanny2005 --space fraud-detection-api
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent

FRONTMATTER = """---
title: Fraud Detection API
emoji: "🛡️"
colorFrom: red
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: FastAPI fraud scoring + WebSocket stream
---

# 🛡️ Fraud Detection API

FastAPI service for **[realtime-fraud-detection](https://github.com/Sanny-Un-Sowadh-Wamik/realtime-fraud-detection)**.

- **Docs:** <https://sanny2005-fraud-detection-api.hf.space/docs>
- `POST /score?explain=true` · `GET /sample` · `GET /models` · `WS /ws/stream`

_Demo on synthetic data — not financial advice._
"""

DOCKERFILE = """FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \\
    PATH=/home/user/.local/bin:$PATH \\
    PYTHONUNBUFFERED=1 \\
    PYTHONPATH=/home/user/app:/home/user/app/src
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt
COPY --chown=user src ./src
COPY --chown=user api ./api
COPY --chown=user config ./config
COPY --chown=user models ./models
COPY --chown=user data/sample ./data/sample

EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
"""


def _stage(stage: Path) -> int:
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for d in ("api", "src", "config"):
        shutil.copytree(ROOT / d, stage / d)
    (stage / "models").mkdir()
    for f in ("fraud_bundle.joblib", "metadata.json"):
        shutil.copy(ROOT / "models" / f, stage / "models" / f)
    shutil.copytree(ROOT / "data" / "sample", stage / "data" / "sample")
    shutil.copy(ROOT / "requirements-serving.txt", stage / "requirements.txt")
    (stage / "README.md").write_text(FRONTMATTER)
    (stage / "Dockerfile").write_text(DOCKERFILE)
    for pat in ("__pycache__", "*.egg-info"):
        for p in stage.rglob(pat):
            shutil.rmtree(p, ignore_errors=True)
    for p in stage.rglob("*.pyc"):
        p.unlink()
    return sum(1 for x in stage.rglob("*") if x.is_file())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="Sanny2005")
    ap.add_argument("--space", default="fraud-detection-api")
    args = ap.parse_args()

    repo_id = f"{args.user}/{args.space}"
    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

    stage = Path("/tmp/hf_fraud_api")
    n = _stage(stage)
    print(f"uploading {n} files…")
    api.upload_folder(folder_path=str(stage), repo_id=repo_id, repo_type="space", commit_message="Deploy fraud API")
    print("SPACE_URL=https://huggingface.co/spaces/" + repo_id)


if __name__ == "__main__":
    main()

# FastAPI serving image (real-time fraud scoring). Deploys to Hugging Face Spaces
# (Docker SDK, port 7860). The frauddet package is imported from ./src via PYTHONPATH
# so the committed models/ + config/ + data/sample/ resolve relative to the repo root.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/home/user/app:/home/user/app/src
WORKDIR /home/user/app

COPY --chown=user requirements-serving.txt .
RUN pip install --user --no-cache-dir -r requirements-serving.txt

COPY --chown=user src ./src
COPY --chown=user api ./api
COPY --chown=user config ./config
COPY --chown=user models ./models
COPY --chown=user data/sample ./data/sample

EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]

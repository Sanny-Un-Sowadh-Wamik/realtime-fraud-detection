.PHONY: help setup data train test lint format api app mlflow docker-build docker-run clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-13s\033[0m %s\n",$$1,$$2}'

setup:  ## Create the venv and install every extra
	uv venv --python 3.11 && uv pip install -e ".[models,api,app,dev]"

data:  ## Fetch dataset (keyless OpenML) + write the committed sample
	python scripts/build_dataset.py

train:  ## Train + evaluate all models, log to MLflow, save the serving bundle
	python scripts/train.py

test:  ## Run the test suite
	pytest -q

lint:  ## Ruff + mypy
	ruff check . && mypy src

format:  ## Auto-format and fix
	ruff format . && ruff check --fix .

api:  ## Run the FastAPI service locally on :8000
	uvicorn api.main:app --reload --port 8000

app:  ## Run the Streamlit dashboard
	streamlit run dashboard/app.py

mlflow:  ## Open the MLflow tracking UI
	mlflow ui --backend-store-uri sqlite:///mlflow.db

docker-build:  ## Build the API serving image
	docker build -t fraud-api .

docker-run:  ## Run the API image on :7860
	docker run -p 7860:7860 fraud-api

clean:  ## Remove caches and local tracking artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache mlruns mlartifacts mlflow.db
	find . -type d -name __pycache__ -exec rm -rf {} +

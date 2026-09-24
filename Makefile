.DEFAULT_GOAL := help
UV ?= uv
export UV_PROJECT_ENVIRONMENT ?= .venv-uv
export OMP_NUM_THREADS ?= 1
export OPENBLAS_NUM_THREADS ?= 1
export MKL_NUM_THREADS ?= 1
export CUDA_VISIBLE_DEVICES :=

.PHONY: help setup test lint check data-status validate-data register-runs mlflow
help:
	@echo "setup        Install the locked Python environment and collaboration tools"
	@echo "test         Run the CPU test suite (no datasets or training required)"
	@echo "lint         Check Python correctness with Ruff"
	@echo "check        Run lint and tests"
	@echo "data-status  Inspect local DVC data versions"
	@echo "validate-data Verify completed data and frozen patient splits with DVC"
	@echo "register-runs Import historical aggregate results into MLflow"
	@echo "mlflow       Open the local MLflow service on http://127.0.0.1:5000"
	@echo "See docs/data-versioning.md and docs/experiment-tracking.md for data and MLflow commands."

setup:
	$(UV) sync --locked --group data --group tracking

test:
	$(UV) run --locked --group data --group tracking python -m pytest -q

lint:
	$(UV) run --locked ruff check ecg_experiment scripts tests

check: lint test

data-status:
	$(UV) run --locked --group data dvc status

validate-data:
	$(UV) run --locked --group data dvc repro validate_completed_data

register-runs:
	$(UV) run --locked --group tracking python -m scripts.import_mlflow_history

mlflow:
	mkdir -p outputs/mlflow
	$(UV) run --locked --group tracking mlflow ui --host 127.0.0.1 --backend-store-uri sqlite:///outputs/mlflow/mlflow.db

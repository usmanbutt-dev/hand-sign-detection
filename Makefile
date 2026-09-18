.DEFAULT_GOAL := help

# ─── Setup ──────────────────────────────────────────────────────────────────
.PHONY: install
install:  ## Install all dependencies (including dev)
	uv pip install -e ".[dev]"

.PHONY: venv
venv:  ## Create virtual environment
	uv venv .venv
	@echo "Activate with: .venv\Scripts\activate"

# ─── Training ───────────────────────────────────────────────────────────────
.PHONY: train-detector
train-detector:  ## Train YOLO26 hand detector
	python src/training/train_yolo.py

.PHONY: train-classifier
train-classifier:  ## Train PyTorch Transformer classifier
	python src/training/train_classifier.py

# ─── Running ────────────────────────────────────────────────────────────────
.PHONY: run-api
run-api:  ## Start FastAPI inference server
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

.PHONY: run-app
run-app:  ## Start Streamlit demo
	streamlit run app/streamlit_app.py

.PHONY: run-webcam
run-webcam:  ## Run real-time webcam inference (no UI)
	python src/inference/pipeline.py

# ─── Data ───────────────────────────────────────────────────────────────────
.PHONY: capture
capture:  ## Collect webcam images for a sign class
	python src/data/capture.py

.PHONY: extract-keypoints
extract-keypoints:  ## Run MediaPipe keypoint extraction on raw images
	python src/data/dataset.py

# ─── Quality ────────────────────────────────────────────────────────────────
.PHONY: lint
lint:  ## Run ruff linter
	ruff check .

.PHONY: format
format:  ## Auto-fix formatting with ruff
	ruff format .

.PHONY: test
test:  ## Run pytest test suite
	pytest --cov=src tests/

# ─── MLflow ─────────────────────────────────────────────────────────────────
.PHONY: mlflow
mlflow:  ## Open MLflow experiment UI
	mlflow ui --port 5000

# ─── Help ───────────────────────────────────────────────────────────────────
.PHONY: help
help:  ## Show this help message
	@echo ""
	@echo "Hand Sign Detection — Available Commands"
	@echo "========================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

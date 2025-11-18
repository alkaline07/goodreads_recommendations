.PHONY: help build up down clean logs shell test pipeline-all
.DEFAULT_GOAL := help

IMAGE_NAME := goodreads-model:latest
COMPOSE_FILE := docker-compose.model.yaml


DOCKER_COMPOSE := $(shell if command -v docker-compose > /dev/null 2>&1; then echo "docker-compose"; elif docker compose version > /dev/null 2>&1; then echo "docker compose"; else echo "docker-compose"; fi)

help:
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  Goodreads Model Development - Docker Commands"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Build Commands:"
	@echo "  make build              Build the Docker image"
	@echo "  make rebuild            Rebuild the Docker image (no cache)"
	@echo ""
	@echo "Run Individual Components:"
	@echo "  make load-data          Load data from BigQuery"
	@echo "  make train              Train ML models"
	@echo "  make predict            Generate prediction tables"
	@echo "  make bias               Run bias detection & mitigation"
	@echo "  make validate           Validate trained models"
	@echo "  make rollback           Rollback model"
	@echo ""
	@echo "Run Pipeline:"
	@echo "  make pipeline-all       Run complete pipeline (all steps)"
	@echo "  make pipeline-train     Run training pipeline (load→train)"
	@echo "  make pipeline-eval      Run evaluation pipeline (predict→bias→validate)"
	@echo ""
	@echo "Development Commands:"
	@echo "  make shell              Open interactive shell in container"
	@echo "  make logs               View logs from last run"
	@echo "  make ps                 Show running containers"
	@echo ""
	@echo "Cleanup Commands:"
	@echo "  make clean              Stop and remove containers"
	@echo "  make clean-volumes      Remove all volumes (data, mlruns, etc.)"
	@echo "  make clean-all          Clean everything (containers, volumes, images)"
	@echo ""
	@echo "Info Commands:"
	@echo "  make check-setup        Check if prerequisites are met"
	@echo "  make version            Show Docker and tool versions"
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"

build:
	@echo "Building Docker image"
	docker build -f Dockerfile.model -t $(IMAGE_NAME) .
	@echo "Build complete!"

rebuild:
	@echo "Rebuilding Docker image (no cache)"
	docker build --no-cache -f Dockerfile.model -t $(IMAGE_NAME) .
	@echo "Rebuild complete!"

load-data:
	@echo "Loading data from BigQuery"
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up data-loader
	@echo "Data loading complete!"

train:
	@echo "Training models"
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up model-training
	@echo "Model training complete!"

predict:
	@echo "Generating prediction tables"
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) run --rm \
		-e GOOGLE_APPLICATION_CREDENTIALS=/app/config/gcp_credentials.json \
		model-training python -m src.generate_prediction_tables
	@echo "Prediction generation complete!"

bias:
	@echo "Running bias detection & mitigation"
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up bias-pipeline
	@echo "Bias pipeline complete!"

validate:
	@echo "Validating models"
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up model-validation
	@echo "Model validation complete!"

rollback:
	@echo "Running rollback manager"
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up model-rollback
	@echo "Rollback manager complete!"


pipeline-all:
	@echo "Running complete pipeline"
	@echo ""
	@echo "Step 1/6: Loading data"
	@make load-data
	@echo ""
	@echo "Step 2/6: Training models"
	@make train
	@echo ""
	@echo "Step 3/6: Generating predictions"
	@make predict
	@echo ""
	@echo "Step 4/6: Running bias analysis"
	@make bias
	@echo ""
	@echo "Step 5/6: Validating models"
	@make validate
	@echo ""
	@echo "Step 6/6: Rollback models"
	@make rollback
	@echo ""
	@echo "Complete pipeline finished!"

pipeline-train:
	@echo "Running training pipeline"
	@make load-data
	@make train
	@echo "Training pipeline finished!"

pipeline-eval:
	@echo "Running evaluation pipeline"
	@make predict
	@make bias
	@make validate
	@echo "Evaluation pipeline finished!"

shell:
	@echo "Opening interactive shell"
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) run --rm model-dev

logs:
	@echo "Showing logs"
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) logs --tail=100

ps:
	@echo "Running containers:"
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) ps

down:
	@echo "Stopping containers"
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) down
	@echo "Containers stopped!"

clean: down
	@echo "🧹 Cleaning up containers"
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) down --remove-orphans
	@echo "Cleanup complete!"

clean-volumes:
	@echo "WARNING: This will delete all data in volumes!"
	@echo "This includes: data/, mlruns/, mlartifacts/, docs/"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf mlruns/* mlartifacts/* docs/bias_reports/* docs/model_analysis/*; \
		echo "Volumes cleaned!"; \
	else \
		echo "Cancelled."; \
	fi

clean-all: clean
	@echo "Removing Docker image"
	docker rmi $(IMAGE_NAME) 2>/dev/null || true
	@echo "Complete cleanup done!"

check-setup:
	@echo "Checking prerequisites"
	@echo ""
	@command -v docker >/dev/null 2>&1 && echo "Docker installed" || echo "Docker not found"
	@if command -v docker-compose > /dev/null 2>&1; then \
		echo "Docker Compose V1 installed (docker-compose)"; \
	elif docker compose version > /dev/null 2>&1; then \
		echo "Docker Compose V2 installed (docker compose)"; \
	else \
		echo "Docker Compose not found"; \
	fi
	@echo "   Using: $(DOCKER_COMPOSE)"
	@test -f config/gcp_credentials.json && echo "GCP credentials found" || echo "GCP credentials missing (config/gcp_credentials.json)"
	@test -f Dockerfile.model && echo "Dockerfile.model found" || echo "Dockerfile.model missing"
	@test -f $(COMPOSE_FILE) && echo "docker-compose.model.yaml found" || echo "docker-compose.model.yaml missing"
	@test -f model_requirements.txt && echo "model_requirements.txt found" || echo "model_requirements.txt missing"
	@echo ""
	@echo "Required directories:"
	@mkdir -p data mlruns mlartifacts docs config
	@echo "All directories ready"

version:
	@echo "Version Information:"
	@echo ""
	@docker --version
	@if command -v docker-compose > /dev/null 2>&1; then \
		docker-compose --version; \
	fi
	@if docker compose version > /dev/null 2>&1; then \
		docker compose version; \
	fi
	@echo ""
	@echo "Using: $(DOCKER_COMPOSE)"
	@echo ""
	@docker images | grep goodreads-model || echo "Docker image not built yet (run 'make build')"

setup-dirs:
	@echo "Creating required directories"
	@mkdir -p data mlruns mlartifacts docs config
	@echo "Directories created!"

test:
	@echo "Running tests in container"
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) run --rm model-dev pytest datapipeline/tests/ -v

mlflow-ui:
	@echo "Starting MLflow UI on http://localhost:5000"
	@echo "   Press Ctrl+C to stop"
	@mlflow ui --backend-store-uri file://$(shell pwd)/mlruns --host 0.0.0.0 --port 5000

run-custom:
	@echo "Running custom command in container"
	@read -p "Enter Python module (e.g., src.bq_model_training): " module; \
	docker run --rm \
		-v $(shell pwd)/config:/app/config \
		-v $(shell pwd)/data:/app/data \
		-v $(shell pwd)/mlruns:/app/mlruns \
		-v $(shell pwd)/mlartifacts:/app/mlartifacts \
		-v $(shell pwd)/docs:/app/docs \
		-e GOOGLE_APPLICATION_CREDENTIALS=/app/config/gcp_credentials.json \
		$(IMAGE_NAME) \
		python -m $$module
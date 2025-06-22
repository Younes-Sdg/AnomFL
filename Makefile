# Makefile for AnomFL

.PHONY: help install install-dev test test-cov lint format clean data train example

# Default target
help:
	@echo "AnomFL - Federated Learning for Aircraft Anomaly Detection"
	@echo ""
	@echo "Available commands:"
	@echo "  install      - Install the package in development mode"
	@echo "  install-dev  - Install development dependencies"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black"
	@echo "  clean        - Clean generated files"
	@echo "  data         - Generate sample data"
	@echo "  train        - Run federated learning training"
	@echo "  example      - Run basic usage example"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e .[dev]

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 src/ tests/ examples/
	mypy src/

format:
	black src/ tests/ examples/
	isort src/ tests/ examples/

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Data generation
data:
	python -m src.data_generation.cli --fleet-id 1 --num-aircraft 5 --num-anomalous 1

# Training
train:
	python -m src.federated.cli --fleet-id 1 --rounds 5 --local-epochs 50

# Example
example:
	python examples/basic_usage.py

# Full workflow
workflow: data train
	@echo "✅ Complete workflow completed!"

# Development setup
dev-setup: install-dev
	@echo "✅ Development environment setup complete!"
	@echo "Run 'make data' to generate sample data"
	@echo "Run 'make train' to start federated learning"
	@echo "Run 'make example' to run the complete example" 
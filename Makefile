.PHONY: help build up down logs test lint format clean

help:
	@echo "Available targets:"
	@echo "  build   - Build Docker image"
	@echo "  up      - Start service with docker-compose"
	@echo "  down    - Stop service"
	@echo "  logs    - Show service logs"
	@echo "  test    - Run tests"
	@echo "  lint    - Run linters"
	@echo "  format  - Format code with ruff"
	@echo "  clean   - Clean up containers and volumes"

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

test:
	pytest tests/

lint:
	ruff check .
	mypy cog_statistics/

format:
	ruff format .

clean:
	docker-compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

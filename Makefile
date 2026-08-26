.DEFAULT_GOAL := help
.PHONY: help setup lint format typecheck test test-all doctor clean

help:  ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Crea el entorno y instala el paquete en modo editable
	uv sync --group dev
	uv run pre-commit install

lint:  ## ruff check
	uv run ruff check src tests

format:  ## ruff format
	uv run ruff format src tests

typecheck:  ## mypy sobre src
	uv run mypy

test:  ## Tests que no requieren datos con licencia
	uv run pytest -m 'not needs_data'

test-all:  ## Todos los tests, incluidos los que requieren datos
	uv run pytest

doctor:  ## Reporta entorno y disponibilidad de proveedores
	uv run finvex doctor

clean:  ## Borra caches de herramientas
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

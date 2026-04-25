# stl-seed developer commands.
# Targets are POSIX-make compatible (works with macOS GNU Make 3.81 and BSD make).
#
# Usage:
#   make help        # list targets
#   make install     # install dev deps via uv
#   make test        # run CPU-only tests
#   make all         # lint + typecheck + firewall + test

PY := uv run python
PYTEST := uv run pytest
RUFF := uv run ruff
PYRIGHT := uv run pyright
COV_PKG := stl_seed
SRC_DIRS := src tests
PKG_DIR := src

.DEFAULT_GOAL := help

.PHONY: help install install-mlx install-cuda test test-cov lint format \
        typecheck firewall build all clean

help:
	@echo "stl-seed developer targets:"
	@echo "  install       uv sync --extra dev"
	@echo "  install-mlx   uv sync --extra dev --extra mlx   (Apple Silicon only)"
	@echo "  install-cuda  uv sync --extra dev --extra cuda  (Linux/CUDA only)"
	@echo "  test          run CPU-only test suite"
	@echo "  test-cov      run tests with coverage report"
	@echo "  lint          ruff check + format check"
	@echo "  format        ruff format apply (rewrites files)"
	@echo "  typecheck     pyright src (warn-only)"
	@echo "  firewall      REDACTED firewall grep check"
	@echo "  build         uv build wheel + sdist"
	@echo "  all           lint + typecheck + firewall + test"
	@echo "  clean         remove caches, build artifacts, coverage"

install:
	uv sync --extra dev

install-mlx:
	uv sync --extra dev --extra mlx

install-cuda:
	uv sync --extra dev --extra cuda

test:
	$(PYTEST) tests/ -q -m "not gpu and not cuda and not mlx"

test-cov:
	$(PYTEST) tests/ -m "not gpu and not cuda and not mlx" \
		--cov=$(COV_PKG) \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		--cov-report=xml:coverage.xml

lint:
	$(RUFF) check $(SRC_DIRS)
	$(RUFF) format --check $(SRC_DIRS)

format:
	$(RUFF) format $(SRC_DIRS)
	$(RUFF) check --fix $(SRC_DIRS)

typecheck:
	-$(PYRIGHT) $(PKG_DIR)

firewall:
	bash scripts/REDACTED.sh

build:
	uv build

all: lint typecheck firewall test

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .pyright \
	       build dist htmlcov \
	       coverage.xml pytest-junit.xml \
	       *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

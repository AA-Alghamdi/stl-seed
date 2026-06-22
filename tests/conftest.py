"""Pytest configuration: auto-skip tests marked for unavailable hardware.

Markers (registered in pyproject.toml):
- @pytest.mark.cuda  → skipped if torch.cuda.is_available() is False
- @pytest.mark.mlx   → skipped if not on Apple Silicon Darwin
- @pytest.mark.gpu   → skipped if neither CUDA nor MLX is available
- @pytest.mark.slow  → run by default; opt out with `pytest -m "not slow"`
"""

from __future__ import annotations

import platform

import pytest


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _mlx_available() -> bool:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        import importlib.util

        return importlib.util.find_spec("mlx") is not None
    except ImportError:
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip tests requiring unavailable hardware."""
    cuda_ok = _cuda_available()
    mlx_ok = _mlx_available()

    skip_cuda = pytest.mark.skip(reason="CUDA not available on this host")
    skip_mlx = pytest.mark.skip(reason="MLX requires Apple Silicon Darwin with mlx installed")
    skip_gpu = pytest.mark.skip(reason="No GPU backend (CUDA or MLX) available")

    for item in items:
        if "cuda" in item.keywords and not cuda_ok:
            item.add_marker(skip_cuda)
        if "mlx" in item.keywords and not mlx_ok:
            item.add_marker(skip_mlx)
        if "gpu" in item.keywords and not (cuda_ok or mlx_ok):
            item.add_marker(skip_gpu)

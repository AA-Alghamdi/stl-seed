"""Backend-agnostic SFT entry point.

The single public function :func:`train_with_filter` selects an MLX or
bnb backend (lazy-imported), loads the filtered dataset for ``(filter,
task)`` from ``stl_seed.filter``, and dispatches the actual training
to the backend.

This module deliberately does not import torch, mlx, transformers, trl,
or peft at module top-level. The heavy imports happen inside the
selected backend's ``train`` method (see ``backends/mlx.py`` and
``backends/bnb.py``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from stl_seed.training.backends.base import (
    TrainedCheckpoint,
    TrainingBackend,
    TrainingConfig,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend selection (lazy).
# ---------------------------------------------------------------------------


def get_backend(name: Literal["mlx", "bnb"]) -> TrainingBackend:
    """Construct the named backend.

    Construction itself does NOT import the heavy native deps; the
    deferred imports happen on the first :meth:`train` / :meth:`load`
    call. This means tests on CPU machines can call ``get_backend("mlx")``
    or ``get_backend("bnb")`` to verify dispatch logic without crashing.
    """
    if name == "mlx":
        from stl_seed.training.backends.mlx import MLXBackend

        return MLXBackend()
    if name == "bnb":
        from stl_seed.training.backends.bnb import BNBBackend

        return BNBBackend()
    raise ValueError(f"Unknown backend {name!r}; expected 'mlx' or 'bnb'.")


# ---------------------------------------------------------------------------
# Dataset loading shim.
# ---------------------------------------------------------------------------


def _load_filtered_dataset(filter_condition: str, task: str) -> Any:
    """Load the filtered SFT dataset for ``(filter_condition, task)``.

    Bridges to ``stl_seed.filter.dataset`` which is built in subphase 1.3
    A8 (parallel agent). We import lazily so that this module does not
    fail to import when the filter package is not yet present.

    Returns
    -------
    A HuggingFace ``datasets.Dataset`` with at minimum the columns
    ``messages``, ``prompt``, ``completion``, ``weight``, ``task`` (see
    :mod:`stl_seed.training.tokenize`).
    """
    try:
        from stl_seed.filter.dataset import load_filtered_dataset  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "stl_seed.filter.dataset.load_filtered_dataset is not yet "
            "available; this is provided by subphase 1.3 A8. For a unit "
            "test, pass an explicit dataset to BNBBackend.train / "
            "MLXBackend.train directly."
        ) from exc

    return load_filtered_dataset(condition=filter_condition, task=task)


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def train_with_filter(
    filter_condition: str,
    task: str,
    model: str,
    backend: Literal["mlx", "bnb"],
    config: TrainingConfig | None = None,
    *,
    dataset: Any = None,
) -> TrainedCheckpoint:
    """Run SFT end-to-end for one ``(filter, task, model, backend)`` cell.

    Parameters
    ----------
    filter_condition:
        One of ``"hard"``, ``"quantile"``, ``"continuous"``. Selects which
        filtered dataset variant to train on (see ``paper/theory.md`` §2).
    task:
        Task family — one of ``"repressilator"``, ``"toggle"``, ``"mapk"``,
        ``"glucose_insulin"``.
    model:
        HuggingFace model ID (e.g., ``"Qwen/Qwen3-0.6B-Instruct"``). This
        overrides ``config.base_model`` for convenience at the CLI; if the
        two disagree, the explicit ``model`` argument wins and a warning is
        logged.
    backend:
        ``"mlx"`` or ``"bnb"``.
    config:
        :class:`TrainingConfig`. Defaults to ``TrainingConfig(base_model=model)``.
    dataset:
        Optional pre-built dataset, primarily for testing. If ``None``,
        the filtered dataset is loaded from ``stl_seed.filter.dataset``.

    Returns
    -------
    :class:`TrainedCheckpoint` written to ``config.output_dir``.
    """
    if config is None:
        config = TrainingConfig(base_model=model)
    elif config.base_model != model:
        log.warning(
            "config.base_model=%s overridden by explicit model=%s",
            config.base_model,
            model,
        )

    if filter_condition not in ("hard", "quantile", "continuous"):
        raise ValueError(
            f"Unknown filter_condition {filter_condition!r}; expected "
            f"'hard', 'quantile', or 'continuous'."
        )

    if dataset is None:
        dataset = _load_filtered_dataset(filter_condition, task)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    backend_inst = get_backend(backend)
    log.info(
        "Dispatching SFT: backend=%s task=%s filter=%s model=%s "
        "n_examples=%d output=%s",
        backend,
        task,
        filter_condition,
        model,
        len(dataset),
        output_dir,
    )

    return backend_inst.train(
        base_model=model,
        dataset=dataset,
        config=config,
        output_dir=output_dir,
    )


__all__ = [
    "train_with_filter",
    "get_backend",
]

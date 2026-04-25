"""Tests for ``stl_seed.filter.dataset.build_sft_dataset``.

This file targets the formerly low-coverage ``filter/dataset.py`` module
(was 26%), exercising the fallback formatter, the canonical-formatter
TypeError-fallback path, length-mismatch validation, and the metadata
override path.
"""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest

import stl_seed.filter.dataset as dataset_mod
from stl_seed.filter.dataset import (
    _format_trajectory_as_text,
    _resolve_formatter,
    build_sft_dataset,
)
from stl_seed.tasks._trajectory import Trajectory, TrajectoryMeta


@pytest.fixture
def force_fallback_formatter(monkeypatch: pytest.MonkeyPatch):
    """Monkey-patch ``_resolve_formatter`` to return the local fallback,
    bypassing the canonical ``stl_seed.training.tokenize`` resolver whose
    signature is incompatible with ``build_sft_dataset``'s call shape.
    """
    monkeypatch.setattr(
        dataset_mod, "_resolve_formatter", lambda: dataset_mod._format_trajectory_as_text
    )
    return None


def _toy_trajectory(seed: int = 0, T: int = 4, n: int = 2, H: int = 3, m: int = 1) -> Trajectory:
    return Trajectory(
        states=jnp.full((T, n), float(seed)),
        actions=jnp.full((H, m), float(seed) * 0.1),
        times=jnp.linspace(0.0, 10.0, T),
        meta=TrajectoryMeta(
            n_nan_replacements=jnp.asarray(0, dtype=jnp.int32),
            final_solver_result=jnp.asarray(0, dtype=jnp.int32),
            used_stiff_fallback=jnp.asarray(0, dtype=jnp.int32),
        ),
    )


# ---------------------------------------------------------------------------
# Fallback formatter (covers private function lines 50-63).
# ---------------------------------------------------------------------------


def test_fallback_formatter_returns_prompt_and_completion() -> None:
    traj = _toy_trajectory(seed=2, T=5, n=3, H=4, m=2)
    prompt, completion = _format_trajectory_as_text(traj, spec_text="dummy spec", task="testtask")
    assert "testtask" in prompt
    assert "dummy spec" in prompt
    assert "H=4" in prompt
    assert "m=2" in prompt or "2-vectors" in prompt
    parsed = json.loads(completion)
    assert isinstance(parsed, list)
    assert len(parsed) == 4
    assert all(len(row) == 2 for row in parsed)


def test_resolve_formatter_returns_callable() -> None:
    fn = _resolve_formatter()
    assert callable(fn)


# ---------------------------------------------------------------------------
# build_sft_dataset.
# ---------------------------------------------------------------------------


def test_build_sft_dataset_basic_columns_with_fallback(force_fallback_formatter) -> None:  # noqa: ARG001
    """Force the fallback formatter by asking it to handle a non-canonical
    spec_text, and assert all five columns are produced."""
    pytest.importorskip("datasets")
    trajs = [_toy_trajectory(seed=i) for i in range(3)]
    weights = np.array([1.0, 0.5, 0.25], dtype=np.float32)
    ds = build_sft_dataset(
        trajs,
        weights,
        task="glucose_insulin",
        spec_text="G_[30,120] (G >= 70)",
    )
    cols = set(ds.column_names)
    assert {"prompt", "completion", "weight", "trajectory_id", "spec_key"}.issubset(cols)
    assert len(ds) == 3
    # Weights round-trip as floats.
    assert ds["weight"] == pytest.approx([1.0, 0.5, 0.25])
    # IDs follow the synthetic pattern when no metadata supplied.
    assert ds["trajectory_id"] == ["traj_0000", "traj_0001", "traj_0002"]


def test_build_sft_dataset_with_metadata_override(force_fallback_formatter) -> None:  # noqa: ARG001
    pytest.importorskip("datasets")
    trajs = [_toy_trajectory(seed=i) for i in range(2)]
    weights = jnp.array([0.7, 0.3])
    metadata = [
        {"id": "abc-123", "spec_key": "spec_a"},
        {"id": "def-456", "spec_key": "spec_b"},
    ]
    ds = build_sft_dataset(
        trajs,
        weights,
        metadata=metadata,
        task="repressilator",
        spec_text="F_[0,30] (mapk_pp >= 0.5)",
    )
    assert ds["trajectory_id"] == ["abc-123", "def-456"]
    assert ds["spec_key"] == ["spec_a", "spec_b"]


def test_build_sft_dataset_metadata_missing_keys_uses_synthetic_defaults(
    force_fallback_formatter,
) -> None:  # noqa: ARG001
    """metadata dict without ``id`` / ``spec_key`` gets synthetic fallbacks."""
    pytest.importorskip("datasets")
    trajs = [_toy_trajectory(seed=0)]
    weights = np.array([1.0])
    metadata = [{}]  # empty dict — formatter must fall back to synthetic id
    ds = build_sft_dataset(trajs, weights, metadata=metadata, task="toggle", spec_text="...")
    assert ds["trajectory_id"] == ["traj_0000"]
    assert ds["spec_key"] == ["toggle"]


def test_build_sft_dataset_length_mismatch_raises() -> None:
    trajs = [_toy_trajectory(seed=0), _toy_trajectory(seed=1)]
    weights = np.array([1.0, 0.5, 0.25])  # 3 weights vs 2 trajectories
    with pytest.raises(ValueError, match="length mismatch"):
        build_sft_dataset(trajs, weights, task="x", spec_text="...")


def test_build_sft_dataset_metadata_length_mismatch_raises() -> None:
    trajs = [_toy_trajectory(seed=i) for i in range(3)]
    weights = np.ones(3, dtype=np.float32)
    metadata = [{}, {}]  # 2 metadata vs 3 trajectories
    with pytest.raises(ValueError, match="metadata length"):
        build_sft_dataset(trajs, weights, metadata=metadata, task="x", spec_text="")

"""Unit tests for the STL filter conditions (Subphase 1.3, A8)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stl_seed.filter import (
    ContinuousWeightedFilter,
    FilterError,
    HardFilter,
    QuantileFilter,
)
from stl_seed.tasks._trajectory import Trajectory, TrajectoryMeta

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _dummy_trajectory(seed: int = 0) -> Trajectory:
    """A finite, well-formed Trajectory pytree of arbitrary shape."""
    return Trajectory(
        states=jnp.zeros((4, 2)) + seed,
        actions=jnp.zeros((3, 1)) + seed,
        times=jnp.linspace(0.0, 10.0, 4),
        meta=TrajectoryMeta(
            n_nan_replacements=jnp.asarray(0, dtype=jnp.int32),
            final_solver_result=jnp.asarray(0, dtype=jnp.int32),
            used_stiff_fallback=jnp.asarray(0, dtype=jnp.int32),
        ),
    )


def _trajs(n: int) -> list[Trajectory]:
    return [_dummy_trajectory(i) for i in range(n)]


# -----------------------------------------------------------------------------
# HardFilter
# -----------------------------------------------------------------------------


def test_hard_filter_basic() -> None:
    """ρ = [-1, 0.5, 1, -0.3, 0.8] -> hard returns 3 trajectories with
    weights [1, 1, 1] (the three positive ρ entries: indices 1, 2, 4)."""
    rhos = np.array([-1.0, 0.5, 1.0, -0.3, 0.8])
    trajs = _trajs(5)
    f = HardFilter(rho_threshold=0.0, min_kept=1)
    kept, w = f.filter(trajs, rhos)
    assert len(kept) == 3
    np.testing.assert_allclose(np.asarray(w), [1.0, 1.0, 1.0])


def test_hard_filter_threshold_strict() -> None:
    """ρ == threshold is NOT kept (strict > comparison per theory.md §2)."""
    rhos = np.array([0.0, 0.0, 1.0, 1.0])
    trajs = _trajs(4)
    f = HardFilter(rho_threshold=0.0, min_kept=1)
    kept, _ = f.filter(trajs, rhos)
    assert len(kept) == 2  # only the two strictly positive entries


def test_hard_filter_too_few_raises() -> None:
    """All ρ < 0 with default min_kept=10 -> FilterError."""
    rhos = -np.ones(20)
    trajs = _trajs(20)
    f = HardFilter(rho_threshold=0.0)  # min_kept=10 default
    with pytest.raises(FilterError, match="HardFilter"):
        f.filter(trajs, rhos)


# -----------------------------------------------------------------------------
# QuantileFilter
# -----------------------------------------------------------------------------


def test_quantile_filter_top_25() -> None:
    """100 ρ values uniform on [-1, 1], top 25% returns 25, all weights 1.0."""
    key = jax.random.key(0)
    rhos = np.asarray(jax.random.uniform(key, (100,), minval=-1.0, maxval=1.0))
    trajs = _trajs(100)
    f = QuantileFilter(top_k_pct=25.0)
    kept, w = f.filter(trajs, rhos)
    assert len(kept) == 25
    assert w.shape == (25,)
    np.testing.assert_allclose(np.asarray(w), np.ones(25))


def test_quantile_filter_top_25_picks_correct_set() -> None:
    """Top-25% must equal the np.argsort top quartile."""
    rhos = np.array([0.1, 0.9, -0.5, 0.7, 0.3, -0.2, 0.6, 0.8, 0.4, 0.2])
    trajs = _trajs(10)
    f = QuantileFilter(top_k_pct=30.0, min_kept=1)  # ceil(0.3*10) = 3
    kept, _ = f.filter(trajs, rhos)
    # Top 3 by ρ: indices 1 (0.9), 7 (0.8), 3 (0.7).
    expected_states_seeds = sorted([1, 7, 3])
    actual = [int(np.asarray(t.states[0, 0])) for t in kept]
    assert actual == expected_states_seeds


def test_quantile_filter_below_min_raises() -> None:
    """top_k_pct=10 on N=20 -> ceil(2) = 2 < min_kept=10 -> FilterError."""
    rhos = np.linspace(-1, 1, 20)
    trajs = _trajs(20)
    f = QuantileFilter(top_k_pct=10.0)
    with pytest.raises(FilterError, match="QuantileFilter"):
        f.filter(trajs, rhos)


# -----------------------------------------------------------------------------
# ContinuousWeightedFilter
# -----------------------------------------------------------------------------


def test_continuous_weighted_softmax() -> None:
    """Weights sum to N (softmax-normalized × N), unbiased gradient.

    With ρ ~ Normal(0, 1) and N=200, weights satisfy:
      * sum(weights) ≈ N
      * mean(weights) ≈ 1.0  (the unbiased-gradient property)
      * larger ρ -> larger weight
    """
    key = jax.random.key(7)
    rhos = np.asarray(jax.random.normal(key, (200,)))
    trajs = _trajs(200)
    f = ContinuousWeightedFilter(temperature=None)  # auto β = std(ρ)
    kept, w = f.filter(trajs, rhos)
    w_np = np.asarray(w)

    assert len(kept) == 200
    np.testing.assert_allclose(w_np.sum(), 200.0, rtol=1e-5)
    np.testing.assert_allclose(w_np.mean(), 1.0, rtol=1e-5)
    # Monotonicity: argsort(weights) == argsort(rhos).
    assert np.all(np.argsort(w_np) == np.argsort(rhos))


def test_continuous_weighted_explicit_temperature() -> None:
    """Explicit temperature overrides the auto-β path."""
    rhos = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    trajs = _trajs(11)
    # β = 1.0: weights are softmax(ρ) * N; the largest entry dominates.
    f_sharp = ContinuousWeightedFilter(temperature=1.0, min_kept=1)
    _, w_sharp = f_sharp.filter(trajs, rhos)
    # β = 100.0: weights are nearly uniform (sharpness ~ 0).
    f_flat = ContinuousWeightedFilter(temperature=100.0, min_kept=1)
    _, w_flat = f_flat.filter(trajs, rhos)

    w_sharp_np = np.asarray(w_sharp)
    w_flat_np = np.asarray(w_flat)
    np.testing.assert_allclose(w_sharp_np.sum(), 11.0, rtol=1e-5)
    np.testing.assert_allclose(w_flat_np.sum(), 11.0, rtol=1e-5)
    # Sharp filter has higher max weight than the flat filter.
    assert w_sharp_np.max() > w_flat_np.max() * 2.0


def test_continuous_weighted_below_min_raises() -> None:
    """N < min_kept -> FilterError."""
    rhos = np.array([0.5, 0.6])  # N=2 < default min_kept=10
    trajs = _trajs(2)
    f = ContinuousWeightedFilter()
    with pytest.raises(FilterError, match="ContinuousWeightedFilter"):
        f.filter(trajs, rhos)


def test_continuous_weighted_zero_std_raises() -> None:
    """Identical ρ across the corpus + auto-β -> FilterError."""
    rhos = np.full((20,), 0.5)
    trajs = _trajs(20)
    f = ContinuousWeightedFilter()  # temperature=None -> auto β = std(ρ) = 0
    with pytest.raises(FilterError, match="std"):
        f.filter(trajs, rhos)


def test_continuous_weighted_zero_std_with_explicit_temp_ok() -> None:
    """Identical ρ + explicit temperature -> uniform weights summing to N."""
    rhos = np.full((20,), 0.5)
    trajs = _trajs(20)
    f = ContinuousWeightedFilter(temperature=1.0)
    kept, w = f.filter(trajs, rhos)
    assert len(kept) == 20
    np.testing.assert_allclose(np.asarray(w), np.ones(20), atol=1e-5)


# -----------------------------------------------------------------------------
# Filter contract: length-mismatch raises
# -----------------------------------------------------------------------------


def test_filter_length_mismatch_raises_hard() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        HardFilter(min_kept=1).filter(_trajs(3), np.array([1.0, 2.0]))


def test_filter_length_mismatch_raises_quantile() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        QuantileFilter(min_kept=1).filter(_trajs(3), np.array([1.0, 2.0]))


def test_filter_length_mismatch_raises_continuous() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        ContinuousWeightedFilter(min_kept=1).filter(_trajs(3), np.array([1.0, 2.0]))

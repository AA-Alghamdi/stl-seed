"""Tests for ``stl_seed.specs.calibration`` (Subphase 1.6, A20).

The calibration module currently has 0% coverage. These tests exercise
both the diagnostic (no-threshold) and sweep (with-threshold) modes
against a stub trajectory sampler / robustness function — no JAX or
Diffrax integration is needed because the module is duck-typed against
``TrajectorySampler`` / ``RobustnessFn`` Protocols.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from stl_seed.specs import REGISTRY, STLSpec
from stl_seed.specs.calibration import (
    CalibrationResult,
    calibrate_spec,
    scan_threshold,
    success_rate,
)

# ---------------------------------------------------------------------------
# Stub sampler / robustness function.
# ---------------------------------------------------------------------------


class _ConstSampler:
    """Returns a fixed list of "trajectories" (treated opaquely by ``rho``)."""

    def __init__(self, n: int) -> None:
        self._items = list(range(n))

    def sample(self, n: int, seed: int) -> list[int]:  # noqa: ARG002
        return self._items[:n]


def _rho_constant(value: float):
    """Return a robustness function that always reports ``value``."""

    def rho(traj, spec):  # noqa: ARG001
        return value

    return rho


def _rho_index_above(threshold: int):
    """rho(traj=i, spec) = +1 iff i >= threshold else -1."""

    def rho(traj, spec):  # noqa: ARG001
        return 1.0 if int(traj) >= threshold else -1.0

    return rho


def _rho_threshold_aware(metadata_key: str, target: float):
    """rho depends on a threshold value pulled from spec.metadata.

    Returns +1 for half the trajectories and -1 for the other half WHEN
    spec.metadata[key] == target; otherwise all -1. Lets tests verify
    that ``apply`` actually mutates the spec and the new metadata is
    consulted.
    """

    def rho(traj, spec):
        if spec.metadata.get(metadata_key) != target:
            return -1.0
        return 1.0 if int(traj) % 2 == 0 else -1.0

    return rho


def _spec(name: str = "stub.spec") -> STLSpec:
    """Borrow a real registered spec; it doesn't matter which — we only
    need the dataclass shape and a mutable metadata dict."""

    return REGISTRY["glucose_insulin.tir.easy"]


def _apply_threshold(spec: STLSpec, value: float) -> STLSpec:
    """Test-side ``apply``: write ``value`` into ``spec.metadata['threshold']``."""
    new_meta = dict(spec.metadata)
    new_meta["threshold"] = value
    return replace(spec, metadata=new_meta)


# ---------------------------------------------------------------------------
# success_rate
# ---------------------------------------------------------------------------


def test_success_rate_all_satisfied() -> None:
    """rho >= 0 for every trajectory -> success_rate == 1.0."""
    spec = _spec()
    sampler = _ConstSampler(20)
    rho = _rho_constant(0.5)
    assert success_rate(spec, sampler, rho, n_samples=20) == 1.0


def test_success_rate_all_violated() -> None:
    spec = _spec()
    sampler = _ConstSampler(20)
    rho = _rho_constant(-0.5)
    assert success_rate(spec, sampler, rho, n_samples=20) == 0.0


def test_success_rate_mixed() -> None:
    """Half of 20 trajectories pass, half fail -> success_rate == 0.5."""
    spec = _spec()
    sampler = _ConstSampler(20)
    rho = _rho_index_above(threshold=10)  # i >= 10 passes
    assert success_rate(spec, sampler, rho, n_samples=20) == pytest.approx(0.5)


def test_success_rate_handles_exception_in_rho() -> None:
    """Exceptions in ``rho`` are caught and counted as failures."""
    spec = _spec()
    sampler = _ConstSampler(5)

    def boom(traj, spec):  # noqa: ARG001
        raise RuntimeError("simulated solver blow-up")

    assert success_rate(spec, sampler, boom, n_samples=5) == 0.0


def test_success_rate_handles_nan_inf() -> None:
    """NaN / +/-inf rho counts as failure (matches CLAUDE.md no-silent-swallow)."""
    spec = _spec()
    sampler = _ConstSampler(4)

    nans = iter([float("nan"), float("inf"), float("-inf"), 1.0])

    def rho(traj, spec):  # noqa: ARG001
        return next(nans)

    # Only the 4th trajectory (rho=1.0) succeeds.
    assert success_rate(spec, sampler, rho, n_samples=4) == 0.25


def test_success_rate_zero_samples_division_safe() -> None:
    """``max(1, n_samples)`` guard: n_samples=0 must not raise ZeroDivision."""
    spec = _spec()

    class _EmptySampler:
        def sample(self, n: int, seed: int) -> list:  # noqa: ARG002
            return []

    rho = _rho_constant(1.0)
    # The sampler returns 0 trajectories; the divisor is clamped to 1.
    assert success_rate(spec, _EmptySampler(), rho, n_samples=0) == 0.0


# ---------------------------------------------------------------------------
# scan_threshold
# ---------------------------------------------------------------------------


def test_scan_threshold_returns_value_rate_pairs() -> None:
    spec = _spec()
    sampler = _ConstSampler(10)
    candidates = [0.0, 0.5, 1.0]

    # rho returns +1 only when threshold metadata == 0.5.
    rho = _rho_threshold_aware("threshold", target=0.5)
    sweep = scan_threshold(
        spec=spec,
        sampler=sampler,
        rho=rho,
        threshold_key="threshold",
        candidates=candidates,
        apply=_apply_threshold,
        n_samples=10,
    )
    assert isinstance(sweep, tuple) and len(sweep) == 3
    # Only threshold=0.5 yields any successes (half the sampler indices).
    by_value = dict(sweep)
    assert by_value[0.0] == 0.0
    assert by_value[1.0] == 0.0
    assert by_value[0.5] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# calibrate_spec — diagnostic mode
# ---------------------------------------------------------------------------


def test_calibrate_diagnostic_in_band() -> None:
    spec = _spec()
    sampler = _ConstSampler(20)
    rho = _rho_index_above(threshold=14)  # 6/20 = 0.30 success rate
    result = calibrate_spec(spec, sampler, rho, n_samples=20, target_range=(0.15, 0.55))
    assert isinstance(result, CalibrationResult)
    assert result.in_band is True
    assert result.success_rate == pytest.approx(0.30)
    assert result.threshold_key is None
    assert "diagnostic" in result.notes


def test_calibrate_diagnostic_out_of_band() -> None:
    spec = _spec()
    sampler = _ConstSampler(20)
    rho = _rho_constant(1.0)  # 100% success — too easy
    result = calibrate_spec(spec, sampler, rho, n_samples=20, target_range=(0.15, 0.55))
    assert result.in_band is False
    assert result.success_rate == 1.0


def test_calibrate_diagnostic_invalid_target_range() -> None:
    """target_range outside [0, 1] or with low >= high -> ValueError."""
    spec = _spec()
    sampler = _ConstSampler(5)
    rho = _rho_constant(0.5)
    with pytest.raises(ValueError, match="target_range"):
        calibrate_spec(spec, sampler, rho, target_range=(0.7, 0.3))
    with pytest.raises(ValueError, match="target_range"):
        calibrate_spec(spec, sampler, rho, target_range=(-0.1, 0.5))
    with pytest.raises(ValueError, match="target_range"):
        calibrate_spec(spec, sampler, rho, target_range=(0.0, 1.5))


# ---------------------------------------------------------------------------
# calibrate_spec — sweep mode
# ---------------------------------------------------------------------------


def test_calibrate_sweep_picks_in_band_threshold() -> None:
    spec = _spec()
    sampler = _ConstSampler(20)

    # Construct a rho that varies with threshold value: each candidate
    # threshold has a known success rate.
    success_by_threshold = {0.0: 0.0, 0.3: 0.30, 0.5: 0.40, 0.8: 1.0}

    def rho(traj, spec):
        th = spec.metadata.get("threshold")
        rate = success_by_threshold.get(th, 0.0)
        # Use traj index to deterministically realize the rate over 20 samples.
        return 1.0 if int(traj) < int(rate * 20) else -1.0

    result = calibrate_spec(
        spec,
        sampler,
        rho,
        threshold_key="threshold",
        candidates=[0.0, 0.3, 0.5, 0.8],
        apply=_apply_threshold,
        n_samples=20,
        target_range=(0.15, 0.55),
    )
    assert result.in_band is True
    # Centre of band is 0.35; closest in-band sweep value is 0.40 (th=0.5).
    assert result.threshold_value == 0.5
    assert result.success_rate == pytest.approx(0.40)
    assert result.threshold_key == "threshold"
    assert result.sweep is not None and len(result.sweep) == 4
    # Calibration metadata is recorded on the returned spec.
    assert result.spec.metadata["calibrated_threshold"] == 0.5
    assert result.spec.metadata["calibrated_success_rate"] == pytest.approx(0.40)
    assert result.spec.metadata["calibrated_n_samples"] == 20
    assert result.spec.metadata["calibrated_target_range"] == (0.15, 0.55)


def test_calibrate_sweep_no_in_band_raises() -> None:
    spec = _spec()
    sampler = _ConstSampler(20)
    # All candidates produce rho = -1 -> success rate 0 -> never in band.
    rho = _rho_constant(-1.0)
    with pytest.raises(RuntimeError, match="no candidate threshold"):
        calibrate_spec(
            spec,
            sampler,
            rho,
            threshold_key="threshold",
            candidates=[0.0, 0.5, 1.0],
            apply=_apply_threshold,
            target_range=(0.15, 0.55),
        )


def test_calibrate_sweep_requires_apply_and_candidates() -> None:
    spec = _spec()
    sampler = _ConstSampler(5)
    rho = _rho_constant(0.5)
    # Pass threshold_key but omit candidates AND apply -> ValueError.
    with pytest.raises(ValueError, match="Sweep mode"):
        calibrate_spec(spec, sampler, rho, threshold_key="threshold")

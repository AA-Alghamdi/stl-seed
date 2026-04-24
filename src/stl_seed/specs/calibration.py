"""STL spec calibration utilities.

A spec ``phi`` is *calibrated* on a (task, simulator) pair iff the random-
policy success rate falls in a target band ``[low, high]`` (default
``[0.15, 0.55]``). A spec that is too easy (``> high``) gives the SFT loop
no useful gradient; one that is too hard (``< low``) gives the loop almost
no positive examples to imitate.

This module provides:

* :func:`success_rate` — compute the empirical random-policy success rate
  for a spec given a trajectory sampler.
* :func:`scan_threshold` — sweep a single threshold and return the success
  rate at each value.
* :func:`calibrate_spec` — adjust one threshold (specified by a key into the
  spec's ``metadata`` dictionary) so the success rate enters the target band.

Phase 1 / subphase 1.2 status: this is the *interface*. The trajectory
sampler is plugged in by subphase 1.4 (random-policy trajectory generation)
and the robustness backend is plugged in by subphase 1.3 (STL evaluator).
The functions below operate against duck-typed protocols so they can be
exercised with a stub sampler in unit tests without pulling JAX in.

REDACTED firewall posture. Calibration only adjusts thresholds *within their
literature-derived plausibility bands*. The default behaviour of
:func:`calibrate_spec` is to fail loudly (``RuntimeError``) when a spec
cannot be brought into the target band without leaving its cited band.
This is deliberate: silently relaxing a clinical threshold (e.g. raising
the severe-hypoglycaemia bound from 54 mg/dL to 60 mg/dL to make the spec
easier) would be a scientific-integrity violation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from stl_seed.specs import STLSpec

# ---------------------------------------------------------------------------
# Protocols.
# ---------------------------------------------------------------------------


class TrajectorySampler(Protocol):
    """Duck-typed sampler that yields ``n`` random-policy trajectories.

    Implementations live in subphase 1.4 (``stl_seed.tasks.<family>``).
    Each trajectory has shape ``(T_steps, signal_dim)`` and is a NumPy
    array.
    """

    def sample(self, n: int, seed: int) -> Sequence:  # noqa: D401
        """Return ``n`` trajectories (NumPy arrays)."""
        ...


class RobustnessFn(Protocol):
    """Duck-typed STL robustness function ``rho(traj, spec) -> float``.

    Implementations live in subphase 1.3.
    """

    def __call__(self, traj, spec: STLSpec) -> float:  # noqa: D401
        ...


# ---------------------------------------------------------------------------
# Calibration result.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationResult:
    """Outcome of a single calibration run."""

    spec: STLSpec
    success_rate: float
    n_samples: int
    target_range: tuple[float, float]
    in_band: bool
    threshold_key: str | None = None
    threshold_value: float | None = None
    sweep: tuple[tuple[float, float], ...] | None = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Core.
# ---------------------------------------------------------------------------


def success_rate(
    spec: STLSpec,
    sampler: TrajectorySampler,
    rho: RobustnessFn,
    n_samples: int = 100,
    seed: int = 0,
) -> float:
    """Empirical random-policy success rate ``Pr[rho(traj, spec) >= 0]``.

    Robust to NaN / Inf in the robustness scalar (common when an ODE solve
    blows up): such trajectories are counted as *failures* (rho < 0
    interpretation). This matches the firewall's "no silent error
    swallowing" rule from ``CLAUDE.md``.
    """

    trajs = sampler.sample(n_samples, seed=seed)
    successes = 0
    for traj in trajs:
        try:
            r = float(rho(traj, spec))
        except Exception:
            r = float("-inf")
        if r != r or r == float("inf") or r == float("-inf"):
            # NaN or +/-inf -> failure.
            continue
        if r >= 0.0:
            successes += 1
    return successes / max(1, n_samples)


def scan_threshold(
    spec: STLSpec,
    sampler: TrajectorySampler,
    rho: RobustnessFn,
    threshold_key: str,
    candidates: Iterable[float],
    apply: Callable[[STLSpec, float], STLSpec],
    n_samples: int = 100,
    seed: int = 0,
) -> tuple[tuple[float, float], ...]:
    """Sweep a threshold and return ``((value, success_rate), ...)``.

    ``apply(spec, value)`` returns a *new* :class:`STLSpec` with the named
    threshold replaced by ``value``. Callers provide ``apply`` so the
    calibration engine does not need to know the per-spec wiring.
    """

    out: list[tuple[float, float]] = []
    for v in candidates:
        candidate = apply(spec, v)
        sr = success_rate(candidate, sampler, rho, n_samples=n_samples, seed=seed)
        out.append((v, sr))
    return tuple(out)


def calibrate_spec(
    spec: STLSpec,
    sampler: TrajectorySampler,
    rho: RobustnessFn,
    *,
    threshold_key: str | None = None,
    candidates: Iterable[float] | None = None,
    apply: Callable[[STLSpec, float], STLSpec] | None = None,
    n_samples: int = 100,
    target_range: tuple[float, float] = (0.15, 0.55),
    seed: int = 0,
) -> CalibrationResult:
    """Adjust ``spec`` so random-policy success rate lies in ``target_range``.

    Two modes:

    * **Diagnostic mode** (``threshold_key is None``): compute the success
      rate for the spec as-is and return whether it is in band. This is the
      mode subphase 1.2 unit tests will use.
    * **Sweep mode** (``threshold_key`` and ``candidates`` provided): sweep
      the named threshold over the candidate list, pick the value whose
      success rate is closest to the centre of the band while remaining
      inside the band, and return a new :class:`STLSpec` with that
      threshold applied. Raises ``RuntimeError`` if no candidate lies in
      the band; this is by design (see module docstring).

    Parameters
    ----------
    spec
        The spec to calibrate.
    sampler
        A trajectory sampler — see :class:`TrajectorySampler`.
    rho
        A robustness evaluator — see :class:`RobustnessFn`.
    threshold_key
        Key into ``spec.metadata`` that names the threshold to vary.
    candidates
        Iterable of candidate threshold values.
    apply
        Callable ``(spec, value) -> spec`` that produces a candidate spec.
    n_samples
        Number of random-policy trajectories per evaluation.
    target_range
        ``(low, high)`` band on the random-policy success rate.
    seed
        Random seed for the trajectory sampler.
    """

    low, high = target_range
    if not (0.0 <= low < high <= 1.0):
        raise ValueError(f"target_range must be 0 <= low < high <= 1, got {target_range}")

    # Diagnostic mode.
    if threshold_key is None:
        sr = success_rate(spec, sampler, rho, n_samples=n_samples, seed=seed)
        return CalibrationResult(
            spec=spec,
            success_rate=sr,
            n_samples=n_samples,
            target_range=target_range,
            in_band=(low <= sr <= high),
            notes="diagnostic mode (no threshold sweep)",
        )

    # Sweep mode.
    if candidates is None or apply is None:
        raise ValueError(
            "Sweep mode requires both `candidates` and `apply` to be provided."
        )
    sweep = scan_threshold(
        spec=spec,
        sampler=sampler,
        rho=rho,
        threshold_key=threshold_key,
        candidates=candidates,
        apply=apply,
        n_samples=n_samples,
        seed=seed,
    )
    in_band = [(v, sr) for v, sr in sweep if low <= sr <= high]
    if not in_band:
        raise RuntimeError(
            f"calibrate_spec: no candidate threshold for key {threshold_key!r} "
            f"yielded a success rate in {target_range}. Sweep: {sweep}. "
            "Refusing to silently relax the spec — see module docstring."
        )
    centre = 0.5 * (low + high)
    best_v, best_sr = min(in_band, key=lambda pair: abs(pair[1] - centre))
    new_spec = apply(spec, best_v)
    new_meta = dict(new_spec.metadata)
    new_meta[f"calibrated_{threshold_key}"] = best_v
    new_meta["calibrated_success_rate"] = best_sr
    new_meta["calibrated_n_samples"] = n_samples
    new_meta["calibrated_target_range"] = target_range
    new_spec = replace(new_spec, metadata=new_meta)
    return CalibrationResult(
        spec=new_spec,
        success_rate=best_sr,
        n_samples=n_samples,
        target_range=target_range,
        in_band=True,
        threshold_key=threshold_key,
        threshold_value=best_v,
        sweep=sweep,
        notes="sweep mode",
    )


__all__ = [
    "TrajectorySampler",
    "RobustnessFn",
    "CalibrationResult",
    "success_rate",
    "scan_threshold",
    "calibrate_spec",
]

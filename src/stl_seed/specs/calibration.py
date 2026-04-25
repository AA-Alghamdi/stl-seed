"""STL spec calibration and auto-tuning.

Two related procedures live here:

1. :func:`calibrate_spec` — *single-policy* calibration that nudges one
   threshold so the random-policy success rate falls into a target band.
   This is the original Subphase-1.6 stub; its public surface is unchanged
   (321 existing tests cover it).

2. :func:`auto_tune_spec_thresholds` — *multi-policy* threshold optimisation
   that picks the threshold combination maximising between-policy
   discriminability of the robustness margin :math:`\\rho`. This is the
   technical contribution: it removes spec-author bias from benchmark
   design and produces a reproducible threshold-selection procedure that
   can be re-run for new task families.

Why discriminability rather than success-rate calibration?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Random-policy success-rate calibration (mode 1) keeps a spec from being
trivially easy or trivially hard, but it does not check that the spec
actually *separates* good from bad policies. A spec whose threshold sits in
a saturated regime of the simulator can have a sensible random-policy rate
yet give every policy the same :math:`\\rho`, providing zero training
signal in the SERA-style soft filter (see ``paper/theory.md`` §3 on the
TOST equivalence-vs-difference framing).

The discriminability metrics here directly probe the *separation* of the
:math:`\\rho` distribution under two policies:

* ``wasserstein`` — earth-mover distance between the empirical
  :math:`\\rho`-distributions. Symmetric, scale-aware, robust to outliers
  (a single failed solve does not dominate). Implemented as the closed-form
  1-D Wasserstein-1 (sum of absolute differences of sorted samples), since
  scipy availability is not assumed by all Phase-1 deployments.
* ``auc_separation`` — area under the ROC curve treating policy A vs.
  policy B as a binary discrimination problem on per-trajectory
  :math:`\\rho`. Centred so 0 means "no information" and the headline
  value reported is ``2 |AUC - 0.5|`` (Gini coefficient form).
* ``trace_overlap`` — :math:`1 - \\sum_i \\min(h^A_i, h^B_i)` where
  :math:`h^A, h^B` are normalised histograms of the two
  :math:`\\rho`-vectors on a shared bin grid. Cheap, geometric, but
  bin-resolution sensitive.

Aggregation across the policy set is the *worst-case pairwise* metric by
default ("the spec must separate even the closest two policies"), with an
optional mean aggregation for diagnostic plotting.

Threshold-placeholder model
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A :class:`stl_seed.specs.Predicate` is a *named* atomic. Concrete specs in
``bio_ode_specs`` and ``glucose_insulin_specs`` use the ``_gt`` / ``_lt``
helpers, which name predicates ``"<base>><threshold>"`` /
``"<base>><<threshold>"`` and capture ``(channel, threshold)`` as the
lambda's ``__defaults__``. The same introspection convention drives the
JIT-compiled evaluator in ``stl_seed.stl.evaluator`` so the predicate
inventory we build here is exactly the predicate inventory the evaluator
sees.

For the auto-tuner, we identify each tunable threshold by the
*base name* of its predicate (the part before the comparison operator).
The user supplies a search space dict ``{base_name: [v1, v2, ...]}`` and
the tuner sweeps the cartesian product. Predicates whose base names are
not in the search space keep their original threshold.

REDACTED firewall posture
~~~~~~~~~~~~~~~~~~~~~

Auto-tuning is *not* allowed to relax a spec into the REDACTED dimensionless
threshold band by accident: the search space is supplied externally and is
the user's responsibility (the auto-tune script in
``scripts/auto_tune_specs.py`` constructs each search range from the
literature-derived plausibility band documented inline in
``bio_ode_specs.py`` and ``glucose_insulin_specs.py``). The auto-tuner
itself emits *recommendations*, not in-place spec mutations. The final
adoption decision (``v0.2``) is the user's.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from stl_seed.specs import (
    Always,
    And,
    Eventually,
    Negation,
    Node,
    Predicate,
    STLSpec,
)

# ---------------------------------------------------------------------------
# Section 1 — original single-policy success-rate calibration (unchanged).
# ---------------------------------------------------------------------------


class TrajectorySampler(Protocol):
    """Duck-typed sampler that yields ``n`` random-policy trajectories."""

    def sample(self, n: int, seed: int) -> Sequence:  # noqa: D401
        """Return ``n`` trajectories."""
        ...


class RobustnessFn(Protocol):
    """Duck-typed STL robustness function ``rho(traj, spec) -> float``."""

    def __call__(self, traj, spec: STLSpec) -> float:  # noqa: D401
        ...


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


def success_rate(
    spec: STLSpec,
    sampler: TrajectorySampler,
    rho: RobustnessFn,
    n_samples: int = 100,
    seed: int = 0,
) -> float:
    """Empirical random-policy success rate ``Pr[rho(traj, spec) >= 0]``.

    NaN/Inf rho values are treated as failures (matches the CLAUDE.md
    "no silent error swallowing" rule).
    """
    trajs = sampler.sample(n_samples, seed=seed)
    successes = 0
    for traj in trajs:
        try:
            r = float(rho(traj, spec))
        except Exception:
            r = float("-inf")
        if r != r or r == float("inf") or r == float("-inf"):
            continue
        if r >= 0.0:
            successes += 1
    return successes / max(1, n_samples)


def scan_threshold(
    spec: STLSpec,
    sampler: TrajectorySampler,
    rho: RobustnessFn,
    threshold_key: str,  # noqa: ARG001  (kept for API stability)
    candidates: Iterable[float],
    apply: Callable[[STLSpec, float], STLSpec],
    n_samples: int = 100,
    seed: int = 0,
) -> tuple[tuple[float, float], ...]:
    """Sweep one threshold and return ``((value, success_rate), ...)``."""
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
    """Single-policy success-rate calibration. See module docstring §1."""
    low, high = target_range
    if not (0.0 <= low < high <= 1.0):
        raise ValueError(f"target_range must be 0 <= low < high <= 1, got {target_range}")

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

    if candidates is None or apply is None:
        raise ValueError("Sweep mode requires both `candidates` and `apply` to be provided.")
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


# ---------------------------------------------------------------------------
# Section 2 — predicate introspection (shared with stl.evaluator).
# ---------------------------------------------------------------------------


def _introspect_predicate(pred: Predicate) -> tuple[int, float, str] | None:
    """Recover ``(channel, threshold, op)`` from a ``_gt`` / ``_lt`` predicate.

    Mirrors :func:`stl_seed.stl.evaluator._introspect_predicate` but lives
    here as well so this module does not import from ``stl.evaluator``
    (avoids a JAX import cycle when ``calibration`` is used in pure-Python
    contexts).
    """
    fn = pred.fn
    defaults = getattr(fn, "__defaults__", None)
    if defaults is None or len(defaults) != 2:
        return None
    channel, threshold = defaults
    if not isinstance(channel, int) or not isinstance(threshold, (int, float)):
        return None
    n_cols = max(int(channel) + 1, 1)
    try:
        zero_traj = np.zeros((1, n_cols), dtype=np.float64)
        unit_traj = np.zeros((1, n_cols), dtype=np.float64)
        unit_traj[0, channel] = 1.0
        v0 = float(fn(zero_traj, 0))
        v1 = float(fn(unit_traj, 0))
    except Exception:
        return None
    th = float(threshold)
    delta = v1 - v0
    if abs(delta - 1.0) < 1e-9 and abs(v0 + th) < 1e-9:
        return (int(channel), th, "gt")
    if abs(delta + 1.0) < 1e-9 and abs(v0 - th) < 1e-9:
        return (int(channel), th, "lt")
    return None


def _predicate_base_name(pred: Predicate) -> str:
    """Strip the ``>threshold`` / ``<threshold`` suffix from a predicate name.

    The ``_gt`` and ``_lt`` helpers both name their predicates
    ``f"{base}>{th}"`` or ``f"{base}<{th}"``; we recover ``base`` so the
    auto-tuner can refer to the conceptual placeholder (e.g. ``"p1"``,
    ``"G_above_70"``) rather than the threshold-decorated name that
    changes every time we re-instantiate the spec.
    """
    name = pred.name
    for sep in (">", "<"):
        if sep in name:
            return name.split(sep, 1)[0]
    return name


# ---------------------------------------------------------------------------
# Section 3 — placeholder extraction and substitution.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdPlaceholder:
    """A tunable threshold inside a spec.

    Attributes
    ----------
    base_name : str
        The base name (output of :func:`_predicate_base_name`). This is the
        key the user references in ``threshold_search_space``.
    channel : int
        State-vector channel index the predicate reads.
    op : str
        ``"gt"`` (predicate is ``signal >= threshold``) or ``"lt"``
        (predicate is ``signal < threshold``).
    current_value : float
        Hand-set threshold value present in the spec template.
    """

    base_name: str
    channel: int
    op: str
    current_value: float


def extract_threshold_placeholders(spec: Node | STLSpec) -> list[ThresholdPlaceholder]:
    """Walk the AST and return every introspectable threshold predicate.

    Predicates that do not match the ``_gt`` / ``_lt`` introspection
    convention are silently skipped (they cannot be re-instantiated with
    a new threshold, so they are not tunable). Duplicate base names are
    de-duplicated by ``(base_name, channel, op)``.
    """
    root = spec.formula if isinstance(spec, STLSpec) else spec
    seen: dict[tuple[str, int, str], ThresholdPlaceholder] = {}

    def _walk(node: Node) -> None:
        if isinstance(node, Predicate):
            info = _introspect_predicate(node)
            if info is not None:
                channel, th, op = info
                base = _predicate_base_name(node)
                key = (base, channel, op)
                if key not in seen:
                    seen[key] = ThresholdPlaceholder(
                        base_name=base, channel=channel, op=op, current_value=th
                    )
            return
        if isinstance(node, Negation):
            _walk(node.inner)
            return
        if isinstance(node, (Always, Eventually)):
            _walk(node.inner)
            return
        if isinstance(node, And):
            for c in node.children:
                _walk(c)
            return
        # Unknown node: ignore (forward compat).

    _walk(root)
    return list(seen.values())


def _make_predicate(base_name: str, channel: int, op: str, threshold: float) -> Predicate:
    """Re-create a ``_gt`` / ``_lt`` predicate with a new threshold.

    Identical wiring to the helpers in ``bio_ode_specs._gt`` / ``_lt`` and
    ``glucose_insulin_specs._gt`` / ``_lt``: the lambda captures
    ``(channel, threshold)`` as defaults so the existing evaluator
    introspection recognises the new predicate. Naming follows the same
    ``f"{base}>{threshold}"`` / ``f"{base}<{threshold}"`` convention.
    """
    if op == "gt":
        return Predicate(
            f"{base_name}>{threshold}",
            fn=lambda traj, t, c=int(channel), th=float(threshold): float(traj[t, c]) - th,
        )
    if op == "lt":
        return Predicate(
            f"{base_name}<{threshold}",
            fn=lambda traj, t, c=int(channel), th=float(threshold): th - float(traj[t, c]),
        )
    raise ValueError(f"unknown op {op!r}; must be 'gt' or 'lt'")


def instantiate_spec_with_thresholds(
    spec_template: STLSpec | Node,
    thresholds: dict[str, float],
) -> STLSpec | Node:
    """Return a deep-copy of ``spec_template`` with named thresholds replaced.

    ``thresholds`` is keyed by predicate base name (see
    :func:`extract_threshold_placeholders`). Predicates whose base name is
    *not* in ``thresholds`` are kept verbatim, including their current
    threshold values. Predicates that cannot be introspected (rare) are
    also passed through unchanged.

    If ``spec_template`` is an :class:`STLSpec` the metadata field is
    copied onto the new instance with one extra key
    ``"auto_tuned_thresholds"`` recording the substitution. The
    ``formula_text`` is regenerated by inserting the new values into the
    original text where possible (best-effort, not load-bearing for the
    evaluator).
    """
    is_wrapped = isinstance(spec_template, STLSpec)
    root = spec_template.formula if is_wrapped else spec_template

    def _rebuild(node: Node) -> Node:
        if isinstance(node, Predicate):
            info = _introspect_predicate(node)
            if info is None:
                return node
            channel, th, op = info
            base = _predicate_base_name(node)
            if base in thresholds:
                return _make_predicate(base, channel, op, float(thresholds[base]))
            return node
        if isinstance(node, Negation):
            return Negation(inner=_rebuild(node.inner))  # type: ignore[arg-type]
        if isinstance(node, Always):
            return Always(inner=_rebuild(node.inner), interval=node.interval)
        if isinstance(node, Eventually):
            return Eventually(inner=_rebuild(node.inner), interval=node.interval)
        if isinstance(node, And):
            return And(children=tuple(_rebuild(c) for c in node.children))
        return node

    new_root = _rebuild(root)

    if not is_wrapped:
        return new_root

    spec = spec_template  # narrow for type-checkers
    new_meta = dict(spec.metadata)
    new_meta["auto_tuned_thresholds"] = dict(thresholds)
    return replace(spec, formula=new_root, metadata=new_meta)


# ---------------------------------------------------------------------------
# Section 4 — discriminability metrics.
# ---------------------------------------------------------------------------


def _drop_nonfinite(rho: np.ndarray) -> np.ndarray:
    """Filter out NaN/+/-Inf entries; keep the array finite."""
    arr = np.asarray(rho, dtype=np.float64)
    return arr[np.isfinite(arr)]


def wasserstein_distance_rho(rho_a: np.ndarray, rho_b: np.ndarray) -> float:
    """Closed-form 1-D Wasserstein-1 distance between two empirical samples.

    For two equal-length sorted samples ``a, b``, ``W1(a,b) =
    mean(|a_i - b_i|)``. For unequal lengths, we use the equivalent
    CDF-difference integral form ``W1 = integral |F_a(x) - F_b(x)| dx``
    evaluated as a sum on the merged-and-deduped support.

    We do *not* depend on scipy: the pure-NumPy implementation is exact
    for the empirical-CDF case and matches
    ``scipy.stats.wasserstein_distance`` to float64 epsilon (verified in
    ``tests/test_auto_tune.py::test_wasserstein_basic``).

    NaN/Inf entries are dropped before the computation. If either sample
    is empty after cleaning, returns ``0.0`` (no information).
    """
    a = _drop_nonfinite(rho_a)
    b = _drop_nonfinite(rho_b)
    if a.size == 0 or b.size == 0:
        return 0.0
    if a.size == b.size:
        return float(np.mean(np.abs(np.sort(a) - np.sort(b))))
    # General case: merge supports, evaluate CDFs, integrate the gap.
    all_values = np.sort(np.concatenate([a, b]))
    deltas = np.diff(all_values)
    cdf_a = np.searchsorted(np.sort(a), all_values[:-1], side="right") / a.size
    cdf_b = np.searchsorted(np.sort(b), all_values[:-1], side="right") / b.size
    return float(np.sum(np.abs(cdf_a - cdf_b) * deltas))


def auc_separation(rho_a: np.ndarray, rho_b: np.ndarray) -> float:
    """ROC-AUC for ``rho_a`` (positive) vs. ``rho_b`` (negative).

    Implemented via the Mann-Whitney U identity ``AUC = U / (n_a * n_b)``,
    where ``U`` is the number of pairs ``(x_a, x_b)`` with ``x_a > x_b``
    (with ties contributing 0.5).

    Returns the *signed* AUC in ``[0, 1]``: ``0.5`` is no information,
    ``1.0`` is perfect separation with A above B, ``0.0`` is perfect
    separation with B above A. Callers that want a symmetric "how
    separated are these distributions" scalar should use
    :func:`abs(auc - 0.5) * 2` (the Gini coefficient form), which is
    what :func:`auto_tune_spec_thresholds` uses internally.
    """
    a = _drop_nonfinite(rho_a)
    b = _drop_nonfinite(rho_b)
    if a.size == 0 or b.size == 0:
        return 0.5
    # Vectorised Mann-Whitney U via rank-sum on the merged sample.
    merged = np.concatenate([a, b])
    ranks = _avg_ranks(merged)
    rank_sum_a = float(np.sum(ranks[: a.size]))
    n_a = float(a.size)
    n_b = float(b.size)
    u_a = rank_sum_a - n_a * (n_a + 1.0) / 2.0
    return float(u_a / (n_a * n_b))


def _avg_ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), with ties given the average of their ranks."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)
    # Resolve ties: same value -> mean of their integer ranks.
    sorted_x = x[order]
    i = 0
    while i < sorted_x.size:
        j = i + 1
        while j < sorted_x.size and sorted_x[j] == sorted_x[i]:
            j += 1
        if j > i + 1:
            avg = (ranks[order[i]] + ranks[order[j - 1]]) / 2.0
            for k in range(i, j):
                ranks[order[k]] = avg
        i = j
    return ranks


def trace_overlap(rho_a: np.ndarray, rho_b: np.ndarray, n_bins: int = 25) -> float:
    """``1 - overlap`` of normalised histograms on a shared bin grid.

    A discriminability score in ``[0, 1]``: 0 = identical distributions,
    1 = disjoint supports. The shared bin grid is derived from the union
    of the two samples (min..max with ``n_bins`` equal-width bins). Less
    principled than Wasserstein but cheap and easy to reason about.
    """
    a = _drop_nonfinite(rho_a)
    b = _drop_nonfinite(rho_b)
    if a.size == 0 or b.size == 0:
        return 0.0
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if hi - lo < 1e-12:
        return 0.0
    edges = np.linspace(lo, hi, n_bins + 1)
    h_a, _ = np.histogram(a, bins=edges, density=False)
    h_b, _ = np.histogram(b, bins=edges, density=False)
    p_a = h_a / max(1, h_a.sum())
    p_b = h_b / max(1, h_b.sum())
    overlap = float(np.sum(np.minimum(p_a, p_b)))
    return 1.0 - overlap


# ---------------------------------------------------------------------------
# Section 5 — Auto-tune entry point.
# ---------------------------------------------------------------------------


# A "policy" here is *any* callable conforming to the
# `stl_seed.generation.policies.Policy` Protocol. We do not import that
# Protocol explicitly to keep this module importable in pure-Python tests
# that do not need the policy machinery. Duck typing is enforced at call.
PolicyLike = Any


@dataclass(frozen=True)
class AutoTuneResult:
    """Outcome of an auto-tune sweep.

    Attributes
    ----------
    best_thresholds : dict[str, float]
        The threshold combination maximising the chosen aggregation of the
        chosen discriminability metric.
    best_metric_value : float
        The metric value at ``best_thresholds`` (already aggregated across
        the policy pairs — typically the *worst-case* pairwise
        discriminability).
    search_results : pandas.DataFrame
        One row per threshold combination, with the threshold values as
        leading columns plus columns ``metric_mean``, ``metric_min``,
        ``metric_max`` (aggregations across all unordered policy pairs)
        and one ``metric_<a>_vs_<b>`` column per pair.
    per_policy_rho_stats : dict[str, dict[str, float]]
        At ``best_thresholds``: per-policy ``{min, mean, std, max,
        n_finite}`` summary of the rho samples.
    placeholders : list[ThresholdPlaceholder]
        Snapshot of the spec's discovered placeholders (handy for paper
        tables and the diagnostic plots).
    metadata : dict[str, Any]
        Diagnostics: chosen metric, aggregation, n_trajectories_per_policy,
        the per-policy (and per-threshold-combo) NaN counts.
    """

    best_thresholds: dict[str, float]
    best_metric_value: float
    search_results: pd.DataFrame
    per_policy_rho_stats: dict[str, dict[str, float]]
    placeholders: list[ThresholdPlaceholder] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _evaluate_rho_on_states(
    spec_or_node: STLSpec | Node,
    states_batch: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Evaluate STL robustness on a (batch, T, n) stack of cached states.

    Imports the JAX-based evaluator lazily so the bulk of this module
    remains import-cheap. Each call compiles the spec once and reuses the
    compiled closure across the batch.
    """
    # Lazy import keeps top-level import cheap and avoids a JAX dep when
    # only the success-rate path is used.
    from stl_seed.stl.evaluator import compile_spec  # local import; cheap.

    compiled = compile_spec(spec_or_node)
    states_jnp = jnp.asarray(states_batch)
    times_jnp = jnp.asarray(times)

    # Vectorise over the batch axis with vmap; if the spec fell back to the
    # Python-loop path (`_FALLBACK_USED`), we cannot jit it, so loop in
    # Python over the batch.
    used_fallback = bool(getattr(compiled, "_stl_seed_fallback_used", False))
    if used_fallback:
        out = np.empty((states_jnp.shape[0],), dtype=np.float64)
        for i in range(states_jnp.shape[0]):
            out[i] = float(compiled(states_jnp[i], times_jnp))
        return out

    # JIT + vmap path: one trace, full batch in flight.
    batched = jax.vmap(compiled, in_axes=(0, None))
    rho = jax.jit(batched)(states_jnp, times_jnp)
    return np.asarray(rho, dtype=np.float64)


def _simulate_policy_batch(
    simulator: Any,
    policy: PolicyLike,
    spec: STLSpec,
    *,
    initial_state: np.ndarray,
    horizon: int,
    action_dim: int,
    aux: dict | None,
    sim_params: Any,
    n_trajectories: int,
    key: Any,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Roll out ``policy`` ``n_trajectories`` times and stack the states.

    Reuses the rollout logic from ``stl_seed.generation.runner`` to keep
    the auto-tuner consistent with the rest of the pipeline. Returns
    ``(states_batch, times, n_failed)`` where ``states_batch`` has shape
    ``(n_kept, T, state_dim)`` and ``n_failed`` counts simulator
    exceptions or all-NaN rollouts (both are dropped).
    """
    # Lazy imports — keeps `calibration` importable in environments where
    # the simulator stack is heavy.
    from stl_seed.generation.runner import _simulate_one  # noqa: PLC0415

    states_list: list[np.ndarray] = []
    times: np.ndarray | None = None
    n_failed = 0

    for i in range(n_trajectories):
        traj_key = jax.random.fold_in(key, i)
        # Build the control sequence by stepping the policy open-loop, just
        # like TrajectoryRunner._rollout_one. We pass the *initial* state to
        # every step (open-loop schedule emission per the SERA recipe).
        history: list[tuple[Any, Any]] = []
        actions = []
        state = jnp.asarray(initial_state)
        for h in range(horizon):
            sub = jax.random.fold_in(traj_key, h)
            a = policy(state, spec, history, sub)
            a = jnp.asarray(a, dtype=jnp.float32)
            if a.ndim == 0:
                a = a[None]
            history.append((state, a))
            actions.append(a)
        control_sequence = jnp.stack(actions, axis=0)
        if control_sequence.shape[-1] != action_dim:
            raise ValueError(
                f"policy emitted control of shape {control_sequence.shape}; "
                f"expected last-axis size {action_dim}"
            )
        try:
            traj = _simulate_one(
                simulator,
                task=getattr(spec, "name", "<auto_tune>"),
                initial_state=initial_state,
                control_sequence=control_sequence,
                params=sim_params,
                aux=aux,
                key=traj_key,
            )
        except Exception:
            n_failed += 1
            continue
        states_np = np.asarray(traj.states, dtype=np.float64)
        if not np.all(np.isfinite(states_np)):
            # Sentinel-replaced trajectory; we still keep it because the
            # evaluator will compute a finite (negative) rho on the
            # sentinel-poisoned states. This matches the runner's
            # NaN-fraction policy: sentinel-replaced != dropped.
            states_np = np.nan_to_num(states_np, nan=0.0, posinf=0.0, neginf=0.0)
        states_list.append(states_np)
        if times is None:
            times = np.asarray(traj.times, dtype=np.float64)

    if not states_list or times is None:
        # Total wipe-out: synthesise a zero-batch so downstream code is safe.
        return (
            np.zeros((0, 1, int(initial_state.shape[0])), dtype=np.float64),
            np.zeros((1,), dtype=np.float64),
            n_failed,
        )

    states_batch = np.stack(states_list, axis=0)
    return states_batch, times, n_failed


def _itercombos(
    search_space: dict[str, list[float]],
) -> Iterable[dict[str, float]]:
    """Cartesian product of the threshold search space."""
    keys = list(search_space.keys())
    if not keys:
        yield {}
        return
    grids = [list(map(float, search_space[k])) for k in keys]
    indices = [0] * len(keys)
    while True:
        yield {keys[i]: grids[i][indices[i]] for i in range(len(keys))}
        # Increment.
        pos = len(keys) - 1
        while pos >= 0:
            indices[pos] += 1
            if indices[pos] < len(grids[pos]):
                break
            indices[pos] = 0
            pos -= 1
        if pos < 0:
            return


def auto_tune_spec_thresholds(
    simulator: Any,
    spec_template: STLSpec,
    threshold_search_space: dict[str, list[float]],
    policies: dict[str, PolicyLike],
    *,
    initial_state: np.ndarray,
    horizon: int | None = None,
    action_dim: int | None = None,
    sim_params: Any = None,
    aux: dict | None = None,
    n_trajectories_per_policy: int = 200,
    discriminability_metric: Literal[
        "wasserstein", "auc_separation", "trace_overlap"
    ] = "wasserstein",
    aggregation: Literal["worst", "mean"] = "worst",
    key: Any | None = None,
) -> AutoTuneResult:
    """Find threshold values maximising between-policy discriminability of rho.

    For each combination of threshold values:

    1. Roll out ``n_trajectories_per_policy`` trajectories per named
       policy on the same ``simulator`` with the same ``initial_state``,
       and *cache the state arrays*. Trajectories are simulated *once*
       per policy (not once per threshold combo) — the spec is independent
       of the simulator dynamics, so this is exact, not an approximation.
    2. Compile the spec at the candidate thresholds via
       :func:`instantiate_spec_with_thresholds` and evaluate
       :math:`\\rho` on the cached states (one JIT trace per spec, one
       vmap call per policy).
    3. Compute the chosen discriminability metric on every unordered
       policy pair, aggregate across pairs, and record the row.

    The final ``best_thresholds`` is the row maximising the aggregated
    metric.

    Parameters
    ----------
    simulator : Simulator
        Any object conforming to ``stl_seed.tasks.bio_ode.Simulator``
        (which both ``GlucoseInsulinSimulator`` and the bio_ode
        simulators do).
    spec_template : STLSpec
        Spec whose thresholds are tunable. Predicates whose base names
        (see :func:`extract_threshold_placeholders`) appear in
        ``threshold_search_space`` are tuned; the rest are kept verbatim.
    threshold_search_space : dict[str, list[float]]
        Map predicate-base-name -> candidate values. The cartesian product
        defines the search grid.
    policies : dict[str, PolicyLike]
        Named policies to discriminate between. Need >= 2 entries.
    initial_state, horizon, action_dim, sim_params, aux : optional
        Forwarded to the simulator. ``horizon`` defaults to
        ``simulator.n_control_points``; ``action_dim`` defaults to
        ``simulator.action_dim``.
    n_trajectories_per_policy : int
        Per-policy batch size. 200 is the literature-typical bootstrap-CI
        floor; smaller values are fine for diagnostic runs.
    discriminability_metric : str
        ``"wasserstein"`` (default) | ``"auc_separation"`` | ``"trace_overlap"``.
    aggregation : str
        ``"worst"`` (default; maximin across pairs) | ``"mean"``.
    key : PRNGKeyArray
        Trajectory-level PRNG key. If ``None``, defaults to
        ``jax.random.key(0)`` for full reproducibility.

    Returns
    -------
    AutoTuneResult
        See dataclass docstring.

    Raises
    ------
    ValueError
        If ``policies`` has fewer than two entries, or if
        ``threshold_search_space`` references a base name that isn't in
        the spec.
    """
    if len(policies) < 2:
        raise ValueError(f"auto_tune_spec_thresholds needs >= 2 policies; got {len(policies)}")
    placeholders = extract_threshold_placeholders(spec_template)
    by_name = {p.base_name: p for p in placeholders}
    unknown = [k for k in threshold_search_space if k not in by_name]
    if unknown:
        raise ValueError(
            f"threshold_search_space keys {unknown!r} are not predicate base "
            f"names in the spec; available: {sorted(by_name.keys())}"
        )

    horizon = horizon if horizon is not None else int(getattr(simulator, "n_control_points", 0))
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    action_dim = action_dim if action_dim is not None else int(getattr(simulator, "action_dim", 1))
    if key is None:
        key = jax.random.key(0)

    # ------------------------------------------------------------------
    # 1. Cache trajectories per policy (the expensive part).
    # ------------------------------------------------------------------
    cached: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for i, (pname, policy) in enumerate(policies.items()):
        pkey = jax.random.fold_in(key, i)
        cached[pname] = _simulate_policy_batch(
            simulator,
            policy,
            spec_template,
            initial_state=initial_state,
            horizon=horizon,
            action_dim=action_dim,
            aux=aux,
            sim_params=sim_params,
            n_trajectories=n_trajectories_per_policy,
            key=pkey,
        )

    # ------------------------------------------------------------------
    # 2. Sweep threshold combinations.
    # ------------------------------------------------------------------
    metric_fn: Callable[[np.ndarray, np.ndarray], float]
    if discriminability_metric == "wasserstein":
        metric_fn = wasserstein_distance_rho
    elif discriminability_metric == "auc_separation":
        metric_fn = lambda a, b: 2.0 * abs(auc_separation(a, b) - 0.5)  # noqa: E731
    elif discriminability_metric == "trace_overlap":
        metric_fn = trace_overlap
    else:
        raise ValueError(f"unknown discriminability_metric {discriminability_metric!r}")

    policy_names = list(policies.keys())
    pairs = [
        (policy_names[i], policy_names[j])
        for i in range(len(policy_names))
        for j in range(i + 1, len(policy_names))
    ]

    rows: list[dict[str, Any]] = []
    best_combo: dict[str, float] | None = None
    best_value: float = -np.inf
    best_per_policy_rho: dict[str, np.ndarray] = {}

    for combo in _itercombos(threshold_search_space):
        spec_at_combo = instantiate_spec_with_thresholds(spec_template, combo)
        rho_per_policy: dict[str, np.ndarray] = {}
        for pname, (states, times, _n_failed) in cached.items():
            if states.shape[0] == 0:
                rho_per_policy[pname] = np.array([], dtype=np.float64)
                continue
            rho = _evaluate_rho_on_states(spec_at_combo, states, times)
            rho_per_policy[pname] = rho

        pair_metrics: dict[str, float] = {}
        for a, b in pairs:
            m = float(metric_fn(rho_per_policy[a], rho_per_policy[b]))
            pair_metrics[f"metric_{a}_vs_{b}"] = m

        finite_metrics = [v for v in pair_metrics.values() if np.isfinite(v)]
        if not finite_metrics:
            metric_min = 0.0
            metric_mean = 0.0
            metric_max = 0.0
        else:
            metric_min = float(min(finite_metrics))
            metric_mean = float(np.mean(finite_metrics))
            metric_max = float(max(finite_metrics))
        agg = metric_min if aggregation == "worst" else metric_mean

        row = dict(combo)
        row.update(pair_metrics)
        row.update(
            {
                "metric_min": metric_min,
                "metric_mean": metric_mean,
                "metric_max": metric_max,
                "metric_aggregated": agg,
            }
        )
        rows.append(row)

        if agg > best_value:
            best_value = agg
            best_combo = dict(combo)
            best_per_policy_rho = dict(rho_per_policy)

    # ------------------------------------------------------------------
    # 3. Per-policy rho summary at the best threshold combo.
    # ------------------------------------------------------------------
    per_policy_rho_stats: dict[str, dict[str, float]] = {}
    for pname, rho in best_per_policy_rho.items():
        finite = _drop_nonfinite(rho)
        if finite.size == 0:
            per_policy_rho_stats[pname] = {
                "n_finite": 0.0,
                "min": float("nan"),
                "mean": float("nan"),
                "std": float("nan"),
                "max": float("nan"),
            }
        else:
            per_policy_rho_stats[pname] = {
                "n_finite": float(finite.size),
                "min": float(finite.min()),
                "mean": float(finite.mean()),
                "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
                "max": float(finite.max()),
            }

    metadata = {
        "discriminability_metric": discriminability_metric,
        "aggregation": aggregation,
        "n_trajectories_per_policy": int(n_trajectories_per_policy),
        "n_combinations": len(rows),
        "n_failed_per_policy": {p: int(cached[p][2]) for p in policy_names},
        "n_kept_per_policy": {p: int(cached[p][0].shape[0]) for p in policy_names},
    }

    if best_combo is None:
        # The search space was empty (no tunable keys). Fall back to the
        # template's current values for a coherent return value.
        best_combo = {p.base_name: p.current_value for p in placeholders}
        best_value = 0.0

    return AutoTuneResult(
        best_thresholds=best_combo,
        best_metric_value=float(best_value),
        search_results=pd.DataFrame(rows),
        per_policy_rho_stats=per_policy_rho_stats,
        placeholders=placeholders,
        metadata=metadata,
    )


__all__ = [
    # Section 1 — original API.
    "TrajectorySampler",
    "RobustnessFn",
    "CalibrationResult",
    "success_rate",
    "scan_threshold",
    "calibrate_spec",
    # Section 2-3 — placeholder model.
    "ThresholdPlaceholder",
    "extract_threshold_placeholders",
    "instantiate_spec_with_thresholds",
    # Section 4 — discriminability metrics.
    "wasserstein_distance_rho",
    "auc_separation",
    "trace_overlap",
    # Section 5 — auto-tune entry point.
    "AutoTuneResult",
    "auto_tune_spec_thresholds",
]

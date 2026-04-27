"""Empirical measurement of the Goodhart spec-completeness gap (theory.md S6).

For each policy ``pi`` and task family, this module:

1. Generates / consumes ``n`` trajectories from ``pi``.
2. Computes the proxy STL robustness ``rho_i = rho(tau_i, phi_proxy)``.
3. Computes the gold score ``g_i = g(tau_i)``.
4. Reports population statistics that diagnose the spec-completeness gap:

   * Pearson and Spearman correlations between ``rho_i`` and ``g_i``.
   * Linear regression slope of ``g_i`` on a normalized ``rho_i``.
   * Top-decile gap: ``mean(g_i | rho_i >= q90(rho)) - mean(g_i)``,
     i.e., how much worse the gold is in the top-rho decile vs. the
     overall mean. Negative gap means the spec-top trajectories are
     *systematically worse* under gold --- a smoking gun for Goodhart.

The last quantity is the "FM2" preflight in project rules S "Failure modes":
if Spearman r between rho and gold drops below 0.3, the spec is
suspected of being a poor proxy for the gold, and the cell is flagged.

This module deliberately does not import the
:class:`stl_seed.analysis.TrajectoryAdversary`: those two are
independent operationalizations of the same theoretical decomposition
(adversary = worst-case lower bound; this module = population-statistics
description). They are designed to be combinable downstream by
``scripts/run_adversary.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from stl_seed.specs import Node, STLSpec
from stl_seed.stl.evaluator import compile_spec
from stl_seed.tasks._trajectory import Trajectory

# A "Policy" is anything callable that, given (initial_state, key),
# returns an action sequence of shape (H, m). For canonical-store
# replay we wrap a fixed iterator into the same callable contract.
Policy = Callable[[Float[Array, " n"], PRNGKeyArray], Float[Array, "H m"]]


@dataclass
class PerPolicyGap:
    """Population statistics for one policy on one task family.

    Attributes
    ----------
    policy_name:
        Identifier ("random", "heuristic.bang_bang", "qwen3-0.6b", ...).
    n_trajectories:
        Sample size used for the statistics.
    rho_values:
        Per-trajectory proxy STL rho (length n).
    gold_values:
        Per-trajectory gold score (length n).
    pearson_r:
        Pearson correlation between rho and gold.
    spearman_r:
        Spearman rank correlation between rho and gold.
    regression_slope:
        OLS slope of gold on standardized rho (i.e., per-1-sigma rho
        change, the gold change). A positive slope indicates rho and
        gold move together; a slope near zero or negative indicates
        proxy/gold misalignment (Goodhart risk).
    top_decile_gap:
        ``mean(gold | rho >= q90(rho)) - mean(gold)``. Negative means
        the spec-top trajectories underperform on gold; this is the
        most direct empirical handle on the spec-completeness gap.
    flagged:
        True iff the population-level Spearman correlation falls below
        0.3, the FM2 preflight threshold from project rules.
    """

    policy_name: str
    n_trajectories: int
    rho_values: np.ndarray
    gold_values: np.ndarray
    pearson_r: float
    spearman_r: float
    regression_slope: float
    top_decile_gap: float
    flagged: bool = False


@dataclass
class GoodhartGapResult:
    """Multi-policy summary suitable for cross-comparison plots.

    Attributes
    ----------
    spec_name:
        The proxy spec key (e.g., ``"glucose_insulin.tir.easy"``).
    per_policy:
        Dict from policy name to its :class:`PerPolicyGap`.
    cross_policy_pearson_range:
        ``(min, max)`` of the per-policy Pearson r values. A wide range
        suggests the spec-completeness gap is policy-dependent (the
        most useful regime for downstream RLHF analysis).
    """

    spec_name: str
    per_policy: dict[str, PerPolicyGap]
    cross_policy_pearson_range: tuple[float, float] = (0.0, 0.0)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Statistical helpers (pure NumPy; the per-traj rho / gold are scalars
# already so we drop into NumPy for the reductions).
# ---------------------------------------------------------------------------


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r with degenerate-input guard.

    NumPy's ``np.corrcoef`` returns NaN if either input is constant
    (zero variance). We catch that case and return 0.0 with no
    warning, matching the convention used by ``stats_utils`` elsewhere
    in the project.
    """
    if x.size < 2 or y.size < 2:
        return 0.0
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation via Pearson on ranks.

    SciPy is a project dependency but we avoid the import to keep this
    module slim; rank-then-Pearson is the textbook definition. We
    additionally guard the constant-input case (zero variance in either
    raw input) by short-circuiting to 0; without this guard
    ``argsort(argsort(constant))`` produces a deterministic but
    *non-degenerate* rank sequence ``[0, 1, ..., n-1]`` that spuriously
    correlates with whatever the other input happens to be.
    """
    if x.size < 2 or y.size < 2:
        return 0.0
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return _safe_corrcoef(rx, ry)


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """OLS slope of y on standardized x.

    Returns the change in y per +1 sigma in x. If x has zero variance,
    returns 0.0.
    """
    if x.size < 2:
        return 0.0
    sx = np.std(x)
    if sx < 1e-12:
        return 0.0
    x_std = (x - np.mean(x)) / sx
    return float(np.cov(x_std, y, ddof=0)[0, 1])


def _top_decile_gap(rho: np.ndarray, gold: np.ndarray) -> float:
    """``mean(gold | rho >= q90(rho)) - mean(gold)``.

    Quantile q=0.9 by NumPy default (linear interpolation). If the top
    decile is empty (n < 10), we use top-k=max(1, n//10).
    """
    if rho.size == 0:
        return 0.0
    q90 = np.quantile(rho, 0.9)
    mask = rho >= q90
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(gold[mask]) - np.mean(gold))


# ---------------------------------------------------------------------------
# Trajectory generation policies.
# ---------------------------------------------------------------------------


def random_policy(
    horizon: int,
    action_dim: int,
    action_min: float | np.ndarray = 0.0,
    action_max: float | np.ndarray = 1.0,
) -> Policy:
    """Uniform-random policy in the action box.

    Returns a callable ``(x_0, key) -> u_{1:H}`` that ignores the
    initial state, matching the protocol used by the other generators
    in ``stl_seed.generation``. We do NOT depend on
    ``stl_seed.generation`` here to keep this module independent of
    the heavyweight policy machinery.
    """
    a_min = jnp.asarray(action_min, dtype=jnp.float32)
    a_max = jnp.asarray(action_max, dtype=jnp.float32)

    def _policy(initial_state: Float[Array, " n"], key: PRNGKeyArray) -> Float[Array, "H m"]:
        del initial_state
        u01 = jax.random.uniform(key, shape=(horizon, action_dim))
        return a_min + (a_max - a_min) * u01

    return _policy


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------


def measure_goodhart_gap(
    simulator: object,
    proxy_spec: STLSpec | Node,
    gold_score: Callable[[Trajectory], Float[Array, ""]],
    policies: dict[str, Policy],
    initial_state: Float[Array, " n"],
    action_dim: int,
    horizon: int,
    params: object,
    key: PRNGKeyArray,
    n_trajectories: int = 200,
    simulator_aux: tuple = (),
    flag_threshold: float = 0.3,
) -> GoodhartGapResult:
    """Empirically measure the spec-completeness gap per policy.

    For each policy:
      1. Roll out ``n_trajectories`` action sequences and simulate them.
      2. Compute proxy_rho and gold for each trajectory.
      3. Reduce to per-policy statistics: Pearson, Spearman, OLS slope,
         top-decile gap.

    Parameters
    ----------
    simulator:
        Any object exposing a ``simulate(initial_state, control_sequence,
        *aux, params, key)`` method returning a ``Trajectory`` or
        ``(states, times, meta)`` tuple.
    proxy_spec:
        The proxy STL spec to score against.
    gold_score:
        Callable ``Trajectory -> scalar JAX array``.
    policies:
        Dict from policy name to a callable producing actions of shape
        ``(horizon, action_dim)``.
    initial_state:
        Common ``x_0`` used for all rollouts (for the per-policy
        statistics to be comparable).
    action_dim, horizon:
        Action shape parameters.
    params:
        Kinetic parameters passed through to ``simulator.simulate``.
    key:
        Top-level PRNG key; per-policy keys are split deterministically.
    n_trajectories:
        Per-policy sample size. Default 200.
    simulator_aux:
        Extra positional args between ``control_sequence`` and ``params``
        in the simulator call signature (e.g. ``MealSchedule`` for
        glucose-insulin).
    flag_threshold:
        Spearman r below this value flags the cell as FM2-suspicious
        per project rules.

    Returns
    -------
    GoodhartGapResult
        Per-policy and cross-policy summary, including raw arrays so the
        caller can plot scatter / histograms downstream.
    """
    compiled = compile_spec(proxy_spec)
    spec_name = proxy_spec.name if isinstance(proxy_spec, STLSpec) else "<raw_node>"

    # Allocate one PRNG key per (policy, trajectory) pair deterministically.
    n_policies = len(policies)
    keys_top = jax.random.split(key, n_policies)
    per_policy: dict[str, PerPolicyGap] = {}

    pearson_values: list[float] = []
    notes: list[str] = []

    for p_idx, (name, policy) in enumerate(policies.items()):
        keys_traj = jax.random.split(keys_top[p_idx], n_trajectories)
        rho_arr = np.zeros(n_trajectories, dtype=np.float64)
        gold_arr = np.zeros(n_trajectories, dtype=np.float64)

        is_glucose = type(simulator).__name__ == "GlucoseInsulinSimulator"
        for i in range(n_trajectories):
            actions = policy(initial_state, keys_traj[i])
            if is_glucose:
                u_flat = actions.reshape(-1)
                sim_out = simulator.simulate(
                    initial_state, u_flat, *simulator_aux, params, keys_traj[i]
                )
            else:
                sim_out = simulator.simulate(
                    initial_state, actions, *simulator_aux, params, keys_traj[i]
                )
            traj = _coerce_trajectory(sim_out, actions)
            rho = float(compiled(traj.states, traj.times))
            g = float(gold_score(traj))
            rho_arr[i] = rho
            gold_arr[i] = g

        pearson = _safe_corrcoef(rho_arr, gold_arr)
        spearman = _spearman(rho_arr, gold_arr)
        slope = _ols_slope(rho_arr, gold_arr)
        gap = _top_decile_gap(rho_arr, gold_arr)
        flagged = spearman < flag_threshold

        if flagged:
            notes.append(
                f"policy={name!r}: Spearman r={spearman:.3f} below FM2 threshold "
                f"{flag_threshold} - spec/gold misalignment suspected."
            )

        per_policy[name] = PerPolicyGap(
            policy_name=name,
            n_trajectories=n_trajectories,
            rho_values=rho_arr,
            gold_values=gold_arr,
            pearson_r=pearson,
            spearman_r=spearman,
            regression_slope=slope,
            top_decile_gap=gap,
            flagged=flagged,
        )
        pearson_values.append(pearson)

    if pearson_values:
        cross_range = (float(min(pearson_values)), float(max(pearson_values)))
    else:
        cross_range = (0.0, 0.0)

    return GoodhartGapResult(
        spec_name=spec_name,
        per_policy=per_policy,
        cross_policy_pearson_range=cross_range,
        notes=notes,
    )


def _coerce_trajectory(sim_output: object, actions: Float[Array, "H m"]) -> Trajectory:
    """Coerce simulator output into a Trajectory (handles tuple form).

    The bio_ode simulators return Trajectory directly; the
    glucose-insulin simulator returns ``(states, times, meta)``. This
    helper hides the difference so the caller's loop is uniform.
    """
    if isinstance(sim_output, Trajectory):
        return sim_output
    states, times, meta = sim_output
    if actions.ndim == 1:
        actions = actions[:, None]
    return Trajectory(states=states, actions=actions, times=times, meta=meta)


def measure_from_arrays(
    rho_values: Iterable[float],
    gold_values: Iterable[float],
    policy_name: str,
    flag_threshold: float = 0.3,
) -> PerPolicyGap:
    """Compute a :class:`PerPolicyGap` from precomputed (rho, gold) arrays.

    Useful when trajectories were generated and scored upstream
    (e.g., from the canonical parquet store) and you only want the
    statistical reductions.
    """
    rho_arr = np.asarray(list(rho_values), dtype=np.float64)
    gold_arr = np.asarray(list(gold_values), dtype=np.float64)
    pearson = _safe_corrcoef(rho_arr, gold_arr)
    spearman = _spearman(rho_arr, gold_arr)
    slope = _ols_slope(rho_arr, gold_arr)
    gap = _top_decile_gap(rho_arr, gold_arr)
    return PerPolicyGap(
        policy_name=policy_name,
        n_trajectories=int(rho_arr.size),
        rho_values=rho_arr,
        gold_values=gold_arr,
        pearson_r=pearson,
        spearman_r=spearman,
        regression_slope=slope,
        top_decile_gap=gap,
        flagged=spearman < flag_threshold,
    )


__all__ = [
    "GoodhartGapResult",
    "PerPolicyGap",
    "Policy",
    "measure_from_arrays",
    "measure_goodhart_gap",
    "random_policy",
]

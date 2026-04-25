"""TrajectoryRunner: orchestrates batched per-task rollouts.

The runner is the single point in `stl-seed` that
1. composes a (`Simulator`, `Policy`, `STLSpec`) triple,
2. rolls out the policy step-by-step against the simulator,
3. computes per-trajectory STL robustness ρ,
4. tags the result with the policy used and persists to a
   `TrajectoryStore`.

It does NOT perform vector-level vmap fusion — the policy interface is
state-by-state, and policies like `MLXModelPolicy` cannot be vmapped. The
inner ODE solve is JIT'd by Diffrax automatically. For pure-JAX policies
(`Random`, `Constant`, `PID`, `BangBang`) the runner could be vmap'd over
the batch dimension; that is a future optimization tracked in
`paper/architecture.md`. The current implementation is correct,
deterministic, and resumable.

NaN policy (per `paper/architecture.md` §"NaN/Inf policy"): trajectories
whose `meta.n_nan_replacements` exceeds `nan_fraction_threshold * T` are
DROPPED from the persisted corpus (and counted in `n_nan_dropped`). The
sentinel-replaced state is computed by the simulator, not by the runner.

REDACTED firewall: this module imports only from `stl_seed.{generation,
specs, tasks}`. No REDACTED artifact is touched.
"""

from __future__ import annotations

import dataclasses
import datetime
import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from stl_seed.generation.policies import Action, History
from stl_seed.specs import (
    Always,
    And,
    Eventually,
    Negation,
    Predicate,
    STLSpec,
)
from stl_seed.tasks._trajectory import Trajectory

# -----------------------------------------------------------------------------
# Inline STL robustness evaluator (Donzé-Maler space-robustness).
# -----------------------------------------------------------------------------
# Subphase 1.3 also delivers a stand-alone `stl_seed.stl.evaluator` module
# (agent A9). Pending that delivery, the runner ships a self-contained
# numpy reference implementation of Donzé-Maler ρ over the AST defined in
# `stl_seed.specs`. The two implementations must agree to within float64
# epsilon when both are present (a regression test for that contract is
# tracked in `tests/test_stl.py` once A9 lands).
# -----------------------------------------------------------------------------


def _interval_indices(
    times_min: np.ndarray, t_lo: float, t_hi: float
) -> np.ndarray:
    """Return integer indices `i` with `t_lo <= times_min[i] <= t_hi`."""
    mask = (times_min >= t_lo) & (times_min <= t_hi)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        # Edge case: spec interval falls outside the trajectory's save grid.
        # We snap to the nearest single sample so robustness is well-defined.
        nearest = int(np.argmin(np.abs(times_min - 0.5 * (t_lo + t_hi))))
        idx = np.array([nearest], dtype=np.int64)
    return idx


def _evaluate_predicate(
    pred: Predicate, traj_states: np.ndarray, t_idx: int
) -> float:
    return float(pred.fn(traj_states, t_idx))


def evaluate_robustness(
    spec: STLSpec,
    states: np.ndarray,
    times_min: np.ndarray,
) -> float:
    """Donzé-Maler space-robustness ρ for `spec` against the trajectory.

    Recursive on the `stl_seed.specs` AST: predicates evaluate via their
    `fn`; `Negation` flips the sign of the wrapped predicate's robustness;
    `Always_I` reduces to `min` over the interval samples; `Eventually_I`
    reduces to `max`; `And` reduces to `min` across children.

    Parameters
    ----------
    spec:
        Top-level `STLSpec` (registered in `stl_seed.specs.REGISTRY`).
    states:
        Shape `(T, n)` numpy array of state samples.
    times_min:
        Shape `(T,)` numpy array of save times in MINUTES (matches the
        simulator-side convention; the spec's `Interval` bounds are
        already in minutes per `paper/REDACTED.md`).

    Returns
    -------
    float ρ. Positive iff the trajectory satisfies the spec.
    """

    def _recurse(node: object) -> float:
        if isinstance(node, Predicate):
            # A predicate evaluated with no temporal context defaults to t=0.
            # In practice, predicates are always wrapped in Always/Eventually
            # so this branch is never directly reached at the top level.
            return _evaluate_predicate(node, states, 0)
        if isinstance(node, Negation):
            inner = node.inner
            return -_evaluate_predicate(inner, states, 0)
        if isinstance(node, And):
            return float(min(_recurse(c) for c in node.children))
        if isinstance(node, Always):
            idx = _interval_indices(times_min, node.interval.t_lo, node.interval.t_hi)
            inner = node.inner
            if isinstance(inner, Predicate):
                vals = np.array(
                    [_evaluate_predicate(inner, states, int(t)) for t in idx]
                )
            elif isinstance(inner, Negation):
                vals = np.array(
                    [-_evaluate_predicate(inner.inner, states, int(t)) for t in idx]
                )
            else:
                # Nested Always/And/Eventually under Always — fall back to
                # full per-step recursion. Our specs never use this nesting
                # but the evaluator handles it for completeness.
                vals = np.array(
                    [_recurse_at(inner, int(t)) for t in idx]
                )
            return float(np.min(vals))
        if isinstance(node, Eventually):
            idx = _interval_indices(times_min, node.interval.t_lo, node.interval.t_hi)
            inner = node.inner
            if isinstance(inner, Predicate):
                vals = np.array(
                    [_evaluate_predicate(inner, states, int(t)) for t in idx]
                )
            elif isinstance(inner, Negation):
                vals = np.array(
                    [-_evaluate_predicate(inner.inner, states, int(t)) for t in idx]
                )
            else:
                vals = np.array(
                    [_recurse_at(inner, int(t)) for t in idx]
                )
            return float(np.max(vals))
        raise TypeError(f"unknown STL node type: {type(node).__name__}")

    def _recurse_at(node: object, t_idx: int) -> float:
        # Variant of `_recurse` that pins the predicate-evaluation time index.
        if isinstance(node, Predicate):
            return _evaluate_predicate(node, states, t_idx)
        if isinstance(node, Negation):
            return -_evaluate_predicate(node.inner, states, t_idx)
        if isinstance(node, And):
            return float(min(_recurse_at(c, t_idx) for c in node.children))
        # Nested temporal under temporal — re-enter the global recursion.
        return _recurse(node)

    return _recurse(spec.formula)


# -----------------------------------------------------------------------------
# Simulator adapter — bridges per-task simulator signatures to a uniform call.
# -----------------------------------------------------------------------------


def _is_glucose_insulin_simulator(simulator: Any) -> bool:
    """Heuristic: does `simulator` look like `GlucoseInsulinSimulator`?"""
    return type(simulator).__name__ == "GlucoseInsulinSimulator"


def _simulate_one(
    simulator: Any,
    task: str,
    initial_state: np.ndarray,
    control_sequence: np.ndarray,
    params: Any,
    aux: Mapping[str, Any] | None,
    key: PRNGKeyArray,
) -> Trajectory:
    """Call `simulator.simulate(...)` with task-appropriate arguments.

    The architecture-spec `Simulator.simulate(initial_state, control_sequence,
    params, key)` is the *normalized* signature. The current
    `GlucoseInsulinSimulator.simulate(...)` takes an extra `meal_schedule`
    argument; we route through `aux["meal_schedule"]` for that family. As
    new task families land, extend this dispatch.

    Returns a canonical `Trajectory` pytree (states, actions, times, meta).
    """
    if _is_glucose_insulin_simulator(simulator):
        # glucose-insulin emits a (states, times, meta) tuple with action axis
        # implicit (scalar insulin); we wrap actions as (H, 1) for consistency.
        from stl_seed.tasks.glucose_insulin import MealSchedule

        meal_schedule = (
            aux.get("meal_schedule") if aux is not None else None
        ) or MealSchedule.empty()
        # control_sequence comes in shape (H, 1) from the policy loop; the
        # glucose simulator wants (H,) for the scalar insulin rate.
        u_flat = jnp.asarray(control_sequence).reshape(-1)
        states_jax, times_jax, meta = simulator.simulate(
            jnp.asarray(initial_state),
            u_flat,
            meal_schedule,
            params,
            key,
        )
        actions_jax = jnp.asarray(control_sequence).reshape(-1, 1)
        return Trajectory(
            states=states_jax,
            actions=actions_jax,
            times=times_jax,
            meta=meta,
        )
    # Generic: the simulator is expected to return a `Trajectory` directly.
    return simulator.simulate(
        jnp.asarray(initial_state),
        jnp.asarray(control_sequence),
        params,
        key,
    )


# -----------------------------------------------------------------------------
# TrajectoryRunner
# -----------------------------------------------------------------------------


_PolicyFactory = Callable[[PRNGKeyArray], Any]


@dataclasses.dataclass
class _RunnerStats:
    n_requested: int = 0
    n_kept: int = 0
    n_nan_dropped: int = 0
    n_failed: int = 0
    nan_rate: float = 0.0


class TrajectoryRunner:
    """Generates trajectories under a configurable policy mix.

    Parameters
    ----------
    simulator:
        An object exposing `simulate(...)` per the architecture-locked
        `Simulator` Protocol. Today the only supported concrete simulator
        is `GlucoseInsulinSimulator`; the dispatch table in
        `_simulate_one(...)` is the extension point.
    spec_registry:
        A dict mapping `spec_key -> STLSpec`. Pass `stl_seed.specs.REGISTRY`
        to use the global registry. Restricted lookups (e.g. only the
        glucose specs for a glucose simulator) keep tests independent.
    output_store:
        Optional `TrajectoryStore`. If supplied, every kept trajectory is
        persisted via `output_store.save(...)`. Pass `None` to keep the
        runner pure (tests use this).
    initial_state:
        Default initial-state vector for the simulator. Required because
        the architecture-locked `Simulator.simulate` signature takes
        `initial_state` as input (i.e. the simulator does not own it).
    horizon:
        Number of control points `H`. Must match the simulator's
        `n_control_points`.
    action_dim:
        Action vector dimensionality `m`.
    aux:
        Task-specific auxiliary kwargs forwarded to the simulator (e.g.
        `meal_schedule` for glucose-insulin).
    nan_fraction_threshold:
        Drop trajectories with more than this fraction of save-time
        samples having been replaced by sentinels. Default 0.10 per
        architecture.md.
    sim_params:
        Kinetic-parameter object passed to the simulator (`BergmanParams`
        for glucose, `RepressilatorParams` etc. for bio_ode).
    """

    def __init__(
        self,
        simulator: Any,
        spec_registry: Mapping[str, STLSpec],
        output_store: Any | None = None,
        *,
        initial_state: np.ndarray | Float[Array, " n"] | None = None,
        horizon: int | None = None,
        action_dim: int = 1,
        aux: Mapping[str, Any] | None = None,
        nan_fraction_threshold: float = 0.10,
        sim_params: Any = None,
    ) -> None:
        self.simulator = simulator
        self.spec_registry = dict(spec_registry)
        self.output_store = output_store
        if initial_state is None:
            raise ValueError(
                "initial_state must be provided to the TrajectoryRunner — "
                "it is part of the locked Simulator.simulate signature."
            )
        self.initial_state = jnp.asarray(initial_state)
        self.horizon = (
            horizon
            if horizon is not None
            else int(getattr(simulator, "n_control_points", 0))
        )
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        self.action_dim = action_dim
        self.aux = aux
        if not (0.0 <= nan_fraction_threshold <= 1.0):
            raise ValueError(
                f"nan_fraction_threshold must be in [0, 1], "
                f"got {nan_fraction_threshold}"
            )
        self.nan_fraction_threshold = nan_fraction_threshold
        self.sim_params = sim_params
        self.last_stats: _RunnerStats | None = None

    # ------------------------------------------------------------------ rollouts
    def _rollout_one(
        self,
        policy: Any,
        spec: STLSpec,
        key: PRNGKeyArray,
    ) -> Trajectory:
        """Run a single policy episode and return the canonical Trajectory."""
        # Build the control sequence step-by-step. The policy sees the *initial*
        # state plus the running history; the simulator integrates the full
        # H-step piecewise-constant control in one shot at the end.
        history: History = []
        actions: list[Action] = []
        state = self.initial_state
        for h in range(self.horizon):
            subkey = jax.random.fold_in(key, h)
            a = policy(state, spec, history, subkey)
            a = jnp.asarray(a, dtype=jnp.float32)
            if a.ndim == 0:
                a = a[None]
            if a.shape[-1] != self.action_dim:
                raise ValueError(
                    f"policy emitted action of shape {a.shape}, "
                    f"expected last-axis size {self.action_dim}"
                )
            history.append((state, a))
            actions.append(a)
            # State update: open-loop policy — we don't update state between
            # control points (the simulator integrates the whole sequence at
            # once). Pass the same `state` to every call. This matches the
            # SERA recipe (one rollout = one full control schedule emission).
        control_sequence = jnp.stack(actions, axis=0)  # (H, m)

        return _simulate_one(
            self.simulator,
            task=getattr(spec, "name", "<unknown>"),
            initial_state=self.initial_state,
            control_sequence=control_sequence,
            params=self.sim_params,
            aux=self.aux,
            key=key,
        )

    # ------------------------------------------------------------- orchestration
    def generate_trajectories(
        self,
        task: str,
        n: int,
        policy_mix: Mapping[str, float],
        key: PRNGKeyArray,
        *,
        spec_key: str | None = None,
        policy_factories: Mapping[str, _PolicyFactory] | None = None,
    ) -> tuple[list[Trajectory], list[dict[str, Any]]]:
        """Generate `n` trajectories under `policy_mix`.

        Parameters
        ----------
        task:
            Task identifier (e.g. ``"glucose_insulin"``). Used by the
            heuristic policy router and persisted in metadata.
        n:
            Total number of trajectories to *attempt*. NaN-dropped or
            failed rollouts reduce the kept count; see `last_stats`.
        policy_mix:
            Dict mapping policy name -> weight. Weights are normalized;
            counts are rounded so the sum equals `n`. Names must appear
            in `policy_factories`. The default factories cover
            `random`, `constant`, `heuristic`. The `model` slot requires
            an explicit factory because MLX is optional.
        key:
            Trajectory-level PRNG key. Split deterministically per
            trajectory so the run is reproducible from `key` alone.
        spec_key:
            Registry key for the STL spec to evaluate against. Defaults
            to `f"{task}.tir.easy"` for glucose and the first matching
            registry key otherwise.
        policy_factories:
            Optional override mapping `name -> (key) -> Policy`. The
            default factories are constructed lazily from `task` and
            `self.action_dim`.

        Returns
        -------
        (trajectories, metadata) — a list of kept `Trajectory` pytrees and
        a parallel list of metadata dicts (id, task, spec_key, policy,
        seed, robustness, nan_count, generated_at). NaN-dropped
        trajectories are recorded in `self.last_stats` but not returned.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if not policy_mix:
            raise ValueError("policy_mix must be non-empty")

        # Resolve spec.
        if spec_key is None:
            # Heuristic default per task.
            for candidate in (
                f"{task}.tir.easy",
                f"{task}.repressilator.easy",
                f"{task}.toggle.medium",
                f"{task}.mapk.hard",
            ):
                if candidate in self.spec_registry:
                    spec_key = candidate
                    break
            if spec_key is None:
                # Fall back: first registered spec whose name starts with task.
                for k in self.spec_registry:
                    if k.startswith(task):
                        spec_key = k
                        break
        if spec_key is None or spec_key not in self.spec_registry:
            raise KeyError(
                f"spec_key={spec_key!r} not found in registry; "
                f"known keys: {sorted(self.spec_registry)}"
            )
        spec = self.spec_registry[spec_key]

        # Build default policy factories.
        factories: dict[str, _PolicyFactory] = {}
        if policy_factories is None:
            from stl_seed.generation.policies import (
                ConstantPolicy,
                HeuristicPolicy,
                RandomPolicy,
            )

            # Use simulator-declared bounds where available; otherwise
            # default to the broad [-1, 1] box.
            if _is_glucose_insulin_simulator(self.simulator):
                from stl_seed.tasks.glucose_insulin import (
                    U_INSULIN_MAX_U_PER_H,
                    U_INSULIN_MIN_U_PER_H,
                )

                lo = U_INSULIN_MIN_U_PER_H
                hi = U_INSULIN_MAX_U_PER_H
            else:
                lo, hi = 0.0, 1.0

            factories["random"] = lambda _key, lo=lo, hi=hi: RandomPolicy(
                action_dim=self.action_dim, action_low=lo, action_high=hi
            )
            factories["constant"] = lambda _key: ConstantPolicy(
                jnp.zeros((self.action_dim,))
            )
            factories["heuristic"] = lambda _key, t=task: HeuristicPolicy(t)
        else:
            factories.update(policy_factories)

        for name in policy_mix:
            if name not in factories:
                raise KeyError(
                    f"policy_mix name {name!r} has no factory; "
                    f"pass policy_factories= to register it"
                )

        # Allocate per-policy counts proportional to weights, summing to n.
        counts = _proportional_split(policy_mix, n)

        stats = _RunnerStats(n_requested=n)
        kept_traj: list[Trajectory] = []
        kept_meta: list[dict[str, Any]] = []

        traj_idx = 0
        for policy_name, count in counts.items():
            for _ in range(count):
                # Per-trajectory subkey: split deterministically from `key`.
                subkey = jax.random.fold_in(key, traj_idx)
                policy = factories[policy_name](subkey)
                traj_id = uuid.uuid4().hex
                generated_at = datetime.datetime.now(
                    datetime.UTC
                ).isoformat()
                # Pack the JAX key (two uint32s) into a SIGNED 64-bit int so
                # downstream Parquet (Arrow int64) can store it without overflow.
                seed_bytes = bytes(np.asarray(jax.random.key_data(subkey)))
                seed_int = int.from_bytes(seed_bytes[:8], "little", signed=True)

                # Run the rollout.
                try:
                    traj = self._rollout_one(policy, spec, subkey)
                except Exception:
                    stats.n_failed += 1
                    traj_idx += 1
                    continue

                # NaN-fraction gate.
                states_np = np.asarray(traj.states)
                T = states_np.shape[0]
                n_bad = int(np.asarray(traj.meta.n_nan_replacements))
                if T == 0 or n_bad / T > self.nan_fraction_threshold:
                    stats.n_nan_dropped += 1
                    traj_idx += 1
                    continue

                # STL robustness.
                times_np = np.asarray(traj.times)
                rho = evaluate_robustness(spec, states_np, times_np)

                meta = {
                    "id": traj_id,
                    "task": task,
                    "spec_key": spec_key,
                    "policy": policy_name,
                    "seed": seed_int,
                    "robustness": float(rho),
                    "nan_count": n_bad,
                    "generated_at": generated_at,
                }
                kept_traj.append(traj)
                kept_meta.append(meta)
                stats.n_kept += 1
                traj_idx += 1

        stats.nan_rate = stats.n_nan_dropped / max(1, stats.n_requested)
        self.last_stats = stats

        if self.output_store is not None and kept_traj:
            self.output_store.save(kept_traj, kept_meta)

        return kept_traj, kept_meta


# -----------------------------------------------------------------------------
# Helpers.
# -----------------------------------------------------------------------------


def _proportional_split(weights: Mapping[str, float], n: int) -> dict[str, int]:
    """Split `n` items proportionally over `weights`, summing to exactly n.

    Uses Hamilton's "largest remainders" method for stability.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    items = list(weights.items())
    total_w = float(sum(w for _, w in items))
    if total_w <= 0.0:
        raise ValueError("policy_mix weights must sum to > 0")

    raw = [(name, n * w / total_w) for name, w in items]
    floors = {name: int(np.floor(v)) for name, v in raw}
    remainder = n - sum(floors.values())
    # Distribute the remainder to the items with the largest fractional parts.
    fracs = sorted(
        ((v - np.floor(v), name) for name, v in raw),
        key=lambda x: -x[0],
    )
    for i in range(remainder):
        floors[fracs[i % len(fracs)][1]] += 1
    return floors


__all__ = [
    "TrajectoryRunner",
    "evaluate_robustness",
]


# -----------------------------------------------------------------------------
# Self-test (manual): `python -m stl_seed.generation.runner`.
# -----------------------------------------------------------------------------


def _self_test() -> None:  # pragma: no cover
    """Minimal end-to-end smoke: glucose-insulin + random policy + STL eval."""
    from stl_seed.specs import REGISTRY
    from stl_seed.tasks.glucose_insulin import (
        BergmanParams,
        GlucoseInsulinSimulator,
        default_normal_subject_initial_state,
    )

    sim = GlucoseInsulinSimulator()
    params = BergmanParams()
    runner = TrajectoryRunner(
        simulator=sim,
        spec_registry=REGISTRY,
        output_store=None,
        initial_state=default_normal_subject_initial_state(params),
        horizon=sim.n_control_points,
        action_dim=1,
        sim_params=params,
    )
    key = jax.random.key(0)
    t0 = time.time()
    traj, meta = runner.generate_trajectories(
        task="glucose_insulin",
        n=8,
        policy_mix={"random": 1.0},
        key=key,
    )
    dt = time.time() - t0
    print(
        f"generated {len(traj)} trajectories in {dt:.2f}s; "
        f"ρ range = "
        f"({min(m['robustness'] for m in meta):.2f}, "
        f"{max(m['robustness'] for m in meta):.2f})"
    )


if __name__ == "__main__":  # pragma: no cover
    _self_test()

"""Eval harness — the evaluation surface registered in
``paper/architecture.md`` (eval/ module, A11 deliverable).

Given a trained checkpoint and a held-out set of STL specs, the harness:

1. For each spec, samples ``n_samples_per_spec`` LLM-controlled
   trajectories under independent seeds.
2. Simulates each (state_0, control_sequence) under the appropriate
   simulator from the registry.
3. Scores each trajectory with the STL evaluator on the spec.
4. Computes per-spec success rates at the registered BoN budgets
   ``{1, 2, 4, 8, 16, 32, 64, 128}`` via *sample reuse*: a single draw
   of ``N_max = n_samples_per_spec`` is sufficient to read off success
   at all budgets ``≤ N_max``.
5. Records ρ distributions, seeds, and per-spec aggregates so
   downstream analysis (``stats.bootstrap``, ``stats.hierarchical_bayes``)
   can ingest the same data without re-simulation.

Design notes
------------

* The harness is *agnostic* to the concrete simulator and STL evaluator
  implementations — it operates on protocols. This lets us unit-test
  it against synthetic stand-ins without spinning up Diffrax. The
  concrete simulators (``glucose_insulin``, ``bio_ode``) and the STL
  evaluator (``stl/evaluator.py``) are sibling A8/A9 deliverables.

* The checkpoint protocol exposes a single ``sample_controls(spec, key)``
  call that returns a ``(H, m)`` control sequence (or, when the
  underlying agent is stochastic, a single sample drawn under ``key``).
  This abstracts away whether the checkpoint is an MLX-LoRA, a
  bnb-quantized HF model, or a heuristic baseline — the harness only
  cares about the action sequence.

* All randomness flows from a single ``jax.random.key`` per checkpoint;
  splits are deterministic so re-running the harness with the same
  ``key`` reproduces every trajectory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from stl_seed.evaluation.metrics import (
    bon_success_curve,
    rho_margin,
    success_rate,
)

# Default BoN budgets (theory.md §5)
DEFAULT_BON_BUDGETS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)


# ---------------------------------------------------------------------------
# Protocols (so the harness is unit-testable without sim/STL implementations)
# ---------------------------------------------------------------------------


class SimulatorProtocol(Protocol):
    """Subset of the ``architecture.md`` Simulator interface used here."""

    def simulate(
        self,
        initial_state: Any,
        control_sequence: Any,
        key: Any,
    ) -> Any: ...

    @property
    def state_dim(self) -> int: ...

    @property
    def action_dim(self) -> int: ...

    @property
    def horizon(self) -> int: ...


class CheckpointProtocol(Protocol):
    """Trained-checkpoint sample interface for the eval harness.

    A ``Checkpoint`` provides only one operation: sample one control
    sequence for a given spec under a given RNG key. The harness wraps
    this into the BoN sampling loop.
    """

    name: str

    def sample_controls(
        self,
        spec: Any,
        initial_state: Any,
        key: Any,
    ) -> Any: ...


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerSpecResult:
    """Eval outcome for a single (checkpoint, spec) pair.

    Fields
    ------
    spec_name:
        Registry key for the spec.
    n_samples:
        Number of trajectories drawn (= ``N_max``).
    rhos:
        Length-``n_samples`` vector of robustness scores (numpy float64).
    seeds:
        Length-``n_samples`` vector of integer seeds used for each draw.
    bon_success:
        ``{N: success_rate}`` mapping for each budget in
        ``budgets``. Reads from ``rhos`` via the sample-reuse rule.
    success_rate_marginal:
        Fraction of *individual* trajectories with ρ > 0 (the BoN-1
        success rate is identical to this).
    rho_mean, rho_iqr:
        ``rho_margin`` summary.
    n_nan:
        Count of trajectories where the simulator reported NaN/Inf in
        states (per the architecture.md NaN policy these trajectories'
        ρ values are recorded as ``nan`` here and dropped from the
        ``success_rate_marginal`` denominator).
    """

    spec_name: str
    n_samples: int
    rhos: np.ndarray  # shape (n_samples,)
    seeds: np.ndarray  # shape (n_samples,)
    bon_success: dict[int, float]
    success_rate_marginal: float
    rho_mean: float
    rho_iqr: float
    n_nan: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "spec_name": self.spec_name,
            "n_samples": self.n_samples,
            "rhos": self.rhos.tolist(),
            "seeds": self.seeds.tolist(),
            "bon_success": dict(self.bon_success),
            "success_rate_marginal": self.success_rate_marginal,
            "rho_mean": self.rho_mean,
            "rho_iqr": self.rho_iqr,
            "n_nan": self.n_nan,
        }


@dataclass(frozen=True)
class EvalResults:
    """Aggregate evaluation result for a single checkpoint.

    Fields
    ------
    checkpoint_name:
        ``checkpoint.name`` of the model evaluated.
    per_spec:
        Mapping from spec name to ``PerSpecResult``.
    budgets:
        Tuple of BoN budgets used.
    n_samples_per_spec:
        ``N_max``.
    """

    checkpoint_name: str
    per_spec: dict[str, PerSpecResult]
    budgets: tuple[int, ...]
    n_samples_per_spec: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def spec_names(self) -> tuple[str, ...]:
        return tuple(self.per_spec.keys())

    def aggregate_bon(self) -> dict[int, float]:
        """Mean BoN success across all evaluated specs."""
        out: dict[int, float] = {}
        if not self.per_spec:
            return out
        for n in self.budgets:
            vals = [r.bon_success.get(n, float("nan")) for r in self.per_spec.values()]
            arr = np.asarray(vals, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            out[int(n)] = float(arr.mean()) if arr.size else float("nan")
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_name": self.checkpoint_name,
            "n_samples_per_spec": self.n_samples_per_spec,
            "budgets": list(self.budgets),
            "per_spec": {k: v.as_dict() for k, v in self.per_spec.items()},
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# EvalHarness
# ---------------------------------------------------------------------------


class EvalHarness:
    """Run BoN evaluation of a checkpoint on a held-out set of STL specs.

    Parameters
    ----------
    simulator_registry:
        Mapping from spec name to a Simulator instance. Different specs
        live on different ODE families (gene-toggle vs predator-prey vs
        glucose-insulin), so the simulator is selected per-spec.
    spec_registry:
        Mapping from spec name to the STL spec object passed to the STL
        evaluator. The harness is agnostic to its concrete type — it
        forwards the value to ``stl_evaluator(spec, trajectory)``.
    stl_evaluator:
        Callable ``(spec, trajectory) -> float`` returning the
        Donzé-Maler robustness ρ. In production this is
        ``stl_seed.stl.evaluate_robustness``; in tests we pass a
        deterministic stub.
    initial_state_fn:
        Callable ``(spec_name, seed) -> initial_state``. Lets the
        harness draw fresh ``x_0`` per (spec, seed) combination, which
        is the registered eval setup (theory.md §5: "n_seeds = 5 trials
        per (m, v, f, i) configuration"). For deterministic ``x_0``
        per spec, return a constant.
    budgets:
        BoN budgets to evaluate. Default ``DEFAULT_BON_BUDGETS``.

    Notes
    -----
    The harness draws all ``n_samples_per_spec`` trajectories per spec,
    then computes BoN success at each budget by sample reuse. This
    matches the design choice in theory.md §5: BoN budgets are not
    independent within a cell — sample reuse keeps the variance
    structure honest.
    """

    def __init__(
        self,
        simulator_registry: Mapping[str, SimulatorProtocol],
        spec_registry: Mapping[str, Any],
        stl_evaluator: Any,
        initial_state_fn: Any,
        budgets: Sequence[int] = DEFAULT_BON_BUDGETS,
    ) -> None:
        self.simulator_registry = dict(simulator_registry)
        self.spec_registry = dict(spec_registry)
        self.stl_evaluator = stl_evaluator
        self.initial_state_fn = initial_state_fn
        self.budgets: tuple[int, ...] = tuple(int(n) for n in budgets)

        # Validate that every spec has a simulator
        missing = [s for s in self.spec_registry if s not in self.simulator_registry]
        if missing:
            raise KeyError(f"specs missing from simulator_registry: {sorted(missing)}")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def evaluate_checkpoint(
        self,
        checkpoint: CheckpointProtocol,
        held_out_specs: Sequence[str],
        n_samples_per_spec: int = 128,
        key: Any = 0,
    ) -> EvalResults:
        """Evaluate ``checkpoint`` on each spec in ``held_out_specs``.

        Parameters
        ----------
        checkpoint:
            Object satisfying :class:`CheckpointProtocol`.
        held_out_specs:
            Iterable of spec names. All names must be in both
            ``spec_registry`` and ``simulator_registry``.
        n_samples_per_spec:
            ``N_max`` — the number of trajectories drawn per spec.
            Must be at least the largest budget in ``self.budgets``.
        key:
            Either an integer seed or a ``jax.random.PRNGKey``. Splits
            deterministically across (spec, sample).
        """
        if n_samples_per_spec < max(self.budgets):
            raise ValueError(
                f"n_samples_per_spec={n_samples_per_spec} must be >= "
                f"max(budgets)={max(self.budgets)}"
            )

        unknown = [s for s in held_out_specs if s not in self.spec_registry]
        if unknown:
            raise KeyError(f"unknown specs in held_out_specs: {sorted(unknown)}")

        rng = jax.random.PRNGKey(key) if isinstance(key, int) else key

        per_spec: dict[str, PerSpecResult] = {}
        spec_keys = jax.random.split(rng, len(held_out_specs))
        for spec_idx, spec_name in enumerate(held_out_specs):
            per_spec[spec_name] = self._evaluate_one_spec(
                checkpoint=checkpoint,
                spec_name=spec_name,
                n_samples=n_samples_per_spec,
                key=spec_keys[spec_idx],
            )
        return EvalResults(
            checkpoint_name=getattr(checkpoint, "name", "<unknown>"),
            per_spec=per_spec,
            budgets=self.budgets,
            n_samples_per_spec=n_samples_per_spec,
        )

    # ------------------------------------------------------------------
    # Single-spec inner loop
    # ------------------------------------------------------------------

    def _evaluate_one_spec(
        self,
        checkpoint: CheckpointProtocol,
        spec_name: str,
        n_samples: int,
        key: jax.Array,
    ) -> PerSpecResult:
        spec = self.spec_registry[spec_name]
        sim = self.simulator_registry[spec_name]

        rhos = np.full(n_samples, np.nan, dtype=np.float64)
        seeds = np.zeros(n_samples, dtype=np.int64)
        sample_keys = jax.random.split(key, n_samples)

        n_nan = 0
        for s in range(n_samples):
            sk = sample_keys[s]
            # Derive a printable seed for provenance. ``jax.random.bits``
            # returns either a 0-d (newer JAX with typed PRNG keys) or
            # 1-d uint32 array depending on the active key impl; coerce
            # both into a plain Python int for record-keeping.
            bits = jnp.asarray(jax.random.bits(sk)).reshape(-1)
            seed_int = int(np.asarray(bits[0], dtype=np.uint32))
            seeds[s] = seed_int

            # Two sub-keys: one for x_0 sampling, one for the policy
            x0_key, policy_key, sim_key = jax.random.split(sk, 3)
            initial_state = self.initial_state_fn(spec_name, x0_key)
            try:
                controls = checkpoint.sample_controls(
                    spec=spec,
                    initial_state=initial_state,
                    key=policy_key,
                )
                trajectory = sim.simulate(
                    initial_state=initial_state,
                    control_sequence=controls,
                    key=sim_key,
                )
                rho = float(self.stl_evaluator(spec, trajectory))
                # Detect NaN/Inf trajectory states (architecture.md NaN policy)
                states = getattr(trajectory, "states", None)
                if states is not None:
                    states_np = np.asarray(states)
                    if not np.all(np.isfinite(states_np)):
                        n_nan += 1
                if not np.isfinite(rho):
                    n_nan += 1
                    rho = float("nan")
                rhos[s] = rho
            except (FloatingPointError, RuntimeError, ValueError):
                # Simulator/evaluator failures: count and mark NaN; do not
                # crash the harness (architecture.md NaN policy: replace
                # with sentinel and continue).
                n_nan += 1
                rhos[s] = float("nan")

        # BoN success via sample reuse: shape (1, n_samples) — single
        # "seed" pool from which we draw N samples, matching the
        # eval-time sample-reuse semantics.
        rhos_2d = rhos[None, :]  # the seed dim is collapsed; bon_success_curve
        # treats each seed as one BoN trial. Here every spec instance
        # contributes one seed-of-1; cross-spec aggregation happens in
        # ``EvalResults.aggregate_bon``.
        # We adopt the convention from theory.md §5: per (m,v,f,i,s) cell,
        # draw N_max samples and read BoN-K success as
        # 1{max(rhos[:K]) > 0}; aggregating across seeds happens at the
        # outer driver level. Since this method is per-spec (one cell),
        # we report the per-cell indicator-vector mean as a scalar in [0,1].
        # bon_success_curve handles arbitrary (n_seeds, K) but here
        # n_seeds = 1, so the result is in {0, 1}.
        bon_curve = bon_success_curve(rhos_2d, budgets=self.budgets)

        sr = success_rate(rhos)
        rmean, riqr = rho_margin(rhos)

        return PerSpecResult(
            spec_name=spec_name,
            n_samples=n_samples,
            rhos=rhos,
            seeds=seeds,
            bon_success=bon_curve,
            success_rate_marginal=sr,
            rho_mean=rmean,
            rho_iqr=riqr,
            n_nan=n_nan,
        )


__all__ = [
    "EvalHarness",
    "EvalResults",
    "PerSpecResult",
    "SimulatorProtocol",
    "CheckpointProtocol",
    "DEFAULT_BON_BUDGETS",
]

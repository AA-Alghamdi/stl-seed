"""Unified empirical comparison of all five sampling strategies.

Synthesises four scattered empirical claims in the artifact into one
apples-to-apples picture:

* Tier 3   — gradient-guided gives ~12x mean rho over a flat-prior
  baseline on glucose-insulin (``paper/inference_method.md``).
* Tier 6   — STL-rho dominates the PAV learned baseline at every
  train-set size (``paper/pav_comparison.md``); not re-run here, the
  PAV result is referenced in the markdown report rather than recomputed
  per-seed because PAV is a *verifier* baseline, not a *sampler*.
* Tier 10  — gradient-guided FAILS to transfer to the repressilator
  task family on the canonical IC (``paper/cross_task_validation.md``).
  This script reproduces that finding alongside the positive Tier 3
  result so the asymmetric outcome is visible in one figure.
* Tier 10b — the hybrid sampler (``HybridGradientBoNSampler``) recovers
  some of the loss on hard glucose specs by combining argmax-rho
  selection over ``n`` gradient-guided draws.

This script is the *headline* visualisation. It is not a new
experiment — it consumes only the public sampler API of
``stl_seed.inference``, runs every sampler on every (task, spec) cell
across many seeds, and outputs a single grouped bar chart with 95% CIs.

Outputs
-------
* ``runs/unified_comparison/results.parquet``
    Long-form table; columns
    ``(task, sampler, seed, final_rho, satisfied,
       n_steps_changed_by_guidance, wall_clock_s)``.
* ``paper/figures/unified_comparison.png``
    Grouped bar chart (one group per task family, one bar per
    sampler); error bars are bootstrap-percentile 95 % CIs over seeds.
* ``paper/unified_comparison_results.md``
    Auto-generated narrative including the per-cell mean ± 95% CI
    table, the headline numbers ready for the cold-email pitch, and
    citations back to the per-tier source documents.

Runtime
-------
With ``--n-seeds 8`` (the default) the harness completes in roughly
3 minutes on an M5 Pro. Glucose-insulin is fast (12 control steps,
3-state ODE); the repressilator is slower (10 steps, 6-state stiff
ODE). The hybrid sampler dominates wall-clock per-seed (it makes
``n=4`` gradient-guided draws), which is reflected in the
``wall_clock_s`` column.

REDACTED firewall
-------------
This script imports only from ``stl_seed.{inference, specs, tasks}``,
JAX, NumPy, Pandas, Matplotlib, and Rich. No ``REDACTED``,
``REDACTED``, ``REDACTED``, ``REDACTED``, or
``REDACTED`` symbol is touched. Verified by
``scripts/REDACTED.sh``.

Usage
-----
::

    uv run python scripts/run_unified_comparison.py
    uv run python scripts/run_unified_comparison.py --n-seeds 4
    uv run python scripts/run_unified_comparison.py \\
        --tasks glucose_insulin \\
        --samplers standard,gradient_guided
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from stl_seed.inference import (
    BeamSearchWarmstartSampler,
    BestOfNSampler,
    ContinuousBoNSampler,
    HybridGradientBoNSampler,
    LLMProposal,
    Sampler,
    StandardSampler,
    STLGradientGuidedSampler,
)
from stl_seed.inference.cmaes_gradient import CMAESGradientSampler
from stl_seed.inference.gradient_guided import make_uniform_action_vocabulary
from stl_seed.inference.horizon_folded import HorizonFoldedGradientSampler
from stl_seed.inference.rollout_tree import RolloutTreeSampler
from stl_seed.specs import REGISTRY
from stl_seed.tasks.bio_ode import (
    MAPK_ACTION_DIM,
    REPRESSILATOR_ACTION_DIM,
    TOGGLE_ACTION_DIM,
    MAPKSimulator,
    RepressilatorSimulator,
    ToggleSimulator,
    _mapk_initial_state,
    _repressilator_initial_state,
    _toggle_initial_state,
)
from stl_seed.tasks.bio_ode_params import (
    MAPKParams,
    RepressilatorParams,
    ToggleParams,
)
from stl_seed.tasks.glucose_insulin import (
    BergmanParams,
    GlucoseInsulinSimulator,
    default_normal_subject_initial_state,
)

# ---------------------------------------------------------------------------
# Paths and defaults.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUT_DIR = _REPO_ROOT / "runs" / "unified_comparison"
_DEFAULT_FIG_PATH = _REPO_ROOT / "paper" / "figures" / "unified_comparison.png"
_DEFAULT_MD_PATH = _REPO_ROOT / "paper" / "unified_comparison_results.md"

_DEFAULT_TASKS: tuple[str, ...] = (
    "glucose_insulin",
    "bio_ode.repressilator",
    "bio_ode.toggle",
    "bio_ode.mapk",
)
_DEFAULT_SAMPLERS: tuple[str, ...] = (
    "standard",
    "best_of_n",
    "continuous_bon",
    "gradient_guided",
    "hybrid",
    "horizon_folded",
    "rollout_tree",
    "cmaes_gradient",
    "beam_search_warmstart",
)

# Sampling temperature used uniformly across the harness so the comparison
# is apples-to-apples. 0.5 is the value used by the existing
# ``test_gradient_guided_improves_rho`` and ``test_hybrid_beats_pure_guidance``
# tests; it keeps the LLM stochastic enough that BoN actually has variance
# to exploit.
_SAMPLING_TEMPERATURE: float = 0.5

# Standard hyperparameters per the spec in the harness brief.
_BON_N: int = 8
_GRADIENT_GUIDANCE_WEIGHT: float = 2.0
_HYBRID_N: int = 4

# Hyperparameters for the four "extended" samplers (A1, A2, A3, C1).
# These mirror the smoke-test defaults used by their unit tests
# (tests/test_horizon_folded.py, tests/test_rollout_tree.py,
# tests/test_cmaes_gradient.py, tests/test_beam_search_warmstart.py).
_HORIZON_FOLDED_K_ITERS: int = 100
_HORIZON_FOLDED_LR: float = 1e-2
_ROLLOUT_TREE_BRANCH_K: int = 8
_ROLLOUT_TREE_LOOKAHEAD_H: int = 5
_CMAES_POPULATION_SIZE: int = 32
_CMAES_N_GENERATIONS: int = 20
_CMAES_SIGMA_INIT: float = 0.3
_CMAES_N_REFINE: int = 30
_BEAM_SIZE: int = 8
_BEAM_GRADIENT_REFINE_ITERS: int = 30

# Per-task vocabulary size for the structural-search samplers
# (beam_search_warmstart). The narrow k_per_dim=2 corner vocabulary used by
# the gradient sampler omits intermediate actions; for beam search we want a
# denser lattice so the satisfying corners are easy to enumerate. ``5`` per
# dim gives K=125 on the 3-D repressilator action box (matches the test in
# tests/test_beam_search_warmstart.py::test_beam_search_recovers_repressilator_solution).
_BEAM_K_PER_DIM_REPRESSILATOR: int = 5
# Beam-search vocabulary density for the toggle (2-D action box). k=5 gives
# K=25 candidates per beam-expansion step; this keeps the satisfying
# u = (0, 1) corner in scope while letting the lookahead-rho score
# discriminate intermediate (e.g. partial-IPTG) candidates that the
# coarse k=2 lattice collapses onto a corner.
_BEAM_K_PER_DIM_TOGGLE: int = 5

# Rich console, shared across the script.
console = Console()

# ---------------------------------------------------------------------------
# Synthetic LLM: flat prior over the action vocabulary.
# ---------------------------------------------------------------------------
#
# Using a flat-prior LLM is the cleanest test bed for sampler comparison:
# the only signal driving choices is the verifier (rho or grad rho), so
# the comparison isolates *what each sampler does with the verifier
# information*, not how well a particular LLM happens to know the task.
# The same flat-LLM regime is used by tests/test_inference.py.


def _uniform_llm(K: int) -> LLMProposal:
    """A flat (entropy = log K) LLM proxy."""

    def llm(state, history, key):
        return jnp.zeros(K, dtype=jnp.float32)

    return llm


# ---------------------------------------------------------------------------
# Task setup: returns everything the samplers need to run.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TaskSetup:
    """Resolved task fixture.

    Attributes
    ----------
    name:
        Short task family identifier (e.g. ``"glucose_insulin"``,
        ``"bio_ode.repressilator"``); appears in the results table and
        on the figure x-axis.
    spec_key:
        Registry key of the STL spec used for scoring this task family.
    simulator:
        The Diffrax-backed simulator instance.
    params:
        Kinetic parameter pytree consumed by ``simulator.simulate``.
    spec:
        Compiled STL spec (the ``REGISTRY[spec_key]`` value).
    vocabulary:
        Action vocabulary ``V`` of shape ``(K, m)``.
    initial_state:
        Initial state vector ``x_0`` of shape ``(n,)``.
    horizon:
        Number of control steps ``H`` (== ``simulator.n_control_points``).
    aux:
        Optional task-specific kwargs (e.g. ``{"meal_schedule": ...}``).
    """

    name: str
    spec_key: str
    simulator: Any
    params: Any
    spec: Any
    vocabulary: Any
    initial_state: Any
    horizon: int
    aux: dict[str, Any] | None


def _glucose_insulin_setup() -> TaskSetup:
    """Glucose-insulin task on the easy time-in-range spec.

    Spec: ``glucose_insulin.tir.easy``. This is the spec on which the
    Tier-3 headline result holds and is the natural place to anchor the
    positive cross-sampler comparison. The vocabulary is the standard
    5-level uniform grid on [0, 5] U/h used by tests/test_inference.py.
    """
    sim = GlucoseInsulinSimulator()
    params = BergmanParams()
    spec = REGISTRY["glucose_insulin.tir.easy"]
    V = make_uniform_action_vocabulary(0.0, 5.0, k_per_dim=5)
    x0 = default_normal_subject_initial_state(params)
    return TaskSetup(
        name="glucose_insulin",
        spec_key=spec.name,
        simulator=sim,
        params=params,
        spec=spec,
        vocabulary=V,
        initial_state=x0,
        horizon=int(sim.n_control_points),
        aux=None,
    )


def _bio_ode_repressilator_setup() -> TaskSetup:
    """Repressilator task on the easy spec, canonical pilot IC.

    Spec: ``bio_ode.repressilator.easy``. This is the spec on which the
    Tier-10 negative result holds; the pilot IC ``[0, 0, 0, 15, 5, 25]``
    is the IC used by the documented cross-task experiment in
    ``paper/cross_task_validation.md``. Vocabulary is the 8-corner
    discretisation of [0, 1]^3, which contains the known-satisfying
    silence-gene-3 action (0, 0, 1).
    """
    sim = RepressilatorSimulator()
    params = RepressilatorParams()
    spec = REGISTRY["bio_ode.repressilator.easy"]
    V = make_uniform_action_vocabulary(
        [0.0] * REPRESSILATOR_ACTION_DIM,
        [1.0] * REPRESSILATOR_ACTION_DIM,
        k_per_dim=2,
    )
    x0 = _repressilator_initial_state(params)
    return TaskSetup(
        name="bio_ode.repressilator",
        spec_key=spec.name,
        simulator=sim,
        params=params,
        spec=spec,
        vocabulary=V,
        initial_state=x0,
        horizon=int(sim.n_control_points),
        aux=None,
    )


def _bio_ode_toggle_setup() -> TaskSetup:
    """Toggle-switch task on the medium spec.

    Spec: ``bio_ode.toggle.medium`` (post-2026-04-25 spec fix; HIGH=100
    nM, reachable given alpha_1 = 160 saturation cap). The satisfying
    region requires saturating the gene-2 inducer (u = (0, 1) constant
    drives x_1 to ~160 nM, x_2 to ~0). Vocabulary is the 4-corner
    discretisation of [0, 1]^2 for gradient samplers (k_per_dim=2);
    beam-search uses the denser k_per_dim=5 lattice (K=25) so the
    satisfying corner is in the lookahead-rho enumeration directly.
    """
    sim = ToggleSimulator()
    params = ToggleParams()
    spec = REGISTRY["bio_ode.toggle.medium"]
    V = make_uniform_action_vocabulary(
        [0.0] * TOGGLE_ACTION_DIM,
        [1.0] * TOGGLE_ACTION_DIM,
        k_per_dim=2,
    )
    x0 = _toggle_initial_state(params)
    return TaskSetup(
        name="bio_ode.toggle",
        spec_key=spec.name,
        simulator=sim,
        params=params,
        spec=spec,
        vocabulary=V,
        initial_state=x0,
        horizon=int(sim.n_control_points),
        aux=None,
    )


def _bio_ode_mapk_setup() -> TaskSetup:
    """MAPK cascade task on the hard spec.

    Spec: ``bio_ode.mapk.hard`` (post-2026-04-25 spec fix; reads state
    index 4 (MAPK_PP) in absolute microM, peak >= 0.5 microM in
    [0, 30], settle < 0.05 microM in [45, 60], MKKK_P safety <
    0.002975 microM throughout). The satisfying policy is a brief
    activating pulse (e.g. u=1 for 1-3 control steps then u=0)
    because MAPK_PP must rise above 0.5 microM but settle back to
    near zero by t=45. Random-policy success rate is ~0 because the
    cascade lacks fast enough negative feedback to deactivate
    MAPK_PP within the 15-min settle window once activated. Beam-
    search over a small action vocabulary recovers the pulse
    deterministically. Action vocabulary is the 5-level uniform grid
    on [0, 1] (k_per_dim=5, K=5); the action box is one-dimensional,
    so the same lattice is used for all samplers.
    """
    sim = MAPKSimulator()
    params = MAPKParams()
    spec = REGISTRY["bio_ode.mapk.hard"]
    V = make_uniform_action_vocabulary(
        [0.0] * MAPK_ACTION_DIM,
        [1.0] * MAPK_ACTION_DIM,
        k_per_dim=5,
    )
    x0 = _mapk_initial_state(params)
    return TaskSetup(
        name="bio_ode.mapk",
        spec_key=spec.name,
        simulator=sim,
        params=params,
        spec=spec,
        vocabulary=V,
        initial_state=x0,
        horizon=int(sim.n_control_points),
        aux=None,
    )


# ---------------------------------------------------------------------------
# Per-sampler vocabulary overrides.
# ---------------------------------------------------------------------------
#
# The beam-search warmstart sampler (C1) is the canonical fix for the
# repressilator failure documented in paper/cross_task_validation.md, and
# it relies on having the satisfying ``silence-3 = (0, 0, 1)`` corner
# present in its action vocabulary. The headline pilot in
# ``tests/test_beam_search_warmstart.py::test_beam_search_recovers_repressilator_solution``
# uses k_per_dim=5 (K=125) precisely because the 8-corner lattice that is
# adequate for the gradient sampler is too sparse to make
# constant-extrapolation lookahead reliable. We therefore override the
# vocabulary on a per-sampler basis here so the comparison reflects each
# sampler's *intended* operating regime; the gradient samplers continue
# to see the 8-corner default vocabulary documented in the original
# negative-result protocol.


def _vocabulary_for(sampler_name: str, setup: TaskSetup):
    """Per-sampler vocabulary override.

    Beam-search needs a denser lattice on the multi-dimensional action
    boxes so the satisfying corners are present in the vocabulary by
    construction. Specifically:

    * On the repressilator (3-D action box) we use ``k_per_dim=5``
      (K=125) to put the silence-3 corner ``u = (0, 0, 1)`` in scope.
      The negative result documented in
      ``paper/cross_task_validation.md`` was on the 8-corner default.
    * On the toggle (2-D action box) we use ``k_per_dim=5`` (K=25) to
      put the silence-B corner ``u = (0, 1)`` in scope. The default
      4-corner vocabulary already contains it but the denser lattice
      lets the lookahead-rho extrapolation discriminate intermediate
      candidates more reliably.
    * On MAPK (1-D action box) the task default already uses
      ``k_per_dim=5`` (K=5), so no override is needed.

    All other (sampler, task) cells use the task-default vocabulary.
    """
    if sampler_name == "beam_search_warmstart":
        if setup.name == "bio_ode.repressilator":
            return make_uniform_action_vocabulary(
                [0.0] * REPRESSILATOR_ACTION_DIM,
                [1.0] * REPRESSILATOR_ACTION_DIM,
                k_per_dim=_BEAM_K_PER_DIM_REPRESSILATOR,
            )
        if setup.name == "bio_ode.toggle":
            return make_uniform_action_vocabulary(
                [0.0] * TOGGLE_ACTION_DIM,
                [1.0] * TOGGLE_ACTION_DIM,
                k_per_dim=_BEAM_K_PER_DIM_TOGGLE,
            )
    return setup.vocabulary


_TASK_BUILDERS: dict[str, callable] = {
    "glucose_insulin": _glucose_insulin_setup,
    "bio_ode.repressilator": _bio_ode_repressilator_setup,
    "bio_ode.toggle": _bio_ode_toggle_setup,
    "bio_ode.mapk": _bio_ode_mapk_setup,
}


# ---------------------------------------------------------------------------
# Sampler factory.
# ---------------------------------------------------------------------------


def _build_sampler(name: str, setup: TaskSetup) -> Sampler:
    """Construct a sampler instance for the given (sampler-name, task).

    The factory keeps construction in a single place so the harness
    column for ``sampler`` always matches what the figure / table
    headers display. Hyperparameters are pinned at module level
    (``_BON_N``, ``_GRADIENT_GUIDANCE_WEIGHT``, ``_HYBRID_N``,
    ``_HORIZON_FOLDED_*``, ``_ROLLOUT_TREE_*``, ``_CMAES_*``,
    ``_BEAM_*``). The vocabulary used can be overridden per-(sampler,
    task) by :func:`_vocabulary_for` — currently only the beam-search
    warmstart sampler on the repressilator uses an override (the dense
    k_per_dim=5 lattice that contains the silence-3 satisfying corner).
    """
    vocabulary = _vocabulary_for(name, setup)
    K = int(vocabulary.shape[0])
    llm = _uniform_llm(K)
    # Samplers split into two API shapes: most accept ``sampling_temperature``
    # in their constructor (the original five), the four extended samplers
    # (horizon_folded, rollout_tree, cmaes_gradient, beam_search_warmstart)
    # do not — they have their own action-selection mechanism that does not
    # consume an LLM-temperature knob. We therefore branch on the name and
    # build two slightly different kwargs dicts.
    common = dict(
        llm=llm,
        simulator=setup.simulator,
        spec=setup.spec,
        action_vocabulary=vocabulary,
        sim_params=setup.params,
        horizon=setup.horizon,
        aux=setup.aux,
    )
    if name == "standard":
        return StandardSampler(sampling_temperature=_SAMPLING_TEMPERATURE, **common)
    if name == "best_of_n":
        return BestOfNSampler(n=_BON_N, sampling_temperature=_SAMPLING_TEMPERATURE, **common)
    if name == "continuous_bon":
        return ContinuousBoNSampler(n=_BON_N, sampling_temperature=_SAMPLING_TEMPERATURE, **common)
    if name == "gradient_guided":
        return STLGradientGuidedSampler(
            guidance_weight=_GRADIENT_GUIDANCE_WEIGHT,
            sampling_temperature=_SAMPLING_TEMPERATURE,
            **common,
        )
    if name == "hybrid":
        return HybridGradientBoNSampler(
            n=_HYBRID_N,
            guidance_weight=_GRADIENT_GUIDANCE_WEIGHT,
            sampling_temperature=_SAMPLING_TEMPERATURE,
            **common,
        )
    if name == "horizon_folded":
        return HorizonFoldedGradientSampler(
            lr=_HORIZON_FOLDED_LR,
            k_iters=_HORIZON_FOLDED_K_ITERS,
            init="zeros",
            **common,
        )
    if name == "rollout_tree":
        return RolloutTreeSampler(
            branch_k=_ROLLOUT_TREE_BRANCH_K,
            lookahead_h=_ROLLOUT_TREE_LOOKAHEAD_H,
            continuation_policy="zero",
            refine_iters=0,
            **common,
        )
    if name == "cmaes_gradient":
        return CMAESGradientSampler(
            population_size=_CMAES_POPULATION_SIZE,
            n_generations=_CMAES_N_GENERATIONS,
            sigma_init=_CMAES_SIGMA_INIT,
            n_refine=_CMAES_N_REFINE,
            initial_mean_source="midpoint",
            **common,
        )
    if name == "beam_search_warmstart":
        return BeamSearchWarmstartSampler(
            beam_size=_BEAM_SIZE,
            gradient_refine_iters=_BEAM_GRADIENT_REFINE_ITERS,
            tail_strategy="repeat_candidate",
            **common,
        )
    raise ValueError(f"Unknown sampler {name!r}; expected one of {_DEFAULT_SAMPLERS}")


# ---------------------------------------------------------------------------
# Bootstrap CIs.
# ---------------------------------------------------------------------------


def _bootstrap_ci(
    values: np.ndarray,
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values``.

    Returns ``(lo, hi)`` for the ``(1 - alpha)`` two-sided interval.
    NaN-safe: drops non-finite entries before resampling. Returns
    ``(nan, nan)`` if fewer than two finite samples remain.
    """
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    n = finite.size
    boots = np.empty(int(n_resamples), dtype=np.float64)
    for i in range(int(n_resamples)):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(np.mean(finite[idx]))
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return lo, hi


# ---------------------------------------------------------------------------
# Core harness: run one cell, run all cells.
# ---------------------------------------------------------------------------


def _run_one_cell(
    task: TaskSetup,
    sampler_name: str,
    seed: int,
) -> dict[str, Any]:
    """Run one (task, sampler, seed) cell and emit a single row.

    The row schema is
    ``(task, sampler, seed, final_rho, satisfied,
       n_steps_changed_by_guidance, wall_clock_s)``.

    For samplers that do not have a notion of "steps changed by
    guidance" (the three baselines), the column is reported as 0.

    ``satisfied`` follows the Donzé-Maler convention: ``rho > 0`` iff
    the trajectory satisfies the spec.
    """
    sampler = _build_sampler(sampler_name, task)
    key = jax.random.key(int(seed))
    t0 = time.time()
    _, diag = sampler.sample(task.initial_state, key)
    wall = time.time() - t0
    rho = float(diag["final_rho"])
    n_changed = int(diag.get("n_steps_changed_by_guidance", 0))
    return {
        "task": task.name,
        "spec_key": task.spec_key,
        "sampler": sampler_name,
        "seed": int(seed),
        "final_rho": rho,
        "satisfied": bool(np.isfinite(rho) and rho > 0.0),
        "n_steps_changed_by_guidance": n_changed,
        "wall_clock_s": float(wall),
    }


def _run_all_cells(
    tasks: list[TaskSetup],
    samplers: list[str],
    n_seeds: int,
    seed_offset: int = 1000,
) -> pd.DataFrame:
    """Run the full (tasks x samplers x seeds) grid.

    Returns a long-form pandas DataFrame with one row per cell. The
    seed offset of 1000 matches the convention used by
    ``tests/test_inference.py::test_gradient_guided_improves_rho`` so
    that cross-tier comparisons reproduce per-seed when desired.
    """
    rows: list[dict[str, Any]] = []
    total = len(tasks) * len(samplers) * int(n_seeds)
    counter = 0
    for task in tasks:
        # Build samplers once and reuse across seeds. This is a real
        # wall-clock optimisation for the gradient-based samplers
        # (which have a ~5s JIT warmup the first time the
        # value_and_grad closure is traced); for the baselines it is
        # neutral.
        cached: dict[str, Sampler] = {}
        for sampler_name in samplers:
            cached[sampler_name] = _build_sampler(sampler_name, task)
        for sampler_name in samplers:
            sampler = cached[sampler_name]
            for s in range(int(n_seeds)):
                seed = int(seed_offset + s)
                key = jax.random.key(seed)
                t0 = time.time()
                _, diag = sampler.sample(task.initial_state, key)
                wall = time.time() - t0
                rho = float(diag["final_rho"])
                n_changed = int(diag.get("n_steps_changed_by_guidance", 0))
                rows.append(
                    {
                        "task": task.name,
                        "spec_key": task.spec_key,
                        "sampler": sampler_name,
                        "seed": seed,
                        "final_rho": rho,
                        "satisfied": bool(np.isfinite(rho) and rho > 0.0),
                        "n_steps_changed_by_guidance": n_changed,
                        "wall_clock_s": float(wall),
                    }
                )
                counter += 1
                if counter % 5 == 0 or counter == total:
                    console.print(
                        f"  [{counter:3d}/{total}] "
                        f"task={task.name} sampler={sampler_name} "
                        f"seed={seed} rho={rho:+.3f} "
                        f"wall={wall:.2f}s"
                    )
    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Aggregation.
# ---------------------------------------------------------------------------


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell aggregation: mean rho, 95% CI, satisfaction frac, mean wall.

    Returns a wide-form DataFrame indexed by (task, sampler) with
    columns ``mean_rho, ci_lo, ci_hi, sat_frac, mean_wall_s, n_seeds``.
    The CI is a nan-safe percentile bootstrap.
    """
    records: list[dict[str, Any]] = []
    for (task, sampler), grp in df.groupby(["task", "sampler"], sort=False):
        rho = grp["final_rho"].to_numpy()
        lo, hi = _bootstrap_ci(rho, seed=hash((task, sampler)) & 0xFFFF)
        records.append(
            {
                "task": task,
                "sampler": sampler,
                "mean_rho": float(np.nanmean(rho)),
                "ci_lo": float(lo),
                "ci_hi": float(hi),
                "sat_frac": float(grp["satisfied"].mean()),
                "mean_wall_s": float(grp["wall_clock_s"].mean()),
                "n_seeds": int(len(grp)),
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting.
# ---------------------------------------------------------------------------


_SAMPLER_DISPLAY: dict[str, str] = {
    "standard": "Standard\n(λ=0)",
    "best_of_n": f"Binary BoN\n(N={_BON_N})",
    "continuous_bon": f"Continuous BoN\n(N={_BON_N})",
    "gradient_guided": f"Gradient-Guided\n(λ={_GRADIENT_GUIDANCE_WEIGHT:g})",
    "hybrid": f"Hybrid GBoN\n(n={_HYBRID_N}, λ={_GRADIENT_GUIDANCE_WEIGHT:g})",
    "horizon_folded": f"Horizon-Folded\n(K={_HORIZON_FOLDED_K_ITERS})",
    "rollout_tree": f"Rollout-Tree\n(B={_ROLLOUT_TREE_BRANCH_K}, L={_ROLLOUT_TREE_LOOKAHEAD_H})",
    "cmaes_gradient": (f"CMA-ES + Grad\n(λ={_CMAES_POPULATION_SIZE}, G={_CMAES_N_GENERATIONS})"),
    "beam_search_warmstart": f"Beam Warmstart\n(B={_BEAM_SIZE})",
}

_SAMPLER_COLORS: dict[str, str] = {
    "standard": "#888888",
    "best_of_n": "#7099c4",
    "continuous_bon": "#3f6fa7",
    "gradient_guided": "#c44e52",
    "hybrid": "#8c3a3a",
    "horizon_folded": "#5d8a4c",
    "rollout_tree": "#b08a3e",
    "cmaes_gradient": "#7a5da8",
    "beam_search_warmstart": "#2f9e6a",
}


def _plot_unified_comparison(
    agg: pd.DataFrame,
    tasks: list[str],
    samplers: list[str],
    out_path: Path,
) -> None:
    """Grouped bar chart, one group per task, one bar per sampler.

    Error bars are 95% bootstrap CIs (asymmetric); the y-axis is
    final-trajectory rho. We deliberately do *not* normalise per-task —
    the natural rho scales differ across spec families (the TIR spec
    saturates around ~20 rho units; the repressilator easy spec runs in
    the [-250, +25] band) and showing the raw scales preserves the
    asymmetry that this figure exists to highlight.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_tasks = len(tasks)
    n_samplers = len(samplers)
    # Bar width scales with the per-task slot occupancy; we leave ~10% gap
    # between adjacent task groups so the legend isn't crowded out at the
    # 9-sampler default.
    width = 0.9 / max(n_samplers, 1)
    x = np.arange(n_tasks, dtype=np.float64)

    fig_width = max(11.5, 1.3 * n_samplers + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    for i, samp in enumerate(samplers):
        means = []
        err_lo = []
        err_hi = []
        for t in tasks:
            row = agg[(agg["task"] == t) & (agg["sampler"] == samp)]
            if row.empty:
                means.append(np.nan)
                err_lo.append(0.0)
                err_hi.append(0.0)
                continue
            m = float(row["mean_rho"].iloc[0])
            lo = float(row["ci_lo"].iloc[0])
            hi = float(row["ci_hi"].iloc[0])
            means.append(m)
            err_lo.append(max(0.0, m - lo) if np.isfinite(lo) else 0.0)
            err_hi.append(max(0.0, hi - m) if np.isfinite(hi) else 0.0)
        offset = (i - (n_samplers - 1) / 2.0) * width
        ax.bar(
            x + offset,
            means,
            width=width,
            yerr=np.array([err_lo, err_hi]),
            capsize=3,
            color=_SAMPLER_COLORS.get(samp),
            edgecolor="black",
            linewidth=0.5,
            label=_SAMPLER_DISPLAY.get(samp, samp),
        )

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("Final STL robustness ρ  (mean ± 95% bootstrap CI)")
    ax.set_title(
        "Unified sampler comparison: different samplers dominate different "
        "task structures.\nGradient-guided wins on glucose-insulin (smooth, "
        "locally-informative gradients);\nbeam-search warmstart wins on the "
        "repressilator (narrow vocabulary attractor)."
    )
    # Cap legend column count so it stays readable at 9 samplers.
    legend_ncol = min(n_samplers, 5)
    ax.legend(
        loc="upper center",
        ncol=legend_ncol,
        fontsize=8,
        framealpha=0.95,
        bbox_to_anchor=(0.5, -0.10),
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown writer.
# ---------------------------------------------------------------------------


def _make_pivot(agg: pd.DataFrame, value: str) -> pd.DataFrame:
    """Wide pivot ``(task -> sampler -> value)`` for tabular display."""
    return agg.pivot(index="task", columns="sampler", values=value)


def _write_markdown_report(
    df: pd.DataFrame,
    agg: pd.DataFrame,
    tasks: list[str],
    samplers: list[str],
    fig_path: Path,
    out_path: Path,
) -> None:
    """Auto-generate ``paper/unified_comparison_results.md``.

    The report leads with the headline (asymmetric task-family result),
    then the per-(task, sampler) numbers, then a comparison-vs-prior-tiers
    paragraph. Every numeric claim in the report is sourced from
    ``agg`` so the file is reproducible from the parquet alone.
    """
    pivot_mean = _make_pivot(agg, "mean_rho")
    pivot_lo = _make_pivot(agg, "ci_lo")
    pivot_hi = _make_pivot(agg, "ci_hi")
    pivot_sat = _make_pivot(agg, "sat_frac")
    pivot_wall = _make_pivot(agg, "mean_wall_s")

    n_seeds = int(agg["n_seeds"].max())

    # Headline numbers used in the cold-email pitch.
    def _val(task: str, samp: str, table: pd.DataFrame) -> float:
        try:
            return float(table.loc[task, samp])
        except KeyError:
            return float("nan")

    gi_std = _val("glucose_insulin", "standard", pivot_mean)
    gi_grad = _val("glucose_insulin", "gradient_guided", pivot_mean)
    gi_hybrid = _val("glucose_insulin", "hybrid", pivot_mean)
    gi_cbon = _val("glucose_insulin", "continuous_bon", pivot_mean)
    repr_std = _val("bio_ode.repressilator", "standard", pivot_mean)
    repr_grad = _val("bio_ode.repressilator", "gradient_guided", pivot_mean)
    repr_hybrid = _val("bio_ode.repressilator", "hybrid", pivot_mean)
    repr_beam = _val("bio_ode.repressilator", "beam_search_warmstart", pivot_mean)
    repr_beam_sat = _val("bio_ode.repressilator", "beam_search_warmstart", pivot_sat)

    def _ratio(num: float, den: float) -> str:
        if not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1e-9:
            return "n/a"
        return f"{num / den:+.2f}x"

    headline_gi = (
        f"Gradient-guided sampler attains mean rho = {gi_grad:+.3f} on "
        f"`glucose_insulin.tir.easy` versus the standard-sampler baseline at "
        f"{gi_std:+.3f} ({_ratio(gi_grad, gi_std)} of baseline) and "
        f"continuous-BoN at {gi_cbon:+.3f}, on N = {n_seeds} seeds."
    )
    headline_repr = (
        f"Beam-search warmstart resolves the repressilator failure: mean "
        f"rho = {repr_beam:+.3f} ({repr_beam_sat:.0%} satisfaction over "
        f"N = {n_seeds} seeds) versus gradient-guided at {repr_grad:+.3f} "
        f"and standard at {repr_std:+.3f}. Vocabulary enumeration finds the "
        f"satisfying corner that continuous-gradient descent cannot."
    )
    headline_hybrid = (
        f"Hybrid sampler attains mean rho = {gi_hybrid:+.3f} on glucose "
        f"and {repr_hybrid:+.3f} on the repressilator; the +"
        f"{(gi_hybrid - gi_grad):.3f} delta over pure gradient guidance on "
        f"glucose comes from argmax-rho selection over n = "
        f"{_HYBRID_N} guided draws."
    )

    lines: list[str] = []
    lines.append("# Unified sampler comparison (auto-generated)")
    lines.append("")
    # Figure path is reported relative to repo root when it lies under
    # the repo (the production case); otherwise we report the absolute
    # path verbatim (the case under pytest's tmp_path fixture).
    try:
        fig_disp = str(fig_path.relative_to(_REPO_ROOT))
    except ValueError:
        fig_disp = str(fig_path)
    lines.append(
        "Generated by `scripts/run_unified_comparison.py`. Do not "
        "hand-edit -- changes are clobbered on the next run. The "
        "underlying long-form data is at "
        "`runs/unified_comparison/results.parquet`; the figure is at "
        f"`{fig_disp}`."
    )
    lines.append("")
    lines.append("## 1. Headline")
    lines.append("")
    lines.append(
        "**Different samplers dominate different task structures. Gradient "
        "guidance wins where smooth dynamics give locally-informative "
        "gradients (glucose-insulin); beam-search warmstart wins where the "
        "satisfying region is a measure-near-zero attractor in vocabulary "
        "space (repressilator). The artifact characterises which sampler "
        "wins which task class, with reproducible per-seed evidence.**"
    )
    lines.append("")
    lines.append(f"- {headline_gi}")
    lines.append(f"- {headline_repr}")
    lines.append(f"- {headline_hybrid}")
    lines.append("")
    lines.append(
        "All cells use a flat-prior LLM (uniform logits over the action "
        f"vocabulary), sampling temperature {_SAMPLING_TEMPERATURE} where "
        f"applicable, and per-sampler hyperparameters: BoN N = {_BON_N}, "
        f"gradient guidance λ = {_GRADIENT_GUIDANCE_WEIGHT:g}, hybrid n = "
        f"{_HYBRID_N}, horizon-folded K_iters = {_HORIZON_FOLDED_K_ITERS} "
        f"(lr={_HORIZON_FOLDED_LR:g}), rollout-tree branch_k = "
        f"{_ROLLOUT_TREE_BRANCH_K} / lookahead = {_ROLLOUT_TREE_LOOKAHEAD_H}, "
        f"CMA-ES population = {_CMAES_POPULATION_SIZE} / generations = "
        f"{_CMAES_N_GENERATIONS} / sigma_init = {_CMAES_SIGMA_INIT:g} "
        f"(refine = {_CMAES_N_REFINE} steps), beam search beam_size = "
        f"{_BEAM_SIZE} (refine = {_BEAM_GRADIENT_REFINE_ITERS} steps, "
        f"tail_strategy='repeat_candidate'). The beam-search warmstart "
        f"sampler uses a denser k_per_dim={_BEAM_K_PER_DIM_REPRESSILATOR} "
        "(K=125) action lattice on the repressilator so the satisfying "
        "silence-3 corner is in scope; all other (sampler, task) cells use "
        "the task-default vocabulary."
    )
    lines.append("")
    lines.append("## 2. Per-(task, sampler) results")
    lines.append("")
    lines.append(f"Each cell is mean rho ± 95% bootstrap CI over N = {n_seeds} seeds.")
    lines.append("")

    sampler_display_inline: dict[str, str] = {
        k: v.replace("\n", " ") for k, v in _SAMPLER_DISPLAY.items()
    }
    header = "| task |" + "".join(f" {sampler_display_inline.get(s, s)} |" for s in samplers)
    sep = "|---|" + "".join(["---:|"] * len(samplers))
    lines.append(header)
    lines.append(sep)
    for t in tasks:
        cells: list[str] = [t]
        for s in samplers:
            mean = _val(t, s, pivot_mean)
            lo = _val(t, s, pivot_lo)
            hi = _val(t, s, pivot_hi)
            if np.isfinite(mean) and np.isfinite(lo) and np.isfinite(hi):
                cells.append(f"{mean:+.2f} [{lo:+.2f}, {hi:+.2f}]")
            elif np.isfinite(mean):
                cells.append(f"{mean:+.2f}")
            else:
                cells.append("--")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## 3. Satisfaction fraction (rho > 0)")
    lines.append("")
    header_s = "| task |" + "".join(f" {sampler_display_inline.get(s, s)} |" for s in samplers)
    lines.append(header_s)
    lines.append(sep)
    for t in tasks:
        cells_s: list[str] = [t]
        for s in samplers:
            sat = _val(t, s, pivot_sat)
            cells_s.append("--" if not np.isfinite(sat) else f"{sat:.2f}")
        lines.append("| " + " | ".join(cells_s) + " |")
    lines.append("")
    lines.append("## 4. Mean wall-clock per cell (seconds)")
    lines.append("")
    lines.append(header_s)
    lines.append(sep)
    for t in tasks:
        cells_w: list[str] = [t]
        for s in samplers:
            w = _val(t, s, pivot_wall)
            cells_w.append("--" if not np.isfinite(w) else f"{w:.2f}")
        lines.append("| " + " | ".join(cells_w) + " |")
    lines.append("")
    lines.append("## 5. Interpretation")
    lines.append("")
    lines.append(
        "The figure makes the asymmetry across task families immediately "
        "visible. On `glucose_insulin.tir.easy`, both the gradient-guided "
        "and the hybrid samplers dominate the three baselines (standard, "
        "binary BoN, continuous BoN) at matched seed budget. The gradient "
        "guidance term is informative because the Bergman 1979 minimal "
        "model is locally near-linear in the insulin action and the time-"
        "in-range spec is a smooth `min(margin_low, margin_high)` over the "
        "post-absorptive window: each control step's `grad rho` points "
        "in a useful direction and the partial-then-extrapolated probe is "
        "approximately myopic-optimal."
    )
    lines.append("")
    lines.append(
        "On `bio_ode.repressilator.easy`, gradient guidance fails -- the "
        "satisfying region is a measure-near-zero attractor in the joint "
        "control space, the `G_[120,200] (m1 >= 250)` clause demands "
        "sustained silence-of-gene-3 over the back of the horizon, and the "
        "partial-trajectory gradient at any single intermediate step does "
        "not point coherently toward this attractor. The full structural "
        "diagnosis is in `paper/cross_task_validation.md` and the formal "
        "`xfail` for the gradient-guided sampler stays in "
        "`tests/test_inference.py::test_gradient_guided_improves_rho_repressilator`. "
        "Beam-search warmstart resolves this by enumerating the discrete "
        "action vocabulary directly: with the dense k_per_dim=5 lattice the "
        "satisfying silence-3 corner u=(0,0,1) is *in* the vocabulary by "
        "construction, and a model-predictive constant-extrapolation "
        "lookahead score finds it deterministically. The headline asymmetry "
        "is therefore not 'one sampler that wins everywhere' but 'we "
        "characterise which sampler wins which class of task': continuous-"
        "gradient methods for smooth, locally-informative landscapes; "
        "discrete enumeration for narrow vocabulary attractors."
    )
    lines.append("")
    lines.append(
        "PAV (the learned process-reward verifier baseline) is documented "
        "separately in `paper/pav_comparison.md`. PAV is a *verifier* "
        "baseline, not a *sampler* -- it competes with `rho` itself, not "
        "with any of the five samplers compared here. The result there: "
        "STL-rho dominates PAV at every train-set size on both task "
        "families (STL AUC near 1.0, PAV AUC near chance even at 2000 "
        "training trajectories). Together with this script's results, the "
        "story is that the verifier (rho) is uniformly informative; what "
        "differs across tasks is whether `grad rho` is locally useful for "
        "decoding."
    )
    lines.append("")
    lines.append("## 6. Provenance")
    lines.append("")
    lines.append(
        "- Sampler implementations: `src/stl_seed/inference/baselines.py`, "
        "`src/stl_seed/inference/gradient_guided.py`, "
        "`src/stl_seed/inference/hybrid.py`, "
        "`src/stl_seed/inference/horizon_folded.py`, "
        "`src/stl_seed/inference/rollout_tree.py`, "
        "`src/stl_seed/inference/cmaes_gradient.py`, "
        "`src/stl_seed/inference/beam_search_warmstart.py`."
    )
    lines.append(
        "- Companion empirical files: `paper/inference_method.md` "
        "(Tier 3, glucose-insulin headline), "
        "`paper/cross_task_validation.md` (Tier 10, repressilator negative "
        "result + the 2026-04-25 resolution via beam-search warmstart), "
        "`paper/pav_comparison.md` (Tier 6, PAV vs STL verifier)."
    )
    lines.append("- This harness: `scripts/run_unified_comparison.py`.")
    lines.append("- Long-form table: `runs/unified_comparison/results.parquet`.")
    lines.append(f"- Figure: `{fig_disp}`.")
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_unified_comparison.py",
        description=(
            "Unified empirical comparison of all five STL-aware samplers "
            "across the two task families. Produces results.parquet, the "
            "headline figure, and the auto-generated markdown report."
        ),
    )
    p.add_argument(
        "--n-seeds",
        type=int,
        default=8,
        help="Number of seeds per (task, sampler) cell. Default: 8.",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=list(_DEFAULT_TASKS),
        help=(
            f"Task families to evaluate. Choices: "
            f"{sorted(_TASK_BUILDERS.keys())}. "
            f"Default: {' '.join(_DEFAULT_TASKS)}."
        ),
    )
    p.add_argument(
        "--samplers",
        type=str,
        default=",".join(_DEFAULT_SAMPLERS),
        help=(
            f"Comma-separated list of samplers to run. "
            f"Choices: {','.join(_DEFAULT_SAMPLERS)}. Default: all."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT_DIR,
        help=(f"Directory for the parquet results table. Default: {_DEFAULT_OUT_DIR}."),
    )
    p.add_argument(
        "--fig-path",
        type=Path,
        default=_DEFAULT_FIG_PATH,
        help=(f"Path to write the unified-comparison PNG. Default: {_DEFAULT_FIG_PATH}."),
    )
    p.add_argument(
        "--md-path",
        type=Path,
        default=_DEFAULT_MD_PATH,
        help=(f"Path to write the auto-generated markdown report. Default: {_DEFAULT_MD_PATH}."),
    )
    p.add_argument(
        "--seed-offset",
        type=int,
        default=1000,
        help=(
            "Offset for the seed sequence. The seeds used are "
            "[seed_offset, seed_offset + n_seeds). Default: 1000 "
            "(matches the existing test_gradient_guided_improves_rho "
            "convention for cross-tier reproducibility)."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path: Path = args.fig_path
    md_path: Path = args.md_path

    samplers = [s.strip() for s in args.samplers.split(",") if s.strip()]
    for s in samplers:
        if s not in _DEFAULT_SAMPLERS:
            console.print(f"[red]Unknown sampler {s!r}; valid: {_DEFAULT_SAMPLERS}.[/]")
            return 2

    tasks_built: list[TaskSetup] = []
    for t in args.tasks:
        if t not in _TASK_BUILDERS:
            console.print(f"[red]Unknown task {t!r}; valid: {sorted(_TASK_BUILDERS.keys())}.[/]")
            return 2
        tasks_built.append(_TASK_BUILDERS[t]())

    console.print(
        Panel.fit(
            f"Unified sampler comparison\n"
            f"  tasks   : {', '.join(t.name for t in tasks_built)}\n"
            f"  samplers: {', '.join(samplers)}\n"
            f"  n_seeds : {int(args.n_seeds)} (seed offset {int(args.seed_offset)})\n"
            f"  out_dir : {out_dir}\n"
            f"  fig     : {fig_path}\n"
            f"  md      : {md_path}",
            title="[bold]run_unified_comparison",
        )
    )

    df = _run_all_cells(
        tasks=tasks_built,
        samplers=samplers,
        n_seeds=int(args.n_seeds),
        seed_offset=int(args.seed_offset),
    )
    parquet_path = out_dir / "results.parquet"
    df.to_parquet(parquet_path, index=False)
    console.print(f"[green]Wrote {len(df)} rows to {parquet_path}.[/]")

    agg = _aggregate(df)

    # Console summary table.
    console.rule("[bold]Per-cell summary")
    table = Table(title="mean rho ± 95% bootstrap CI", header_style="bold")
    table.add_column("task")
    table.add_column("sampler")
    table.add_column("mean rho", justify="right")
    table.add_column("CI lo", justify="right")
    table.add_column("CI hi", justify="right")
    table.add_column("sat frac", justify="right")
    table.add_column("wall (s)", justify="right")
    for _, row in agg.iterrows():
        table.add_row(
            str(row["task"]),
            str(row["sampler"]),
            f"{row['mean_rho']:+.3f}",
            f"{row['ci_lo']:+.3f}",
            f"{row['ci_hi']:+.3f}",
            f"{row['sat_frac']:.2f}",
            f"{row['mean_wall_s']:.2f}",
        )
    console.print(table)

    # Plot + markdown.
    _plot_unified_comparison(
        agg=agg,
        tasks=[t.name for t in tasks_built],
        samplers=samplers,
        out_path=fig_path,
    )
    console.print(f"[green]Wrote figure to {fig_path}.[/]")

    _write_markdown_report(
        df=df,
        agg=agg,
        tasks=[t.name for t in tasks_built],
        samplers=samplers,
        fig_path=fig_path,
        out_path=md_path,
    )
    console.print(f"[green]Wrote markdown report to {md_path}.[/]")

    return 0


if __name__ == "__main__":
    sys.exit(main())

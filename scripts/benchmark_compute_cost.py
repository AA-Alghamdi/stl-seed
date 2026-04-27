"""Compute-cost vs quality Pareto benchmark across all nine samplers.

This script complements ``scripts/run_unified_comparison.py``. The
unified-comparison harness answers *which sampler reaches what rho*; this
harness answers *at what wall-clock and memory cost*. The output is a
Pareto front in (wall-clock, mean-rho) space, in the spirit of Dettmers'
cost-per-capability framing in the bitsandbytes / k-bit line of work.

Quantities measured per (task, sampler) cell across N seeds
-----------------------------------------------------------

* ``wall_clock_s_cold``    . wall-clock of seed 0 (includes JIT trace).
* ``wall_clock_s_warm``    . mean wall-clock over seeds 1..N-1 (post
  JIT). For samplers without a JIT path, equals the ``cold`` value
  modulo measurement noise.
* ``peak_rss_delta_mb``    . peak resident-set-size delta during the
  warm sample call, measured via ``psutil.Process().memory_info().rss``
  taken before and after with a JAX block_until_ready barrier in between.
  Coarse upper bound on additional working memory; JAX preallocation
  means absolute RSS is dominated by the device pool.
* ``mean_rho``, ``std_rho``. over all N seeds.
* ``sat_frac``             . fraction with ``rho > 0`` (Donzé-Maler).
* ``n_simulator_calls_proxy``. *analytical* count of ODE integrations
  needed by the sampler at its configured hyperparameters. Documented in
  :data:`_SIM_CALL_FORMULA`. This is a structural proxy, not a runtime
  counter (no in-source counter exists on the sampler classes).
* ``time_to_target_s``     . projected wall-clock to first satisfy
  ``mean_rho >= TARGET_RHO``. For samplers whose mean already clears the
  target, equals ``wall_clock_s_warm``; otherwise reported as ``inf``.

Pareto frontier
---------------

A point ``(wall_i, rho_i)`` is Pareto-dominated iff there exists ``j``
with ``wall_j <= wall_i`` AND ``rho_j >= rho_i`` and at least one of the
inequalities is strict. Non-dominated points form the frontier and are
plotted in bold.

Outputs
-------

* ``runs/cost_benchmark/results.parquet``. long-form one-row-per-cell
  table with all measured columns plus the per-seed individual rows
  preserved in ``runs/cost_benchmark/per_seed.parquet`` for downstream
  bootstrap CIs.
* ``paper/figures/compute_cost_pareto.png``. Pareto plot, x = warm
  wall-clock (log), y = mean rho with SEM error bars; frontier in bold.
* ``paper/compute_cost_results.md``. auto-generated technical writeup
  with headline numbers.

-------------

Imports only ``stl_seed.{inference, specs, tasks}``, JAX, NumPy, Pandas,
Matplotlib, psutil, and the standard library. Verified by

Usage
-----
::

    uv run python scripts/benchmark_compute_cost.py
    uv run python scripts/benchmark_compute_cost.py --n-seeds 4
    uv run python scripts/benchmark_compute_cost.py --tasks glucose_insulin
    uv run python scripts/benchmark_compute_cost.py --quick  # 2 seeds, fast
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import os
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import psutil
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
from stl_seed.tasks.cardiac_ap import (
    CARDIAC_ACTION_DIM,
    CardiacAPSimulator,
    FitzHughNagumoParams,
    default_cardiac_initial_state,
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
_DEFAULT_OUT_DIR = _REPO_ROOT / "runs" / "cost_benchmark"
_DEFAULT_FIG_PATH = _REPO_ROOT / "paper" / "figures" / "compute_cost_pareto.png"
_DEFAULT_MD_PATH = _REPO_ROOT / "paper" / "compute_cost_results.md"

# Default tasks: all four task families (matches scripts/run_unified_comparison.py).
# Per-task per-sampler runtimes on M5 Pro (2026-04-25, N=8 seeds, all 9
# samplers) are bounded by hybrid + cmaes_gradient on the bio_ode tasks;
# total wall-clock for the full 4 x 9 x 8 = 288-cell benchmark is ~10 min.
# Use ``--tasks glucose_insulin`` to recover the legacy single-task mode.
#
# The smooth-dynamics vs narrow-attractor axis is what this benchmark
# refines: glucose_insulin and bio_ode.mapk are smooth-dynamics tasks
# where gradient-aware samplers are competitive; bio_ode.repressilator
# and bio_ode.toggle are narrow-attractor tasks where only structural
# enumeration (beam_search_warmstart) reliably reaches the satisfying
# region.
_DEFAULT_TASKS: tuple[str, ...] = (
    "glucose_insulin",
    "bio_ode.repressilator",
    "bio_ode.toggle",
    "bio_ode.mapk",
    "cardiac_ap",
)

# Targets the "near-saturating" rho on glucose_insulin (the legacy single-
# task default, used when --target-rho is not set on the CLI). The
# empirically observed rho ceiling on glucose_insulin.tir.easy is ~20.75
# (rollout_tree, beam_search); 19.0 is well clear of the standard / BoN
# cluster but reachable by the gradient-guided family. Justification for
# the 19.0 threshold: the spec is min(margin_low, margin_high) over a
# 12-step horizon; values above ~19.0 indicate consistent in-band tracking
# with sub-1.0 unit slack at every interior knot, which is the qualitative
# regime "all knots unambiguously satisfy the TIR predicate".
_DEFAULT_TARGET_RHO: float = 19.0

# Per-task target_rho overrides used when --target-rho is not explicitly
# set on the CLI. The rho scales of the four task families differ by
# orders of magnitude (glucose_insulin saturates at ~+20.75; bio_ode.mapk
# saturates at ~+0.0025; bio_ode.repressilator at ~+25 with most samplers
# at -250; bio_ode.toggle at ~+30 with most at -100). The natural
# operationally-interesting threshold on the bio_ode tasks is the Donzé-
# Maler satisfaction threshold rho > 0; the headline target_rho is set to
# 0.0 there so ``time_to_target`` is the wall-clock to first satisfy the
# spec at all. Glucose-insulin keeps the legacy 19.0 because every
# gradient-aware sampler clears 0.0 trivially on that task and the
# discriminator we want is "near the spec ceiling vs unguided floor".
_PER_TASK_TARGET_RHO: dict[str, float] = {
    "glucose_insulin": 19.0,
    "bio_ode.repressilator": 0.0,
    "bio_ode.toggle": 0.0,
    "bio_ode.mapk": 0.0,
    # cardiac_ap on the easy spec saturates rho ~ +1.16 under sustained
    # suprathreshold drive (the spec's Eventually clause has bounded
    # margin once V exceeds the firing threshold V = +1, since the
    # predicate is V - 1 and V_max ~ 2.16). 0.0 is the natural Donzé-
    # Maler satisfaction threshold and matches the bio_ode-task convention.
    "cardiac_ap": 0.0,
}

# Same nine samplers as the unified comparison harness.
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

# Match unified-comparison hyperparameters so the (rho, wall) numbers are
# directly comparable across the two harnesses.
_SAMPLING_TEMPERATURE: float = 0.5
_BON_N: int = 8
_GRADIENT_GUIDANCE_WEIGHT: float = 2.0
_HYBRID_N: int = 4
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
# Beam-search vocabulary density on the multi-dimensional bio_ode action boxes.
# Mirrors scripts/run_unified_comparison.py: k_per_dim=5 on the 3-D
# repressilator (K=125) puts the silence-3 corner u=(0,0,1) in scope, and
# k_per_dim=5 on the 2-D toggle (K=25) puts the silence-B corner u=(0,1) in
# scope while letting the lookahead-rho discriminator score intermediate
# candidates that the coarse k=2 lattice collapses onto a single corner.
# MAPK is 1-D and already uses k_per_dim=5 (K=5) at task default, so no
# beam-search override is needed.
_BEAM_K_PER_DIM_REPRESSILATOR: int = 5
_BEAM_K_PER_DIM_TOGGLE: int = 5

# Console.
console = Console()

# ---------------------------------------------------------------------------
# n_simulator_calls analytical proxy.
# ---------------------------------------------------------------------------
#
# No sampler in src/stl_seed/inference/ exposes a runtime ODE-call
# counter, so we approximate it analytically from the sampler's config.
# The formulas below are *upper bounds* on the number of full-horizon ODE
# integrations equivalent to one sample; each formula is justified inline
# from the sampler's source. ``H`` is the task's control horizon and
# ``K`` is the action vocabulary size at that task.
#
# NB: This is a structural proxy for cross-sampler comparison, not a
# wall-clock surrogate. JIT compilation, batched dispatch, and JAX
# scan-vmap can collapse many "logical" simulator invocations into one
# device kernel, so e.g. horizon_folded reports 100 calls but only ~0.23s
# wall on glucose-insulin because the gradient-step loop is JITed.

_SIM_CALL_FORMULA: dict[str, str] = {
    "standard": "1  -- one autoregressive rollout, one ODE solve",
    "best_of_n": f"{_BON_N}  -- N independent rollouts",
    "continuous_bon": f"{_BON_N}  -- N independent rollouts",
    "gradient_guided": (
        "H + 1  -- one partial-then-extrapolated probe per control step"
        " (autodiff over a single full integration)"
    ),
    "hybrid": (f"{_HYBRID_N} * (H + 1)  -- {_HYBRID_N} gradient-guided draws"),
    "horizon_folded": (
        f"{_HORIZON_FOLDED_K_ITERS}  -- {_HORIZON_FOLDED_K_ITERS} gradient-descent"
        " steps over the joint control vector (one ODE per step)"
    ),
    "rollout_tree": (
        f"H * {_ROLLOUT_TREE_BRANCH_K} + 1  -- per step, branch_k partial"
        f" rollouts of length {_ROLLOUT_TREE_LOOKAHEAD_H}, then one final"
        " full integration"
    ),
    "cmaes_gradient": (
        f"{_CMAES_POPULATION_SIZE} * {_CMAES_N_GENERATIONS} + {_CMAES_N_REFINE}"
        f"  -- pop * gens evaluations + {_CMAES_N_REFINE} gradient refine steps"
    ),
    "beam_search_warmstart": (
        f"H * {_BEAM_SIZE} * K + {_BEAM_GRADIENT_REFINE_ITERS}"
        f"  -- per step, expand {_BEAM_SIZE} beams over K vocab candidates,"
        f" then {_BEAM_GRADIENT_REFINE_ITERS} gradient refine steps"
    ),
}


def _sim_call_proxy(sampler_name: str, horizon: int, k_vocab: int) -> int:
    """Closed-form upper bound on per-sample ODE-integration count.

    See :data:`_SIM_CALL_FORMULA` for the documented formula per sampler.
    """
    H = int(horizon)
    K = int(k_vocab)
    if sampler_name == "standard":
        return 1
    if sampler_name == "best_of_n":
        return _BON_N
    if sampler_name == "continuous_bon":
        return _BON_N
    if sampler_name == "gradient_guided":
        return H + 1
    if sampler_name == "hybrid":
        return _HYBRID_N * (H + 1)
    if sampler_name == "horizon_folded":
        return _HORIZON_FOLDED_K_ITERS
    if sampler_name == "rollout_tree":
        return H * _ROLLOUT_TREE_BRANCH_K + 1
    if sampler_name == "cmaes_gradient":
        return _CMAES_POPULATION_SIZE * _CMAES_N_GENERATIONS + _CMAES_N_REFINE
    if sampler_name == "beam_search_warmstart":
        return H * _BEAM_SIZE * K + _BEAM_GRADIENT_REFINE_ITERS
    raise ValueError(f"Unknown sampler {sampler_name!r}")


# ---------------------------------------------------------------------------
# Synthetic flat-prior LLM (matches unified comparison).
# ---------------------------------------------------------------------------


def _uniform_llm(K: int) -> LLMProposal:
    def llm(state, history, key):
        return jnp.zeros(K, dtype=jnp.float32)

    return llm


def _make_llm(llm_name: str, setup: TaskSetup, vocabulary: Any) -> LLMProposal:
    """Build the LLM proposal for one (task, sampler-vocab) pair.

    Mirrors :func:`scripts.run_unified_comparison._make_llm`. ``uniform``
    returns the synthetic flat-prior proxy; ``qwen3-{0.6b,1.7b,4b}``
    wraps the corresponding mlx-community Qwen3-bf16 checkpoint via
    :class:`MLXLLMProposal` (Apple Silicon only).
    """
    if llm_name == "uniform":
        return _uniform_llm(int(np.asarray(vocabulary).shape[0]))
    if llm_name in {"qwen3-0.6b", "qwen3-1.7b", "qwen3-4b"}:
        from stl_seed.inference.mlx_llm_proposal import MLXLLMProposal

        x0 = np.asarray(setup.initial_state)
        return MLXLLMProposal(
            action_vocabulary=vocabulary,
            spec=setup.spec,
            task=setup.name,
            initial_state=x0,
            horizon=setup.horizon,
            state_dim=int(x0.shape[0]),
            model_id=llm_name,
        )
    raise ValueError(
        f"Unknown LLM backend {llm_name!r}; expected one of "
        "{uniform, qwen3-0.6b, qwen3-1.7b, qwen3-4b}."
    )


# ---------------------------------------------------------------------------
# Task setup.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TaskSetup:
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

    Spec: ``bio_ode.toggle.medium`` (post-2026-04-25 fix; HIGH=100 nM,
    reachable given alpha_1 = 160 saturation cap). The satisfying region
    requires saturating the gene-2 inducer (u = (0, 1) constant drives
    x_1 to ~160 nM, x_2 to ~0). Vocabulary is the 4-corner discretisation
    of [0, 1]^2 for gradient samplers (k_per_dim=2); beam-search uses the
    denser k_per_dim=5 lattice (K=25) via :func:`_vocabulary_for` so the
    satisfying corner is in the lookahead-rho enumeration directly.
    Mirrors ``scripts/run_unified_comparison._bio_ode_toggle_setup``.
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

    Spec: ``bio_ode.mapk.hard`` (post-2026-04-25 fix; reads MAPK_PP in
    absolute microM, peak >= 0.5 microM in [0, 30], settle < 0.05 microM
    in [45, 60], MKKK_P safety < 0.002975 microM throughout). The
    satisfying policy is a brief activating pulse. Action vocabulary is
    the 5-level uniform grid on [0, 1] (k_per_dim=5, K=5); the action box
    is one-dimensional, so the same lattice is used for all samplers
    (no beam-search override). Mirrors
    ``scripts/run_unified_comparison._bio_ode_mapk_setup``.
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


def _cardiac_ap_setup() -> TaskSetup:
    """FitzHugh-Nagumo cardiac action potential task on the easy spec.

    Spec: ``cardiac.depolarize.easy`` (single Eventually clause -- fire at
    least once in [0, 50] dimensionless time units). The 1-D action box
    uses a 5-level uniform grid on [0, 1] for ALL samplers (no beam-
    search override needed on a 1-D action box; mirrors
    bio_ode.mapk). Adds the millisecond-time-scale cardiac task to the
    benchmark, demonstrating the methodology generalises across orders
    of magnitude of physical time-scale and stiffness. Mirrors
    ``scripts/run_unified_comparison._cardiac_ap_setup``.
    """
    sim = CardiacAPSimulator()
    params = FitzHughNagumoParams()
    spec = REGISTRY["cardiac.depolarize.easy"]
    V = make_uniform_action_vocabulary(
        [0.0] * CARDIAC_ACTION_DIM,
        [1.0] * CARDIAC_ACTION_DIM,
        k_per_dim=5,
    )
    x0 = default_cardiac_initial_state(params)
    return TaskSetup(
        name="cardiac_ap",
        spec_key=spec.name,
        simulator=sim,
        params=params,
        spec=spec,
        vocabulary=V,
        initial_state=x0,
        horizon=int(sim.n_control_points),
        aux=None,
    )


_TASK_BUILDERS: dict[str, callable] = {
    "glucose_insulin": _glucose_insulin_setup,
    "bio_ode.repressilator": _bio_ode_repressilator_setup,
    "bio_ode.toggle": _bio_ode_toggle_setup,
    "bio_ode.mapk": _bio_ode_mapk_setup,
    "cardiac_ap": _cardiac_ap_setup,
}


def _vocabulary_for(sampler_name: str, setup: TaskSetup):
    """Per-sampler vocabulary override for beam-search on bio_ode tasks.

    Mirrors ``run_unified_comparison._vocabulary_for`` exactly so the
    cost-benchmark and the unified-comparison harness produce
    apples-to-apples (rho, wall) numbers per cell. The override is:

    * ``beam_search_warmstart`` on the 3-D repressilator -> k_per_dim=5
      (K=125) so the silence-3 corner u=(0,0,1) is in the vocabulary.
    * ``beam_search_warmstart`` on the 2-D toggle -> k_per_dim=5 (K=25)
      so the silence-B corner u=(0,1) is in the vocabulary and the
      lookahead-rho discriminator can score intermediate candidates.
    * MAPK is 1-D and already uses k_per_dim=5 at task default.
    * All other (sampler, task) cells use the task-default vocabulary.
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


# ---------------------------------------------------------------------------
# Sampler factory (mirrors unified comparison).
# ---------------------------------------------------------------------------


def _build_sampler(name: str, setup: TaskSetup, llm_name: str = "uniform") -> Sampler:
    vocabulary = _vocabulary_for(name, setup)
    K = int(vocabulary.shape[0])  # noqa: F841 (clarity; future use)
    llm = _make_llm(llm_name, setup, vocabulary)
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
# Per-cell measurement.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BenchmarkResult:
    """Aggregate result for one (task, sampler) cell."""

    task: str
    sampler: str
    n_seeds: int
    wall_clock_s_cold: float
    wall_clock_s_warm: float
    wall_clock_s_warm_std: float
    peak_rss_delta_mb: float
    mean_rho: float
    std_rho: float
    sem_rho: float
    sat_frac: float
    target_rho: float
    target_hit_frac: float
    n_simulator_calls_proxy: int
    sim_call_formula: str
    time_to_target_s: float


def _process_rss_mb() -> float:
    """Current RSS of this process in MB."""
    return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)


def _block_until_ready(sample_result) -> None:
    """Force JAX device buffers to materialise so timing isn't dispatch-only.

    ``sample_result`` is whatever ``sampler.sample(...)`` returned. We
    walk the (Trajectory, dict) pair and call ``.block_until_ready()``
    on every JAX array we find; non-array entries are ignored. This
    guarantees the warm wall-clock measurement covers actual compute,
    not async device dispatch.
    """
    traj, diag = sample_result
    # Trajectory is a frozen dataclass with .states, .actions, .times,
    # .meta. Only the array fields need a barrier.
    for field in ("states", "actions", "times"):
        x = getattr(traj, field, None)
        if x is not None and hasattr(x, "block_until_ready"):
            x.block_until_ready()
    # Diagnostics may contain stray arrays (e.g. all_rho lists with JAX
    # scalars). We don't deep-walk them. final_rho is a Python float
    # and that's the only field the harness consumes downstream.


def benchmark_sampler(
    sampler_name: str,
    task: TaskSetup,
    n_seeds: int,
    target_rho: float,
    seed_offset: int = 1000,
    verbose: bool = True,
    llm_name: str = "uniform",
) -> tuple[BenchmarkResult, list[dict[str, Any]]]:
    """Measure wall-clock, peak memory, and final-rho for one sampler.

    Runs ``sampler.sample(...)`` ``n_seeds`` times on the same
    ``(task, initial_state)`` cell. The first call (cold) is timed
    separately because it includes the JAX JIT trace; subsequent calls
    (warm) are averaged. Returns the aggregate :class:`BenchmarkResult`
    plus a list of per-seed row dicts for the parquet file.

    Parameters
    ----------
    sampler_name:
        One of :data:`_DEFAULT_SAMPLERS`.
    task:
        Resolved task fixture from :func:`_glucose_insulin_setup` etc.
    n_seeds:
        Number of seeds to run. ``n_seeds >= 2`` gives separate cold /
        warm timings; with ``n_seeds == 1`` only the cold timing exists
        and warm is reported as nan.
    target_rho:
        Quality threshold used to compute ``time_to_target_s``. A seed
        "hits the target" iff its final rho is at least ``target_rho``.
    seed_offset:
        Base for the per-seed PRNG keys; matches the unified-comparison
        harness convention (1000) so cross-tier reproducibility holds.
    verbose:
        If True, prints per-seed rho and wall on the rich console.
    """
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1, got {n_seeds}")
    sampler = _build_sampler(sampler_name, task, llm_name=llm_name)

    # Warm-up GC + RSS baseline before measurement.
    gc.collect()
    rss_before = _process_rss_mb()
    rss_peak = rss_before

    per_seed_rows: list[dict[str, Any]] = []
    wall_clocks: list[float] = []
    rhos: list[float] = []
    sats: list[bool] = []

    for s in range(n_seeds):
        seed = int(seed_offset + s)
        key = jax.random.key(seed)
        # Tight measurement window: barrier on inputs, time, barrier on
        # outputs. The JIT trace happens inside the first .sample() call.
        gc.collect()
        rss_pre = _process_rss_mb()
        t0 = time.perf_counter()
        result = sampler.sample(task.initial_state, key)
        _block_until_ready(result)
        t1 = time.perf_counter()
        wall = float(t1 - t0)
        traj, diag = result
        rho = float(diag["final_rho"])
        sat = bool(np.isfinite(rho) and rho > 0.0)
        rss_post = _process_rss_mb()
        rss_peak = max(rss_peak, rss_post)
        wall_clocks.append(wall)
        rhos.append(rho)
        sats.append(sat)
        per_seed_rows.append(
            {
                "task": task.name,
                "spec_key": task.spec_key,
                "sampler": sampler_name,
                "seed": seed,
                "seed_index": s,
                "is_cold": s == 0,
                "wall_clock_s": wall,
                "rss_pre_mb": rss_pre,
                "rss_post_mb": rss_post,
                "rss_delta_mb": float(rss_post - rss_pre),
                "final_rho": rho,
                "satisfied": sat,
                "hits_target": bool(np.isfinite(rho) and rho >= target_rho),
            }
        )
        if verbose:
            tag = "cold" if s == 0 else "warm"
            console.print(
                f"  [{sampler_name:>22s}] seed={seed} ({tag}) "
                f"rho={rho:+.3f} wall={wall:.3f}s rss_delta={rss_post - rss_pre:+.1f}MB"
            )

    cold = wall_clocks[0]
    if n_seeds >= 2:
        warm_arr = np.asarray(wall_clocks[1:], dtype=np.float64)
        warm = float(np.mean(warm_arr))
        warm_std = float(np.std(warm_arr, ddof=1)) if warm_arr.size >= 2 else 0.0
    else:
        warm = float("nan")
        warm_std = float("nan")

    rho_arr = np.asarray(rhos, dtype=np.float64)
    finite_mask = np.isfinite(rho_arr)
    if finite_mask.any():
        mean_rho = float(np.mean(rho_arr[finite_mask]))
        std_rho = float(np.std(rho_arr[finite_mask], ddof=1)) if finite_mask.sum() >= 2 else 0.0
        sem_rho = std_rho / float(np.sqrt(finite_mask.sum())) if finite_mask.sum() >= 2 else 0.0
    else:
        mean_rho = float("nan")
        std_rho = float("nan")
        sem_rho = float("nan")

    target_hit_frac = float(
        np.mean([1.0 if (np.isfinite(r) and r >= target_rho) else 0.0 for r in rhos])
    )
    sat_frac = float(np.mean([1.0 if s_ else 0.0 for s_ in sats]))

    sim_calls = _sim_call_proxy(
        sampler_name,
        horizon=task.horizon,
        k_vocab=int(_vocabulary_for(sampler_name, task).shape[0]),
    )

    # time_to_target: warm wall-clock per sample, divided by hit fraction
    # (geometric expectation of attempts to first success). Reported as
    # inf if the cell never hits the target.
    if target_hit_frac > 0.0 and np.isfinite(warm):
        time_to_target = warm / target_hit_frac
    elif target_hit_frac > 0.0:
        time_to_target = cold / target_hit_frac
    else:
        time_to_target = float("inf")

    result_obj = BenchmarkResult(
        task=task.name,
        sampler=sampler_name,
        n_seeds=int(n_seeds),
        wall_clock_s_cold=cold,
        wall_clock_s_warm=warm,
        wall_clock_s_warm_std=warm_std,
        peak_rss_delta_mb=float(rss_peak - rss_before),
        mean_rho=mean_rho,
        std_rho=std_rho,
        sem_rho=sem_rho,
        sat_frac=sat_frac,
        target_rho=float(target_rho),
        target_hit_frac=target_hit_frac,
        n_simulator_calls_proxy=sim_calls,
        sim_call_formula=_SIM_CALL_FORMULA[sampler_name],
        time_to_target_s=time_to_target,
    )
    return result_obj, per_seed_rows


# ---------------------------------------------------------------------------
# Pareto frontier.
# ---------------------------------------------------------------------------


def _pareto_front(points: list[tuple[float, float, str]]) -> set[str]:
    """Return labels of Pareto-optimal points in (x_cost, y_quality) space.

    A point ``(x_i, y_i)`` is dominated iff some ``(x_j, y_j)`` has
    ``x_j <= x_i`` AND ``y_j >= y_i`` with at least one strict
    inequality. Non-dominated points are on the frontier.

    Both axes are treated as floats; non-finite x or y collapses the
    point off the frontier (we can't compare an inf-cost or nan-quality
    sampler against finite ones).
    """
    on_front: set[str] = set()
    for i, (xi, yi, label_i) in enumerate(points):
        if not (np.isfinite(xi) and np.isfinite(yi)):
            continue
        dominated = False
        for j, (xj, yj, _) in enumerate(points):
            if i == j:
                continue
            if not (np.isfinite(xj) and np.isfinite(yj)):
                continue
            if xj <= xi and yj >= yi and (xj < xi or yj > yi):
                dominated = True
                break
        if not dominated:
            on_front.add(label_i)
    return on_front


# ---------------------------------------------------------------------------
# Plotting.
# ---------------------------------------------------------------------------


_SAMPLER_DISPLAY: dict[str, str] = {
    "standard": "Standard",
    "best_of_n": f"Binary BoN (N={_BON_N})",
    "continuous_bon": f"Continuous BoN (N={_BON_N})",
    "gradient_guided": f"Gradient-Guided (λ={_GRADIENT_GUIDANCE_WEIGHT:g})",
    "hybrid": f"Hybrid GBoN (n={_HYBRID_N})",
    "horizon_folded": f"Horizon-Folded (K={_HORIZON_FOLDED_K_ITERS})",
    "rollout_tree": f"Rollout-Tree (B={_ROLLOUT_TREE_BRANCH_K},L={_ROLLOUT_TREE_LOOKAHEAD_H})",
    "cmaes_gradient": f"CMA-ES + Grad (pop={_CMAES_POPULATION_SIZE})",
    "beam_search_warmstart": f"Beam Warmstart (B={_BEAM_SIZE})",
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


def _grid_shape(n_tasks: int) -> tuple[int, int]:
    """Choose a (nrows, ncols) grid for ``n_tasks`` panels.

    Layout policy:
    * 1 task -> 1x1 (single panel, legacy default).
    * 2 tasks -> 1x2 (side-by-side).
    * 3 tasks -> 1x3 (single row).
    * 4 tasks -> 2x2 (the headline 4-task layout).
    * >4 tasks -> ceil(sqrt(n)) x ceil(n / ceil(sqrt(n))).

    The 2x2 grid for 4 tasks is the headline figure described in
    ``paper/compute_cost_results.md``: each panel is one task's compute-
    cost vs quality Pareto, side by side, so the smooth-dynamics vs
    narrow-attractor dominance pattern is immediately readable.
    """
    if n_tasks <= 1:
        return (1, 1)
    if n_tasks == 2:
        return (1, 2)
    if n_tasks == 3:
        return (1, 3)
    if n_tasks == 4:
        return (2, 2)
    if n_tasks == 5:
        # 5 tasks fit cleanly in a 2x3 grid (one panel is hidden via the
        # "Hide any unused axes" loop downstream). 2x3 keeps the panel
        # aspect ratio close to the 4-task headline figure so the cardiac
        # addition does not visually distort the existing layout.
        return (2, 3)
    import math

    ncols = int(math.ceil(math.sqrt(n_tasks)))
    nrows = int(math.ceil(n_tasks / ncols))
    return (nrows, ncols)


def _plot_pareto(
    results: list[BenchmarkResult],
    target_rho: float | None,
    out_path: Path,
    target_rho_per_task: dict[str, float] | None = None,
) -> None:
    """Pareto plot: x = warm wall-clock (log), y = mean rho.

    One labeled point per sampler. Vertical error bars are standard error
    of the mean over seeds; horizontal error bars are warm-wall std.
    Pareto frontier is connected by a bold black dashed line; off-frontier
    points are translucent.

    Layout adapts to the number of tasks via :func:`_grid_shape`. The
    headline 4-task case produces a 2x2 grid; single-task invocations
    produce the legacy single-panel figure.

    The horizontal target-rho dotted line is drawn per panel using
    ``target_rho_per_task[task]`` if available, else the global
    ``target_rho`` argument.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Order panels: glucose_insulin first (legacy headline cell), then
    # bio_ode tasks alphabetised so the layout is deterministic.
    tasks = sorted({r.task for r in results}, key=lambda t: (t != "glucose_insulin", t))
    n_tasks = len(tasks)
    nrows, ncols = _grid_shape(n_tasks)

    panel_w = 7.0
    panel_h = 5.5
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_w * ncols, panel_h * nrows),
        squeeze=False,
    )

    for idx, task in enumerate(tasks):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        cell = [r for r in results if r.task == task]
        # Per-panel target rho: prefer the per-task override, fall back
        # to the global default. Cells in ``results`` already carry the
        # ``target_rho`` field used for ``time_to_target`` so we read it
        # straight off the first cell to keep the figure-vs-table
        # consistency.
        if target_rho_per_task is not None and task in target_rho_per_task:
            panel_target = float(target_rho_per_task[task])
        elif cell and np.isfinite(cell[0].target_rho):
            panel_target = float(cell[0].target_rho)
        else:
            panel_target = float(target_rho) if target_rho is not None else 0.0
        # Build (x, y, label) tuples for Pareto computation. We use warm
        # wall-clock as the cost axis when available (n_seeds >= 2),
        # falling back to cold for a single-seed run.
        points: list[tuple[float, float, str]] = []
        for r in cell:
            x = r.wall_clock_s_warm if np.isfinite(r.wall_clock_s_warm) else r.wall_clock_s_cold
            points.append((float(x), float(r.mean_rho), r.sampler))
        front = _pareto_front(points)

        # Sort frontier points by x for the connecting line.
        front_pts = sorted(
            [(x, y, lbl) for (x, y, lbl) in points if lbl in front],
            key=lambda p: p[0],
        )
        if front_pts:
            xs = [p[0] for p in front_pts]
            ys = [p[1] for p in front_pts]
            ax.plot(
                xs,
                ys,
                color="black",
                linewidth=2.0,
                linestyle="--",
                alpha=0.85,
                zorder=1,
                label="Pareto frontier",
            )

        # Label-placement strategy: a substantial fraction of the
        # interesting samplers cluster near the spec ceiling (rho ~
        # 20.7 on glucose), so a default upper-right nudge causes
        # heavy overlap. We compute a per-point offset based on the
        # sampler name to spread labels across the four cardinal
        # directions; this is deterministic so the figure is
        # reproducible across runs.
        # Bespoke per-sampler nudges. Most ceiling-rho samplers cluster
        # tightly in (x, y); we spread their labels into distinct text
        # regions (above, below, far-right) and rely on the marker
        # markeredgecolor to keep dots visually separable.
        label_offsets: dict[str, tuple[int, int]] = {
            "standard": (10, -12),
            "best_of_n": (8, -14),
            "continuous_bon": (8, 10),
            "gradient_guided": (8, -16),
            "hybrid": (-8, 10),
            "horizon_folded": (10, -10),
            "rollout_tree": (-12, 14),
            "cmaes_gradient": (-8, -16),
            "beam_search_warmstart": (8, 12),
        }
        ha_overrides: dict[str, str] = {
            "rollout_tree": "right",
            "hybrid": "right",
            "cmaes_gradient": "right",
        }

        for r, (x, y, lbl) in zip(cell, points, strict=True):
            color = _SAMPLER_COLORS.get(lbl, "#444444")
            on_front = lbl in front
            ax.errorbar(
                x,
                y,
                yerr=(r.sem_rho if np.isfinite(r.sem_rho) else 0.0),
                xerr=(r.wall_clock_s_warm_std if np.isfinite(r.wall_clock_s_warm_std) else 0.0),
                fmt="o",
                markersize=12 if on_front else 9,
                color=color,
                ecolor=color,
                alpha=1.0 if on_front else 0.55,
                markeredgecolor="black",
                markeredgewidth=1.2 if on_front else 0.5,
                capsize=3,
                zorder=3 if on_front else 2,
            )
            offset = label_offsets.get(lbl, (8, 6))
            ax.annotate(
                _SAMPLER_DISPLAY.get(lbl, lbl),
                xy=(x, y),
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
                fontweight="bold" if on_front else "normal",
                color=color,
                ha=ha_overrides.get(lbl, "left"),
            )

        ax.axhline(
            panel_target,
            color="grey",
            linewidth=0.8,
            linestyle=":",
            label=f"Target ρ = {panel_target:g}",
        )
        # On bio_ode tasks the satisfaction threshold (rho > 0) is the
        # operationally interesting line; draw it explicitly when it is
        # not already the panel target so the reader sees both the
        # "any-satisfaction" and the "near-ceiling" reference.
        if abs(panel_target - 0.0) > 1e-9:
            ax.axhline(
                0.0,
                color="black",
                linewidth=0.6,
                linestyle="--",
                alpha=0.4,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Warm wall-clock per sample (s, log scale)")
        if col == 0:
            ax.set_ylabel("Mean final ρ ± SEM")
        ax.set_title(f"{task}\nCompute-cost vs quality Pareto")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.95)

    # Hide any unused axes (e.g. 3 tasks in a 2x2 layout).
    for idx in range(n_tasks, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown writer.
# ---------------------------------------------------------------------------


def _classify_task(task: str) -> str:
    """Return the structural class label for a task family.

    The cost-benchmark refines the headline asymmetry on the
    smooth-dynamics vs narrow-attractor axis. Glucose-insulin and the
    MAPK cascade have smooth, locally-informative dynamics; the
    repressilator and toggle have measure-near-zero satisfying
    attractors in vocabulary space.
    """
    if task in ("glucose_insulin", "bio_ode.mapk", "cardiac_ap"):
        return "smooth-dynamics"
    if task in ("bio_ode.repressilator", "bio_ode.toggle"):
        return "narrow-attractor"
    return "unclassified"


def _useful_pareto_winner(
    cell: dict[str, BenchmarkResult],
    front: set[str],
    target: float,
) -> BenchmarkResult | None:
    """Cheapest Pareto-frontier sampler that meets the target rho.

    The cheapest frontier point is often a dominated-but-cheap baseline
    (e.g. Standard at 0.01s with rho = +0.16) that is technically Pareto
    but useless. The "useful winner" is the cheapest frontier sampler
    whose mean rho clears ``target``; if no frontier sampler does, falls
    back to the highest-rho frontier point (closest to useful).
    """

    def _wall(r: BenchmarkResult) -> float:
        return r.wall_clock_s_warm if np.isfinite(r.wall_clock_s_warm) else r.wall_clock_s_cold

    front_cell = [cell[s] for s in front if s in cell]
    if not front_cell:
        return None
    useful = [r for r in front_cell if r.mean_rho >= target]
    if useful:
        return min(useful, key=lambda r: _wall(r) if np.isfinite(_wall(r)) else float("inf"))
    return max(front_cell, key=lambda r: r.mean_rho)


def _write_markdown_report(
    results: list[BenchmarkResult],
    target_rho_per_task: dict[str, float],
    fig_path: Path,
    out_path: Path,
    n_seeds: int,
    total_wall_clock_s: float | None = None,
) -> None:
    """Auto-generate the technical writeup at ``out_path``.

    The 4-task version of the report leads with a per-task headline
    (Pareto winner, cost, mean rho, satisfaction fraction), surfaces the
    smooth-dynamics vs narrow-attractor split that this benchmark
    refines, and then drops into per-task tables and frontier listings.
    Single-task invocations still produce a coherent (if shorter) report
    because every section iterates over whatever tasks are present.
    """
    tasks = sorted({r.task for r in results}, key=lambda t: (t != "glucose_insulin", t))

    # Compute frontier per task.
    fronts: dict[str, set[str]] = {}
    cells: dict[str, dict[str, BenchmarkResult]] = {}
    for task in tasks:
        task_results = [r for r in results if r.task == task]
        cells[task] = {r.sampler: r for r in task_results}
        pts = [
            (
                (
                    float(r.wall_clock_s_warm)
                    if np.isfinite(r.wall_clock_s_warm)
                    else float(r.wall_clock_s_cold)
                ),
                float(r.mean_rho),
                r.sampler,
            )
            for r in task_results
        ]
        fronts[task] = _pareto_front(pts)

    def _wall(r: BenchmarkResult | None) -> float:
        if r is None:
            return float("nan")
        return r.wall_clock_s_warm if np.isfinite(r.wall_clock_s_warm) else r.wall_clock_s_cold

    def _ratio(num: float, den: float) -> str:
        if not (np.isfinite(num) and np.isfinite(den)) or abs(den) < 1e-9:
            return "n/a"
        return f"{num / den:.1f}x"

    # Per-task winners and ceilings.
    winners: dict[str, BenchmarkResult | None] = {}
    ceilings: dict[str, BenchmarkResult | None] = {}
    for task in tasks:
        cell = cells[task]
        if not cell:
            winners[task] = None
            ceilings[task] = None
            continue
        ordered = sorted(cell.values(), key=lambda r: -r.mean_rho)
        ceilings[task] = ordered[0]
        winners[task] = _useful_pareto_winner(
            cell,
            fronts.get(task, set()),
            float(target_rho_per_task.get(task, 0.0)),
        )

    lines: list[str] = []
    lines.append("# Compute-cost vs quality Pareto (auto-generated)")
    lines.append("")
    try:
        fig_disp = str(fig_path.relative_to(_REPO_ROOT))
    except ValueError:
        fig_disp = str(fig_path)
    lines.append(
        "Generated by `scripts/benchmark_compute_cost.py`. Do not "
        "hand-edit -- changes are clobbered on the next run. The "
        "long-form per-cell data is at "
        "`runs/cost_benchmark/results.parquet`; the per-seed table is at "
        "`runs/cost_benchmark/per_seed.parquet`; the figure is at "
        f"`{fig_disp}`."
    )
    lines.append("")
    lines.append("## 1. Headline")
    lines.append("")
    # Build the headline narrative *from the data* so it cannot drift
    # out of sync with the per-task table on rerun. We summarise:
    #   (a) which samplers are the cheapest *useful* frontier point per
    #       task (the "Pareto winners" table just below);
    #   (b) which samplers reach ρ > 0 (Donzé-Maler satisfaction) on
    #       each task -- the more discriminative metric on the
    #       narrow-attractor tasks where most samplers fail entirely.
    sat_per_task: dict[str, list[BenchmarkResult]] = {}
    for task in tasks:
        sat_per_task[task] = [r for r in cells.get(task, {}).values() if r.sat_frac > 0.5]
    winner_names: list[str] = []
    for task in tasks:
        w = winners.get(task)
        if w is not None:
            winner_names.append(f"`{task}` -> {_SAMPLER_DISPLAY.get(w.sampler, w.sampler)}")
    winners_phrase = "; ".join(winner_names) if winner_names else "(no winners identified)"

    lines.append(
        "**The compute-cost Pareto frontier is task-dependent.** Across the "
        f"{len(tasks)} task families benchmarked here, no single sampler "
        "dominates the frontier on every task; the cheapest Pareto-frontier "
        f"sampler that meets the per-task target ρ is: {winners_phrase}. "
        "The structural class of the task predicts which sampler wins: on "
        "smooth-dynamics tasks (locally-informative gradients, dense "
        "satisfying regions) the cheap structural-lookahead samplers "
        "(rollout-tree in particular) dominate at sub-second wall-clock. "
        "On narrow-attractor tasks (measure-near-zero satisfying regions in "
        "vocabulary space) most samplers fail entirely; beam-search "
        "warmstart is consistently the rho ceiling, but the cheapest "
        "*satisfying* frontier point is sometimes a different sampler whose "
        "joint-control gradient happens to align with the satisfying corner "
        "(e.g. horizon-folded gradient on the toggle reaches ρ > 0 at much "
        "lower wall-clock than beam-search there). The honest takeaway is "
        "that the *Pareto-dominant sampler is task-dependent*; the same "
        "qualitative finding as the binary satisfaction analysis in "
        "`paper/cross_task_validation.md`, now refined with the cost axis."
    )
    lines.append("")
    # Per-task satisfaction summary -- which samplers reach rho > 0,
    # not just which is cheapest. Crucial for the narrow-attractor
    # tasks where most samplers fail by tens to hundreds of rho units.
    lines.append("Samplers reaching ρ > 0 (Donzé-Maler satisfaction) on each task, in cost order:")
    lines.append("")
    for task in tasks:
        sat_samps = sat_per_task.get(task, [])
        if not sat_samps:
            lines.append(f"- `{task}`: none.")
            continue
        ordered = sorted(
            sat_samps,
            key=lambda r: _wall(r) if np.isfinite(_wall(r)) else float("inf"),
        )
        names = ", ".join(
            f"{_SAMPLER_DISPLAY.get(r.sampler, r.sampler)} "
            f"(ρ={r.mean_rho:+.3f}, sat={r.sat_frac:.0%}, wall={_wall(r):.2f}s)"
            for r in ordered
        )
        lines.append(f"- `{task}`: {names}.")
    lines.append("")
    if total_wall_clock_s is not None and np.isfinite(total_wall_clock_s):
        lines.append(
            f"Total benchmark wall-clock: {total_wall_clock_s:.1f}s "
            f"({total_wall_clock_s / 60.0:.1f} min) across "
            f"{len(tasks)} tasks x {len({r.sampler for r in results})} samplers x "
            f"{n_seeds} seeds = {len(tasks) * len({r.sampler for r in results}) * n_seeds} cells."
        )
        lines.append("")
    lines.append("Per-task headline (Pareto winner = cheapest frontier sampler clearing target ρ):")
    lines.append("")
    lines.append(
        "| task | class | target ρ | Pareto winner | warm wall (s) | mean ρ | sat frac | rho ceiling sampler | ceiling ρ |"
    )
    lines.append("|---|---|---:|---|---:|---:|---:|---|---:|")
    for task in tasks:
        winner = winners.get(task)
        ceil = ceilings.get(task)
        cls = _classify_task(task)
        target = float(target_rho_per_task.get(task, 0.0))
        if winner is None or ceil is None:
            lines.append(f"| `{task}` | {cls} | {target:g} | -- | -- | -- | -- | -- | -- |")
            continue
        lines.append(
            f"| `{task}` | {cls} | {target:g} | "
            f"{_SAMPLER_DISPLAY.get(winner.sampler, winner.sampler)} | "
            f"{_wall(winner):.3f} | {winner.mean_rho:+.3f} | "
            f"{winner.sat_frac:.2f} | "
            f"{_SAMPLER_DISPLAY.get(ceil.sampler, ceil.sampler)} | "
            f"{ceil.mean_rho:+.3f} |"
        )
    lines.append("")
    lines.append("Per-task headline bullets, in measured cost-quality terms:")
    lines.append("")
    for task in tasks:
        winner = winners.get(task)
        ceil = ceilings.get(task)
        if winner is None or ceil is None:
            lines.append(f"- `{task}`: insufficient data to identify a Pareto winner.")
            continue
        cls = _classify_task(task)
        cell = cells[task]
        standard = cell.get("standard")
        std_rho_str = f"{standard.mean_rho:+.3f}" if standard is not None else "n/a"
        std_wall_str = f"{_wall(standard):.3f}s" if standard is not None else "n/a"
        target = float(target_rho_per_task.get(task, 0.0))
        if winner.mean_rho >= target:
            useful_word = "reaches the target"
        else:
            useful_word = "is the highest-quality frontier point but does not clear the target"
        if winner.sampler == ceil.sampler:
            lines.append(
                f"- **`{task}`** ({cls}): "
                f"{_SAMPLER_DISPLAY.get(winner.sampler, winner.sampler)} "
                f"is *both* the Pareto winner and the rho ceiling at "
                f"ρ = {winner.mean_rho:+.3f}, {winner.sat_frac:.0%} sat-frac, "
                f"{_wall(winner):.3f}s warm wall-clock vs the unguided "
                f"baseline at ρ = {std_rho_str} in {std_wall_str} "
                f"({_ratio(_wall(winner), _wall(standard)) if standard is not None else 'n/a'} "
                f"the wall-clock for "
                f"{(winner.mean_rho - standard.mean_rho if standard is not None else float('nan')):+.2f} "
                f"ρ improvement)."
            )
        else:
            lines.append(
                f"- **`{task}`** ({cls}): Pareto winner = "
                f"{_SAMPLER_DISPLAY.get(winner.sampler, winner.sampler)} "
                f"at ρ = {winner.mean_rho:+.3f} ({winner.sat_frac:.0%} sat) in "
                f"{_wall(winner):.3f}s ({useful_word}); rho ceiling = "
                f"{_SAMPLER_DISPLAY.get(ceil.sampler, ceil.sampler)} at "
                f"ρ = {ceil.mean_rho:+.3f} in {_wall(ceil):.3f}s "
                f"({_ratio(_wall(ceil), _wall(winner))} the cost for a "
                f"{ceil.mean_rho - winner.mean_rho:+.2f} ρ improvement)."
            )
    lines.append("")
    lines.append("## 2. Methodology")
    lines.append("")
    lines.append(
        "**Hardware:** Apple M5 Pro (local). All measurements include the JIT "
        "trace on the first seed (the `cold` column) and average the "
        "remaining seeds for warm-state cost. JAX device dispatch is forced "
        "to materialise via `block_until_ready()` on the trajectory arrays "
        "before stopping the wall-clock timer."
    )
    lines.append("")
    lines.append(
        f"**Seeds:** N = {n_seeds} per (task, sampler) cell, with key offset "
        "1000 for cross-tier reproducibility against "
        "`scripts/run_unified_comparison.py`. Per-seed rho values are "
        "preserved in `runs/cost_benchmark/per_seed.parquet`."
    )
    lines.append("")
    lines.append(
        "**LLM proxy:** flat-prior LLM (uniform logits over the action "
        "vocabulary), so the only signal driving sampler choices is the "
        "STL verifier or its gradient. This isolates *what each sampler "
        "does with the verifier information*, not how a particular LLM "
        "happens to know the task."
    )
    lines.append("")
    target_rho_phrase = ", ".join(f"`{t}` -> {target_rho_per_task.get(t, 0.0):g}" for t in tasks)
    lines.append(
        f"**Target ρ for `time_to_target` per task:** {target_rho_phrase}. "
        "On `glucose_insulin.tir.easy` the rho ceiling is approximately +20.75 "
        "(rollout-tree, beam-search) and 19.0 is well clear of the standard / "
        "BoN cluster (~+0.16 to +11.5) but reachable by every gradient-aware "
        "sampler in the suite, so 19.0 is the natural near-ceiling discriminator. "
        "On the bio_ode tasks the rho scales differ qualitatively (ceilings of "
        "+25 on repressilator, +30 on toggle, +0.0024 on MAPK; failure floors of "
        "-250 on repressilator, -100 on toggle, -1.17 on MAPK), so the natural "
        "operationally-interesting threshold is the Donzé-Maler satisfaction "
        "boundary ρ > 0 and we use 0.0 as the target. `time_to_target` is the "
        "warm wall-clock per sample divided by the per-seed hit fraction "
        "(geometric expectation of attempts to first success), so a 1.0-second "
        "sampler with 0.5 hit-fraction is reported as 2.0 s, equivalent on "
        "average to a deterministic 2.0-second sampler with hit-fraction 1.0."
    )
    lines.append("")
    lines.append(
        "**Per-task vocabulary overrides:** the beam-search warmstart sampler "
        "uses a denser k_per_dim=5 lattice on the multi-dimensional bio_ode "
        "action boxes (repressilator: K=125; toggle: K=25) so the satisfying "
        "corners are present in the vocabulary by construction. All other "
        "(sampler, task) cells use the task-default vocabulary. This mirrors "
        "the convention in `scripts/run_unified_comparison.py` exactly so the "
        "two harnesses produce apples-to-apples (rho, wall) numbers per cell."
    )
    lines.append("")
    lines.append(
        "**Memory:** `peak_rss_delta_mb` is the peak resident-set-size delta "
        "above the pre-benchmark baseline, measured via "
        "`psutil.Process().memory_info().rss`. This is a coarse upper bound "
        "on additional working memory; absolute RSS is dominated by JAX's "
        "device-pool preallocation and is not informative for cross-sampler "
        "comparison. The delta does capture the per-cell allocation "
        "footprint (e.g. CMA-ES' population matrix, beam-search's beam "
        "buffer)."
    )
    lines.append("")
    lines.append(
        "**`n_simulator_calls_proxy`:** structural upper bound on the number "
        "of full-horizon ODE integrations equivalent to one sample, "
        "computed analytically from the sampler's documented control flow "
        "(see `_SIM_CALL_FORMULA` in the script). This is *not* a runtime "
        "counter -- JIT compilation, batched dispatch, and `scan/vmap` can "
        "collapse many logical solves into one device kernel, so the "
        "wall-clock measurement is the correct cost axis for the Pareto "
        "plot. The proxy is reported because it characterises the sampler's "
        "asymptotic compute scaling independent of JAX's compilation magic."
    )
    lines.append("")
    lines.append("## 3. Per-(task, sampler) results")
    lines.append("")
    lines.append(
        "Each row reports cold wall-clock (1st-seed, includes JIT trace), "
        "warm wall-clock (mean over seeds 1..N-1), peak RSS delta, mean ρ "
        "± std-dev over seeds, satisfaction fraction (ρ > 0), target-hit "
        "fraction (ρ ≥ task-target), the structural simulator-call proxy, and "
        "the projected `time_to_target` (warm wall / hit-fraction)."
    )
    lines.append("")
    for task in tasks:
        target = float(target_rho_per_task.get(task, 0.0))
        lines.append(
            f"### 3.{tasks.index(task) + 1} `{task}` (target ρ = {target:g}, class = {_classify_task(task)})"
        )
        lines.append("")
        header = (
            "| sampler | cold (s) | warm (s) | RSS Δ (MB) | mean ρ | "
            "std ρ | sat | hit | n_sim | t_to_target (s) | Pareto |"
        )
        sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|"
        lines.append(header)
        lines.append(sep)
        ordered = sorted(
            [r for r in results if r.task == task],
            key=lambda r: (
                r.wall_clock_s_warm if np.isfinite(r.wall_clock_s_warm) else r.wall_clock_s_cold
            ),
        )
        for r in ordered:
            on_front = "**Y**" if r.sampler in fronts.get(task, set()) else " "
            warm_disp = (
                f"{r.wall_clock_s_warm:.3f} ± {r.wall_clock_s_warm_std:.3f}"
                if np.isfinite(r.wall_clock_s_warm)
                else "--"
            )
            ttt_disp = f"{r.time_to_target_s:.3f}" if np.isfinite(r.time_to_target_s) else "∞"
            lines.append(
                f"| {r.sampler} | {r.wall_clock_s_cold:.3f} | "
                f"{warm_disp} | {r.peak_rss_delta_mb:+.1f} | "
                f"{r.mean_rho:+.3f} | {r.std_rho:.3f} | "
                f"{r.sat_frac:.2f} | {r.target_hit_frac:.2f} | "
                f"{r.n_simulator_calls_proxy} | {ttt_disp} | {on_front} |"
            )
        lines.append("")
    lines.append("## 4. Pareto frontier per task")
    lines.append("")
    for task in tasks:
        front = fronts.get(task, set())
        if not front:
            lines.append(
                f"- `{task}`: no frontier identified (all points dominated or non-finite)."
            )
            continue
        ordered = sorted(
            [r for r in results if r.task == task and r.sampler in front],
            key=lambda r: (
                r.wall_clock_s_warm if np.isfinite(r.wall_clock_s_warm) else r.wall_clock_s_cold
            ),
        )
        names = ", ".join(
            f"{_SAMPLER_DISPLAY.get(r.sampler, r.sampler)} "
            f"(ρ={r.mean_rho:+.2f}, wall={_wall(r):.2f}s)"
            for r in ordered
        )
        lines.append(f"- `{task}`: {names}.")
    lines.append("")
    lines.append("## 5. Interpretation")
    lines.append("")
    # Smooth-dynamics tasks: gradient-aware structural samplers tend to
    # win. Narrow-attractor tasks: only structural enumeration reaches
    # the satisfying region.
    smooth = [t for t in tasks if _classify_task(t) == "smooth-dynamics"]
    narrow = [t for t in tasks if _classify_task(t) == "narrow-attractor"]
    if smooth:
        smooth_winners = []
        for t in smooth:
            w = winners.get(t)
            if w is not None:
                smooth_winners.append(
                    f"`{t}` -> {_SAMPLER_DISPLAY.get(w.sampler, w.sampler)} "
                    f"(ρ={w.mean_rho:+.2f} at {_wall(w):.2f}s)"
                )
        if smooth_winners:
            lines.append(
                "**Smooth-dynamics tasks** ("
                + ", ".join(f"`{t}`" for t in smooth)
                + "): the Pareto winners are "
                + "; ".join(smooth_winners)
                + ". The locally-informative gradient and dense "
                "satisfying region make structural lookahead samplers (rollout-"
                "tree, gradient-guided variants) immediately effective; the "
                "wall-clock cost of a single backward pass through the "
                "Diffrax-evaluator-spec stack is dominated by the cheaper "
                "branched lookahead in rollout-tree on most cells. Beam-search "
                "is competitive in rho but pays an enumeration cost the smooth-"
                "gradient methods do not have."
            )
            lines.append("")
    if narrow:
        narrow_winners = []
        for t in narrow:
            w = winners.get(t)
            if w is not None:
                narrow_winners.append(
                    f"`{t}` -> {_SAMPLER_DISPLAY.get(w.sampler, w.sampler)} "
                    f"(ρ={w.mean_rho:+.2f} at {_wall(w):.2f}s)"
                )
        if narrow_winners:
            # Compute the actual beam-search costs from the data so the
            # interpretation paragraph cannot lie about them.
            beam_phrases: list[str] = []
            for t in narrow:
                beam = cells.get(t, {}).get("beam_search_warmstart")
                if beam is not None:
                    beam_phrases.append(
                        f"{_wall(beam):.2f}s on `{t}` "
                        f"(ρ={beam.mean_rho:+.2f}, sat={beam.sat_frac:.0%})"
                    )
            beam_cost_str = "; ".join(beam_phrases) if beam_phrases else "(no measurement)"
            lines.append(
                "**Narrow-attractor tasks** ("
                + ", ".join(f"`{t}`" for t in narrow)
                + "): the Pareto winners are "
                + "; ".join(narrow_winners)
                + ". The satisfying region is a measure-near-zero "
                "corner of the action box (silence-3 on the repressilator, "
                "silence-B on the toggle); continuous-gradient methods "
                "trained against the full-horizon rho fail almost entirely. "
                "Beam-search warmstart is the rho ceiling on both tasks "
                "(measured wall: " + beam_cost_str + ") because it "
                "enumerates the satisfying corner directly from a denser "
                "action vocabulary (k_per_dim=5) and scores each candidate "
                "under a model-predictive constant-extrapolation lookahead. "
                "On the toggle the joint-control horizon-folded gradient "
                "*also* finds the satisfying region (with its 100 gradient "
                "steps over the joint control vector, the global rho gradient "
                "is informative enough to silence gene B even though no "
                "myopic per-step probe is); on the repressilator it does "
                "not. The cheapest *satisfying* frontier point on each "
                "narrow-attractor task is therefore the per-task Pareto "
                "winner above, not unconditionally beam-search."
            )
            lines.append("")
    lines.append(
        "The cross-task headline -- 'no single sampler dominates everywhere; "
        "structural class predicts which sampler wins' -- is the same "
        "qualitative claim made by the unified-comparison harness "
        "(`paper/unified_comparison_results.md`), now refined with the cost "
        "axis. A practitioner choosing a sampler for a new control problem "
        "should first ask: does my dynamics give locally-informative "
        "gradients toward a dense satisfying region, or is the satisfying "
        "region a narrow attractor in vocabulary space? The first case "
        "selects rollout-tree / gradient-guided; the second selects beam-"
        "search warmstart with a vocabulary that contains the satisfying "
        "corner."
    )
    lines.append("")
    lines.append(
        "**Connection to Dettmers' framing.** The k-bit / bitsandbytes "
        "literature characterises model performance as a Pareto front in "
        "(memory, accuracy) or (FLOPs, accuracy) space; the same logic "
        "applies to inference-time decoding strategies. The artifact's "
        "headline is therefore not 'one sampler that wins everywhere' but "
        "'characterised cost-per-capability frontier across nine "
        "inference-time strategies, on four task families spanning two "
        "structural classes', enabling a reader to pick the operating point "
        "that matches their compute budget *and* their problem geometry."
    )
    lines.append("")
    lines.append("## 6. Provenance")
    lines.append("")
    lines.append("- Sampler implementations: `src/stl_seed/inference/`.")
    lines.append("- Companion harness (quality-only): `scripts/run_unified_comparison.py`.")
    lines.append("- This harness: `scripts/benchmark_compute_cost.py`.")
    lines.append("- Per-cell aggregate: `runs/cost_benchmark/results.parquet`.")
    lines.append("- Per-seed long-form: `runs/cost_benchmark/per_seed.parquet`.")
    lines.append(f"- Figure: `{fig_disp}`.")
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="benchmark_compute_cost.py",
        description=(
            "Compute-cost vs quality Pareto benchmark across all nine "
            "inference-time samplers. Produces results.parquet, per_seed."
            "parquet, the Pareto figure, and the auto-generated markdown report."
        ),
    )
    p.add_argument(
        "--n-seeds",
        type=int,
        default=8,
        help="Seeds per (task, sampler) cell. Default: 8.",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=list(_DEFAULT_TASKS),
        help=(
            f"Task families to benchmark. Choices: "
            f"{sorted(_TASK_BUILDERS.keys())}. Default: "
            f"{' '.join(_DEFAULT_TASKS)}."
        ),
    )
    p.add_argument(
        "--samplers",
        type=str,
        default=",".join(_DEFAULT_SAMPLERS),
        help=(f"Comma-separated samplers. Choices: {','.join(_DEFAULT_SAMPLERS)}. Default: all."),
    )
    p.add_argument(
        "--target-rho",
        type=float,
        default=None,
        help=(
            "Quality threshold for time-to-target, applied to every task. "
            "If omitted, uses per-task targets (see _PER_TASK_TARGET_RHO): "
            "glucose_insulin -> 19.0 (near-ceiling discriminator); "
            "bio_ode.{repressilator,toggle,mapk} -> 0.0 (Donzé-Maler "
            "satisfaction threshold)."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT_DIR,
        help=(f"Directory for parquet outputs. Default: {_DEFAULT_OUT_DIR}."),
    )
    p.add_argument(
        "--fig-path",
        type=Path,
        default=_DEFAULT_FIG_PATH,
        help=(f"Path to write the Pareto PNG. Default: {_DEFAULT_FIG_PATH}."),
    )
    p.add_argument(
        "--md-path",
        type=Path,
        default=_DEFAULT_MD_PATH,
        help=(f"Path to write the markdown report. Default: {_DEFAULT_MD_PATH}."),
    )
    p.add_argument(
        "--seed-offset",
        type=int,
        default=1000,
        help="Seed offset (matches unified-comparison convention). Default: 1000.",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 2 seeds, single task; for smoke / dev iteration.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-seed verbose logging.",
    )
    p.add_argument(
        "--llm",
        type=str,
        default="uniform",
        choices=["uniform", "qwen3-0.6b", "qwen3-1.7b", "qwen3-4b"],
        help=(
            "LLM proposal backend. 'uniform' uses a flat-prior synthetic "
            "LLM (default; matches the v0.1 cost benchmark); the qwen3-* "
            "options wrap the corresponding mlx-community Qwen3-bf16 "
            "checkpoint via MLXLLMProposal. Apple Silicon only. "
            "Default: uniform."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.quick:
        args.n_seeds = 2
        args.tasks = ["glucose_insulin"]
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

    # Per-task target_rho. If --target-rho is supplied on the CLI it
    # overrides the per-task defaults (used for sweeps where a single
    # threshold is wanted across tasks); otherwise the per-task map
    # supplies natural thresholds for each task family.
    target_rho_per_task: dict[str, float] = {}
    for task in tasks_built:
        if args.target_rho is not None:
            target_rho_per_task[task.name] = float(args.target_rho)
        else:
            target_rho_per_task[task.name] = float(
                _PER_TASK_TARGET_RHO.get(task.name, _DEFAULT_TARGET_RHO)
            )
    target_rho_disp = ", ".join(
        f"{name}={target_rho_per_task[name]:g}" for name in (t.name for t in tasks_built)
    )

    console.print(
        Panel.fit(
            f"Compute-cost vs quality Pareto benchmark\n"
            f"  tasks      : {', '.join(t.name for t in tasks_built)}\n"
            f"  samplers   : {', '.join(samplers)}\n"
            f"  n_seeds    : {int(args.n_seeds)} (offset {int(args.seed_offset)})\n"
            f"  target_rho : {target_rho_disp}\n"
            f"  out_dir    : {out_dir}\n"
            f"  fig        : {fig_path}\n"
            f"  md         : {md_path}",
            title="[bold]benchmark_compute_cost",
        )
    )

    all_results: list[BenchmarkResult] = []
    all_per_seed: list[dict[str, Any]] = []
    t_run = time.perf_counter()
    for task in tasks_built:
        console.rule(f"[bold]task = {task.name}")
        task_target = float(target_rho_per_task[task.name])
        for samp in samplers:
            console.print(f"[bold cyan]>>> sampler = {samp}[/]")
            res, rows = benchmark_sampler(
                sampler_name=samp,
                task=task,
                n_seeds=int(args.n_seeds),
                target_rho=task_target,
                seed_offset=int(args.seed_offset),
                verbose=not args.quiet,
                llm_name=str(args.llm),
            )
            all_results.append(res)
            all_per_seed.extend(rows)
    t_run = time.perf_counter() - t_run
    console.print(f"[green]Total benchmark wall-clock: {t_run:.1f}s.[/]")

    # Persist outputs.
    res_df = pd.DataFrame([dataclasses.asdict(r) for r in all_results])
    per_seed_df = pd.DataFrame(all_per_seed)
    parquet_path = out_dir / "results.parquet"
    per_seed_path = out_dir / "per_seed.parquet"
    res_df.to_parquet(parquet_path, index=False)
    per_seed_df.to_parquet(per_seed_path, index=False)
    console.print(f"[green]Wrote {len(res_df)} cells to {parquet_path}.[/]")
    console.print(f"[green]Wrote {len(per_seed_df)} per-seed rows to {per_seed_path}.[/]")

    # Console summary table.
    console.rule("[bold]Per-cell summary")
    table = Table(title="cost vs quality", header_style="bold")
    table.add_column("task")
    table.add_column("sampler")
    table.add_column("cold (s)", justify="right")
    table.add_column("warm (s)", justify="right")
    table.add_column("RSS Δ MB", justify="right")
    table.add_column("mean ρ", justify="right")
    table.add_column("std ρ", justify="right")
    table.add_column("sat", justify="right")
    table.add_column("hit", justify="right")
    table.add_column("n_sim", justify="right")
    table.add_column("t_to_tgt s", justify="right")
    for r in all_results:
        warm_disp = f"{r.wall_clock_s_warm:.3f}" if np.isfinite(r.wall_clock_s_warm) else "--"
        ttt_disp = f"{r.time_to_target_s:.3f}" if np.isfinite(r.time_to_target_s) else "inf"
        table.add_row(
            r.task,
            r.sampler,
            f"{r.wall_clock_s_cold:.3f}",
            warm_disp,
            f"{r.peak_rss_delta_mb:+.1f}",
            f"{r.mean_rho:+.3f}",
            f"{r.std_rho:.3f}",
            f"{r.sat_frac:.2f}",
            f"{r.target_hit_frac:.2f}",
            str(r.n_simulator_calls_proxy),
            ttt_disp,
        )
    console.print(table)

    _plot_pareto(
        all_results,
        target_rho=None,
        target_rho_per_task=target_rho_per_task,
        out_path=fig_path,
    )
    console.print(f"[green]Wrote figure to {fig_path}.[/]")

    _write_markdown_report(
        results=all_results,
        target_rho_per_task=target_rho_per_task,
        fig_path=fig_path,
        out_path=md_path,
        n_seeds=int(args.n_seeds),
        total_wall_clock_s=float(t_run),
    )
    console.print(f"[green]Wrote markdown report to {md_path}.[/]")

    return 0


if __name__ == "__main__":
    sys.exit(main())

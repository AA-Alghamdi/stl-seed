"""Generate the Phase-2 *canonical* trajectory stores (locally on M5 Pro).

Produces the 2,500-trajectory-per-task corpus that the Phase-2 RunPod
sweep (`scripts/run_canonical_sweep.py`) consumes. Generation runs on the
free local M5 Pro under JAX, NOT on the paid RunPod GPU — the simulators
are CPU-bound at this scale and pre-generating saves ~30 min of GPU time
per sweep.

Differences vs. `scripts/generate_pilot.py` (Phase 1):

* **Per-task buffer:** 2,500 (vs. 2,000 pilot). The 25% buffer above the
  paper-registered 2,000 absorbs the hard-filter dropout when the spec
  satisfaction floor is just above 30% (the validation threshold below).
* **Enriched policy mix (per-task):** for ``bio_ode.repressilator``,
  {random: 0.4, heuristic: 0.4, perturbed_heuristic: 0.2}. The
  "perturbed_heuristic" leg adds small Gaussian noise (σ = 0.1 of the
  action range) on top of the heuristic controller, boosting SFT
  example diversity without breaking structural correctness — the
  perturbed actions stay inside the simulator's declared action box
  (the wrapper clips to bounds), so the heuristic's spec-satisfaction
  property degrades gracefully rather than collapsing. For
  ``glucose_insulin``, the pilot mix {random: 0.5, heuristic: 0.5} is
  retained because empirical generation found that the perturbed leg
  on glucose_insulin destabilizes the Diffrax integrator (see
  `_POLICY_MIX_BY_TASK` source for the full empirical justification).
* **Output dir:** `data/canonical/<task>/` (per-task subdirs), separate
  from `data/pilot/` so Phase-1 reproducibility is preserved.
* **Validation gates:** NaN-rate < 1% AND positive-ρ fraction >= 30% per
  task family, so the hard filter has at least 750 satisfying
  trajectories per task to feed into SFT.

`stl_seed.{generation,specs,stl,tasks}` plus stdlib + numpy + JAX +
`scripts/generate_pilot.py`.

Usage::

    uv run python scripts/generate_canonical.py 2>&1 | tee scripts/generate_canonical.log
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import jax
import numpy as np
from rich.console import Console
from rich.table import Table

from stl_seed.generation.policies import (
    HeuristicPolicy,
    PerturbedHeuristicPolicy,
    RandomPolicy,
)
from stl_seed.generation.runner import TrajectoryRunner
from stl_seed.generation.store import TrajectoryStore
from stl_seed.specs import REGISTRY
from stl_seed.tasks.bio_ode import (
    MAPKSimulator,
    RepressilatorSimulator,
    ToggleSimulator,
    default_mapk_initial_state,
    default_repressilator_initial_state,
    default_toggle_initial_state,
)
from stl_seed.tasks.bio_ode_params import (
    MAPKParams,
    RepressilatorParams,
    ToggleParams,
)
from stl_seed.tasks.glucose_insulin import (
    U_INSULIN_MAX_U_PER_H,
    U_INSULIN_MIN_U_PER_H,
    BergmanParams,
    GlucoseInsulinSimulator,
    MealSchedule,
    default_normal_subject_initial_state,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_ROOT = _REPO_ROOT / "data" / "canonical"
_SEED = 20260424
_N_PER_TASK = 2500
# Per-task policy mixes. The deliverable spec called for a uniform
# {random: 0.4, heuristic: 0.4, perturbed_heuristic: 0.2} mix at
# σ_frac = 0.1, but empirical generation (see
# scripts/generate_canonical.log on 2026-04-24) found that this mix
# breaks the architecture's 1% NaN-drop budget on both task families,
# for different reasons:
#
# 1. glucose_insulin: the PID heuristic emits actions near the lower
#    bound (0 U/h) most of the time, so additive Gaussian noise
#    produces a *biased* effective insulin signal under clipping. The
#    resulting low-amplitude but persistent insulin injection drives
#    glucose to ~90 mg/dL (lower edge of the integrator's stable
#    regime) and Diffrax then returns NaN on the descending leg. We
#    swept σ_frac ∈ {0.005, 0.01, 0.02, 0.05, 0.1} and observed
#    66–76% per-trajectory drop rate on the perturbed leg in all
#    cases — the instability is *not* primarily σ-dependent. We
#    therefore drop the perturbed_heuristic leg from this family and
#    fall back to the pilot mix {random: 0.5, heuristic: 0.5}, which
#    yields ~0% NaN drops.
#
# 2. bio_ode.repressilator: perturbing the topology-aware controller
#    around its 0/1 switch causes oscillations the Diffrax solver
#    occasionally cannot resolve. Drop rate is monotone in σ_frac:
#    {σ=0.02: 0.4%, σ=0.03: 2.2%, σ=0.05: 5.8%, σ=0.1: 18.8%}. We
#    use σ_frac = 0.02 (the largest value that keeps the corpus-level
#    NaN budget under 1% with random=1000, heuristic=1000,
#    perturbed=500). This deviates from the deliverable's σ = 0.1
#    spec.
#
# Both deviations are generator-side hyperparameter calibrations
# documented as such, NOT spec relaxations: the STL spec thresholds
# (P_LOW, P_HIGH, TIR band) are unchanged from the pilot. The
# glucose_insulin corpus loses the diversity the perturbed leg would
# have added; the random leg already covers the full action box
# uniformly, so SFT example diversity at this scale is dominated by
# the random leg in both families.
_POLICY_MIX_BY_TASK: dict[str, dict[str, float]] = {
    "bio_ode.repressilator": {
        "random": 0.4,
        "heuristic": 0.4,
        "perturbed_heuristic": 0.2,
    },
    # Toggle: random leg DROPPED because uniform random inducer sequences
    # drive the bistable Gardner-Cantor-Collins integrator into a regime
    # where the state goes slightly negative (numerical step error) and
    # the Hill term ``A^n_BA`` with non-integer ``n_BA = 1.5`` then
    # returns NaN. Empirically (2026-04-24, 50-sample probe) the random
    # leg has ~48% NaN-drop rate while the heuristic and perturbed legs
    # at sigma_frac=0.01 have ~1.5% drop rate; the corpus-level NaN
    # budget of 1% cannot be met with any non-trivial random fraction.
    # We therefore drop the random leg entirely and rely on the
    # topology-aware heuristic + its low-sigma perturbed variant for
    # action diversity. Fixing the underlying NaN propagation requires
    # a `jnp.maximum(state, 0)` clip in `bio_ode._toggle_vector_field`;
    # that is a separate work item tracked outside this canonical-
    # generator change.
    "bio_ode.toggle": {
        "heuristic": 0.65,
        "perturbed_heuristic": 0.35,
    },
    # MAPK: same enriched mix as repressilator. The MAPK simulator is
    # numerically well-conditioned (no NaN drops observed in the 2026-04-24
    # 2,500-trajectory canonical run); the only failure mode is spec
    # satisfaction (a separate spec/simulator state-index mismatch
    # documented in the controller config in `_HEURISTIC_DEFAULTS`).
    "bio_ode.mapk": {
        "random": 0.4,
        "heuristic": 0.4,
        "perturbed_heuristic": 0.2,
    },
    "glucose_insulin": {
        "random": 0.5,
        "heuristic": 0.5,
    },
}
# σ as a fraction of the action range, applied per-task to
# perturbed_heuristic legs only. The default of 0.1 matches the
# deliverable spec; per-task overrides shrink it where needed (see
# `_POLICY_MIX_BY_TASK` rationale comment above for the empirical
# justification).
_SIGMA_FRAC_BY_TASK: dict[str, float] = {
    "bio_ode.repressilator": 0.02,  # 0.1 spec'd; reduced for solver stability
    # Toggle: reduced to 0.01 because the bistable Hill term ``A^n_BA`` with
    # non-integer n_BA = 1.5 amplifies any small negative excursion into NaN
    # propagation. Empirically (2026-04-24, 200-sample probe):
    #   sigma=0.05 -> 71% NaN; sigma=0.02 -> 13%; sigma=0.01 -> 1.5%.
    "bio_ode.toggle": 0.01,
    "bio_ode.mapk": 0.05,  # MAPK PID; ultrasensitive but bounded
    "glucose_insulin": 0.1,  # unused (perturbed leg not in this family's mix)
}
_SIGMA_FRAC_DEFAULT = 0.1
_NAN_RATE_THRESHOLD = 0.01  # architecture.md pre-registered budget
_SAT_RATE_FLOOR = 0.30  # so the hard filter has >= 750 examples per task

console = Console()


# ---------------------------------------------------------------------------
# Per-task action bounds (used to scale the perturbed-heuristic σ).
# ---------------------------------------------------------------------------


def _action_bounds_for_task(task: str) -> tuple[float, float]:
    """Return ``(action_low, action_high)`` for the canonical action box.

    The bio_ode family's inducer is dimensionless on [0, 1] (see
    `RepressilatorSimulator` docstring); the glucose-insulin family's
    insulin rate is in U/h on
    [U_INSULIN_MIN_U_PER_H, U_INSULIN_MAX_U_PER_H] (the simulator clips
    to this box, see `tasks/glucose_insulin.py:simulate`).
    """
    if task.startswith("bio_ode"):
        return 0.0, 1.0
    if task == "glucose_insulin" or task.startswith("glucose_insulin."):
        return float(U_INSULIN_MIN_U_PER_H), float(U_INSULIN_MAX_U_PER_H)
    raise KeyError(f"no action bounds registered for task={task!r}")


# ---------------------------------------------------------------------------
# Task-family runner factories.
# ---------------------------------------------------------------------------


def _build_repressilator_runner(store: TrajectoryStore) -> TrajectoryRunner:
    """Wire a TrajectoryRunner around the Elowitz-Leibler repressilator."""
    sim = RepressilatorSimulator()
    params = RepressilatorParams()
    return TrajectoryRunner(
        simulator=sim,
        spec_registry={"bio_ode.repressilator.easy": REGISTRY["bio_ode.repressilator.easy"]},
        output_store=store,
        initial_state=default_repressilator_initial_state(params),
        horizon=sim.n_control_points,
        action_dim=sim.action_dim,
        sim_params=params,
    )


def _build_toggle_runner(store: TrajectoryStore) -> TrajectoryRunner:
    """Wire a TrajectoryRunner around the Gardner-Cantor-Collins toggle."""
    sim = ToggleSimulator()
    params = ToggleParams()
    return TrajectoryRunner(
        simulator=sim,
        spec_registry={"bio_ode.toggle.medium": REGISTRY["bio_ode.toggle.medium"]},
        output_store=store,
        initial_state=default_toggle_initial_state(params),
        horizon=sim.n_control_points,
        action_dim=sim.action_dim,
        sim_params=params,
    )


def _build_mapk_runner(store: TrajectoryStore) -> TrajectoryRunner:
    """Wire a TrajectoryRunner around the Huang-Ferrell MAPK cascade."""
    sim = MAPKSimulator()
    params = MAPKParams()
    return TrajectoryRunner(
        simulator=sim,
        spec_registry={"bio_ode.mapk.hard": REGISTRY["bio_ode.mapk.hard"]},
        output_store=store,
        initial_state=default_mapk_initial_state(params),
        horizon=sim.n_control_points,
        action_dim=sim.action_dim,
        sim_params=params,
    )


def _build_glucose_runner(store: TrajectoryStore) -> TrajectoryRunner:
    """Wire a TrajectoryRunner around the Bergman+Dalla-Man glucose-insulin sim."""
    sim = GlucoseInsulinSimulator()
    params = BergmanParams()
    return TrajectoryRunner(
        simulator=sim,
        spec_registry={"glucose_insulin.tir.easy": REGISTRY["glucose_insulin.tir.easy"]},
        output_store=store,
        initial_state=default_normal_subject_initial_state(params),
        horizon=sim.n_control_points,
        action_dim=1,
        aux={"meal_schedule": MealSchedule.empty()},
        sim_params=params,
    )


# ---------------------------------------------------------------------------
# Policy factory builder (adds the new "perturbed_heuristic" leg).
# ---------------------------------------------------------------------------


def _build_policy_factories(task: str, action_dim: int) -> dict[str, Any]:
    """Construct the three-leg policy factory dict for `task`.

    The default factories inside `TrajectoryRunner` only cover
    {random, constant, heuristic}; the canonical generator needs a fourth
    "perturbed_heuristic" leg, so we override the factory registry
    entirely (matching the runner's `policy_factories=` override hook).

    All three factories are *deterministic in the per-trajectory key* and
    delegate per-step PRNG handling to the policy itself (the runner's
    `_rollout_one` folds in the step index). This matches the runner's
    Reproducibility contract (`paper/architecture.md` §"PRNG flow").
    """
    lo, hi = _action_bounds_for_task(task)
    sigma_frac = _SIGMA_FRAC_BY_TASK.get(task, _SIGMA_FRAC_DEFAULT)

    def _random(_key: Any) -> RandomPolicy:
        return RandomPolicy(action_dim=action_dim, action_low=lo, action_high=hi)

    def _heuristic(_key: Any) -> HeuristicPolicy:
        return HeuristicPolicy(task)

    def _perturbed(_key: Any) -> PerturbedHeuristicPolicy:
        # Wrap a fresh heuristic so cache state on the wrapper is
        # task-local and doesn't leak across trajectories.
        base = HeuristicPolicy(task)
        return PerturbedHeuristicPolicy(
            base_policy=base,
            sigma_frac=sigma_frac,
            action_low=lo,
            action_high=hi,
        )

    return {
        "random": _random,
        "heuristic": _heuristic,
        "perturbed_heuristic": _perturbed,
    }


# ---------------------------------------------------------------------------
# Reporting (mirrors generate_pilot.py for cross-script comparability).
# ---------------------------------------------------------------------------


def _existing_per_task(store: TrajectoryStore) -> Counter:
    """Count trajectories already present per task in the store."""
    counts: Counter = Counter()
    for shard in sorted(store.root.glob("trajectories-*.parquet")):
        import pyarrow.parquet as pq

        tbl = pq.read_table(shard, columns=["task"])
        for t in tbl.column("task").to_pylist():
            counts[str(t)] += 1
    return counts


def _rho_histogram(rhos: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Return (counts, edges) for an equal-width histogram of ρ."""
    if rhos.size == 0:
        return np.zeros(n_bins, dtype=np.int64), np.zeros(n_bins + 1)
    edges = np.linspace(rhos.min(), rhos.max(), n_bins + 1)
    counts, _ = np.histogram(rhos, bins=edges)
    return counts, edges


def _print_task_report(
    task: str,
    spec_key: str,
    rhos: np.ndarray,
    policies: list[str],
    n_kept: int,
    n_requested: int,
    n_nan_dropped: int,
    n_failed: int,
    wall_clock: float,
) -> None:
    nan_rate = n_nan_dropped / max(1, n_requested)
    sat_rate = float((rhos > 0).mean()) if rhos.size else 0.0
    policy_split = Counter(policies)
    table = Table(
        title=f"[bold]{task}  ({spec_key})",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("metric")
    table.add_column("value")
    table.add_row("trajectories saved (kept)", f"{n_kept:,}")
    table.add_row("trajectories requested", f"{n_requested:,}")
    table.add_row("nan-dropped", f"{n_nan_dropped:,}")
    table.add_row("solver-failed", f"{n_failed:,}")
    table.add_row(
        "nan-rate (kept fraction failing)",
        f"{nan_rate:.4%}  (budget {_NAN_RATE_THRESHOLD:.0%})",
    )
    table.add_row(
        "policy split (kept)",
        ", ".join(f"{p}={c}" for p, c in sorted(policy_split.items())),
    )
    table.add_row("wall-clock", f"{wall_clock:.1f} s")
    if rhos.size:
        table.add_row("rho min", f"{rhos.min():+.4e}")
        table.add_row("rho max", f"{rhos.max():+.4e}")
        table.add_row("rho mean", f"{rhos.mean():+.4e}")
        table.add_row("rho median", f"{float(np.median(rhos)):+.4e}")
    table.add_row(
        "rho satisfaction (rho>0)",
        f"{sat_rate:.2%}  ({int((rhos > 0).sum()):,} / {rhos.size:,})  "
        f"(floor {_SAT_RATE_FLOOR:.0%})",
    )
    console.print(table)

    if rhos.size:
        counts, edges = _rho_histogram(rhos, n_bins=10)
        hist_table = Table(
            title="[bold]rho histogram (10 bins, equal-width)",
            show_header=True,
            header_style="bold cyan",
        )
        hist_table.add_column("bin")
        hist_table.add_column("range")
        hist_table.add_column("count")
        hist_table.add_column("bar")
        max_count = max(counts.max(), 1)
        for i, c in enumerate(counts):
            bar_len = int(40 * c / max_count)
            hist_table.add_row(
                str(i),
                f"[{edges[i]:+.3e}, {edges[i + 1]:+.3e}]",
                f"{int(c):,}",
                "#" * bar_len,
            )
        console.print(hist_table)
    if nan_rate >= _NAN_RATE_THRESHOLD:
        console.print(
            f"  [red]FAIL[/]: nan-rate {nan_rate:.2%} >= "
            f"pre-registered budget {_NAN_RATE_THRESHOLD:.0%}."
        )
    if sat_rate < _SAT_RATE_FLOOR:
        console.print(
            f"  [red]FAIL[/]: satisfaction {sat_rate:.2%} < "
            f"floor {_SAT_RATE_FLOOR:.0%} — hard filter will starve."
        )


# ---------------------------------------------------------------------------
# Per-task driver.
# ---------------------------------------------------------------------------


def _generate_one_task(
    task: str,
    spec_key: str,
    runner_factory: Any,
    seed: int,
) -> dict[str, Any]:
    """Generate `_N_PER_TASK` trajectories for `task` into per-task subdir.

    Idempotent in the per-task subdir: if the canonical store already
    contains >= `_N_PER_TASK` trajectories for `task`, only the report
    is regenerated (no new shard is written). This matches the
    `generate_pilot.py` resumability contract.
    """
    task_dir = _DATA_ROOT / task
    task_dir.mkdir(parents=True, exist_ok=True)
    store = TrajectoryStore(task_dir)

    policy_mix = _POLICY_MIX_BY_TASK[task]

    existing = _existing_per_task(store).get(task, 0)
    to_generate = max(0, _N_PER_TASK - existing)
    console.rule(f"[bold]{task}")
    console.print(f"  output dir: {task_dir}")
    console.print(f"  existing in store: {existing:,};  to generate: {to_generate:,}")
    console.print(f"  policy mix: {policy_mix}")

    if to_generate == 0:
        # Pull existing rhos / policies for the report from disk.
        import pyarrow.parquet as pq

        rhos: list[float] = []
        policies: list[str] = []
        for shard in sorted(store.root.glob("trajectories-*.parquet")):
            tbl = pq.read_table(shard, columns=["task", "robustness", "policy"])
            for t, r, p in zip(
                tbl.column("task").to_pylist(),
                tbl.column("robustness").to_pylist(),
                tbl.column("policy").to_pylist(),
                strict=True,
            ):
                if t == task:
                    rhos.append(float(r))
                    policies.append(str(p))
        rhos_np = np.asarray(rhos, dtype=np.float64)
        _print_task_report(
            task=task,
            spec_key=spec_key,
            rhos=rhos_np,
            policies=policies,
            n_kept=int(rhos_np.size),
            n_requested=int(rhos_np.size),
            n_nan_dropped=0,
            n_failed=0,
            wall_clock=0.0,
        )
        sat_rate = float((rhos_np > 0).mean()) if rhos_np.size else 0.0
        return {
            "task": task,
            "n_kept": int(rhos_np.size),
            "nan_rate": 0.0,
            "sat_rate": sat_rate,
            "wall_clock_s": 0.0,
        }

    runner = runner_factory(store)
    factories = _build_policy_factories(task, runner.action_dim)
    key = jax.random.key(seed)
    t0 = time.perf_counter()
    _, meta = runner.generate_trajectories(
        task=task,
        n=to_generate,
        policy_mix=policy_mix,
        key=key,
        spec_key=spec_key,
        policy_factories=factories,
    )
    wall = time.perf_counter() - t0
    stats = runner.last_stats
    rhos_np = np.asarray([m["robustness"] for m in meta], dtype=np.float64)
    policies = [m["policy"] for m in meta]
    _print_task_report(
        task=task,
        spec_key=spec_key,
        rhos=rhos_np,
        policies=policies,
        n_kept=stats.n_kept,
        n_requested=stats.n_requested,
        n_nan_dropped=stats.n_nan_dropped,
        n_failed=stats.n_failed,
        wall_clock=wall,
    )
    sat_rate = float((rhos_np > 0).mean()) if rhos_np.size else 0.0
    return {
        "task": task,
        "n_kept": int(stats.n_kept),
        "nan_rate": float(stats.nan_rate),
        "sat_rate": sat_rate,
        "wall_clock_s": float(wall),
    }


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> int:
    console.rule("[bold]Generate Phase-2 canonical trajectory store")
    console.print(f"output root: {_DATA_ROOT}")
    console.print(f"seed: {_SEED}")
    console.print(f"per-task target: {_N_PER_TASK:,}")
    console.print("policy mix (per-task; see module docstring for rationale):")
    for t, mix in _POLICY_MIX_BY_TASK.items():
        console.print(f"  {t}: {mix}")
    console.print(
        f"validation: nan-rate < {_NAN_RATE_THRESHOLD:.0%} AND "
        f"satisfaction >= {_SAT_RATE_FLOOR:.0%} per task"
    )

    summaries: list[dict[str, Any]] = []
    summaries.append(
        _generate_one_task(
            task="bio_ode.repressilator",
            spec_key="bio_ode.repressilator.easy",
            runner_factory=_build_repressilator_runner,
            seed=_SEED + 1,
        )
    )
    summaries.append(
        _generate_one_task(
            task="glucose_insulin",
            spec_key="glucose_insulin.tir.easy",
            runner_factory=_build_glucose_runner,
            seed=_SEED + 2,
        )
    )
    summaries.append(
        _generate_one_task(
            task="bio_ode.toggle",
            spec_key="bio_ode.toggle.medium",
            runner_factory=_build_toggle_runner,
            seed=_SEED + 3,
        )
    )
    summaries.append(
        _generate_one_task(
            task="bio_ode.mapk",
            spec_key="bio_ode.mapk.hard",
            runner_factory=_build_mapk_runner,
            seed=_SEED + 4,
        )
    )

    total_wall = sum(s["wall_clock_s"] for s in summaries)
    console.rule("[bold]Summary")
    console.print(f"total wall-clock (generation only): {total_wall:.1f} s")

    fail = False
    for s in summaries:
        if s["nan_rate"] >= _NAN_RATE_THRESHOLD:
            console.print(
                f"[red]FAIL[/] {s['task']} nan-rate {s['nan_rate']:.2%} "
                f">= {_NAN_RATE_THRESHOLD:.0%}"
            )
            fail = True
        if s["sat_rate"] < _SAT_RATE_FLOOR:
            console.print(
                f"[red]FAIL[/] {s['task']} satisfaction {s['sat_rate']:.2%} < {_SAT_RATE_FLOOR:.0%}"
            )
            fail = True

    if not fail:
        console.print(
            "[green]PASS[/] both task families satisfy the canonical "
            "generation gates (NaN-rate < 1%, satisfaction >= 30%)."
        )
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())

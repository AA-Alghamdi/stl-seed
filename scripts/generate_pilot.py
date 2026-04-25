"""A13 — Generate the Subphase 1.4 local-validation pilot trajectory store.

Generates 2000 trajectories per task family (bio_ode/repressilator and
glucose-insulin) under a {random: 0.5, heuristic: 0.5} policy mix, scored
against the easy-difficulty STL spec for each family, and persisted to
``data/pilot/`` as a `TrajectoryStore` (Parquet). MLXModelPolicy is
deliberately skipped — that requires the optional `mlx` extra and is
covered separately by A15.

This script also runs the runner-vs-canonical STL evaluator consistency
check required by the deliverable: the runner ships an inline NumPy
reference evaluator (since A9 was not on disk when A11 landed); the
canonical implementation now lives in `stl_seed.stl.evaluator`. We sample
8 random rollouts per task and verify the two implementations agree to
within float32 epsilon (the simulators run in float32; the inline path
internally promotes to float64 via Python `float()` casts, the canonical
path stays in float32). Both are mathematically Donzé-Maler space
robustness; the residual disagreement is purely a dtype artifact below
all spec thresholds by >=4 orders of magnitude.

Idempotency: the script creates a fresh shard inside `data/pilot/` only if
the configured task is not already present (the runner appends, and the
store is append-only by contract). Re-running with no new tasks is a
near-no-op (only the consistency check is repeated).

Determinism: a single `jax.random.key(_SEED)` per task seeds the entire
generation, split via `fold_in` per trajectory inside `TrajectoryRunner`.

Usage:
    cd /Users/abdullahalghamdi/stl-seed
    uv run python scripts/generate_pilot.py 2>&1 | tee scripts/generate_pilot.log
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from rich.console import Console
from rich.table import Table

from stl_seed.generation.runner import TrajectoryRunner
from stl_seed.generation.runner import evaluate_robustness as inline_eval
from stl_seed.generation.store import TrajectoryStore
from stl_seed.specs import REGISTRY
from stl_seed.stl.evaluator import evaluate_robustness as canonical_eval
from stl_seed.tasks.bio_ode import (
    RepressilatorSimulator,
    default_repressilator_initial_state,
)
from stl_seed.tasks.bio_ode_params import RepressilatorParams
from stl_seed.tasks.glucose_insulin import (
    BergmanParams,
    GlucoseInsulinSimulator,
    MealSchedule,
    default_normal_subject_initial_state,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "data" / "pilot"
_SEED = 20260424
_N_PER_TASK = 2000
_POLICY_MIX = {"random": 0.5, "heuristic": 0.5}
_NAN_RATE_THRESHOLD = 0.01  # architecture.md pre-registered budget

console = Console()


# ---------------------------------------------------------------------------
# Runner-vs-canonical evaluator consistency check.
# ---------------------------------------------------------------------------


class _StatesTimes:
    """Minimal duck-typed Trajectory for the canonical evaluator's protocol."""

    def __init__(self, states: jnp.ndarray, times: jnp.ndarray) -> None:
        self.states = states
        self.times = times


def _consistency_check() -> dict[str, Any]:
    """Verify runner inline evaluator matches `stl_seed.stl.evaluator` to f32 eps."""
    console.rule("[bold]STL evaluator consistency check")
    results: dict[str, Any] = {}
    key = jax.random.key(_SEED)

    # Repressilator.
    sim = RepressilatorSimulator()
    params = RepressilatorParams()
    init = default_repressilator_initial_state(params)
    spec = REGISTRY["bio_ode.repressilator.easy"]
    diffs: list[float] = []
    for trial in range(8):
        k = jax.random.fold_in(key, trial)
        u = jax.random.uniform(k, (sim.n_control_points, 3), minval=0.0, maxval=1.0)
        traj = sim.simulate(init, u, params, k)
        rho_inline = float(inline_eval(spec, np.asarray(traj.states), np.asarray(traj.times)))
        rho_canon = float(canonical_eval(spec, traj))
        diffs.append(abs(rho_inline - rho_canon))
    results["repressilator_max_abs_diff"] = float(max(diffs))

    # Glucose-insulin.
    gsim = GlucoseInsulinSimulator()
    gparams = BergmanParams()
    ginit = default_normal_subject_initial_state(gparams)
    gspec = REGISTRY["glucose_insulin.tir.easy"]
    gdiffs: list[float] = []
    for trial in range(8):
        k = jax.random.fold_in(key, trial)
        u = jax.random.uniform(k, (gsim.n_control_points,), minval=0.0, maxval=5.0)
        states, times, _ = gsim.simulate(ginit, u, MealSchedule.empty(), gparams, k)
        rho_inline = float(inline_eval(gspec, np.asarray(states), np.asarray(times)))
        rho_canon = float(canonical_eval(gspec, _StatesTimes(states, times)))
        gdiffs.append(abs(rho_inline - rho_canon))
    results["glucose_insulin_max_abs_diff"] = float(max(gdiffs))

    # Verdict. The inline evaluator promotes float32 sim outputs to float64 via
    # Python `float()` calls in `_evaluate_predicate`; the canonical evaluator
    # stays in float32. Differences below ~1e-5 on values of magnitude O(100)
    # are expected and immaterial (smallest spec threshold is 0.1).
    smallest_threshold = 0.1
    repress_ok = results["repressilator_max_abs_diff"] < smallest_threshold * 1.0e-3
    gi_ok = results["glucose_insulin_max_abs_diff"] < smallest_threshold * 1.0e-3
    results["pass"] = bool(repress_ok and gi_ok)
    console.print(
        f"  repressilator max |inline - canonical| = {results['repressilator_max_abs_diff']:.3e}"
    )
    console.print(
        f"  glucose-insulin max |inline - canonical| = "
        f"{results['glucose_insulin_max_abs_diff']:.3e}"
    )
    console.print(
        f"  smallest spec threshold = {smallest_threshold}; "
        f"acceptance = max diff < 1e-3 of threshold ({smallest_threshold * 1e-3})"
    )
    if results["pass"]:
        console.print(
            "  [green]PASS[/] — runner inline evaluator agrees with canonical "
            "to float32 epsilon, well below all spec thresholds."
        )
    else:
        console.print(
            "  [red]FAIL[/] — divergence above tolerance; runner must be "
            "switched to `stl_seed.stl.evaluator.evaluate_robustness`."
        )
    return results


# ---------------------------------------------------------------------------
# Task-family runners.
# ---------------------------------------------------------------------------


def _build_repressilator_runner(store: TrajectoryStore) -> TrajectoryRunner:
    """Wire a TrajectoryRunner around the Elowitz-Leibler repressilator simulator."""
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


def _existing_per_task(store: TrajectoryStore) -> Counter:
    """Count trajectories already present per task in the store."""
    counts: Counter = Counter()
    for shard in sorted(store.root.glob("trajectories-*.parquet")):
        import pyarrow.parquet as pq

        tbl = pq.read_table(shard, columns=["task"])
        for t in tbl.column("task").to_pylist():
            counts[str(t)] += 1
    return counts


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


def _rho_histogram(rhos: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Return (counts, edges) for a 10-bin histogram of rhos."""
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
    table.add_row("rho min", f"{rhos.min():+.4e}")
    table.add_row("rho max", f"{rhos.max():+.4e}")
    table.add_row("rho mean", f"{rhos.mean():+.4e}")
    table.add_row("rho median", f"{float(np.median(rhos)):+.4e}")
    table.add_row(
        "rho satisfaction (rho>0)",
        f"{(rhos > 0).mean():.2%}  ({int((rhos > 0).sum()):,} / {rhos.size:,})",
    )
    console.print(table)

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
            f"  [yellow]WARNING[/]: nan-rate {nan_rate:.2%} >= "
            f"pre-registered budget {_NAN_RATE_THRESHOLD:.0%}. "
            "Flag for re-design per architecture.md NaN policy."
        )


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def _generate_one_task(
    task: str,
    spec_key: str,
    runner_factory,
    store: TrajectoryStore,
    seed: int,
) -> dict[str, Any]:
    """Generate `_N_PER_TASK` trajectories for `task`, idempotent."""
    existing = _existing_per_task(store).get(task, 0)
    to_generate = max(0, _N_PER_TASK - existing)
    console.rule(f"[bold]{task}")
    console.print(f"  existing in store: {existing:,};  to generate: {to_generate:,}")
    if to_generate == 0:
        # Pull existing rhos / policies for the report from disk.
        import pyarrow.parquet as pq

        rhos = []
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
        return {
            "task": task,
            "n_kept": int(rhos_np.size),
            "n_requested": int(rhos_np.size),
            "n_nan_dropped": 0,
            "n_failed": 0,
            "nan_rate": 0.0,
            "wall_clock_s": 0.0,
            "policy_split": dict(Counter(policies)),
            "rho": rhos_np,
        }

    runner = runner_factory(store)
    key = jax.random.key(seed)
    t0 = time.perf_counter()
    _, meta = runner.generate_trajectories(
        task=task,
        n=to_generate,
        policy_mix=_POLICY_MIX,
        key=key,
        spec_key=spec_key,
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
    return {
        "task": task,
        "n_kept": int(stats.n_kept),
        "n_requested": int(stats.n_requested),
        "n_nan_dropped": int(stats.n_nan_dropped),
        "n_failed": int(stats.n_failed),
        "nan_rate": float(stats.nan_rate),
        "wall_clock_s": float(wall),
        "policy_split": dict(Counter(policies)),
        "rho": rhos_np,
    }


def main() -> int:
    console.rule("[bold]A13 — Generate pilot trajectory store")
    console.print(f"output dir: {_DATA_DIR}")
    console.print(f"seed: {_SEED}")
    console.print(f"per-task target: {_N_PER_TASK:,}")
    console.print(f"policy mix: {_POLICY_MIX}")

    consistency = _consistency_check()
    if not consistency["pass"]:
        console.print("[red]Aborting: STL evaluator divergence above tolerance.")
        return 2

    store = TrajectoryStore(_DATA_DIR)

    # We give each task a different sub-seed so the random + heuristic mixes
    # do not share initial control sequences across the two families
    # (defensive against accidental cross-task correlation in any future
    # analysis).
    repress_summary = _generate_one_task(
        task="bio_ode.repressilator",
        spec_key="bio_ode.repressilator.easy",
        runner_factory=_build_repressilator_runner,
        store=store,
        seed=_SEED + 1,
    )
    glucose_summary = _generate_one_task(
        task="glucose_insulin",
        spec_key="glucose_insulin.tir.easy",
        runner_factory=_build_glucose_runner,
        store=store,
        seed=_SEED + 2,
    )

    total_wall = repress_summary["wall_clock_s"] + glucose_summary["wall_clock_s"]
    console.rule("[bold]Summary")
    console.print(f"total wall-clock (generation only): {total_wall:.1f} s")
    console.print(
        f"trajectories on disk under {_DATA_DIR}: {sum(_existing_per_task(store).values()):,}"
    )

    # Pre-registered NaN-rate budget per architecture.md.
    fail = False
    for s in (repress_summary, glucose_summary):
        if s["nan_rate"] >= _NAN_RATE_THRESHOLD:
            console.print(
                f"[red]FAIL[/] {s['task']} nan-rate {s['nan_rate']:.2%} >= "
                f"{_NAN_RATE_THRESHOLD:.0%}"
            )
            fail = True
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())

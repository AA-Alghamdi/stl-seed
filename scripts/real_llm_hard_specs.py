"""Real-LLM hard-spec comparison: does our methodology beat Qwen3 alone?

Why this script exists
----------------------

The 2026-04-25 audit of the unified-comparison harness uncovered a
straw-man baseline: ``Qwen3-0.6B`` standard sampling already saturates
``glucose_insulin.tir.easy`` at rho=+20.747 on every seed. The "+128x
lift over uniform-baseline standard sampling" headline measured the
sampling methodology against a uniform-flat synthetic LLM, not against a
real language-model prior. If a real LLM also saturates the harder
benchmarks, the methodology contribution is moot.

This script runs a *pre-registered, falsification-shaped* head-to-head:

* **Tasks**: four (task, spec) pairs designed to be hard for the LLM:

  1. ``bio_ode.repressilator`` + ``bio_ode.repressilator.easy``
     (narrow vocabulary attractor; documented sampler-failure landscape).
  2. ``bio_ode.toggle`` + ``bio_ode.toggle.medium``
     (narrow attractor on a 2-D action box, harder threshold).
  3. ``bio_ode.mapk`` + ``bio_ode.mapk.hard``
     (a temporal pulse pattern: must spike then settle).
  4. ``cardiac_ap`` + ``cardiac.suppress_after_two.hard``
     (FHN cell on a millisecond time-scale; two spikes then suppress).

* **Samplers**: ``StandardSampler`` (does the LLM solve it on its own?)
  and ``BeamSearchWarmstartSampler`` (does our methodology beat the LLM?).

* **LLM**: ``mlx-community/Qwen3-0.6B-bf16`` via :class:`MLXLLMProposal`
  (chunked-K to fit Metal memory; see the wrapper docstring).

* **Seeds**: three fixed seeds per (task, sampler) cell, totalling 24 runs.

* **Outcome rule** (pre-registered):

  - ``METHODOLOGY MATTERS`` iff beam-search reaches rho>0 on >= 2 of the 4
    tasks where StandardSampler+Qwen3 reaches rho<=0 on a majority of seeds.
  - ``METHODOLOGY DOES NOT MATTER`` if StandardSampler+Qwen3 already
    reaches rho>0 on a majority of seeds across all 4 tasks.
  - ``METHODOLOGY MAYBE MATTERS`` for any in-between outcome.

Pre-registered in this docstring before the runs were executed; the
harness reports whichever outcome is true without softening or retrying.

Outputs
-------

* ``runs/real_llm_hard_specs/results.parquet``. per-(task, sampler, seed)
  row with final rho, satisfaction flag, and wall-clock time.
* ``paper/real_llm_comparison.md``. written by hand by the operator
  (this script prints the table that goes into it).

Usage
-----

::

    uv run python scripts/real_llm_hard_specs.py
    uv run python scripts/real_llm_hard_specs.py --tasks bio_ode.repressilator
    uv run python scripts/real_llm_hard_specs.py --seeds 3000,3001,3002

Wall-clock: ~30-60 minutes on M5 Pro for the default 4 x 2 x 3 = 24 runs.
The repressilator beam-search cell dominates (K=125 vocabulary forces
~125 LLM-score forward passes per control step, x 8-beam x 10 horizon =
~10000 vocabulary scorings per beam call). MAPK and cardiac are fast.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from stl_seed.inference import (
    BeamSearchWarmstartSampler,
    LLMProposal,
    Sampler,
    StandardSampler,
)
from stl_seed.inference.gradient_guided import make_uniform_action_vocabulary
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

# ---------------------------------------------------------------------------
# Paths and defaults.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUT_DIR = _REPO_ROOT / "runs" / "real_llm_hard_specs"

# Three fixed seeds (offset 3000 to avoid collision with the unified
# harness's 1000-block; that block is logged in
# scripts/run_unified_comparison_qwen3_0p6b.log).
_DEFAULT_SEEDS: tuple[int, ...] = (3000, 3001, 3002)

# Sampling temperature: 0.5 matches the unified-comparison harness so
# the head-to-head numbers are commensurable.
_SAMPLING_TEMPERATURE: float = 0.5

# Beam-search hyperparameters: same as the unified-comparison defaults.
_BEAM_SIZE: int = 8
_BEAM_GRADIENT_REFINE_ITERS: int = 30

# Vocabulary density per task, per sampler. Same per-sampler logic as
# scripts/run_unified_comparison.py: the gradient/standard samplers see
# corner-only k_per_dim=2 vocabularies on the multi-D action boxes; beam
# search sees k_per_dim=5 so the satisfying corners are visible to the
# constant-extrapolation lookahead.
_BEAM_K_PER_DIM_REPRESSILATOR: int = 5  # K = 5**3 = 125
_BEAM_K_PER_DIM_TOGGLE: int = 5  # K = 5**2 = 25

# Real LLM backend.
_LLM_NAME: str = "qwen3-0.6b"

console = Console()


# ---------------------------------------------------------------------------
# Task setup (mirrors scripts/run_unified_comparison.py but selects the
# *hard* spec variants).
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TaskSetup:
    """Resolved task fixture; mirrors scripts/run_unified_comparison.TaskSetup."""

    name: str
    spec_key: str
    simulator: Any
    params: Any
    spec: Any
    vocabulary: Any
    initial_state: Any
    horizon: int
    aux: dict[str, Any] | None


def _bio_ode_repressilator_setup() -> TaskSetup:
    """Repressilator on ``bio_ode.repressilator.easy``.

    The "easy" spec is a misnomer relative to the other bio_ode tasks --
    the satisfying region is a single narrow vocabulary corner
    ``u = (0, 0, 1)``, which is *easy* for beam-search to enumerate but
    *hard* for any continuous / random sampler. The unified-comparison
    log shows StandardSampler+Qwen3-0.6B at rho=-247 on this task: the
    LLM does not solve it on its own. This is the canonical test of
    whether methodology rescues real-LLM sampling.
    """
    sim = RepressilatorSimulator()
    params = RepressilatorParams()
    spec = REGISTRY["bio_ode.repressilator.easy"]
    V = make_uniform_action_vocabulary(
        [0.0] * REPRESSILATOR_ACTION_DIM,
        [1.0] * REPRESSILATOR_ACTION_DIM,
        k_per_dim=2,  # standard sampler default; beam-search overrides below
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
    """Toggle switch on ``bio_ode.toggle.medium``.

    Narrow attractor on a 2-D action box: the satisfying region requires
    saturating the gene-2 inducer (constant ``u = (0, 1)``) to drive
    ``x_1 -> 160 nM`` and ``x_2 -> 0``. Threshold tighter than the easy
    variant; selected to test medium-difficulty narrow-attractor regime.
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
    """MAPK cascade on ``bio_ode.mapk.hard`` (pulse-pattern requirement).

    The satisfying policy is a brief activating pulse (u=1 for ~1-3
    control steps then u=0): MAPK_PP must rise above 0.5 microM but
    settle back near zero by t=45. Random-policy success rate ~0 because
    the cascade lacks fast-enough negative feedback to deactivate
    MAPK_PP within the 15-min settle window once activated. 1-D action
    box; same K=5 vocabulary for all samplers.
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


def _cardiac_setup() -> TaskSetup:
    """FHN cardiac AP on ``cardiac.suppress_after_two.hard``.

    Temporal-pattern requirement: two spikes (V > 1) within the first 60
    time units, then sustained suppression (V < 0.5) for [70, 100]. The
    spikes need timed depolarising current; the suppression needs the
    current driven low after the second spike. 1-D action box; same K=5
    vocabulary for all samplers.
    """
    sim = CardiacAPSimulator()
    params = FitzHughNagumoParams()
    spec = REGISTRY["cardiac.suppress_after_two.hard"]
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


_TASK_BUILDERS = {
    "bio_ode.repressilator": _bio_ode_repressilator_setup,
    "bio_ode.toggle": _bio_ode_toggle_setup,
    "bio_ode.mapk": _bio_ode_mapk_setup,
    "cardiac_ap": _cardiac_setup,
}


# ---------------------------------------------------------------------------
# Per-sampler vocabulary override (beam search needs denser lattices).
# ---------------------------------------------------------------------------


def _vocabulary_for(sampler_name: str, setup: TaskSetup):
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
# LLM constructor.
# ---------------------------------------------------------------------------


def _make_llm(setup: TaskSetup, vocabulary: Any) -> LLMProposal:
    """Build an MLXLLMProposal wrapping Qwen3-0.6B for this (task, vocab)."""
    from stl_seed.inference.mlx_llm_proposal import MLXLLMProposal

    x0 = np.asarray(setup.initial_state)
    return MLXLLMProposal(
        action_vocabulary=vocabulary,
        spec=setup.spec,
        task=setup.name,
        initial_state=x0,
        horizon=setup.horizon,
        state_dim=int(x0.shape[0]),
        model_id=_LLM_NAME,
        chunk_size=16,  # required to fit K=125 repressilator vocabulary on Metal
    )


# ---------------------------------------------------------------------------
# Sampler factory.
# ---------------------------------------------------------------------------


def _build_sampler(name: str, setup: TaskSetup) -> Sampler:
    vocabulary = _vocabulary_for(name, setup)
    llm = _make_llm(setup, vocabulary)
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
    if name == "beam_search_warmstart":
        return BeamSearchWarmstartSampler(
            beam_size=_BEAM_SIZE,
            gradient_refine_iters=_BEAM_GRADIENT_REFINE_ITERS,
            tail_strategy="repeat_candidate",
            **common,
        )
    raise ValueError(f"Unknown sampler {name!r}")


# ---------------------------------------------------------------------------
# Per-cell runner.
# ---------------------------------------------------------------------------


def _run_one_cell(
    task: TaskSetup,
    sampler_name: str,
    seed: int,
) -> dict[str, Any]:
    """Run one (task, sampler, seed) cell; return one row of the results table."""
    sampler = _build_sampler(sampler_name, task)
    key = jax.random.key(int(seed))
    t0 = time.time()
    try:
        _, diag = sampler.sample(task.initial_state, key)
        wall = time.time() - t0
        rho = float(diag["final_rho"])
        ok = True
        err = ""
    except Exception as exc:  # noqa: BLE001. surface failure rather than crash sweep
        wall = time.time() - t0
        rho = float("nan")
        ok = False
        err = repr(exc)
        console.print(
            f"[red]ERROR[/red] task={task.name} sampler={sampler_name} seed={seed}: {err}"
        )
    return {
        "task": task.name,
        "spec": task.spec_key,
        "sampler": sampler_name,
        "seed": int(seed),
        "final_rho": rho,
        "satisfied": bool(rho > 0.0) if ok else False,
        "wall_clock_s": float(wall),
        "ok": ok,
        "error": err,
    }


# ---------------------------------------------------------------------------
# Aggregation and verdict.
# ---------------------------------------------------------------------------


def _per_cell_table(df: pd.DataFrame) -> Table:
    """Render the per-(task, sampler) summary table.

    Columns: task, sampler, n_seeds, sat_count (rho > 0), final_rho_mean,
    final_rho_min, final_rho_max, wall_s_mean.
    """
    table = Table(title="Per-cell summary (3 seeds per cell)", show_lines=False)
    table.add_column("task")
    table.add_column("sampler")
    table.add_column("sat / n", justify="right")
    table.add_column("rho_mean", justify="right")
    table.add_column("rho_min", justify="right")
    table.add_column("rho_max", justify="right")
    table.add_column("wall_s_mean", justify="right")
    for (task, sampler), grp in df.groupby(["task", "sampler"], sort=False):
        n = len(grp)
        sat = int(grp["satisfied"].sum())
        rmean = float(grp["final_rho"].mean())
        rmin = float(grp["final_rho"].min())
        rmax = float(grp["final_rho"].max())
        wmean = float(grp["wall_clock_s"].mean())
        table.add_row(
            task,
            sampler,
            f"{sat} / {n}",
            f"{rmean:+.3f}",
            f"{rmin:+.3f}",
            f"{rmax:+.3f}",
            f"{wmean:.1f}",
        )
    return table


def _verdict(df: pd.DataFrame) -> tuple[str, str, dict[str, dict[str, int]]]:
    """Apply the pre-registered outcome rule.

    Returns
    -------
    verdict, explanation, per_task
        - ``verdict`` in {"METHODOLOGY MATTERS", "DOESN'T MATTER", "MIXED"}.
        - ``explanation`` is a one-paragraph natural-language summary of
          which tasks each sampler did/didn't solve.
        - ``per_task`` is a dict ``{task: {"standard_sat": int,
          "beam_sat": int, "n_seeds": int}}`` for downstream reporting.
    """
    per_task: dict[str, dict[str, int]] = {}
    tasks = sorted(df["task"].unique())
    for t in tasks:
        sub = df[df["task"] == t]
        n = int(len(sub) / sub["sampler"].nunique())
        std_sub = sub[sub["sampler"] == "standard"]
        beam_sub = sub[sub["sampler"] == "beam_search_warmstart"]
        per_task[t] = {
            "n_seeds": n,
            "standard_sat": int(std_sub["satisfied"].sum()),
            "beam_sat": int(beam_sub["satisfied"].sum()),
        }
    # Majority threshold: ceil(n/2). For n=3 this is 2.
    n_per_cell = list({v["n_seeds"] for v in per_task.values()})
    assert len(n_per_cell) == 1, "Inconsistent seed counts across tasks"
    n = n_per_cell[0]
    majority = (n + 2) // 2  # ceil(n/2) + 0; e.g. n=3 -> 2

    n_tasks = len(per_task)
    n_std_solves = sum(1 for v in per_task.values() if v["standard_sat"] >= majority)
    n_beam_only = sum(
        1 for v in per_task.values() if v["beam_sat"] >= majority and v["standard_sat"] < majority
    )

    if n_std_solves == n_tasks:
        verdict = "METHODOLOGY DOESN'T MATTER"
        explanation = (
            f"Qwen3-0.6B + StandardSampler reaches rho>0 on >= {majority}/{n} seeds for "
            f"all {n_tasks} hard tasks; beam-search has nothing to add."
        )
    elif n_beam_only >= 2:
        rescued = [
            t
            for t, v in per_task.items()
            if v["beam_sat"] >= majority and v["standard_sat"] < majority
        ]
        verdict = "METHODOLOGY MATTERS"
        explanation = (
            f"Beam-search rescues {n_beam_only}/{n_tasks} tasks where Qwen3-0.6B + "
            f"StandardSampler fails on a majority of seeds. Rescued tasks: {rescued}."
        )
    else:
        rescued = [
            t
            for t, v in per_task.items()
            if v["beam_sat"] >= majority and v["standard_sat"] < majority
        ]
        not_rescued = [
            t
            for t, v in per_task.items()
            if v["beam_sat"] < majority and v["standard_sat"] < majority
        ]
        already_easy = [t for t, v in per_task.items() if v["standard_sat"] >= majority]
        verdict = "METHODOLOGY MAYBE MATTERS"
        explanation = (
            f"Mixed outcome: Qwen3+Standard already solves {already_easy}; "
            f"beam-search additionally rescues {rescued}; neither solves {not_rescued}."
        )
    return verdict, explanation, per_task


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def _parse_csv(s: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in s.split(",") if x.strip())


def _parse_seeds(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 2)[0])
    parser.add_argument(
        "--tasks",
        type=_parse_csv,
        default=tuple(_TASK_BUILDERS.keys()),
        help="Comma-separated task names. Default: all four hard tasks.",
    )
    parser.add_argument(
        "--samplers",
        type=_parse_csv,
        default=("standard", "beam_search_warmstart"),
        help="Comma-separated sampler names. Default: standard,beam_search_warmstart.",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=_DEFAULT_SEEDS,
        help=f"Comma-separated seeds. Default: {','.join(map(str, _DEFAULT_SEEDS))}.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT_DIR,
        help=f"Output directory. Default: {_DEFAULT_OUT_DIR}.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel.fit(
            "Real-LLM hard-spec comparison\n"
            f"  tasks   : {', '.join(args.tasks)}\n"
            f"  samplers: {', '.join(args.samplers)}\n"
            f"  seeds   : {', '.join(map(str, args.seeds))}\n"
            f"  llm     : {_LLM_NAME}\n"
            f"  out_dir : {args.out_dir}",
            title="real_llm_hard_specs",
        )
    )

    setups = {name: _TASK_BUILDERS[name]() for name in args.tasks}

    rows: list[dict[str, Any]] = []
    n_total = len(args.tasks) * len(args.samplers) * len(args.seeds)
    n_done = 0
    for task_name, setup in setups.items():
        for sampler_name in args.samplers:
            for seed in args.seeds:
                n_done += 1
                row = _run_one_cell(setup, sampler_name, int(seed))
                rows.append(row)
                console.print(
                    f"  [{n_done:3d}/{n_total:3d}] task={task_name} "
                    f"sampler={sampler_name} seed={seed} "
                    f"rho={row['final_rho']:+.3f} sat={row['satisfied']} "
                    f"wall={row['wall_clock_s']:.1f}s"
                )

    df = pd.DataFrame(rows)
    parquet_path = args.out_dir / "results.parquet"
    df.to_parquet(parquet_path, index=False)
    console.print(f"Wrote results to {parquet_path}")

    json_path = args.out_dir / "results.jsonl"
    with json_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    console.print(f"Wrote JSON-lines mirror to {json_path}")

    # Print the per-cell summary table.
    table = _per_cell_table(df)
    console.print(table)

    # Apply the pre-registered verdict rule.
    verdict, explanation, per_task = _verdict(df)
    console.print(
        Panel.fit(
            f"VERDICT: {verdict}\n\n{explanation}",
            title="Pre-registered outcome rule",
            border_style="bold yellow",
        )
    )

    # Persist verdict alongside results so the markdown writer can read it.
    (args.out_dir / "verdict.json").write_text(
        json.dumps(
            {"verdict": verdict, "explanation": explanation, "per_task": per_task},
            indent=2,
        )
    )
    console.print(f"Wrote verdict to {args.out_dir / 'verdict.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Quantization x model-size sweep on the four hard specs.

Why this script exists
----------------------

After scripts/real_llm_hard_specs.py established METHODOLOGY MATTERS on
real ``Qwen3-0.6B-bf16`` (4/4 hard tasks rescued by beam-search warmstart
where standard sampling fails on a majority of seeds), two natural
questions remain:

1. **Does the methodology gap survive quantization?** The original
   real-LLM run was bf16. NF4 / 4-bit / 8-bit quantization (the
   ``mlx-community/Qwen3-0.6B-{4bit, 6bit, 8bit}`` variants) shrinks the
   activation precision available to the LLM scoring step in
   ``MLXLLMProposal._score_batch``. Does the rescue still work, or does
   the LLM-prior + quantization-induced noise wash out the verifier-
   derived advantage of beam-search warmstart? This is the most
   Dettmers-coded question we can ask the artifact.

2. **Does the methodology gap survive scaling to 1.7B?** The bigger LLM
   has more entropy in its logits at temperature 0.5 and may saturate
   more of the easy tasks on its own (the SERA-saturation transition
   that the Limitations section flags). If standard sampling at 1.7B
   already solves the easier of the four hard tasks, that is a finding
   in itself: methodology mattering is a small-LLM phenomenon, hardly
   present at 1.7B. If it persists, that is a structural finding.

This script combines both into a single (precision, size) factorial:

   models = {
     Qwen3-0.6B-bf16,    # canonical baseline (Day 1 reproducibility)
     Qwen3-0.6B-8bit,    # mid-quantization
     Qwen3-0.6B-4bit,    # aggressive (NF4-equivalent on MLX)
     Qwen3-1.7B-bf16,    # size scaling
     Qwen3-1.7B-4bit,    # size scaling under aggressive quantization
   }

5 models x 2 samplers (standard, beam_search_warmstart) x 4 hard tasks x
3 fixed seeds = 120 runs total. Wall-clock ~ 60-90 minutes on M5 Pro.

The script reuses the per-cell runner from
``scripts.real_llm_hard_specs`` so the reported metrics (final rho,
satisfied, wall-clock, error) are commensurable with the Day 1 verdict.

Outputs
-------

* ``runs/quant_size_sweep/results.parquet`` -- per (model, task, sampler,
  seed) row.
* ``runs/quant_size_sweep/results.jsonl`` -- mirror.
* ``runs/quant_size_sweep/summary.json`` -- per-model verdict (does the
  Day 1 METHODOLOGY MATTERS rule still fire at this (precision, size)?).
* ``paper/quant_size_results.md`` -- written by hand by the operator,
  using the table this script prints.

Usage
-----

::

    uv run python scripts/quant_size_sweep.py
    uv run python scripts/quant_size_sweep.py --models qwen3-0.6b-4bit,qwen3-1.7b-bf16
    uv run python scripts/quant_size_sweep.py --tasks bio_ode.repressilator

Notes
-----

* The aliases ``qwen3-0.6b-{4bit,6bit,8bit}`` and ``qwen3-1.7b-{4bit,...}``
  are added to ``MLXLLMProposal._MODEL_ALIASES`` in this commit so the
  existing wrapper resolves them. If a model id is not in the alias map,
  it is passed through unchanged (so direct HF ids like
  ``mlx-community/Qwen3-0.6B-4bit`` also work).

* Each (model, task, sampler, seed) cell is run independently; the LLM
  weights are loaded once per model id via ``_MODEL_CACHE`` in
  mlx_llm_proposal.py, but rebuilt across model ids. Memory usage peaks
  during the largest model load (Qwen3-1.7B-bf16 is ~3.4 GB on disk;
  weights live in unified memory while in scope).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import jax
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from scripts import real_llm_hard_specs as _rlh  # noqa: E402

# Reuse the Day 1 cell runner verbatim. We re-import after monkey-patching
# the LLM name so the wrapper picks up the new model id per cell.
from scripts.real_llm_hard_specs import (  # type: ignore[import-not-found]
    _TASK_BUILDERS,
    _build_sampler,
    _verdict,
)

# ---------------------------------------------------------------------------
# Paths and defaults.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUT_DIR = _REPO_ROOT / "runs" / "quant_size_sweep"

#: Reuse the Day 1 seed block (3000, 3001, 3002) so the bf16-0.6B cells
#: in this sweep are a subset of the Day 1 results (sanity check). This
#: does mean three of the bf16-0.6B (model x task) rows are
#: re-computations of Day 1 cells; we keep them for an explicit
#: regression-floor and to confirm the wrapper round-trip is identical.
_DEFAULT_SEEDS: tuple[int, ...] = (3000, 3001, 3002)

#: The five model variants we sweep across. Mix of bf16 and quantized,
#: across two model sizes. Aliases are resolved by MLXLLMProposal.
_DEFAULT_MODELS: tuple[str, ...] = (
    "qwen3-0.6b",  # baseline (bf16) -- Day 1 reproducibility
    "qwen3-0.6b-8bit",  # mid-quantization
    "qwen3-0.6b-4bit",  # aggressive (NF4-equivalent in mlx)
    "qwen3-1.7b",  # size scaling, bf16
    "qwen3-1.7b-4bit",  # size scaling under aggressive quantization
)

console = Console()


# ---------------------------------------------------------------------------
# Per-cell runner: thin wrapper over the Day 1 cell runner that swaps the
# LLM name in.
# ---------------------------------------------------------------------------


def _run_one_cell_with_model(
    task_name: str,
    sampler_name: str,
    seed: int,
    model_id: str,
) -> dict[str, Any]:
    """Run one cell; the LLM is set via monkey-patch on _LLM_NAME."""
    setup = _TASK_BUILDERS[task_name]()
    # Monkey-patch the module-level constant so _make_llm picks it up.
    # This is intentional and isolated to this driver script.
    _rlh._LLM_NAME = model_id
    sampler = _build_sampler(sampler_name, setup)
    key = jax.random.key(int(seed))
    t0 = time.time()
    try:
        _, diag = sampler.sample(setup.initial_state, key)
        wall = time.time() - t0
        rho = float(diag["final_rho"])
        ok = True
        err = ""
    except Exception as exc:  # noqa: BLE001
        wall = time.time() - t0
        rho = float("nan")
        ok = False
        err = repr(exc)
        console.print(
            f"[red]ERROR[/red] model={model_id} task={task_name} "
            f"sampler={sampler_name} seed={seed}: {err}"
        )
    return {
        "model": model_id,
        "task": setup.name,
        "spec": setup.spec_key,
        "sampler": sampler_name,
        "seed": int(seed),
        "final_rho": rho,
        "satisfied": bool(rho > 0.0) if ok else False,
        "wall_clock_s": float(wall),
        "ok": ok,
        "error": err,
    }


# ---------------------------------------------------------------------------
# Aggregation tables.
# ---------------------------------------------------------------------------


def _per_model_table(df: pd.DataFrame) -> Table:
    """Per-(model, task, sampler) summary table, sorted by model then task."""
    table = Table(
        title="Per-(model, task, sampler) summary",
        show_lines=False,
    )
    table.add_column("model")
    table.add_column("task")
    table.add_column("sampler")
    table.add_column("sat / n", justify="right")
    table.add_column("rho_mean", justify="right")
    table.add_column("rho_min", justify="right")
    table.add_column("rho_max", justify="right")
    table.add_column("wall_s_mean", justify="right")
    for (model, task, sampler), grp in df.groupby(["model", "task", "sampler"], sort=False):
        n = len(grp)
        sat = int(grp["satisfied"].sum())
        rmean = float(grp["final_rho"].mean())
        rmin = float(grp["final_rho"].min())
        rmax = float(grp["final_rho"].max())
        wmean = float(grp["wall_clock_s"].mean())
        table.add_row(
            model,
            task,
            sampler,
            f"{sat} / {n}",
            f"{rmean:+.3f}",
            f"{rmin:+.3f}",
            f"{rmax:+.3f}",
            f"{wmean:.1f}",
        )
    return table


def _per_model_verdicts(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """For each model id, run the Day 1 outcome rule on its cells."""
    out: dict[str, dict[str, Any]] = {}
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        verdict, explanation, per_task = _verdict(sub)
        out[str(model)] = {
            "verdict": verdict,
            "explanation": explanation,
            "per_task": per_task,
        }
    return out


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
        "--models",
        type=_parse_csv,
        default=_DEFAULT_MODELS,
        help=f"Comma-separated model ids. Default: {','.join(_DEFAULT_MODELS)}.",
    )
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

    n_total = len(args.models) * len(args.tasks) * len(args.samplers) * len(args.seeds)
    console.print(
        Panel.fit(
            "Quantization x model-size sweep\n"
            f"  models  : {', '.join(args.models)}\n"
            f"  tasks   : {', '.join(args.tasks)}\n"
            f"  samplers: {', '.join(args.samplers)}\n"
            f"  seeds   : {', '.join(map(str, args.seeds))}\n"
            f"  cells   : {n_total}\n"
            f"  out_dir : {args.out_dir}",
            title="quant_size_sweep",
        )
    )

    rows: list[dict[str, Any]] = []
    n_done = 0
    for model_id in args.models:
        # Outer model loop loads weights once per model via _MODEL_CACHE.
        for task_name in args.tasks:
            for sampler_name in args.samplers:
                for seed in args.seeds:
                    n_done += 1
                    row = _run_one_cell_with_model(task_name, sampler_name, int(seed), model_id)
                    rows.append(row)
                    console.print(
                        f"  [{n_done:3d}/{n_total:3d}] model={model_id} "
                        f"task={task_name} sampler={sampler_name} "
                        f"seed={seed} rho={row['final_rho']:+.3f} "
                        f"sat={row['satisfied']} wall={row['wall_clock_s']:.1f}s"
                    )
                    # Append-write per-cell so a kill-9 partway through
                    # leaves a recoverable JSONL on disk.
                    with (args.out_dir / "results.jsonl").open("a") as f:
                        f.write(json.dumps(row) + "\n")

    df = pd.DataFrame(rows)
    parquet_path = args.out_dir / "results.parquet"
    df.to_parquet(parquet_path, index=False)
    console.print(f"Wrote results to {parquet_path}")

    table = _per_model_table(df)
    console.print(table)

    verdicts = _per_model_verdicts(df)
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(verdicts, indent=2))
    console.print(f"Wrote per-model verdicts to {summary_path}")

    panel_lines = []
    for model, info in verdicts.items():
        panel_lines.append(f"{model}: {info['verdict']}")
    console.print(
        Panel.fit(
            "\n".join(panel_lines),
            title="Per-model verdicts (Day-1 outcome rule)",
            border_style="bold yellow",
        )
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

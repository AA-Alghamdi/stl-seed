"""End-to-end PAV vs STL-rho comparison on the canonical trajectory store.

Procedure
---------
For each task family with available canonical trajectories:

1. Load the corpus from ``data/canonical/<task>/`` via :class:`TrajectoryStore`.
2. Compute terminal-success indicators by re-evaluating ``rho > 0`` on the
   spec the corpus was generated against (the metadata's ``spec_key``).
3. Train PAV at varying train-set sizes (the ``--n-train-grid`` argument,
   default ``[100, 500, 1000, 2000]``).
4. Score the held-out test split with both PAV and STL-rho; record AUC,
   Spearman, and per-train-size sample-efficiency points.
5. Plot AUC curve + write ``paper/pav_comparison.md``.

Outputs
-------
* ``runs/pav_comparison/<task>__sample_efficiency.png``
* ``runs/pav_comparison/<task>__results.json``
* ``paper/pav_comparison.md``  (regenerated each run)

REDACTED firewall: imports only ``stl_seed.{baselines,generation,specs,stl}``
plus stdlib + numpy + matplotlib + rich. Verified by
``scripts/REDACTED.sh``.

Usage
-----
::

    uv run python scripts/run_pav_comparison.py
    uv run python scripts/run_pav_comparison.py --tasks bio_ode.repressilator
    uv run python scripts/run_pav_comparison.py --n-train-grid 100 500 1000

"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from stl_seed.baselines.comparison import compare_pav_vs_stl, result_to_summary_dict
from stl_seed.generation.store import TrajectoryStore
from stl_seed.specs import REGISTRY
from stl_seed.stl.evaluator import evaluate_robustness

# ---------------------------------------------------------------------------
# Paths and defaults.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CANONICAL_ROOT = _REPO_ROOT / "data" / "canonical"
_DEFAULT_OUT_DIR = _REPO_ROOT / "runs" / "pav_comparison"
_DEFAULT_RESULTS_MD = _REPO_ROOT / "paper" / "pav_comparison.md"
_DEFAULT_TASKS = (
    "bio_ode.repressilator",
    "glucose_insulin",
)

console = Console()


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _load_corpus(canonical_root: Path, task: str) -> tuple[list[Any], list[dict[str, Any]]]:
    """Load all trajectories for a task family from its canonical store."""
    task_dir = canonical_root / task
    if not task_dir.exists():
        raise FileNotFoundError(f"Canonical store for task {task!r} not found: {task_dir}")
    store = TrajectoryStore(task_dir)
    pairs = store.load()
    if not pairs:
        raise RuntimeError(f"Canonical store for task {task!r} is empty: {task_dir}")
    trajectories = [p[0] for p in pairs]
    metadata = [p[1] for p in pairs]
    return trajectories, metadata


def _terminal_success(
    trajectories: list[Any],
    metadata: list[dict[str, Any]],
    spec_key_override: str | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Compute (success, rho, spec_key) for each trajectory.

    The success label is ``rho > 0`` on the spec stored in the metadata
    (or ``spec_key_override`` if provided). ``rho`` itself is returned for
    Spearman correlation downstream. The spec key is returned so the
    comparison report can show which spec was used.
    """
    spec_keys = {m["spec_key"] for m in metadata if m.get("spec_key")}
    if spec_key_override is not None:
        spec_key = spec_key_override
    elif len(spec_keys) == 1:
        spec_key = next(iter(spec_keys))
    elif len(spec_keys) > 1:
        spec_key = sorted(spec_keys)[0]
        console.print(
            f"[yellow]Multiple spec_keys found ({sorted(spec_keys)}); using {spec_key!r}.[/]"
        )
    else:
        raise RuntimeError("No spec_key in metadata and no override provided.")
    if spec_key not in REGISTRY:
        raise KeyError(f"Spec key {spec_key!r} not in REGISTRY.")
    spec = REGISTRY[spec_key]
    rhos: list[float] = []
    for traj in trajectories:
        rho = float(evaluate_robustness(spec, traj))
        if not np.isfinite(rho):
            rho = 1e9 if rho > 0 else -1e9
        rhos.append(rho)
    rhos_arr = np.asarray(rhos, dtype=np.float64)
    success = (rhos_arr > 0.0).astype(np.float64)
    return success, rhos_arr, spec_key


def _plot_sample_efficiency(
    task: str,
    points: list[dict[str, Any]],
    stl_auc: float,
    out_path: Path,
) -> None:
    """Save an AUC-vs-train-size plot. STL-rho is a horizontal line."""
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    pts_sorted = sorted(points, key=lambda p: p["n_train"])
    xs = [p["n_train"] for p in pts_sorted]
    ys = [p["pav_auc"] for p in pts_sorted]
    ax.plot(xs, ys, "o-", label="PAV (learned)")
    ax.axhline(stl_auc, color="black", linestyle="--", label=f"STL-rho (AUC={stl_auc:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("PAV training trajectories")
    ax.set_ylabel("Held-out AUC")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Sample efficiency: {task}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown writer.
# ---------------------------------------------------------------------------


def _write_results_md(
    out_path: Path,
    summaries: list[dict[str, Any]],
    artifact_dir: Path,
) -> None:
    """Generate ``paper/pav_comparison.md`` from per-task summary dicts.

    The Markdown is regenerated on every run; do not hand-edit.
    """
    lines: list[str] = []
    lines.append("# PAV vs STL-rho: empirical comparison (auto-generated)")
    lines.append("")
    lines.append(
        "This file is regenerated by ``scripts/run_pav_comparison.py``. "
        "Do not hand-edit --- changes will be clobbered on the next run."
    )
    lines.append("")
    lines.append(
        "PAV (Process Advantage Verifier; Setlur et al. 2024, "
        "[arXiv:2410.08146](https://arxiv.org/abs/2410.08146)) is the "
        "learned process-reward baseline. We compare its held-out "
        "discriminability against the closed-form STL-rho verifier on the "
        "canonical trajectory store."
    )
    lines.append("")
    lines.append("## Headline (per task family)")
    lines.append("")
    lines.append(
        "| task | spec | n_train | n_test | STL AUC | PAV AUC | "
        "PAV - STL | crossover_n | PAV train (s) | STL score (s) |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        delta = (
            s["pav_auc"] - s["stl_auc"]
            if (np.isfinite(s["pav_auc"]) and np.isfinite(s["stl_auc"]))
            else float("nan")
        )
        cross = s.get("crossover_n_train")
        cross_disp = "-" if cross is None else str(cross)
        lines.append(
            f"| {s['task']} | `{s['spec_key']}` | {s['n_train']} | {s['n_test']} | "
            f"{s['stl_auc']:.3f} | {s['pav_auc']:.3f} | {delta:+.3f} | "
            f"{cross_disp} | {s['pav_train_seconds']:.2f} | {s['stl_score_seconds']:.4f} |"
        )
    lines.append("")
    lines.append("## Spearman rank correlation")
    lines.append("")
    lines.append("| task | PAV vs success | STL vs success | PAV vs rho | STL vs rho |")
    lines.append("|---|---:|---:|---:|---:|")
    for s in summaries:
        lines.append(
            f"| {s['task']} | {s['pav_spearman_success']:+.3f} | "
            f"{s['stl_spearman_success']:+.3f} | {s['pav_spearman_rho']:+.3f} | "
            f"{s['stl_spearman_rho']:+.3f} |"
        )
    lines.append("")
    lines.append("## Sample efficiency")
    lines.append("")
    lines.append("Each row is a PAV refit at the indicated train-set size.")
    lines.append("STL AUC is the same as in the headline table (constant in train size).")
    lines.append("")
    for s in summaries:
        lines.append(f"### {s['task']}  (`{s['spec_key']}`)")
        lines.append("")
        lines.append(
            "| n_train | PAV AUC | PAV train (s) | PAV final train MSE | PAV final val MSE |"
        )
        lines.append("|---:|---:|---:|---:|---:|")
        for pt in sorted(s["sample_efficiency"], key=lambda p: p["n_train"]):
            lines.append(
                f"| {pt['n_train']} | {pt['pav_auc']:.3f} | "
                f"{pt['pav_train_seconds']:.2f} | {pt['pav_train_loss_final']:.5f} | "
                f"{pt['pav_val_loss_final']:.5f} |"
            )
        lines.append("")
        rel_plot = artifact_dir / f"{s['task']}__sample_efficiency.png"
        try:
            rel = rel_plot.relative_to(out_path.parent)
        except ValueError:
            rel = rel_plot
        lines.append(f"![sample efficiency: {s['task']}]({rel})")
        lines.append("")
    lines.append("## Computational cost asymmetry")
    lines.append("")
    lines.append(
        "STL-rho needs zero training and a single closed-form pass per "
        "trajectory (see ``stl_seed/stl/evaluator.py``). PAV needs the full "
        "training pool, the per-step MC label computation (kNN), and an "
        "MLP fit. The ``PAV train (s)`` column above is the wall-clock time "
        "for the canonical PAV fit at the largest train size."
    )
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append("- PAV implementation: ``src/stl_seed/baselines/pav.py``")
    lines.append("- Comparison harness: ``src/stl_seed/baselines/comparison.py``")
    lines.append("- This script: ``scripts/run_pav_comparison.py``")
    lines.append("- Canonical trajectories: ``data/canonical/``")
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_pav_comparison.py",
        description="PAV vs STL-rho empirical comparison on canonical trajectories.",
    )
    p.add_argument(
        "--canonical-root",
        type=Path,
        default=_DEFAULT_CANONICAL_ROOT,
        help="Root directory of the canonical trajectory stores.",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=list(_DEFAULT_TASKS),
        help="Task families to evaluate. Default: bio_ode.repressilator and glucose_insulin.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT_DIR,
        help="Directory for per-task JSON results and PNG plots.",
    )
    p.add_argument(
        "--results-md",
        type=Path,
        default=_DEFAULT_RESULTS_MD,
        help="paper/pav_comparison.md path (regenerated each run).",
    )
    p.add_argument(
        "--n-train-grid",
        nargs="+",
        type=int,
        default=[100, 500, 1000, 2000],
        help="PAV train-set sizes for the sample-efficiency sweep.",
    )
    p.add_argument(
        "--n-test",
        type=int,
        default=400,
        help="Held-out test set size per task (capped by corpus size).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=20260424,
        help="RNG seed (date as int by default).",
    )
    p.add_argument(
        "--pav-epochs",
        type=int,
        default=30,
        help="PAV training epochs.",
    )
    p.add_argument(
        "--pav-lr",
        type=float,
        default=1e-3,
        help="PAV Adam learning rate.",
    )
    p.add_argument(
        "--pav-hidden",
        type=int,
        default=256,
        help="PAV MLP hidden width.",
    )
    p.add_argument(
        "--pav-dropout",
        type=float,
        default=0.1,
        help="PAV MLP dropout probability.",
    )
    p.add_argument(
        "--k-neighbors",
        type=int,
        default=16,
        help="kNN pool size for the per-step MC estimator.",
    )
    p.add_argument(
        "--max-corpus",
        type=int,
        default=2500,
        help="Cap on per-task corpus size (random subsample if larger).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-epoch PAV training losses.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel.fit(
            f"PAV vs STL-rho comparison\n"
            f"  canonical: {args.canonical_root}\n"
            f"  tasks: {', '.join(args.tasks)}\n"
            f"  n_train grid: {args.n_train_grid}\n"
            f"  n_test: {args.n_test}\n"
            f"  out_dir: {out_dir}\n"
            f"  results_md: {args.results_md}",
            title="[bold]run_pav_comparison",
        )
    )

    summaries: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(args.seed))

    for task in args.tasks:
        console.rule(f"[bold]{task}")
        try:
            trajs, meta = _load_corpus(Path(args.canonical_root), task)
        except (FileNotFoundError, RuntimeError) as exc:
            console.print(f"[red]Skipping {task}: {exc}[/]")
            continue

        # Optional cap on corpus size for tractable wall-clock.
        if len(trajs) > int(args.max_corpus):
            idx = rng.choice(len(trajs), size=int(args.max_corpus), replace=False)
            trajs = [trajs[int(i)] for i in idx]
            meta = [meta[int(i)] for i in idx]
            console.print(
                f"[dim]Subsampled corpus to {len(trajs)} trajectories (cap={args.max_corpus}).[/]"
            )

        try:
            success, rhos, spec_key = _terminal_success(trajs, meta, spec_key_override=None)
        except (KeyError, RuntimeError) as exc:
            console.print(f"[red]Skipping {task}: terminal_success failed: {exc}[/]")
            continue

        n_corpus = len(trajs)
        n_pos = int(success.sum())
        n_neg = int((1 - success).sum())
        console.print(
            f"  corpus={n_corpus} | spec={spec_key} | rho mean={rhos.mean():+.3f} | "
            f"succ frac={n_pos / max(1, n_corpus):.3f} ({n_pos}/{n_corpus})"
        )
        if n_pos == 0 or n_neg == 0:
            console.print(
                f"[yellow]All-{['positive' if n_pos else 'negative'][0]} corpus; "
                "AUC undefined. Skipping.[/]"
            )
            continue

        # Cap the headline n_train at the largest grid point that the corpus
        # supports given n_test.
        max_train_avail = max(0, n_corpus - int(args.n_test))
        headline_n_train = max(
            (n for n in args.n_train_grid if n <= max_train_avail),
            default=max_train_avail,
        )
        if headline_n_train < 2:
            console.print(f"[yellow]Corpus too small for n_test={args.n_test}; skipping.[/]")
            continue
        n_test = min(int(args.n_test), n_corpus - headline_n_train)

        sample_grid = sorted({n for n in args.n_train_grid if n <= max_train_avail})

        t0 = time.time()
        result = compare_pav_vs_stl(
            trajectories=trajs,
            terminal_success=success,
            spec=spec_key,
            n_train=int(headline_n_train),
            n_test=int(n_test),
            seed=int(args.seed),
            sample_efficiency_grid=sample_grid,
            pav_n_epochs=int(args.pav_epochs),
            pav_lr=float(args.pav_lr),
            pav_hidden=int(args.pav_hidden),
            pav_dropout=float(args.pav_dropout),
            k_neighbors=int(args.k_neighbors),
            task_name=task,
            verbose=bool(args.verbose),
            terminal_rho=rhos,
        )
        wall = time.time() - t0
        console.print(
            f"  done in {wall:.1f}s: PAV AUC={result.pav_auc:.3f}, "
            f"STL AUC={result.stl_auc:.3f}, "
            f"crossover_n={result.crossover_n_train}"
        )

        # Persist per-task JSON.
        summary = result_to_summary_dict(result)
        (out_dir / f"{task}__results.json").write_text(json.dumps(summary, indent=2))

        # Plot.
        _plot_sample_efficiency(
            task=task,
            points=summary["sample_efficiency"],
            stl_auc=result.stl_auc,
            out_path=out_dir / f"{task}__sample_efficiency.png",
        )

        summaries.append(summary)

    if not summaries:
        console.print("[red]No tasks produced valid results.[/]")
        return 1

    # Markdown report.
    _write_results_md(
        out_path=Path(args.results_md),
        summaries=summaries,
        artifact_dir=out_dir,
    )

    # Console summary table.
    table = Table(title="PAV vs STL summary", header_style="bold")
    table.add_column("task")
    table.add_column("STL AUC", justify="right")
    table.add_column("PAV AUC", justify="right")
    table.add_column("delta", justify="right")
    table.add_column("crossover n", justify="right")
    for s in summaries:
        delta = s["pav_auc"] - s["stl_auc"]
        cross = s.get("crossover_n_train")
        table.add_row(
            s["task"],
            f"{s['stl_auc']:.3f}",
            f"{s['pav_auc']:.3f}",
            f"{delta:+.3f}",
            "-" if cross is None else str(cross),
        )
    console.print(table)

    return 0


if __name__ == "__main__":
    sys.exit(main())

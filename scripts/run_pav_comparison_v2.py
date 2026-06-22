"""V2 PAV vs STL-rho comparison: model selection + on-policy MC labels.

What this differs from v1
-------------------------
* Uses :func:`stl_seed.baselines.comparison.compare_pav_v2_vs_stl`, which
  in turn calls :meth:`PAVProcessRewardModel.fit_with_selection` (a
  hidden-width / weight-decay grid search with early stopping on val MSE
  plateau). This addresses audit weakness #1 (no model selection).
* Defaults the MC label source to *on-policy fresh rollouts* under a
  uniform-random tail policy; falls back to kNN only when explicitly
  requested or when the rollout simulator is not wired for the task.
  Addresses audit weakness #2 (offline kNN labels, not Setlur §3.2).
* Writes ``paper/pav_v2.md`` directly (separate file from
  ``pav_comparison.md``); does not mutate the v1 artifact.

Outputs
-------
* ``runs/pav_comparison_v2/<task>__results_v2.json``
* ``runs/pav_comparison_v2/<task>__selection_grid.png`` (val MSE
  heatmap over the (hidden, wd) grid)
* ``paper/pav_v2.md``

Usage
-----
::

    .venv/bin/python scripts/run_pav_comparison_v2.py
    .venv/bin/python scripts/run_pav_comparison_v2.py --label-source knn
    .venv/bin/python scripts/run_pav_comparison_v2.py --tasks glucose_insulin --n-train 1000

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

from stl_seed.baselines.comparison import (
    compare_pav_v2_vs_stl,
    result_v2_to_summary_dict,
)
from stl_seed.generation.store import TrajectoryStore
from stl_seed.specs import REGISTRY
from stl_seed.stl.evaluator import evaluate_robustness

# ---------------------------------------------------------------------------
# Paths and defaults.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CANONICAL_ROOT = _REPO_ROOT / "data" / "canonical"
_DEFAULT_OUT_DIR = _REPO_ROOT / "runs" / "pav_comparison_v2"
_DEFAULT_RESULTS_MD = _REPO_ROOT / "paper" / "pav_v2.md"
_DEFAULT_TASKS = (
    "glucose_insulin",
    "bio_ode.repressilator",
)

console = Console()


def _load_corpus(canonical_root: Path, task: str) -> tuple[list[Any], list[dict[str, Any]]]:
    task_dir = canonical_root / task
    if not task_dir.exists():
        raise FileNotFoundError(f"Canonical store for task {task!r} not found: {task_dir}")
    store = TrajectoryStore(task_dir)
    pairs = store.load()
    if not pairs:
        raise RuntimeError(f"Canonical store for task {task!r} is empty: {task_dir}")
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _terminal_success(
    trajectories: list[Any], metadata: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray, str]:
    spec_keys = {m["spec_key"] for m in metadata if m.get("spec_key")}
    if len(spec_keys) == 1:
        spec_key = next(iter(spec_keys))
    elif len(spec_keys) > 1:
        spec_key = sorted(spec_keys)[0]
        console.print(
            f"[yellow]Multiple spec_keys found ({sorted(spec_keys)}); using {spec_key!r}.[/]"
        )
    else:
        raise RuntimeError("No spec_key in metadata.")
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


def _plot_selection_grid(
    task: str,
    grid: list[dict[str, Any]],
    hidden_grid: list[int],
    wd_grid: list[float],
    out_path: Path,
) -> None:
    """Heat-map of val MSE over the (hidden, weight_decay) sweep."""
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    H = len(hidden_grid)
    W = len(wd_grid)
    grid_arr = np.full((H, W), np.nan, dtype=np.float64)
    cell_lookup = {(int(g["hidden"]), float(g["weight_decay"])): g for g in grid}
    for i, h in enumerate(hidden_grid):
        for j, wd in enumerate(wd_grid):
            cell = cell_lookup.get((int(h), float(wd)))
            if cell is not None and np.isfinite(cell.get("best_val_mse", np.nan)):
                grid_arr[i, j] = float(cell["best_val_mse"])
    fig, ax = plt.subplots(figsize=(5, 3.2))
    im = ax.imshow(grid_arr, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(W))
    ax.set_xticklabels([f"{wd:g}" for wd in wd_grid])
    ax.set_yticks(range(H))
    ax.set_yticklabels([str(h) for h in hidden_grid])
    ax.set_xlabel("weight decay")
    ax.set_ylabel("hidden width")
    ax.set_title(f"PAV val MSE: {task}")
    fig.colorbar(im, ax=ax, label="val MSE")
    # Annotate.
    for i in range(H):
        for j in range(W):
            val = grid_arr[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.4f}", ha="center", va="center", color="white", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_results_md(
    out_path: Path,
    summaries: list[dict[str, Any]],
    artifact_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Generate paper/pav_v2.md from per-task summary dicts."""
    lines: list[str] = []
    lines.append("# PAV vs STL-rho V2: model selection + on-policy MC labels")
    lines.append("")
    lines.append("Auto-generated by `scripts/run_pav_comparison_v2.py`. Do not hand-edit.")
    lines.append("")
    lines.append("## Audit (2026-04-26)")
    lines.append("")
    lines.append(
        "The math-rigor audit on 2026-04-26 flagged the original PAV "
        "comparison (`paper/pav_comparison.md`) as weak in three specific "
        "ways. Citations refer to the file at the time of the audit."
    )
    lines.append("")
    lines.append(
        "1. **No model selection.** PAV training was a fixed 30 epochs with "
        "no early stopping on val MSE and no hidden-width sweep "
        "(`scripts/run_pav_comparison.py:319-339`, "
        "`src/stl_seed/baselines/pav.py:521-553`). The non-monotonic AUC "
        "across `n_train` (100 -> 0.912, 500 -> 0.026, 2000 -> 0.774 on "
        "glucose-insulin) is a textbook sign of training instability."
    )
    lines.append("")
    lines.append(
        "2. **Offline-kNN labels, not on-policy MC.** "
        "`src/stl_seed/baselines/pav.py:38-53` documents that the per-step "
        "MC labels are estimated by kNN pooling over the canonical store "
        "rather than by drawing fresh on-policy continuations as "
        "Setlur et al. (arXiv:2410.08146 §3.2) prescribe. This was "
        "explicitly labeled a deliberate weakening."
    )
    lines.append("")
    lines.append(
        "3. **Degenerate corpus on repressilator.** AUC = 0.000 because "
        "0% of trajectories satisfy the spec, so labels are all zero. The "
        'original markdown phrased this as "anti-informative" before being '
        'softened to "degenerate corpus" (no diff in the underlying '
        "comparison)."
    )
    lines.append("")
    lines.append("## Implementation changes")
    lines.append("")
    lines.append(
        "* `PAVProcessRewardModel.fit(...)` now accepts `weight_decay` "
        "(decoupled AdamW), `early_stopping_patience` (best-val-MSE "
        "checkpoint + restore on plateau), and `precomputed_train` / "
        "`precomputed_val` (so a sweep can reuse one MC-label computation)."
    )
    lines.append("")
    lines.append(
        "* `PAVProcessRewardModel.fit_with_selection(...)` is a new "
        "classmethod that sweeps `hidden in {64, 128, 256, 512}` x "
        "`weight_decay in {0, 1e-4, 1e-3, 1e-2}`, picks the cell with the "
        "lowest *trajectory-disjoint* val MSE, and refits at the best "
        "cell. Each cell uses early stopping with patience 5."
    )
    lines.append("")
    lines.append(
        "* `stl_seed.baselines.pav_rollout` is a new module. It "
        "reconstructs the canonical simulator from in-repo defaults and "
        "computes per-step MC values by drawing K=5 i.i.d. random tails "
        "per (trajectory, prefix-length), re-integrating the ODE, and "
        "averaging the rho > 0 indicator. This is the Setlur §3.2 "
        "estimator faithfully implemented; cost is O(N_traj * H * K) "
        "ODE integrations vs. zero for kNN."
    )
    lines.append("")
    lines.append(
        "* `compare_pav_v2_vs_stl(...)` (in "
        "`src/stl_seed/baselines/comparison.py`) glues the two together. "
        "The legacy `compare_pav_vs_stl` is unchanged."
    )
    lines.append("")
    lines.append("## Configuration for this run")
    lines.append("")
    lines.append(f"* label_source: `{args.label_source}` (K_rollout = {args.K_rollout})")
    lines.append(f"* hidden_grid: `{list(args.hidden_grid)}`")
    lines.append(f"* weight_decay_grid: `{list(args.weight_decay_grid)}`")
    lines.append(
        f"* pav_n_epochs: {args.pav_epochs}, lr: {args.pav_lr}, "
        f"early_stopping_patience: {args.early_stopping_patience}"
    )
    lines.append(f"* n_train: {args.n_train}, n_test: {args.n_test}")
    lines.append("")
    lines.append("## Headline results")
    lines.append("")
    lines.append(
        "| task | spec | label src | n_train | n_test | STL AUC | PAV-v2 AUC | "
        "PAV - STL | best (h, wd) | val MSE | label sec | fit sec | sims |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|")
    for s in summaries:
        delta = (
            s["pav_auc"] - s["stl_auc"]
            if (np.isfinite(s["pav_auc"]) and np.isfinite(s["stl_auc"]))
            else float("nan")
        )
        wd = s["pav_best_weight_decay"]
        wd_str = "0" if wd == 0.0 else f"{wd:g}"
        lines.append(
            f"| {s['task']} | `{s['spec_key']}` | {s['label_source']} | "
            f"{s['n_train']} | {s['n_test']} | {s['stl_auc']:.3f} | "
            f"{s['pav_auc']:.3f} | {delta:+.3f} | "
            f"({s['pav_best_hidden']}, {wd_str}) | "
            f"{s['pav_best_val_mse']:.5f} | "
            f"{s['label_compute_seconds']:.1f} | "
            f"{s['pav_train_seconds']:.1f} | "
            f"{s['n_onpolicy_simulations']} |"
        )
    lines.append("")
    lines.append("## Spearman rank correlation")
    lines.append("")
    lines.append("| task | PAV-v2 vs success | STL vs success | PAV-v2 vs rho | STL vs rho |")
    lines.append("|---|---:|---:|---:|---:|")
    for s in summaries:
        lines.append(
            f"| {s['task']} | {s['pav_spearman_success']:+.3f} | "
            f"{s['stl_spearman_success']:+.3f} | {s['pav_spearman_rho']:+.3f} | "
            f"{s['stl_spearman_rho']:+.3f} |"
        )
    lines.append("")
    lines.append("## Selection grid (val MSE)")
    lines.append("")
    for s in summaries:
        lines.append(f"### {s['task']}  (`{s['spec_key']}`)")
        lines.append("")
        lines.append(
            "| hidden | weight_decay | best val MSE | best epoch | early stop | wall (s) |"
        )
        lines.append("|---:|---:|---:|---:|---|---:|")
        for cell in sorted(
            s["selection_grid"], key=lambda c: (int(c["hidden"]), float(c["weight_decay"]))
        ):
            wd = float(cell["weight_decay"])
            wd_str = "0" if wd == 0.0 else f"{wd:g}"
            lines.append(
                f"| {cell['hidden']} | {wd_str} | "
                f"{cell['best_val_mse']:.5f} | "
                f"{cell['best_epoch']} | "
                f"{'yes' if cell['stopped_early'] else 'no'} | "
                f"{cell['wall_time_s']:.1f} |"
            )
        rel_plot = artifact_dir / f"{s['task']}__selection_grid.png"
        try:
            rel = rel_plot.relative_to(out_path.parent)
        except ValueError:
            rel = rel_plot
        lines.append("")
        lines.append(f"![selection grid: {s['task']}]({rel})")
        lines.append("")
    lines.append("## Caveats and what is still a strawman")
    lines.append("")
    lines.append(
        "* **Rollout policy.** Setlur §3.2 rolls out under the policy the "
        "verifier will score. Here the canonical store mixes "
        "{random, constant, heuristic} policies and downstream methods "
        "may use anything; we use *uniform-random* as the no-prior "
        "reference distribution, which is the worst-case proposer "
        "assumption. A future variant could re-roll under each candidate "
        "method's actual policy, at higher cost."
    )
    lines.append("")
    lines.append(
        "* **K_rollout.** Setlur sweeps M in {8, 16, 32}; we use K=5 "
        "default to keep the wall-clock tractable on the M-series. "
        "Variance of the MC estimate scales as 1/K. Bumping K shrinks "
        "variance at the cost of linear sim time."
    )
    lines.append("")
    lines.append(
        "* **Degenerate corpus on bio_ode.repressilator.** Probe runs "
        "(30 trajectories x 10 random tails x 11 prefix lengths = 3300 "
        "fresh ODE integrations) yielded 0 satisfying rollouts. The "
        "Elowitz-Leibler repressilator under uniform-random transcription "
        "control simply cannot satisfy the easy spec --- this is a "
        "property of the proposer / spec pair, not of PAV. STL-rho can "
        "still discriminate because rho is continuous; PAV's binary "
        "advantage signal collapses to zero. We report this honestly "
        "rather than excluding the cell."
    )
    lines.append("")
    lines.append(
        "* **Simulator-side determinism.** On-policy rollout reconstructs "
        "the simulator from in-repo defaults (`GlucoseInsulinSimulator()`, "
        "`BergmanParams()`, `MealSchedule.empty()`, normal-subject IS for "
        "GI; analogous for bio_ode). If those defaults diverge from what "
        "the canonical store was generated against, the MC estimates "
        "become biased; the v2 driver verifies horizon and action_dim "
        "agreement before rolling out as a guardrail."
    )
    lines.append("")
    lines.append(
        "* **Compute asymmetry remains.** STL-rho still requires zero "
        "training and ~0.05s to score 400 test trajectories. PAV-v2 "
        "needs the on-policy rollout pass (charged in the `label sec` "
        "column above) plus the model-selection sweep (`fit sec`)."
    )
    lines.append("")
    lines.append("## Headline interpretation")
    lines.append("")
    if summaries:
        gi = next((s for s in summaries if s["task"] == "glucose_insulin"), None)
        rep = next((s for s in summaries if s["task"] == "bio_ode.repressilator"), None)
        if gi is not None:
            delta_gi = gi["pav_auc"] - gi["stl_auc"]
            if delta_gi >= -0.02:
                verdict_gi = (
                    "On glucose-insulin the strengthened PAV-v2 essentially "
                    f"matches STL-rho ({gi['pav_auc']:.3f} vs {gi['stl_auc']:.3f}, "
                    f"delta = {delta_gi:+.3f}). The v1 narrative that "
                    f'"STL-rho dominates on glucose-insulin" softens once '
                    f"PAV is properly tuned."
                )
            else:
                verdict_gi = (
                    f"On glucose-insulin, even after model selection and "
                    f"on-policy rollouts, PAV-v2 trails STL-rho by "
                    f"{-delta_gi:.3f} AUC ({gi['pav_auc']:.3f} vs "
                    f"{gi['stl_auc']:.3f}). The original headline survives "
                    f"the strengthened comparison."
                )
            lines.append(verdict_gi)
            lines.append("")
        if rep is not None:
            lines.append(
                f"On bio_ode.repressilator the corpus is degenerate "
                f"(success rate 0 under the canonical proposer), so PAV "
                f"AUC is undefined / pinned at random "
                f"(reported {rep['pav_auc']:.3f}). STL-rho still "
                f"achieves {rep['stl_auc']:.3f} because rho is continuous "
                f"and ranks trajectories by margin, not by rho > 0. This "
                f"is a *task* property, not a verifier-quality property."
            )
            lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append("- PAV: `src/stl_seed/baselines/pav.py`")
    lines.append("- On-policy rollouts: `src/stl_seed/baselines/pav_rollout.py`")
    lines.append("- Comparison harness: `src/stl_seed/baselines/comparison.py`")
    lines.append("- This script: `scripts/run_pav_comparison_v2.py`")
    lines.append("- Canonical trajectories: `data/canonical/`")
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_pav_comparison_v2.py",
        description="V2 PAV vs STL-rho: model selection + on-policy MC labels.",
    )
    p.add_argument(
        "--canonical-root",
        type=Path,
        default=_DEFAULT_CANONICAL_ROOT,
    )
    p.add_argument("--tasks", nargs="+", default=list(_DEFAULT_TASKS))
    p.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR)
    p.add_argument("--results-md", type=Path, default=_DEFAULT_RESULTS_MD)
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-test", type=int, default=400)
    p.add_argument("--seed", type=int, default=20260426)
    p.add_argument(
        "--label-source",
        choices=("knn", "onpolicy"),
        default="onpolicy",
        help="MC label estimator: 'onpolicy' (Setlur §3.2) or 'knn' (legacy).",
    )
    p.add_argument("--K-rollout", type=int, default=5)
    p.add_argument(
        "--hidden-grid",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512],
    )
    p.add_argument(
        "--weight-decay-grid",
        type=float,
        nargs="+",
        default=[0.0, 1e-4, 1e-3, 1e-2],
    )
    p.add_argument("--pav-epochs", type=int, default=100)
    p.add_argument("--pav-lr", type=float, default=1e-3)
    p.add_argument("--pav-dropout", type=float, default=0.1)
    p.add_argument("--early-stopping-patience", type=int, default=5)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--k-neighbors", type=int, default=16)
    p.add_argument("--max-corpus", type=int, default=2500)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel.fit(
            f"PAV-v2 vs STL-rho\n"
            f"  canonical: {args.canonical_root}\n"
            f"  tasks: {', '.join(args.tasks)}\n"
            f"  label_source: {args.label_source} (K={args.K_rollout})\n"
            f"  n_train={args.n_train} n_test={args.n_test}\n"
            f"  hidden_grid={args.hidden_grid}\n"
            f"  weight_decay_grid={args.weight_decay_grid}\n"
            f"  out_dir: {out_dir}\n"
            f"  results_md: {args.results_md}",
            title="[bold]run_pav_comparison_v2",
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

        if len(trajs) > int(args.max_corpus):
            idx = rng.choice(len(trajs), size=int(args.max_corpus), replace=False)
            trajs = [trajs[int(i)] for i in idx]
            meta = [meta[int(i)] for i in idx]
            console.print(f"[dim]Subsampled corpus to {len(trajs)} trajectories.[/]")

        try:
            success, rhos, spec_key = _terminal_success(trajs, meta)
        except (KeyError, RuntimeError) as exc:
            console.print(f"[red]Skipping {task}: {exc}[/]")
            continue

        n_corpus = len(trajs)
        n_pos = int(success.sum())
        n_neg = int((1 - success).sum())
        console.print(
            f"  corpus={n_corpus} | spec={spec_key} | succ={n_pos}/{n_corpus} ({n_pos / max(1, n_corpus):.3f})"
        )

        n_train_used = min(int(args.n_train), n_corpus - int(args.n_test))
        n_test_used = min(int(args.n_test), n_corpus - n_train_used)
        if n_train_used < 4 or n_test_used < 4:
            console.print(
                f"[yellow]Corpus too small (n_train_used={n_train_used}, "
                f"n_test_used={n_test_used}); skipping.[/]"
            )
            continue
        if n_pos == 0 or n_neg == 0:
            console.print(
                f"[yellow]Degenerate corpus (succ_frac={n_pos / max(1, n_corpus):.3f}); "
                "PAV will collapse to zero advantage. Reporting anyway.[/]"
            )

        t0 = time.time()
        result = compare_pav_v2_vs_stl(
            trajectories=trajs,
            terminal_success=success,
            spec=spec_key,
            task=task,
            n_train=int(n_train_used),
            n_test=int(n_test_used),
            seed=int(args.seed),
            label_source=str(args.label_source),
            K_rollout=int(args.K_rollout),
            hidden_grid=tuple(int(h) for h in args.hidden_grid),
            weight_decay_grid=tuple(float(w) for w in args.weight_decay_grid),
            pav_n_epochs=int(args.pav_epochs),
            pav_lr=float(args.pav_lr),
            pav_dropout=float(args.pav_dropout),
            early_stopping_patience=int(args.early_stopping_patience),
            val_frac=float(args.val_frac),
            k_neighbors=int(args.k_neighbors),
            task_name=task,
            verbose=bool(args.verbose),
            terminal_rho=rhos,
        )
        wall = time.time() - t0
        console.print(
            f"  done in {wall:.1f}s: PAV-v2 AUC={result.pav_auc:.3f}, "
            f"STL AUC={result.stl_auc:.3f}, "
            f"best_(hidden, wd)=({result.pav_best_hidden}, {result.pav_best_weight_decay})"
        )

        summary = result_v2_to_summary_dict(result)
        (out_dir / f"{task}__results_v2.json").write_text(json.dumps(summary, indent=2))

        _plot_selection_grid(
            task=task,
            grid=summary["selection_grid"],
            hidden_grid=list(args.hidden_grid),
            wd_grid=list(args.weight_decay_grid),
            out_path=out_dir / f"{task}__selection_grid.png",
        )

        summaries.append(summary)

    if not summaries:
        console.print("[red]No tasks produced valid results.[/]")
        return 1

    _write_results_md(
        out_path=Path(args.results_md),
        summaries=summaries,
        artifact_dir=out_dir,
        args=args,
    )

    table = Table(title="PAV-v2 vs STL summary", header_style="bold")
    table.add_column("task")
    table.add_column("STL AUC", justify="right")
    table.add_column("PAV-v2 AUC", justify="right")
    table.add_column("delta", justify="right")
    table.add_column("best (h, wd)")
    table.add_column("label sec", justify="right")
    table.add_column("fit sec", justify="right")
    for s in summaries:
        delta = s["pav_auc"] - s["stl_auc"]
        table.add_row(
            s["task"],
            f"{s['stl_auc']:.3f}",
            f"{s['pav_auc']:.3f}",
            f"{delta:+.3f}",
            f"({s['pav_best_hidden']}, {s['pav_best_weight_decay']})",
            f"{s['label_compute_seconds']:.1f}",
            f"{s['pav_train_seconds']:.1f}",
        )
    console.print(table)

    return 0


if __name__ == "__main__":
    sys.exit(main())

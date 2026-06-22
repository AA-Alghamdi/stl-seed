"""Phase-3 statistical analysis of the Phase-2 canonical sweep.

Consumes ``runs/canonical/eval_results.parquet`` produced by
``scripts/run_canonical_eval.py`` and runs the registered hierarchical
Bayes model from ``src/stl_seed/stats/hierarchical_bayes.py`` plus the
TOST equivalence test from ``src/stl_seed/stats/tost.py``.

Outputs (written under ``--output-dir``)::

    posterior.nc                # ArviZ NetCDF (raw posterior draws)
    posterior_summary.csv       # delta_v_A, delta_v_b summaries
    convergence.json            # R-hat / ESS per parameter
    tost_results.json           # equivalence-test decisions
    figures/bon_curves.png      # 3 (model) x 2 (task), lines per filter
    figures/posterior_summary.png
    figures/goodhart_decomposition.png  # placeholder; v2

Also writes ``paper/results.md`` with the canonical numbers, replacing
pilot/smoke placeholders. The Markdown is regenerated from the CSVs on
every run, so paper/results.md is *derived* and should not be hand-edited.


Usage::

    uv run python scripts/canonical_analysis.py \\
        --eval-results runs/canonical/eval_results.parquet \\
        --output-dir runs/canonical/analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ---------------------------------------------------------------------------
# Paths.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_EVAL = _REPO_ROOT / "runs" / "canonical" / "eval_results.parquet"
_DEFAULT_OUT = _REPO_ROOT / "runs" / "canonical" / "analysis"
_DEFAULT_RESULTS_MD = _REPO_ROOT / "paper" / "results.md"

# Pre-registered settings (paper/theory.md §3, §4).
_TOST_DELTA = 0.05  # equivalence margin (theory.md §3 H1)
_TOST_ALPHA = 0.05
_HEADLINE_BON = 128

console = Console()


# ---------------------------------------------------------------------------
# Data preparation.
# ---------------------------------------------------------------------------


@dataclass
class CodingMaps:
    """Index-to-label maps for the hierarchical model."""

    models: list[str]
    filters: list[str]  # index 0 must be the baseline ("hard")
    families: list[str]
    instances: list[str]


def build_hierarchical_data(df: pd.DataFrame) -> tuple[Any, CodingMaps]:
    """Convert a long-form eval DataFrame into ``HierarchicalData``.

    Filter index 0 is forced to ``hard`` so ``delta_v`` indices 1, 2
    correspond to the {quantile, continuous} contrasts referenced in
    paper/theory.md §4.
    """
    from stl_seed.stats.hierarchical_bayes import HierarchicalData

    if df.empty:
        raise ValueError("eval DataFrame is empty; nothing to analyze.")

    # Force the canonical baseline-first ordering.
    filters = ["hard"] + sorted({f for f in df["filter"].unique() if f != "hard"})
    models = sorted(df["model"].unique())
    families = sorted(df["task"].unique())

    # Instances are nested in (family, instance_idx); we flatten to a
    # global integer for the model. The hierarchical_bayes module
    # treats each (family, instance) as its own observation.
    instance_keys = sorted({(t, int(i)) for t, i in zip(df["task"], df["instance"], strict=True)})
    instance_index = {k: i for i, k in enumerate(instance_keys)}

    model_to_idx = {m: i for i, m in enumerate(models)}
    filter_to_idx = {f: i for i, f in enumerate(filters)}
    family_to_idx = {f: i for i, f in enumerate(families)}

    rows = df.dropna(subset=["success"]).copy()
    coding = CodingMaps(
        models=models,
        filters=filters,
        families=families,
        instances=[f"{t}::{i}" for t, i in instance_keys],
    )
    data = HierarchicalData(
        model_idx=rows["model"].map(model_to_idx).to_numpy(dtype=np.int32),
        verifier_idx=rows["filter"].map(filter_to_idx).to_numpy(dtype=np.int32),
        family_idx=rows["task"].map(family_to_idx).to_numpy(dtype=np.int32),
        instance_idx=np.array(
            [
                instance_index[(t, int(i))]
                for t, i in zip(rows["task"], rows["instance"], strict=True)
            ],
            dtype=np.int32,
        ),
        seed=rows["seed"].to_numpy(dtype=np.int32),
        N=rows["N"].to_numpy(dtype=np.int32),
        Y=rows["success"].to_numpy(dtype=np.int32),
        n_models=len(models),
        n_verifiers=len(filters),
        n_families=len(families),
        n_instances=len(instance_keys),
        coords={
            "models": models,
            "filters": filters,
            "families": families,
        },
    )
    return data, coding


# ---------------------------------------------------------------------------
# Hierarchical Bayes runner.
# ---------------------------------------------------------------------------


def run_bayes(
    data: Any,
    *,
    n_chains: int = 4,
    n_warmup: int = 2000,
    n_samples: int = 2000,
    seed: int = 20260424,
) -> Any:
    """Fit the registered hierarchical model. Lazy-imports numpyro."""
    from stl_seed.stats.hierarchical_bayes import fit

    return fit(
        data=data,
        n_chains=n_chains,
        n_warmup=n_warmup,
        n_samples=n_samples,
        target_accept=0.9,
        max_tree_depth=10,
        key=int(seed),
        progress_bar=True,
    )


def summarize_posterior(idata: Any, coding: CodingMaps) -> pd.DataFrame:
    """Return a DataFrame summarizing the contrast parameters of interest."""
    from stl_seed.stats.hierarchical_bayes import summarize

    summ = summarize(idata, hdi_prob=0.89)
    # Attach human-readable filter labels: contrast_idx ∈ {1,...,n_filters-1}
    # corresponds to coding.filters[contrast_idx].
    if not summ.empty:
        summ["filter_label"] = summ["contrast_idx"].apply(
            lambda i: coding.filters[int(i)] if int(i) < len(coding.filters) else "?"
        )
    return summ


def convergence(idata: Any) -> dict[str, Any]:
    from stl_seed.stats.hierarchical_bayes import convergence_check

    return convergence_check(idata)


# ---------------------------------------------------------------------------
# TOST equivalence test (per cell).
# ---------------------------------------------------------------------------


def tost_per_cell(df: pd.DataFrame, headline_N: int = _HEADLINE_BON) -> list[dict[str, Any]]:
    """Per-(model, task, soft-filter) TOST against the hard baseline at N=headline.

    Implements H1 from paper/theory.md §3: equivalence at Δ = 0.05.
    """
    from stl_seed.stats.tost import tost_equivalence

    out: list[dict[str, Any]] = []
    if "filter" not in df.columns:
        return out
    sub = df[df["N"] == headline_N]
    for (model, task), block in sub.groupby(["model", "task"]):
        baseline = block[block["filter"] == "hard"]
        if baseline.empty:
            continue
        for soft in ["quantile", "continuous"]:
            soft_block = block[block["filter"] == soft]
            if soft_block.empty:
                continue
            try:
                result = tost_equivalence(
                    treatment=soft_block["success"].to_numpy(dtype=np.float64),
                    control=baseline["success"].to_numpy(dtype=np.float64),
                    margin=_TOST_DELTA,
                    alpha=_TOST_ALPHA,
                )
                out.append(
                    {
                        "model": model,
                        "task": task,
                        "soft_filter": soft,
                        "headline_N": headline_N,
                        "delta": float(result.diff),
                        "tost_p": float(result.p_value),
                        "equivalent": bool(result.equivalent),
                    }
                )
            except (ValueError, RuntimeError) as exc:
                out.append(
                    {
                        "model": model,
                        "task": task,
                        "soft_filter": soft,
                        "headline_N": headline_N,
                        "delta": float("nan"),
                        "tost_p": float("nan"),
                        "equivalent": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return out


# ---------------------------------------------------------------------------
# Figures.
# ---------------------------------------------------------------------------


def plot_bon_curves(df: pd.DataFrame, out_path: Path) -> None:
    """3 (model) x 2 (task) grid; one line per filter."""
    import matplotlib.pyplot as plt

    models = sorted(df["model"].unique())
    tasks = sorted(df["task"].unique())
    n_rows = max(len(models), 1)
    n_cols = max(len(tasks), 1)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True, sharey=True, squeeze=False
    )
    for r, model in enumerate(models):
        for c, task in enumerate(tasks):
            ax = axes[r][c]
            sub = df[(df["model"] == model) & (df["task"] == task)]
            for filt in sorted(sub["filter"].unique()):
                cell = sub[sub["filter"] == filt]
                if cell.empty:
                    continue
                grouped = cell.groupby("N")["success"].mean()
                ax.plot(grouped.index, grouped.values, marker="o", label=filt)
            ax.set_xscale("log")
            ax.set_xlabel("N (BoN budget)")
            ax.set_ylabel("success rate")
            ax.set_title(f"{model} | {task}", fontsize=9)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_posterior_summary(summ: pd.DataFrame, out_path: Path) -> None:
    """Forest-style plot of delta_v_A and delta_v_b across contrasts."""
    import matplotlib.pyplot as plt

    if summ.empty:
        return
    fig, ax = plt.subplots(figsize=(6, max(2, 0.4 * len(summ))))
    y = np.arange(len(summ))
    ax.errorbar(
        summ["mean"].values,
        y,
        xerr=[
            summ["mean"].values - summ["hdi_low"].values,
            summ["hdi_high"].values - summ["mean"].values,
        ],
        fmt="o",
        capsize=3,
    )
    ax.axvline(0, color="gray", linestyle="--", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            f"{r['parameter']}[{int(r['contrast_idx'])}={r.get('filter_label', '?')}]"
            for _, r in summ.iterrows()
        ]
    )
    ax.set_xlabel("posterior mean (89% HDI)")
    ax.set_title("Filter contrasts vs. baseline (hard)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_goodhart_placeholder(out_path: Path) -> None:
    """Placeholder for the v2 Goodhart-decomposition figure (theory.md §6)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.text(
        0.5,
        0.5,
        "Goodhart decomposition\n(R_gold - R_proxy)\n\nv2 figure -- requires phi_gold runs\nsee paper/theory.md §6",
        ha="center",
        va="center",
        fontsize=11,
    )
    ax.set_axis_off()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown writer.
# ---------------------------------------------------------------------------


def write_results_md(
    out_path: Path,
    summary: pd.DataFrame,
    tost_rows: list[dict[str, Any]],
    convergence_dict: dict[str, Any],
    df: pd.DataFrame,
) -> None:
    """Generate paper/results.md from the analysis outputs.

    The structure mirrors paper/power_analysis_empirical.md and
    """
    lines: list[str] = []
    lines.append("# Phase-2 canonical results (auto-generated)")
    lines.append("")
    lines.append("This file is regenerated by ``scripts/canonical_analysis.py``.")
    lines.append("Do not hand-edit. changes will be clobbered on the next run.")
    lines.append("")
    lines.append("## Headline numbers (BoN @ N=128)")
    lines.append("")
    if not df.empty:
        head = (
            df[df["N"] == _HEADLINE_BON]
            .groupby(["model", "filter", "task"])["success"]
            .mean()
            .reset_index()
            .sort_values(["task", "model", "filter"])
        )
        lines.append("| model | filter | task | success_rate |")
        lines.append("|---|---|---|---:|")
        for _, r in head.iterrows():
            lines.append(
                f"| {r['model']} | {r['filter']} | {r['task']} | {float(r['success']):.3f} |"
            )
    else:
        lines.append("_no data_")
    lines.append("")
    lines.append("## Posterior summary (delta_v_A, delta_v_b)")
    lines.append("")
    if not summary.empty:
        lines.append("| parameter | contrast (filter) | mean | sd | 89% HDI | P(>0) |")
        lines.append("|---|---|---:|---:|---|---:|")
        for _, r in summary.iterrows():
            lines.append(
                f"| {r['parameter']} | {r.get('filter_label', '?')} | "
                f"{r['mean']:+.3f} | {r['sd']:.3f} | "
                f"[{r['hdi_low']:+.3f}, {r['hdi_high']:+.3f}] | "
                f"{r['P(>0)']:.3f} |"
            )
    else:
        lines.append("_posterior not fit yet_")
    lines.append("")
    lines.append("## TOST equivalence (H1, paper/theory.md §3)")
    lines.append("")
    lines.append(f"- Margin Delta = {_TOST_DELTA}")
    lines.append(f"- alpha = {_TOST_ALPHA}")
    lines.append("")
    if tost_rows:
        lines.append("| model | task | soft filter | delta_p | TOST p | equivalent |")
        lines.append("|---|---|---|---:|---:|:---:|")
        for r in tost_rows:
            eq = "Y" if r.get("equivalent") else "N"
            lines.append(
                f"| {r['model']} | {r['task']} | {r['soft_filter']} | "
                f"{r['delta']:+.3f} | {r['tost_p']:.3f} | {eq} |"
            )
    else:
        lines.append("_no TOST results computed_")
    lines.append("")
    lines.append("## Convergence diagnostics")
    lines.append("")
    if convergence_dict:
        lines.append("| parameter | R-hat (max) | ESS bulk (min) | ESS tail (min) |")
        lines.append("|---|---:|---:|---:|")
        for p, d in convergence_dict.items():
            lines.append(
                f"| {p} | {d.get('r_hat_max', float('nan')):.3f} | "
                f"{d.get('ess_bulk_min', float('nan')):.0f} | "
                f"{d.get('ess_tail_min', float('nan')):.0f} |"
            )
    else:
        lines.append("_convergence diagnostics not computed_")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    lines.append("- `figures/bon_curves.png` -- 3x2 grid (model x task), lines per filter")
    lines.append("- `figures/posterior_summary.png` -- forest plot of contrast posteriors")
    lines.append("- `figures/goodhart_decomposition.png` -- placeholder, see paper/theory.md §6")
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="canonical_analysis.py",
        description="Phase-3 hierarchical-Bayes analysis of canonical eval results.",
    )
    p.add_argument(
        "--eval-results",
        type=Path,
        default=_DEFAULT_EVAL,
        help="Parquet emitted by run_canonical_eval.py.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUT,
        help="Directory for posterior.nc, summaries, and figures.",
    )
    p.add_argument(
        "--results-md",
        type=Path,
        default=_DEFAULT_RESULTS_MD,
        help="paper/results.md path (regenerated each run).",
    )
    p.add_argument("--n-chains", type=int, default=4, help="NUTS chains (paper/theory.md §4: 4).")
    p.add_argument("--n-warmup", type=int, default=2000, help="NUTS warmup (theory.md §4: 2000).")
    p.add_argument("--n-samples", type=int, default=2000, help="NUTS draws (theory.md §4: 2000).")
    p.add_argument(
        "--skip-bayes",
        action="store_true",
        help="Skip MCMC (figures + TOST only). Useful for quick iteration on figures.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    eval_path = Path(args.eval_results)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not eval_path.exists():
        console.print(f"[red]eval results parquet not found at {eval_path}[/]")
        return 1

    df = pd.read_parquet(eval_path)
    console.print(
        Panel.fit(
            f"Phase-3 analysis\n"
            f"  eval results: {eval_path}  ({len(df)} rows)\n"
            f"  output: {out_dir}\n"
            f"  results.md: {args.results_md}",
            title="[bold]canonical_analysis",
        )
    )

    # Always: BoN figure + Goodhart placeholder.
    figs_dir = out_dir / "figures"
    plot_bon_curves(df, figs_dir / "bon_curves.png")
    plot_goodhart_placeholder(figs_dir / "goodhart_decomposition.png")

    # TOST per cell.
    tost_rows = tost_per_cell(df)
    (out_dir / "tost_results.json").write_text(json.dumps(tost_rows, indent=2))

    summary = pd.DataFrame()
    convergence_dict: dict[str, Any] = {}

    if not args.skip_bayes:
        console.rule("[bold]Hierarchical Bayes (NumPyro NUTS)")
        data, coding = build_hierarchical_data(df)
        idata = run_bayes(
            data,
            n_chains=int(args.n_chains),
            n_warmup=int(args.n_warmup),
            n_samples=int(args.n_samples),
        )
        # Persist raw posterior + summaries.
        idata.to_netcdf(str(out_dir / "posterior.nc"))
        summary = summarize_posterior(idata, coding)
        summary.to_csv(out_dir / "posterior_summary.csv", index=False)
        convergence_dict = convergence(idata)
        (out_dir / "convergence.json").write_text(json.dumps(convergence_dict, indent=2))
        plot_posterior_summary(summary, figs_dir / "posterior_summary.png")
    else:
        console.print("[yellow]--skip-bayes[/] set; skipping NumPyro fit.")

    write_results_md(
        out_path=Path(args.results_md),
        summary=summary,
        tost_rows=tost_rows,
        convergence_dict=convergence_dict,
        df=df,
    )

    # Final summary.
    summary_table = Table(title="Analysis artifacts", header_style="bold")
    summary_table.add_column("file")
    summary_table.add_column("present?", justify="center")
    for f in [
        out_dir / "posterior.nc",
        out_dir / "posterior_summary.csv",
        out_dir / "convergence.json",
        out_dir / "tost_results.json",
        figs_dir / "bon_curves.png",
        figs_dir / "posterior_summary.png",
        figs_dir / "goodhart_decomposition.png",
        Path(args.results_md),
    ]:
        summary_table.add_row(str(f), "[green]Y[/]" if f.exists() else "-")
    console.print(summary_table)
    return 0


if __name__ == "__main__":
    sys.exit(main())

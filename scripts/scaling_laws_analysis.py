"""Inference-time compute scaling-law analysis.

Loads runs/unified_comparison/results.parquet and runs/cost_benchmark/results.parquet
and fits power laws of the form ``rho_bar = a * t^b + c`` per task across the 9 samplers,
where t is warm wall-clock per call (seconds).

We deliberately avoid fitting per (sampler, task) within-cell because each cell has
N=4 seeds running at *near-identical* wall-clock cost (sampler determinism + warm JIT),
so within-cell variance is in rho, not in t. The legible compute-vs-quality axis
is *across samplers within a task*: each sampler is a different operating point
on the cost-quality frontier.

Outputs:
- paper/figures/scaling_laws_per_task.png  (log-log rho vs warm-wall, per task)
- paper/figures/scaling_laws_exponents.png (fitted exponents per task with bootstrap CI)
- prints fit summary to stdout in markdown-table form
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]

# Color palette: matched to scripts/run_unified_comparison.py and
# scripts/benchmark_compute_cost.py for consistency across paper figures.
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

_SAMPLER_LABELS: dict[str, str] = {
    "standard": "Standard (lambda=0)",
    "best_of_n": "Binary BoN (N=8)",
    "continuous_bon": "Continuous BoN (N=8)",
    "gradient_guided": "Gradient-Guided (lambda=2)",
    "hybrid": "Hybrid GBoN (n=4)",
    "horizon_folded": "Horizon-Folded (K=100)",
    "rollout_tree": "Rollout-Tree (B=8,L=5)",
    "cmaes_gradient": "CMA-ES + Grad (pop=32)",
    "beam_search_warmstart": "Beam Warmstart (B=8)",
}

_SAMPLER_ORDER = [
    "standard",
    "best_of_n",
    "continuous_bon",
    "gradient_guided",
    "hybrid",
    "horizon_folded",
    "rollout_tree",
    "cmaes_gradient",
    "beam_search_warmstart",
]

_TASK_ORDER = [
    "glucose_insulin",
    "bio_ode.mapk",
    "cardiac_ap",
    "bio_ode.toggle",
    "bio_ode.repressilator",
]

_TASK_LABEL: dict[str, str] = {
    "glucose_insulin": "glucose_insulin (smooth)",
    "bio_ode.mapk": "bio_ode.mapk (smooth)",
    "cardiac_ap": "cardiac_ap (smooth)",
    "bio_ode.toggle": "bio_ode.toggle (narrow-attractor)",
    "bio_ode.repressilator": "bio_ode.repressilator (narrow-attractor)",
}


def _power_law(t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """rho_bar(t) = a * t^b + c.

    a controls scale, b is the exponent we report, c is the asymptote
    (ceiling for a<0,b>0 saturation curves; floor otherwise).
    """
    return a * np.power(t, b) + c


def _fit_per_task(
    walls: np.ndarray,
    rhos: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Fit rho = a*t^b + c via curve_fit on per-cell means; bootstrap CI on b.

    Bootstrap: resample the (t_i, rho_i) cells with replacement and refit.
    With only 9 cells per task this is a coarse CI but it captures whether
    the exponent sign is robust to which sampler happens to be in the bag.
    """
    walls = np.asarray(walls, dtype=float)
    rhos = np.asarray(rhos, dtype=float)
    mask = np.isfinite(walls) & np.isfinite(rhos) & (walls > 0)
    if mask.sum() < 4:
        return {"ok": False, "reason": "insufficient_finite_points"}
    walls = walls[mask]
    rhos = rhos[mask]

    # Initial guess: c = max(rho) (asymptote), a = -(max-min), b = -0.3 (slow saturation)
    rho_max = rhos.max()
    rho_min = rhos.min()
    p0 = (-(rho_max - rho_min) - 1e-6, -0.3, rho_max)
    bounds = ([-np.inf, -3.0, -np.inf], [np.inf, 3.0, np.inf])
    try:
        popt, pcov = curve_fit(
            _power_law,
            walls,
            rhos,
            p0=p0,
            bounds=bounds,
            maxfev=20000,
        )
    except Exception as exc:  # pragma: no cover -- catch all here is intentional
        return {"ok": False, "reason": f"curve_fit_failed: {exc}"}

    pred = _power_law(walls, *popt)
    ss_res = float(np.sum((rhos - pred) ** 2))
    ss_tot = float(np.sum((rhos - rhos.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    rng = np.random.default_rng(seed)
    boot_b = []
    boot_r2 = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(walls), size=len(walls))
        try:
            popt_b, _ = curve_fit(
                _power_law,
                walls[idx],
                rhos[idx],
                p0=popt,
                bounds=bounds,
                maxfev=20000,
            )
            boot_b.append(popt_b[1])
            pred_b = _power_law(walls[idx], *popt_b)
            ss_res_b = float(np.sum((rhos[idx] - pred_b) ** 2))
            ss_tot_b = float(np.sum((rhos[idx] - rhos[idx].mean()) ** 2))
            boot_r2.append(1.0 - ss_res_b / ss_tot_b if ss_tot_b > 0 else float("nan"))
        except Exception:
            continue
    boot_b_arr = np.asarray(boot_b)
    if boot_b_arr.size < 50:
        b_ci_low, b_ci_high = float("nan"), float("nan")
        b_sign_neg_frac = float("nan")
    else:
        b_ci_low = float(np.quantile(boot_b_arr, 0.025))
        b_ci_high = float(np.quantile(boot_b_arr, 0.975))
        b_sign_neg_frac = float((boot_b_arr < 0).mean())

    return {
        "ok": True,
        "a": float(popt[0]),
        "b": float(popt[1]),
        "c": float(popt[2]),
        "b_ci_low": b_ci_low,
        "b_ci_high": b_ci_high,
        "b_sign_neg_frac": b_sign_neg_frac,
        "r2": float(r2),
        "n_points": int(len(walls)),
        "n_boot_succ": int(boot_b_arr.size),
        "boundary_hit": bool(abs(popt[1]) >= 2.99),
    }


def main() -> None:
    cost = pd.read_parquet(ROOT / "runs/cost_benchmark/results.parquet")
    per_seed = pd.read_parquet(ROOT / "runs/cost_benchmark/per_seed.parquet")

    # Compose per-cell aggregate: (task, sampler) -> (warm wall, mean rho, std rho)
    cells = cost[["task", "sampler", "wall_clock_s_warm", "mean_rho", "std_rho", "sat_frac"]].copy()
    cells = cells.rename(columns={"wall_clock_s_warm": "wall"})

    print("\n=== Per-task power-law fits: rho_bar = a * t^b + c ===")
    print("(across 9 samplers per task; bootstrap CI on exponent b)\n")
    fit_rows = []
    for task in _TASK_ORDER:
        sub = cells[cells["task"] == task]
        # Order by sampler so the colour palette aligns deterministically
        sub = sub.set_index("sampler").reindex(_SAMPLER_ORDER).reset_index()
        fit = _fit_per_task(sub["wall"].values, sub["mean_rho"].values)
        fit["task"] = task
        fit_rows.append(fit)
        if fit["ok"]:
            print(
                f"{task:24s}  a={fit['a']:+10.3g}  b={fit['b']:+.3f} "
                f"[{fit['b_ci_low']:+.3f}, {fit['b_ci_high']:+.3f}]  "
                f"c={fit['c']:+10.3g}  R^2={fit['r2']:.3f}  N={fit['n_points']}"
            )
        else:
            print(f"{task:24s}  FIT FAILED: {fit['reason']}")
    fit_df = pd.DataFrame(fit_rows)

    # Save raw fits for the markdown
    out_json = ROOT / "runs/cost_benchmark/scaling_law_fits.json"
    fit_df.to_json(out_json, orient="records", indent=2)
    print(f"\nWrote {out_json}")

    # ------------------------------------------------------------------
    # Figure 1: per-task log-log rho vs wall-clock with all 9 samplers
    # ------------------------------------------------------------------
    n_tasks = len(_TASK_ORDER)
    n_cols = 3
    n_rows = (n_tasks + 1 + n_cols - 1) // n_cols  # +1 for legend panel
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.2 * n_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i_task, task in enumerate(_TASK_ORDER):
        ax = axes[i_task]
        sub = cells[cells["task"] == task].copy()
        sub = sub.set_index("sampler").reindex(_SAMPLER_ORDER).reset_index()
        # Per-seed scatter for variance visibility
        per = per_seed[per_seed["task"] == task]
        for samp in _SAMPLER_ORDER:
            samp_pts = per[(per["sampler"] == samp) & (~per["is_cold"])]
            if not samp_pts.empty:
                ax.scatter(
                    samp_pts["wall_clock_s"],
                    samp_pts["final_rho"],
                    s=18,
                    alpha=0.45,
                    color=_SAMPLER_COLORS.get(samp, "#444444"),
                    edgecolors="none",
                )
            cell = sub[sub["sampler"] == samp]
            if not cell.empty and np.isfinite(cell["wall"].values[0]):
                ax.scatter(
                    cell["wall"].values[0],
                    cell["mean_rho"].values[0],
                    s=85,
                    color=_SAMPLER_COLORS.get(samp, "#444444"),
                    edgecolor="black",
                    linewidth=0.7,
                    label=_SAMPLER_LABELS.get(samp, samp),
                    zorder=5,
                )
        # Overlay the fit if defensible (R^2 > 0.4)
        fit = next((f for f in fit_rows if f["task"] == task), None)
        if fit is not None and fit.get("ok") and fit.get("r2", 0) > 0.4:
            tt = np.geomspace(
                max(sub["wall"].min(), 1e-3),
                sub["wall"].max(),
                200,
            )
            yy = _power_law(tt, fit["a"], fit["b"], fit["c"])
            ax.plot(
                tt,
                yy,
                color="black",
                linestyle="--",
                linewidth=1.2,
                alpha=0.7,
                label=f"fit: b={fit['b']:+.2f} (R^2={fit['r2']:.2f})",
                zorder=4,
            )
        ax.set_xscale("log")
        # Use symlog y-axis for narrow-attractor tasks where rho range spans
        # >100 units (otherwise the +25 winners get crushed against the floor).
        rho_range = sub["mean_rho"].max() - sub["mean_rho"].min()
        if rho_range > 50:
            ax.set_yscale("symlog", linthresh=1.0)
        ax.axhline(0.0, color="black", linewidth=0.6, linestyle=":", alpha=0.5)
        ax.set_title(_TASK_LABEL.get(task, task))
        ax.set_xlabel("warm wall-clock per call (s, log)")
        ax.set_ylabel("rho (mean over N=4 seeds)")
        ax.grid(True, alpha=0.25)

    # Dedicated legend panel
    legend_ax = axes[len(_TASK_ORDER)]
    legend_ax.axis("off")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=_SAMPLER_COLORS[s],
            markeredgecolor="black",
            markersize=10,
            label=_SAMPLER_LABELS[s],
        )
        for s in _SAMPLER_ORDER
    ]
    handles.append(
        plt.Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="power-law fit (R^2 > 0.4)",
        )
    )
    legend_ax.legend(handles=handles, loc="center", fontsize=10, frameon=True, title="Samplers")

    # Hide any other unused axes
    for j in range(len(_TASK_ORDER) + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Inference-time compute scaling, per task (M5 Pro local, warm wall-clock)",
        fontsize=12,
    )
    fig.tight_layout()
    out_fig1 = ROOT / "paper/figures/scaling_laws_per_task.png"
    fig.savefig(out_fig1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_fig1}")

    # ------------------------------------------------------------------
    # Figure 2: fitted exponents bar chart, per task with CI
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.0))
    tasks_ok = [f for f in fit_rows if f.get("ok")]
    xs = np.arange(len(tasks_ok))
    bs = [f["b"] for f in tasks_ok]
    los = [max(f["b"] - f["b_ci_low"], 0.0) for f in tasks_ok]
    his = [max(f["b_ci_high"] - f["b"], 0.0) for f in tasks_ok]
    # Bar colour by task class; alpha encodes fit quality (R^2)
    bar_colors = [
        "#5d8a4c" if "smooth" in _TASK_LABEL.get(f["task"], "") else "#c44e52" for f in tasks_ok
    ]
    bar_alphas = [0.95 if f["r2"] >= 0.7 else (0.55 if f["r2"] >= 0.3 else 0.25) for f in tasks_ok]
    bars = ax.bar(
        xs,
        bs,
        yerr=[los, his],
        color=bar_colors,
        edgecolor="black",
        linewidth=0.8,
        capsize=4,
    )
    for bar, a in zip(bars, bar_alphas, strict=True):
        bar.set_alpha(a)
    for i, f in enumerate(tasks_ok):
        # Annotation: R^2 + boundary-hit + sign-of-b consistency.
        # Position just above the zero line (above the negative-going bars).
        annot = f"R$^2$={f['r2']:.2f}"
        if f.get("boundary_hit"):
            annot += "  (bound hit)"
        elif f["r2"] < 0.3:
            annot += "  (weak fit)"
        ax.text(
            xs[i],
            0.6,
            annot,
            ha="center",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.9),
        )
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [_TASK_LABEL.get(f["task"], f["task"]) for f in tasks_ok],
        rotation=20,
        ha="right",
        fontsize=9,
    )
    ax.set_ylabel("fitted exponent b  (rho ~ a t^b + c)")
    ax.set_title(
        "Power-law exponent of rho with warm wall-clock, per task\n"
        "(green = smooth-dynamics; red = narrow-attractor; "
        "alpha = fit quality; error bars = 95% bootstrap CI over the 9 samplers)"
    )
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_ylim(-3.4, 1.6)
    fig.tight_layout()
    out_fig2 = ROOT / "paper/figures/scaling_laws_exponents.png"
    fig.savefig(out_fig2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_fig2}")


if __name__ == "__main__":
    main()

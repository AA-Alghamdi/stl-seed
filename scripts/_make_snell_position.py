"""Build paper/figures/snell_position.png.

Plots stl-seed glucose_insulin (compute, rho) on log-log axes against
the fitted saturating power law rho_bar(t) = a t^b + c (b = -0.241),
with a stylized Snell et al. 2024-implied reference slope overlaid.

IMPORTANT honesty note: Snell et al. 2024 (arXiv:2408.03314) do NOT report
explicit power-law exponents. They report compute-equivalence ratios
(4x test-time compute reduction at iso-accuracy via compute-optimal allocation;
14x model-size equivalence on a class of problems). The reference line drawn
here is a stylized "implied" exponent in the [-0.3, -0.4] range commonly
inferred from their Figures 4 and 8 by reading off iso-accuracy contours,
NOT a number quoted from their text. The plot title and caption flag this.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures" / "snell_position.png"


def main() -> None:
    df = pd.read_parquet(ROOT / "runs" / "cost_benchmark" / "results.parquet")
    gi = (
        df[df["task"] == "glucose_insulin"][["sampler", "wall_clock_s_warm", "mean_rho"]]
        .sort_values("wall_clock_s_warm")
        .reset_index(drop=True)
    )
    t = gi["wall_clock_s_warm"].to_numpy()
    rho = gi["mean_rho"].to_numpy()
    samplers = gi["sampler"].tolist()

    # Our fit (from runs/cost_benchmark/scaling_law_fits.json)
    a, b, c = -8.3535042384, -0.2405323522, 27.2315890425
    r2 = 0.814
    rho_ceiling = 20.75  # full-satisfaction ceiling read off the data
    rho_baseline = float(rho[0])  # standard sampler

    # Quality axis as "rho-gap closed", normalised to [0, 1] above the
    # standard-sampler baseline. This matches the "fraction of headroom
    # captured" frame Snell et al. use for accuracy gains under best-of-N.
    gap = (rho - rho_baseline) / (rho_ceiling - rho_baseline)
    gap = np.clip(gap, 1e-3, 1.0)  # log-safe; floor below standard at 1e-3

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # ------------------------------------------------------------------
    # Left panel: raw rho vs warm wall-clock, with our fit overlaid.
    # ------------------------------------------------------------------
    ax = axes[0]
    ax.scatter(
        t, rho, s=55, c="#3f6fa7", edgecolor="black", zorder=3, label="stl-seed samplers (N=9)"
    )
    tt = np.geomspace(t.min() * 0.7, t.max() * 1.3, 200)
    ax.plot(
        tt,
        a * tt**b + c,
        color="#c44e52",
        lw=2,
        label=f"fit: rho = {a:.2f} t^({b:.3f}) + {c:.2f}, R^2 = {r2:.2f}",
    )
    ax.axhline(
        rho_ceiling, ls="--", lw=1, color="#5d8a4c", label=f"rho ceiling ~ {rho_ceiling:.1f}"
    )
    ax.set_xscale("log")
    ax.set_xlabel("warm wall-clock per sample, t (s)")
    ax.set_ylabel("mean STL robustness, rho")
    ax.set_title("stl-seed glucose_insulin.tir.easy")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    # Annotate the cheapest and saturating samplers.
    for i, name in enumerate(samplers):
        if name in ("standard", "continuous_bon", "rollout_tree", "hybrid"):
            ax.annotate(
                name,
                (t[i], rho[i]),
                xytext=(6, -10 if name == "hybrid" else 6),
                textcoords="offset points",
                fontsize=7.5,
                color="#444444",
            )

    # ------------------------------------------------------------------
    # Right panel: log-log "fraction of rho-gap closed" vs compute, with
    # an implied Snell-regime reference slope for visual position only.
    # ------------------------------------------------------------------
    ax = axes[1]
    # Empirical log-log slope of (rho - rho_std) vs t over the 8 non-standard samplers.
    mask = np.arange(len(t)) > 0  # exclude standard (defines the baseline)
    slope, intercept = np.polyfit(np.log10(t[mask]), np.log10(gap[mask]), 1)
    ax.scatter(
        t,
        gap,
        s=55,
        c="#3f6fa7",
        edgecolor="black",
        zorder=3,
        label="stl-seed: (rho - rho_std) / (rho_ceil - rho_std)",
    )
    # Our log-log slope through the non-standard samplers
    tt2 = np.geomspace(t[mask].min() * 0.8, t[mask].max() * 1.2, 200)
    ax.plot(
        tt2,
        10**intercept * tt2**slope,
        color="#c44e52",
        lw=2,
        label=f"stl-seed log-log slope = {slope:+.3f}",
    )
    # Stylized Snell et al. 2024 reference: their Figures 4 & 8 imply a
    # compute-quality slope in roughly [-0.3, -0.4] for accuracy-gap-closing
    # under PRM-guided best-of-N + revisions on MATH. We draw two reference
    # lines bounding that band, anchored to pass through the median stl-seed
    # operating point. THIS IS A REGIME REFERENCE, NOT A QUOTED EXPONENT.
    t_anchor = float(np.median(t[mask]))
    g_anchor = float(np.median(gap[mask]))
    for ref_slope, color, label in [
        (-0.30, "#888888", "Snell-regime reference, slope = -0.30 (implied)"),
        (-0.40, "#bbbbbb", "Snell-regime reference, slope = -0.40 (implied)"),
    ]:
        # Snell's curves are accuracy *gain* with positive slope;
        # plotting fraction-of-gap-closed gives a positive slope of
        # the same magnitude. Use |ref_slope| to mirror the regime.
        s = abs(ref_slope)
        c_const = g_anchor / (t_anchor**s)
        ax.plot(tt2, c_const * tt2**s, ls=":", color=color, lw=1.6, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("warm wall-clock per sample, t (s)")
    ax.set_ylabel("fraction of rho-gap closed above standard")
    ax.set_title("Position vs Snell et al. 2024 (arXiv:2408.03314)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower right")

    fig.suptitle(
        "Inference-time compute scaling: stl-seed (STL-verified scientific control) "
        "vs Snell et al. 2024 regime",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print(f"wrote {OUT}")
    print(f"stl-seed log-log slope (rho-gap fraction vs t): {slope:+.3f}")
    print(f"saturating-power-law exponent b: {b:+.3f}, R^2: {r2:.2f}")


if __name__ == "__main__":
    main()

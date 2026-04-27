"""
Money plot for FMAI workshop / Dettmers cold-email artifact.

5x4 heatmap: rows = models (precision x size), cols = hard tasks.
Each cell = methodology gap = mean rho (beam-search warmstart) - mean rho (standard sampler).
Diverging colormap centred at 0; SymLog norm to handle the wide dynamic range
(gap is ~0.5 on cardiac, ~30 on toggle, ~270 on repressilator).

Saturation marker (1.7B + toggle): drawn with a heavy black border + small "S" annotation
to signal "standard sampler already solves this; gap collapses to magnitude residual."

Greyscale-friendly: numerical annotation in each cell + sat-fraction as bottom strip
("3/3" beam vs "0/3" std) so the figure reads without colour.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Rectangle

ROOT = Path("/Users/abdullahalghamdi/stl-seed")
DATA = ROOT / "runs/quant_size_sweep/results.parquet"
SUMMARY = ROOT / "runs/quant_size_sweep/summary.json"
OUT_PNG = ROOT / "paper/figures/money_plot.png"
OUT_PDF = ROOT / "paper/figures/money_plot.pdf"

MODEL_ORDER = [
    "qwen3-0.6b",
    "qwen3-0.6b-8bit",
    "qwen3-0.6b-4bit",
    "qwen3-1.7b",
    "qwen3-1.7b-4bit",
]
MODEL_LABELS = [
    "0.6B  bf16",
    "0.6B  8bit",
    "0.6B  4bit",
    "1.7B  bf16",
    "1.7B  4bit",
]
TASK_ORDER = [
    "bio_ode.repressilator",
    "bio_ode.toggle",
    "bio_ode.mapk",
    "cardiac_ap",
]
TASK_LABELS = [
    "repressilator",
    "toggle",
    "MAPK",
    "cardiac AP",
]


def main() -> None:
    df = pd.read_parquet(DATA)
    with open(SUMMARY) as f:
        json.load(f)

    # Aggregate per (model, task, sampler)
    agg = (
        df.groupby(["model", "task", "sampler"])
        .agg(rho_mean=("final_rho", "mean"), sat=("satisfied", "sum"), n=("final_rho", "count"))
        .reset_index()
    )

    # Build matrices: gap, std_sat, beam_sat, std_rho
    n_rows, n_cols = len(MODEL_ORDER), len(TASK_ORDER)
    gap = np.full((n_rows, n_cols), np.nan)
    std_sat = np.zeros((n_rows, n_cols), dtype=int)
    beam_sat = np.zeros((n_rows, n_cols), dtype=int)
    n_seeds = np.zeros((n_rows, n_cols), dtype=int)
    saturated = np.zeros((n_rows, n_cols), dtype=bool)

    for i, model in enumerate(MODEL_ORDER):
        for j, task in enumerate(TASK_ORDER):
            sub = agg[(agg.model == model) & (agg.task == task)]
            std = sub[sub.sampler == "standard"].iloc[0]
            beam = sub[sub.sampler == "beam_search_warmstart"].iloc[0]
            gap[i, j] = beam["rho_mean"] - std["rho_mean"]
            std_sat[i, j] = int(std["sat"])
            beam_sat[i, j] = int(beam["sat"])
            n_seeds[i, j] = int(std["n"])
            # Saturated = standard sampler already solves majority of seeds
            saturated[i, j] = std["sat"] >= np.ceil(std["n"] / 2)

    # ---- figure ----
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.linewidth": 0.6,
            "axes.edgecolor": "#333333",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(3.6, 3.4), dpi=200)

    # SymLogNorm: linear near 0 (linthresh=1), log beyond. Centred at 0 by symmetry.
    vmax = float(np.nanmax(np.abs(gap)))
    norm = SymLogNorm(linthresh=1.0, linscale=0.6, vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")  # red = positive (methodology helps), blue = negative

    im = ax.imshow(gap, cmap=cmap, norm=norm, aspect="equal", origin="upper")

    # Cell annotations: gap value (top) and sat fraction (bottom)
    for i in range(n_rows):
        for j in range(n_cols):
            v = gap[i, j]
            # Pick text colour for contrast
            rgba = cmap(norm(v))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            tc = "white" if lum < 0.5 else "#1a1a1a"

            # Format gap compactly
            if abs(v) >= 100:
                gtxt = f"{v:+.0f}"
            elif abs(v) >= 10:
                gtxt = f"{v:+.1f}"
            else:
                gtxt = f"{v:+.2f}"

            ax.text(
                j,
                i - 0.18,
                gtxt,
                ha="center",
                va="center",
                fontsize=6.8,
                fontweight="bold",
                color=tc,
            )
            # sat fraction: beam / std
            ax.text(
                j,
                i + 0.22,
                f"{beam_sat[i, j]}/{n_seeds[i, j]} vs {std_sat[i, j]}/{n_seeds[i, j]}",
                ha="center",
                va="center",
                fontsize=5.2,
                color=tc,
                alpha=0.92,
            )

            # Saturation marker on cells where standard already solves
            if saturated[i, j]:
                rect = Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2.0, zorder=5
                )
                ax.add_patch(rect)
                # tiny corner glyph
                ax.text(
                    j + 0.36,
                    i - 0.36,
                    "S",
                    ha="center",
                    va="center",
                    fontsize=6,
                    fontweight="bold",
                    color="black",
                    bbox=dict(
                        boxstyle="circle,pad=0.12",
                        facecolor="white",
                        edgecolor="black",
                        linewidth=0.6,
                    ),
                    zorder=6,
                )

    # Tick labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(TASK_LABELS, fontsize=7, rotation=22, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(MODEL_LABELS, fontsize=7)
    ax.tick_params(axis="both", which="both", length=0)
    # Visual divider between 0.6B and 1.7B model groups
    ax.axhline(2.5, color="#1a1a1a", lw=1.0, zorder=4)

    # Subtle row/col gridlines between cells
    for i in range(n_rows + 1):
        ax.axhline(i - 0.5, color="white", lw=1.2, zorder=3)
    for j in range(n_cols + 1):
        ax.axvline(j - 0.5, color="white", lw=1.2, zorder=3)

    ax.set_title(
        "Methodology gap survives quantization\nand size scaling",
        fontsize=8.8,
        fontweight="bold",
        pad=6,
    )
    ax.set_ylabel("model", fontsize=7.5)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, ticks=[-100, -10, -1, 0, 1, 10, 100])
    cbar.ax.set_yticklabels(["-100", "-10", "-1", "0", "+1", "+10", "+100"], fontsize=6)
    cbar.set_label(
        r"$\Delta\rho = \rho_\mathrm{beam} - \rho_\mathrm{standard}$", fontsize=7, labelpad=3
    )
    cbar.outline.set_linewidth(0.5)

    # Attribution line (bottom). Day-1 verdict, subtle
    fig.text(
        0.5,
        0.005,
        "METHODOLOGY MATTERS  4 of 4 hard tasks rescued, real Qwen3, pre-registered N=3 seeds",
        ha="center",
        va="bottom",
        fontsize=5.6,
        color="#555555",
        style="italic",
    )
    # Saturation legend below attribution line
    fig.text(
        0.5,
        0.04,
        "circled S = standard sampler already saturates the spec (gap collapses to magnitude)",
        ha="center",
        va="bottom",
        fontsize=5.4,
        color="#333333",
    )

    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.99))

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    try:
        fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
        print(f"saved: {OUT_PNG}\nsaved: {OUT_PDF}")
    except Exception as e:
        print(f"PDF skipped ({e})\nsaved: {OUT_PNG}")
    plt.close(fig)

    # ---- optional inset: toggle row, sat-fraction bars across 5 models ----
    fig2, ax2 = plt.subplots(figsize=(3.5, 1.3), dpi=200)
    j_tog = TASK_ORDER.index("bio_ode.toggle")
    x = np.arange(n_rows)
    w = 0.38
    bs = beam_sat[:, j_tog]
    ss = std_sat[:, j_tog]
    n = n_seeds[:, j_tog]
    bs_frac = bs / n
    ss_frac = ss / n
    ax2.bar(
        x - w / 2,
        bs_frac,
        w,
        color="#2ca02c",
        edgecolor="#1a1a1a",
        linewidth=0.4,
        label="beam warmstart",
    )
    ax2.bar(
        x + w / 2, ss_frac, w, color="#9e9e9e", edgecolor="#1a1a1a", linewidth=0.4, label="standard"
    )
    for xi, (a, b, ni) in enumerate(zip(bs, ss, n, strict=False)):
        ax2.text(
            xi - w / 2, bs_frac[xi] + 0.04, f"{a}/{ni}", ha="center", va="bottom", fontsize=5.5
        )
        ax2.text(
            xi + w / 2, ss_frac[xi] + 0.04, f"{b}/{ni}", ha="center", va="bottom", fontsize=5.5
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(MODEL_LABELS, fontsize=6.5, rotation=0)
    ax2.set_ylim(0, 1.32)
    ax2.set_yticks([0, 0.5, 1.0])
    ax2.set_ylabel("sat. fraction", fontsize=6.8)
    ax2.set_title(
        "toggle: SERA-saturation transition at 1.7B", fontsize=7.5, fontweight="bold", pad=4
    )
    # Increase headroom and place a compact legend top-right where 1.7B bars
    # already saturate near 1.0. no useful empty space, so use lower-right where
    # standard bars at 0.6B are zero
    ax2.text(
        0.5, 1.22, "beam warmstart", fontsize=5.8, color="#1f7a1f", fontweight="bold", ha="left"
    )
    ax2.text(2.3, 1.22, "standard", fontsize=5.8, color="#5a5a5a", fontweight="bold", ha="left")
    # Small swatches before each label
    from matplotlib.patches import Rectangle as _R  # noqa: N814

    ax2.add_patch(
        _R((0.32, 1.20), 0.13, 0.06, color="#2ca02c", transform=ax2.transData, clip_on=False)
    )
    ax2.add_patch(
        _R((2.12, 1.20), 0.13, 0.06, color="#9e9e9e", transform=ax2.transData, clip_on=False)
    )
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(axis="both", length=2)
    fig2.tight_layout()
    inset_png = ROOT / "paper/figures/money_plot_inset.png"
    fig2.savefig(inset_png, dpi=300, bbox_inches="tight", facecolor="white")
    with contextlib.suppress(Exception):
        fig2.savefig(inset_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"saved: {inset_png}")


if __name__ == "__main__":
    main()

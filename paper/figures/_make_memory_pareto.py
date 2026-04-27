"""Memory-vs-quality Pareto plot for the quant-size sweep.

Reads `runs/quant_size_sweep/results.parquet` and produces
`paper/figures/memory_pareto.png`.

Memory model (back-of-envelope, see paper/memory_pareto.md, Section 1):
    peak_GB = weight_GB + activation_GB
where weight_GB = du -sh of the cached MLX model directory and
activation_GB approximates the chunked-scoring working set:
    activation_GB = chunk_size * T * |vocab| * 2 bytes (fp16 logits)
                  = 16 * 400 * 151936 * 2 bytes
                  ~= 1.81 GB

We add a 1.2x slop factor for KV-cache and intermediate activations to
get an "engineering" estimate, yielding ~2.17 GB activation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "runs" / "quant_size_sweep" / "results.parquet"
OUT_PNG = ROOT / "paper" / "figures" / "memory_pareto.png"

# Disk-measured weight footprint (du -sh on cached HF blobs, 2026-04-26).
WEIGHT_GB = {
    "qwen3-0.6b": 1.1,  # bf16
    "qwen3-0.6b-8bit": 0.619,
    "qwen3-0.6b-4bit": 0.335,
    "qwen3-1.7b": 3.2,  # bf16
    "qwen3-1.7b-4bit": 0.938,
}

# Activation working set during MLXLLMProposal._score_batch.
CHUNK_SIZE = 16
T_HORIZON = 400
VOCAB = 151_936
BYTES_FP16 = 2
SLOP = 1.2  # KV cache + intermediates

ACTIVATION_GB_PER_MODEL = {
    "qwen3-0.6b": CHUNK_SIZE * T_HORIZON * VOCAB * BYTES_FP16 / 1e9 * SLOP,
    "qwen3-0.6b-8bit": CHUNK_SIZE * T_HORIZON * VOCAB * BYTES_FP16 / 1e9 * SLOP,
    "qwen3-0.6b-4bit": CHUNK_SIZE * T_HORIZON * VOCAB * BYTES_FP16 / 1e9 * SLOP,
    "qwen3-1.7b": CHUNK_SIZE * T_HORIZON * VOCAB * BYTES_FP16 / 1e9 * SLOP,
    "qwen3-1.7b-4bit": CHUNK_SIZE * T_HORIZON * VOCAB * BYTES_FP16 / 1e9 * SLOP,
}

# Pretty labels.
LABEL = {
    "qwen3-0.6b": "0.6B bf16",
    "qwen3-0.6b-8bit": "0.6B 8-bit",
    "qwen3-0.6b-4bit": "0.6B 4-bit",
    "qwen3-1.7b": "1.7B bf16",
    "qwen3-1.7b-4bit": "1.7B 4-bit",
}


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, sub in df.groupby("model"):
        std = sub[sub.sampler == "standard"]
        beam = sub[sub.sampler == "beam_search_warmstart"]
        rows.append(
            {
                "model": model,
                "rho_mean_standard": std.final_rho.mean(),
                "rho_mean_beam": beam.final_rho.mean(),
                "sat_standard": int(std.satisfied.sum()),
                "sat_beam": int(beam.satisfied.sum()),
                "n_standard": len(std),
                "n_beam": len(beam),
                "wall_clock_total_s": sub.wall_clock_s.sum(),
            }
        )
    return pd.DataFrame(rows).set_index("model")


def main() -> None:
    df = pd.read_parquet(PARQUET)
    agg = aggregate(df)

    # Memory column.
    agg["weight_GB"] = agg.index.map(WEIGHT_GB)
    agg["activation_GB"] = agg.index.map(ACTIVATION_GB_PER_MODEL)
    agg["peak_GB"] = agg["weight_GB"] + agg["activation_GB"]

    # Quality column = sat-fraction on the standard sampler (most informative
    # for the methodology-gap thesis: which models can satisfy specs *without*
    # beam-search rescue?). Beam-search column is unanimous (12/12 on every
    # model) so it carries no Pareto signal.
    agg["sat_frac_standard"] = agg["sat_standard"] / agg["n_standard"]

    print(
        agg[
            [
                "weight_GB",
                "activation_GB",
                "peak_GB",
                "rho_mean_standard",
                "sat_standard",
                "sat_frac_standard",
            ]
        ]
    )

    # Plot.
    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    # Color by precision; size by params.
    palette = {
        "bf16": "#1b4f72",
        "8bit": "#7d6608",
        "4bit": "#cb4335",
    }

    def precision_of(name: str) -> str:
        if "4bit" in name:
            return "4bit"
        if "8bit" in name:
            return "8bit"
        return "bf16"

    def params_of(name: str) -> float:
        return 1.7 if "1.7b" in name else 0.6

    for model in agg.index:
        prec = precision_of(model)
        params = params_of(model)
        x = agg.loc[model, "peak_GB"]
        y = agg.loc[model, "sat_frac_standard"]
        ax.scatter(
            x,
            y,
            s=80 + 120 * params,
            c=palette[prec],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
            zorder=3,
        )
        # Label offset chosen per-point for readability.
        dx, dy = 0.05, 0.02
        if model == "qwen3-0.6b-4bit":
            dy = -0.04
        if model == "qwen3-1.7b":
            dx = -0.02
            dy = 0.02
        ax.annotate(
            LABEL[model],
            (x, y),
            xytext=(x * (1 + dx), y + dy),
            fontsize=9.5,
            zorder=4,
        )

    # Highlight bf16 baseline (1.7B bf16, the highest-quality / highest-cost
    # anchor) and the 4-bit-1.7B sweet spot.
    sweet = agg.loc["qwen3-1.7b-4bit"]
    baseline = agg.loc["qwen3-1.7b"]
    ax.scatter(
        [sweet.peak_GB],
        [sweet.sat_frac_standard],
        s=320,
        facecolor="none",
        edgecolor="#cb4335",
        linewidth=2.2,
        zorder=2,
        label="sweet spot (1.7B 4-bit)",
    )
    ax.scatter(
        [baseline.peak_GB],
        [baseline.sat_frac_standard],
        s=320,
        facecolor="none",
        edgecolor="#1b4f72",
        linewidth=2.2,
        linestyle="--",
        zorder=2,
        label="bf16 baseline (1.7B)",
    )

    # Pareto frontier: lower-left dominates. Sort by memory ascending, take
    # running-max of sat-frac.
    ordered = agg.sort_values("peak_GB")
    frontier_x, frontier_y = [], []
    best = -np.inf
    for _, row in ordered.iterrows():
        if row.sat_frac_standard >= best:
            frontier_x.append(row.peak_GB)
            frontier_y.append(row.sat_frac_standard)
            best = row.sat_frac_standard
    ax.plot(
        frontier_x,
        frontier_y,
        color="grey",
        linestyle=":",
        linewidth=1.2,
        label="Pareto frontier",
        zorder=1,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Peak memory during scoring (GB, log scale)")
    ax.set_ylabel("Sat-fraction, standard sampler (out of 12)")
    ax.set_title(
        "Memory vs quality Pareto, Qwen3 quant x size sweep\n"
        "(beam-search rescues 12/12 on every model; standard-sampler "
        "quality shown)"
    )
    ax.set_ylim(-0.05, 0.4)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160)
    print(f"wrote {OUT_PNG}")

    # Stash aggregated table for the markdown.
    agg.to_csv(OUT_PNG.with_suffix(".csv"))
    print(f"wrote {OUT_PNG.with_suffix('.csv')}")


if __name__ == "__main__":
    main()

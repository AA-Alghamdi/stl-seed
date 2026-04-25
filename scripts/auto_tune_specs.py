"""Run STL spec threshold auto-tuning across the registered specs.

Output:

* ``paper/REDACTED.md`` — markdown table comparing the
  hand-set thresholds to the auto-tuned recommendations, plus per-policy
  rho summaries at the recommended thresholds.
* ``paper/figures/spec_calibration_<task>.png`` — discriminability plot
  per spec (1-D line plot if a single threshold is tuned; 2-D heatmap
  if two are tuned).

Performance: trajectories are cached *per policy* in
``auto_tune_spec_thresholds``, so the cost is dominated by simulation
(O(n_policies * n_trajectories)), not by the threshold sweep
(O(n_combinations * n_policies * n_trajectories), where each evaluation
is just a vmap over cached states). On an M5 Pro the full 6-spec sweep
runs in roughly 2-4 minutes.

REDACTED firewall: every search range is constructed inline below from the
literature-derived plausibility bands documented in
``stl_seed.specs.{bio_ode,glucose_insulin}_specs``. No REDACTED artifact is
used and no run-time mutation of the registered specs is performed
(the script emits *recommendations* — adoption is a separate v0.2 step).
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import matplotlib

matplotlib.use("Agg")  # headless backend for CI / batch runs
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from stl_seed.generation.policies import (  # noqa: E402
    HeuristicPolicy,
    RandomPolicy,
)
from stl_seed.specs import REGISTRY  # noqa: E402
from stl_seed.specs.calibration import (  # noqa: E402
    AutoTuneResult,
    auto_tune_spec_thresholds,
    extract_threshold_placeholders,
)
from stl_seed.tasks.bio_ode import (  # noqa: E402
    MAPKSimulator,
    RepressilatorSimulator,
    ToggleSimulator,
    default_mapk_initial_state,
    default_repressilator_initial_state,
    default_toggle_initial_state,
)
from stl_seed.tasks.bio_ode_params import (  # noqa: E402
    MAPKParams,
    RepressilatorParams,
    ToggleParams,
)
from stl_seed.tasks.glucose_insulin import (  # noqa: E402
    BergmanParams,
    GlucoseInsulinSimulator,
    default_normal_subject_initial_state,
    single_meal_schedule,
)

# ---------------------------------------------------------------------------
# Per-spec configuration: simulator, policies, search space.
# ---------------------------------------------------------------------------


@dataclass
class _SpecConfig:
    spec_key: str
    simulator: Any
    sim_params: Any
    initial_state: Any
    aux: dict | None
    action_dim: int
    # Map base_name -> candidate values. Documentation of the choice of
    # range lives in the call-site below.
    threshold_search_space: dict[str, list[float]]
    # Map policy_name -> Policy. The runner's `HeuristicPolicy` and
    # `RandomPolicy` cover the textbook baselines.
    policy_factory: Any  # callable() -> dict[str, Policy]


def _gi_random() -> RandomPolicy:
    return RandomPolicy(action_dim=1, action_low=0.0, action_high=5.0)


def _bio_random(action_dim: int) -> RandomPolicy:
    return RandomPolicy(action_dim=action_dim, action_low=0.0, action_high=1.0)


def _build_configs() -> list[_SpecConfig]:
    """Construct the per-spec auto-tune configuration table."""
    configs: list[_SpecConfig] = []

    # ------------------------------------------------------------------
    # glucose_insulin.tir.easy — sweep G_below_180 in 140..250 mg/dL.
    # Range derived from ADA 2024 Standards: 140 mg/dL is the
    # 2-h-postprandial target; 250 mg/dL is Battelino 2019 "Level 2"
    # severe hyperglycaemia. The hand-set value is 180.
    # ------------------------------------------------------------------
    gi_params = BergmanParams()
    gi_init = np.asarray(default_normal_subject_initial_state(gi_params))
    gi_aux = {"meal_schedule": single_meal_schedule(onset_min=15.0, carb_grams=50.0)}
    configs.append(
        _SpecConfig(
            spec_key="glucose_insulin.tir.easy",
            simulator=GlucoseInsulinSimulator(),
            sim_params=gi_params,
            initial_state=gi_init,
            aux=gi_aux,
            action_dim=1,
            threshold_search_space={
                "G_below_180": [140.0, 160.0, 180.0, 200.0, 220.0, 250.0],
                "G_above_70": [60.0, 70.0, 80.0, 90.0],
            },
            policy_factory=lambda: {
                "pid": HeuristicPolicy("glucose_insulin"),
                "random": _gi_random(),
            },
        )
    )

    # ------------------------------------------------------------------
    # glucose_insulin.no_hypo.medium — sweep TIR-upper + severe-hypo bound.
    # Severe-hypo bound: 50..70 mg/dL, bracketing the Battelino 2019
    # Level-1 (70) and Level-2 (54) lines.
    # ------------------------------------------------------------------
    configs.append(
        _SpecConfig(
            spec_key="glucose_insulin.no_hypo.medium",
            simulator=GlucoseInsulinSimulator(),
            sim_params=gi_params,
            initial_state=gi_init,
            aux=gi_aux,
            action_dim=1,
            threshold_search_space={
                "G_below_180": [160.0, 180.0, 200.0, 220.0],
                "G_severe_hypo": [50.0, 54.0, 60.0, 70.0],
            },
            policy_factory=lambda: {
                "pid": HeuristicPolicy("glucose_insulin"),
                "random": _gi_random(),
            },
        )
    )

    # ------------------------------------------------------------------
    # glucose_insulin.dawn.hard — sweep insulin-bolus-min in 20..80 µU/mL.
    # Polonsky 1988 Fig. 1: postprandial insulin peak 40-80 µU/mL in
    # healthy subjects. 20 is sub-clinical; 80 is the upper end. The
    # hand-set value is 40.
    # ------------------------------------------------------------------
    configs.append(
        _SpecConfig(
            spec_key="glucose_insulin.dawn.hard",
            simulator=GlucoseInsulinSimulator(),
            sim_params=gi_params,
            initial_state=gi_init,
            aux=gi_aux,
            action_dim=1,
            threshold_search_space={
                "I_bolus": [20.0, 40.0, 60.0, 80.0],
                "G_post_below_140": [120.0, 140.0, 160.0, 180.0],
            },
            policy_factory=lambda: {
                "pid": HeuristicPolicy("glucose_insulin"),
                "random": _gi_random(),
            },
        )
    )

    # ------------------------------------------------------------------
    # bio_ode.repressilator.easy — sweep P_HIGH (p1) in 100..600 nM.
    # Müller 2007 Table 1 K_LacI ~ 30-60 nM; "fully on" is canonically
    # 5-10x K, i.e., 150-600 nM. Hand-set 250.
    # ------------------------------------------------------------------
    rep_params = RepressilatorParams()
    rep_init = np.asarray(default_repressilator_initial_state(rep_params))
    configs.append(
        _SpecConfig(
            spec_key="bio_ode.repressilator.easy",
            simulator=RepressilatorSimulator(),
            sim_params=rep_params,
            initial_state=rep_init,
            aux=None,
            action_dim=3,
            threshold_search_space={
                "p1": [100.0, 200.0, 300.0, 400.0, 600.0],
                "p2": [10.0, 25.0, 50.0, 75.0],
            },
            policy_factory=lambda: {
                "heuristic": HeuristicPolicy("bio_ode.repressilator"),
                "random": _bio_random(3),
            },
        )
    )

    # ------------------------------------------------------------------
    # bio_ode.toggle.medium — sweep HIGH (x1) in 100..300 nM, LOW in 10..50.
    # Gardner 2000 Fig. 5a stable states ~200 (high) / ~20 (low).
    # ------------------------------------------------------------------
    tog_params = ToggleParams()
    tog_init = np.asarray(default_toggle_initial_state(tog_params))
    configs.append(
        _SpecConfig(
            spec_key="bio_ode.toggle.medium",
            simulator=ToggleSimulator(),
            sim_params=tog_params,
            initial_state=tog_init,
            aux=None,
            action_dim=2,
            threshold_search_space={
                "x1": [100.0, 150.0, 200.0, 250.0],
                "x2": [10.0, 20.0, 30.0, 50.0],
            },
            policy_factory=lambda: {
                "heuristic": HeuristicPolicy("bio_ode.toggle"),
                "random": _bio_random(2),
            },
        )
    )

    # ------------------------------------------------------------------
    # bio_ode.mapk.hard — sweep MAPK_PEAK (channel 2 in current spec).
    # Huang & Ferrell 1996 Fig. 4: EC50 fraction 0.5; sweeping 0.2..0.8
    # spans the Hill-curve sigmoid. Hand-set 0.5.
    # ------------------------------------------------------------------
    mapk_params = MAPKParams()
    mapk_init = np.asarray(default_mapk_initial_state(mapk_params))
    configs.append(
        _SpecConfig(
            spec_key="bio_ode.mapk.hard",
            simulator=MAPKSimulator(),
            sim_params=mapk_params,
            initial_state=mapk_init,
            aux=None,
            action_dim=1,
            threshold_search_space={
                "mapk_pp": [0.2, 0.4, 0.5, 0.6, 0.8],
                "mapk_pp_settle": [0.05, 0.10, 0.15, 0.20],
            },
            policy_factory=lambda: {
                "heuristic": HeuristicPolicy("bio_ode.mapk"),
                "random": _bio_random(1),
            },
        )
    )

    return configs


# ---------------------------------------------------------------------------
# Plotting.
# ---------------------------------------------------------------------------


def _plot_search_results(result: AutoTuneResult, spec_key: str, out_path: Path) -> None:
    """1-D line plot if one threshold is tuned; 2-D heatmap if two."""
    df = result.search_results.copy()
    threshold_cols = [
        c
        for c in df.columns
        if c
        not in {
            "metric_min",
            "metric_mean",
            "metric_max",
            "metric_aggregated",
        }
        and not c.startswith("metric_")
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    if len(threshold_cols) == 1:
        col = threshold_cols[0]
        df_sorted = df.sort_values(col)
        ax.plot(
            df_sorted[col],
            df_sorted["metric_aggregated"],
            marker="o",
            color="tab:blue",
        )
        ax.axvline(
            result.best_thresholds[col],
            color="tab:red",
            linestyle="--",
            label=f"best = {result.best_thresholds[col]:g}",
        )
        ax.set_xlabel(col)
        ax.set_ylabel("worst-case pairwise Wasserstein-1 of rho")
        ax.set_title(f"{spec_key}\nspec auto-tune (1 threshold)")
        ax.legend()
    elif len(threshold_cols) == 2:
        a, b = threshold_cols
        # Pivot to a heatmap.
        heat = df.pivot_table(index=b, columns=a, values="metric_aggregated", aggfunc="mean")
        im = ax.imshow(
            heat.values,
            origin="lower",
            aspect="auto",
            cmap="viridis",
        )
        ax.set_xticks(range(len(heat.columns)))
        ax.set_xticklabels([f"{v:g}" for v in heat.columns])
        ax.set_yticks(range(len(heat.index)))
        ax.set_yticklabels([f"{v:g}" for v in heat.index])
        ax.set_xlabel(a)
        ax.set_ylabel(b)
        # Mark the best.
        ax_best = (
            list(heat.columns).index(result.best_thresholds[a]),
            list(heat.index).index(result.best_thresholds[b]),
        )
        ax.scatter([ax_best[0]], [ax_best[1]], color="red", marker="*", s=200, label="best")
        fig.colorbar(im, ax=ax, label="worst-case pairwise Wasserstein-1 of rho")
        ax.set_title(f"{spec_key}\nspec auto-tune (2 thresholds)")
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            f"can't plot >2 thresholds ({len(threshold_cols)} tuned)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(spec_key)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown reporting.
# ---------------------------------------------------------------------------


def _format_results_markdown(
    results: Mapping[str, AutoTuneResult],
    elapsed_seconds: float,
) -> str:
    """Build the REDACTED.md content."""
    lines: list[str] = []
    lines.append("# STL Spec Auto-Tuning Results")
    lines.append("")
    lines.append(
        "Threshold values that maximise between-policy discriminability of "
        "the STL robustness margin rho. Discriminability is the worst-case "
        "pairwise 1-D Wasserstein distance between policies' rho "
        "distributions. Generated by `scripts/auto_tune_specs.py`."
    )
    lines.append("")
    lines.append(f"Total runtime: {elapsed_seconds:.1f} s.")
    lines.append("")
    lines.append("## Recommended threshold table")
    lines.append("")
    lines.append(
        "| Spec | Threshold | Hand-set | Auto-tuned | Discriminability gain (best vs worst sweep) |"
    )
    lines.append("|---|---|---|---|---|")
    for spec_key, result in results.items():
        df = result.search_results
        worst = float(df["metric_aggregated"].min())
        best = float(df["metric_aggregated"].max())
        for placeholder in result.placeholders:
            base = placeholder.base_name
            if base not in result.best_thresholds:
                continue
            hand = placeholder.current_value
            tuned = result.best_thresholds[base]
            gain = best - worst
            lines.append(
                f"| `{spec_key}` | `{base}` | {hand:g} | {tuned:g} | "
                f"{gain:.3g} (worst {worst:.3g} -> best {best:.3g}) |"
            )
    lines.append("")
    lines.append("## Per-policy rho summary at the auto-tuned thresholds")
    lines.append("")
    for spec_key, result in results.items():
        lines.append(f"### `{spec_key}`")
        lines.append("")
        lines.append("| Policy | n_finite | min | mean | std | max |")
        lines.append("|---|---|---|---|---|---|")
        for policy, stats in result.per_policy_rho_stats.items():
            lines.append(
                f"| `{policy}` | {int(stats['n_finite'])} | "
                f"{stats['min']:.3g} | {stats['mean']:.3g} | "
                f"{stats['std']:.3g} | {stats['max']:.3g} |"
            )
        lines.append("")
        meta = result.metadata
        lines.append(
            f"- metric: `{meta['discriminability_metric']}`, "
            f"aggregation: `{meta['aggregation']}`, "
            f"n_trajectories_per_policy: {meta['n_trajectories_per_policy']}, "
            f"n_combinations: {meta['n_combinations']}, "
            f"n_kept_per_policy: {meta['n_kept_per_policy']}, "
            f"n_failed_per_policy: {meta['n_failed_per_policy']}"
        )
        lines.append("")

    lines.append("## Adoption note")
    lines.append("")
    lines.append(
        "These are *recommendations*. Adoption (rewriting the literature-"
        "anchored thresholds in `bio_ode_specs.py` / `glucose_insulin_specs.py`) "
        "is a v0.2 decision and must be cross-checked against the cited "
        "biological / clinical sources. The current spec values remain in "
        "place for backwards compatibility."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=20,
        help="Trajectories per policy (default 20; literature-typical 200 floor "
        "for full-confidence runs).",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("paper/REDACTED.md"),
        help="Path to write the markdown summary.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("paper/figures"),
        help="Directory for per-spec discriminability plots.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="If set, restrict to a single spec key (e.g., glucose_insulin.tir.easy).",
    )
    args = parser.parse_args(argv)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    configs = _build_configs()
    if args.only is not None:
        configs = [c for c in configs if c.spec_key == args.only]
        if not configs:
            raise SystemExit(f"--only={args.only!r} matched no spec configs")

    key = jax.random.key(0)
    results: dict[str, AutoTuneResult] = {}
    t0 = time.time()
    for cfg in configs:
        spec = REGISTRY[cfg.spec_key]
        # Filter the search space down to predicates that actually exist
        # in the spec — guards against typos in `_build_configs`.
        ph_names = {p.base_name for p in extract_threshold_placeholders(spec)}
        search_space = {k: v for k, v in cfg.threshold_search_space.items() if k in ph_names}
        if not search_space:
            print(f"[skip] {cfg.spec_key}: no placeholders matched search space")
            continue
        policies = cfg.policy_factory()
        spec_t0 = time.time()
        result = auto_tune_spec_thresholds(
            simulator=cfg.simulator,
            spec_template=spec,
            threshold_search_space=search_space,
            policies=policies,
            initial_state=cfg.initial_state,
            sim_params=cfg.sim_params,
            aux=cfg.aux,
            action_dim=cfg.action_dim,
            n_trajectories_per_policy=args.n_trajectories,
            discriminability_metric="wasserstein",
            aggregation="worst",
            key=jax.random.fold_in(key, hash(cfg.spec_key) & 0xFFFF),
        )
        spec_dt = time.time() - spec_t0
        print(
            f"[{cfg.spec_key}] best={result.best_thresholds} "
            f"metric={result.best_metric_value:.3g} ({spec_dt:.1f}s)"
        )
        results[cfg.spec_key] = result
        # Plot.
        out_png = args.figures_dir / f"spec_calibration_{cfg.spec_key.replace('.', '_')}.png"
        try:
            _plot_search_results(result, cfg.spec_key, out_png)
        except Exception as exc:  # pragma: no cover -- plotting is best-effort
            print(f"[{cfg.spec_key}] plot failed: {exc}")
    elapsed = time.time() - t0
    md = _format_results_markdown(results, elapsed)
    args.out_md.write_text(md, encoding="utf-8")
    print(f"[done] {len(results)} specs auto-tuned in {elapsed:.1f}s")
    print(f"[done] wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the trajectory adversary across task families and report findings.

For each task family + proxy spec combination, this script:

1. Runs the adversary with multiple random restarts to find the
   highest-spec / lowest-gold trajectory.
2. Generates a baseline pool of random trajectories and calls
   ``measure_goodhart_gap`` to characterize the population-level
   spec/gold relationship under the random policy.
3. Emits a scatter plot of spec_rho vs. gold (random pool overlay +
   adversary point) at ``runs/adversary/<task>_<spec>.png``.
4. Writes a per-family report to ``paper/adversary_findings.md`` with
   the worst found divergence, the random-policy correlation, and the
   top-decile gap.

Usage
-----
    uv run python scripts/run_adversary.py

Configurable defaults are at the top of ``main()``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from stl_seed.analysis import (
    TrajectoryAdversary,
    bio_ode_repressilator_gold,
    bio_ode_toggle_gold,
    glucose_insulin_gold_score,
    measure_goodhart_gap,
)
from stl_seed.analysis.decomposition import GoodhartGapResult, random_policy
from stl_seed.specs import REGISTRY
from stl_seed.tasks import (
    BergmanParams,
    GlucoseInsulinSimulator,
    RepressilatorParams,
    RepressilatorSimulator,
    ToggleParams,
    ToggleSimulator,
    default_normal_subject_initial_state,
    default_repressilator_initial_state,
    default_toggle_initial_state,
    single_meal_schedule,
)

# ---------------------------------------------------------------------------
# Per-task-family configuration.
# ---------------------------------------------------------------------------
#
# Each entry encodes everything the adversary + decomposition need:
# the simulator, the proxy spec, the gold scorer, the action box, and
# any task-specific aux args. Keep the dict explicit so that adding a
# new task family is a matter of appending one record (no import-time
# magic).


def _glucose_insulin_config() -> dict[str, Any]:
    sim = GlucoseInsulinSimulator()
    params = BergmanParams()
    x0 = default_normal_subject_initial_state(params)
    meal = single_meal_schedule(onset_min=15.0, carb_grams=60.0)
    return {
        "task_family": "glucose_insulin",
        "simulator": sim,
        "params": params,
        "initial_state": x0,
        "spec": REGISTRY["glucose_insulin.tir.easy"],
        "gold_score": glucose_insulin_gold_score,
        "action_dim": 1,
        "horizon": 12,
        "action_min": jnp.asarray([0.0]),
        "action_max": jnp.asarray([5.0]),
        "simulator_aux": (meal,),
    }


def _bio_ode_repressilator_config() -> dict[str, Any]:
    sim = RepressilatorSimulator()
    params = RepressilatorParams()
    x0 = default_repressilator_initial_state(params)
    return {
        "task_family": "bio_ode.repressilator",
        "simulator": sim,
        "params": params,
        "initial_state": x0,
        "spec": REGISTRY["bio_ode.repressilator.easy"],
        "gold_score": bio_ode_repressilator_gold,
        "action_dim": 3,
        "horizon": 10,
        "action_min": jnp.asarray([0.0, 0.0, 0.0]),
        "action_max": jnp.asarray([1.0, 1.0, 1.0]),
        "simulator_aux": (),
    }


def _bio_ode_toggle_config() -> dict[str, Any]:
    sim = ToggleSimulator()
    params = ToggleParams()
    x0 = default_toggle_initial_state(params)
    return {
        "task_family": "bio_ode.toggle",
        "simulator": sim,
        "params": params,
        "initial_state": x0,
        "spec": REGISTRY["bio_ode.toggle.medium"],
        "gold_score": bio_ode_toggle_gold,
        "action_dim": 2,
        "horizon": 10,
        "action_min": jnp.asarray([0.0, 0.0]),
        "action_max": jnp.asarray([1.0, 1.0]),
        "simulator_aux": (),
    }


def task_family_configs() -> list[dict[str, Any]]:
    """Return the canonical list of task-family adversary configurations."""
    return [
        _glucose_insulin_config(),
        _bio_ode_repressilator_config(),
        _bio_ode_toggle_config(),
    ]


# ---------------------------------------------------------------------------
# Main per-family driver.
# ---------------------------------------------------------------------------


def run_one_family(
    cfg: dict[str, Any],
    *,
    n_restarts: int,
    max_iters: int,
    n_random: int,
    rng_seed: int,
) -> dict[str, Any]:
    """Run the adversary + decomposition for a single task family.

    Returns a JSON-serializable summary dict suitable for the markdown
    report and the JSONL log.
    """
    key_master = jax.random.PRNGKey(rng_seed)
    key_adv, key_pop = jax.random.split(key_master)

    adv = TrajectoryAdversary(
        simulator=cfg["simulator"],
        spec=cfg["spec"],
        gold_score=cfg["gold_score"],
        params=cfg["params"],
        lambda_satisfaction=0.5,
        learning_rate=0.3,
        max_iters=max_iters,
        project_actions=True,
        action_min=cfg["action_min"],
        action_max=cfg["action_max"],
        simulator_aux=cfg["simulator_aux"],
    )
    t_adv = time.time()
    adv_result = adv.find_adversary(
        initial_state=cfg["initial_state"],
        action_dim=cfg["action_dim"],
        horizon=cfg["horizon"],
        key=key_adv,
        n_restarts=n_restarts,
    )
    adv_seconds = time.time() - t_adv

    rand_pol = random_policy(
        horizon=cfg["horizon"],
        action_dim=cfg["action_dim"],
        action_min=np.asarray(cfg["action_min"]),
        action_max=np.asarray(cfg["action_max"]),
    )
    t_pop = time.time()
    gap_result = measure_goodhart_gap(
        simulator=cfg["simulator"],
        proxy_spec=cfg["spec"],
        gold_score=cfg["gold_score"],
        policies={"random": rand_pol},
        initial_state=cfg["initial_state"],
        action_dim=cfg["action_dim"],
        horizon=cfg["horizon"],
        params=cfg["params"],
        key=key_pop,
        n_trajectories=n_random,
        simulator_aux=cfg["simulator_aux"],
    )
    pop_seconds = time.time() - t_pop

    rand_gap: GoodhartGapResult = gap_result
    rand_stats = rand_gap.per_policy["random"]

    # Population reference: the worst gold seen under random sampling
    # AMONG trajectories with rho > 0 (spec-satisfying). The adversary
    # is "winning" if its (spec_rho > 0, gold) point is below this floor.
    rho_arr = rand_stats.rho_values
    gold_arr = rand_stats.gold_values
    sat_mask = rho_arr > 0.0
    if sat_mask.any():
        rand_min_gold_in_sat = float(np.min(gold_arr[sat_mask]))
        rand_mean_gold_in_sat = float(np.mean(gold_arr[sat_mask]))
        rand_n_satisfying = int(sat_mask.sum())
    else:
        rand_min_gold_in_sat = float("nan")
        rand_mean_gold_in_sat = float("nan")
        rand_n_satisfying = 0

    summary = {
        "task_family": cfg["task_family"],
        "spec": cfg["spec"].name,
        "adversary": {
            "best_spec_rho": adv_result.best_spec_rho,
            "best_gold_score": adv_result.best_gold_score,
            "n_restarts": n_restarts,
            "n_iter": max_iters,
            "n_nan_events": adv_result.n_nan_events,
            "converged": adv_result.converged,
            "restart_finals": adv_result.restart_histories,
            "wall_seconds": adv_seconds,
        },
        "random_population": {
            "n_trajectories": n_random,
            "n_spec_satisfying": rand_n_satisfying,
            "pearson_rho_gold": rand_stats.pearson_r,
            "spearman_rho_gold": rand_stats.spearman_r,
            "regression_slope": rand_stats.regression_slope,
            "top_decile_gap": rand_stats.top_decile_gap,
            "min_gold_in_satisfying": rand_min_gold_in_sat,
            "mean_gold_in_satisfying": rand_mean_gold_in_sat,
            "flagged_fm2": rand_stats.flagged,
            "wall_seconds": pop_seconds,
        },
        "gap_lower_bound": {
            "adv_gold_minus_random_mean_satisfying": (
                adv_result.best_gold_score - rand_mean_gold_in_sat
                if rand_n_satisfying > 0
                else float("nan")
            ),
            "adv_gold_minus_random_min_satisfying": (
                adv_result.best_gold_score - rand_min_gold_in_sat
                if rand_n_satisfying > 0
                else float("nan")
            ),
        },
        "rho_array": rho_arr.tolist(),
        "gold_array": gold_arr.tolist(),
    }
    return summary


# ---------------------------------------------------------------------------
# Plotting (matplotlib is a dependency; degrade gracefully if headless).
# ---------------------------------------------------------------------------


def _plot_scatter(
    summary: dict[str, Any],
    out_path: Path,
) -> None:
    """Scatter plot of (rho, gold) with adversary overlay.

    Defensive: matplotlib is a project dependency, but if the backend
    raises (e.g., headless DISPLAY-less environment with no Agg
    fallback) we just skip the plot rather than crashing the run.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [warn] matplotlib unavailable ({e}); skipping plot")
        return

    rho = np.asarray(summary["rho_array"])
    gold = np.asarray(summary["gold_array"])
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=120)
    ax.scatter(rho, gold, s=14, alpha=0.4, color="steelblue", label="random")
    ax.scatter(
        [summary["adversary"]["best_spec_rho"]],
        [summary["adversary"]["best_gold_score"]],
        s=80,
        marker="X",
        color="crimson",
        label="adversary",
        zorder=5,
    )
    ax.axvline(0.0, color="grey", linestyle=":", linewidth=0.7)
    ax.set_xlabel("proxy STL robustness rho")
    ax.set_ylabel("gold score (higher = better unstated quality)")
    ax.set_title(f"{summary['task_family']} :: {summary['spec']}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [plot] {out_path}")


# ---------------------------------------------------------------------------
# Markdown report writer.
# ---------------------------------------------------------------------------


def _write_report(
    summaries: list[dict[str, Any]],
    out_md: Path,
    config_meta: dict[str, Any],
) -> None:
    """Write a paper-ready markdown report of the adversary findings.

    Format mirrors other ``paper/*.md`` artifacts in the repo: front
    matter prose, then per-task-family numerical tables.
    """
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Trajectory adversary findings")
    lines.append("")
    lines.append(
        "Empirical operationalization of the Goodhart spec-completeness gap "
        "(see `paper/theory.md` S6). For each task family + proxy spec, the "
        "trajectory adversary searches for control sequences that satisfy "
        "the proxy STL spec (high rho) yet violate an unstated gold "
        "objective (low gold score). A successful find is a direct "
        "lower-bound on the per-task spec-completeness gap."
    )
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Generated: {config_meta['generated_at']}")
    lines.append(f"- Random seed: {config_meta['rng_seed']}")
    lines.append(f"- Adversary restarts: {config_meta['n_restarts']}")
    lines.append(f"- Adversary iterations per restart: {config_meta['max_iters']}")
    lines.append(f"- Random population size: {config_meta['n_random']}")
    lines.append("")

    for summ in summaries:
        lines.append(f"## {summ['task_family']} -- {summ['spec']}")
        lines.append("")
        adv = summ["adversary"]
        rand = summ["random_population"]
        gap = summ["gap_lower_bound"]
        lines.append("### Adversary")
        lines.append("")
        lines.append(f"- best spec rho       : `{adv['best_spec_rho']:+.4f}`")
        lines.append(f"- best gold score     : `{adv['best_gold_score']:+.4f}`")
        lines.append(
            f"- restarts            : `{adv['n_restarts']}` (iter/restart: `{adv['n_iter']}`)"
        )
        lines.append(f"- NaN/Inf events      : `{adv['n_nan_events']}`")
        lines.append(f"- adversary converged : `{adv['converged']}`")
        lines.append(
            f"- per-restart finals  : "
            f"{[(round(r, 3), round(g, 3)) for r, g in adv['restart_finals']]}"
        )
        lines.append(f"- wall time (s)       : `{adv['wall_seconds']:.2f}`")
        lines.append("")
        lines.append("### Random-population reference")
        lines.append("")
        lines.append(f"- n trajectories               : `{rand['n_trajectories']}`")
        lines.append(f"- n spec-satisfying            : `{rand['n_spec_satisfying']}`")
        lines.append(f"- Pearson r(rho, gold)         : `{rand['pearson_rho_gold']:+.4f}`")
        lines.append(f"- Spearman r(rho, gold)        : `{rand['spearman_rho_gold']:+.4f}`")
        lines.append(f"- regression slope (per sigma) : `{rand['regression_slope']:+.4f}`")
        lines.append(f"- top-decile gap (gold)        : `{rand['top_decile_gap']:+.4f}`")
        lines.append(f"- min gold among satisfying    : `{rand['min_gold_in_satisfying']:+.4f}`")
        lines.append(f"- mean gold among satisfying   : `{rand['mean_gold_in_satisfying']:+.4f}`")
        lines.append(f"- FM2 flagged (Spearman < 0.3) : `{rand['flagged_fm2']}`")
        lines.append(f"- wall time (s)                : `{rand['wall_seconds']:.2f}`")
        lines.append("")
        lines.append("### Empirical gap lower bound")
        lines.append("")
        lines.append(
            f"- adv gold - mean(satisfying random gold): "
            f"`{gap['adv_gold_minus_random_mean_satisfying']:+.4f}`"
        )
        lines.append(
            f"- adv gold - min(satisfying random gold) : "
            f"`{gap['adv_gold_minus_random_min_satisfying']:+.4f}`"
        )
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Per `paper/theory.md` S6, the spec-completeness term "
        "`R_gold(tau) - R_spec(tau)` is the Goodhart-relevant residual once "
        "the verifier-fidelity term is collapsed by the choice of formal "
        "STL robustness as the verifier. The adversary above provides a "
        "constructive lower bound on `sup_tau [R_spec(tau) - R_gold(tau)]` "
        "restricted to the spec-satisfying half-space. A negative gap "
        "(adv gold < mean satisfying random gold) indicates the spec is "
        "*locally* exploitable: there exist high-rho trajectories that score "
        "WORSE under the unstated gold than a random spec-satisfying baseline."
    )
    lines.append("")

    out_md.write_text("\n".join(lines))
    print(f"[report] {out_md}")


# ---------------------------------------------------------------------------
# CLI entry point.
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-restarts", type=int, default=8)
    parser.add_argument("--max-iters", type=int, default=80)
    parser.add_argument("--n-random", type=int, default=200)
    parser.add_argument("--rng-seed", type=int, default=2026)
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "paper" / "adversary_findings.md",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "runs"
        / "adversary"
        / "adversary_findings.jsonl",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "runs" / "adversary",
    )
    args = parser.parse_args()

    summaries: list[dict[str, Any]] = []
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    # Append-only JSONL artifact per project rules "results files are
    # append-only artifacts -- never rewrite them".
    with args.output_jsonl.open("a") as fh:
        for cfg in task_family_configs():
            print(f"[{cfg['task_family']}] running adversary...")
            summary = run_one_family(
                cfg,
                n_restarts=args.n_restarts,
                max_iters=args.max_iters,
                n_random=args.n_random,
                rng_seed=args.rng_seed,
            )
            summaries.append(summary)

            adv = summary["adversary"]
            rand = summary["random_population"]
            print(
                f"  adv: rho={adv['best_spec_rho']:+.3f} gold={adv['best_gold_score']:+.3f} "
                f"| random: pearson={rand['pearson_rho_gold']:+.3f} "
                f"top10gap={rand['top_decile_gap']:+.3f}"
            )

            # Strip large arrays from the JSONL log for size; keep per-traj
            # data in the in-memory summary used for plotting and the .md.
            summary_for_log = {
                k: v for k, v in summary.items() if k not in {"rho_array", "gold_array"}
            }
            fh.write(json.dumps(summary_for_log) + "\n")

            plot_path = args.plot_dir / f"{summary['task_family']}.png"
            _plot_scatter(summary, plot_path)

    config_meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "rng_seed": args.rng_seed,
        "n_restarts": args.n_restarts,
        "max_iters": args.max_iters,
        "n_random": args.n_random,
    }
    _write_report(summaries, args.output_md, config_meta)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

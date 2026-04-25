"""Example 02 — Generate a corpus and filter by STL robustness.

50 glucose-insulin trajectories under a 50/50 random+heuristic policy
mix, scored under the ADA 2024 Time-in-Range spec, then compared
side-by-side under all three filter conditions (`HardFilter`,
`QuantileFilter`, `ContinuousWeightedFilter`). Run from the repo root:

    uv run python examples/02_stl_filtering.py
"""

from __future__ import annotations

import jax
import numpy as np

from stl_seed.filter.conditions import (
    ContinuousWeightedFilter,
    FilterError,
    HardFilter,
    QuantileFilter,
)
from stl_seed.generation.runner import TrajectoryRunner
from stl_seed.specs import REGISTRY
from stl_seed.tasks.glucose_insulin import (
    BergmanParams,
    GlucoseInsulinSimulator,
    MealSchedule,
    default_normal_subject_initial_state,
)

_SEED = 20260424
_N_TRAJECTORIES = 50
_POLICY_MIX = {"random": 0.5, "heuristic": 0.5}
_SPEC_KEY = "glucose_insulin.tir.easy"


def _summary(rhos: np.ndarray, label: str) -> str:
    return (
        f"  {label:<28s} "
        f"N={rhos.size:>3d}  "
        f"rho min={rhos.min():+8.3f}  "
        f"median={float(np.median(rhos)):+8.3f}  "
        f"max={rhos.max():+8.3f}  "
        f"sat={(rhos > 0).mean():.0%}"
    )


def main() -> int:
    # 1. Wire up the generation pipeline. The TrajectoryRunner glues a
    # simulator + spec registry + initial state + meal schedule into a
    # single object that produces (trajectories, metadata) under the
    # configured policy mix.
    sim = GlucoseInsulinSimulator()
    params = BergmanParams()
    runner = TrajectoryRunner(
        simulator=sim,
        spec_registry={_SPEC_KEY: REGISTRY[_SPEC_KEY]},
        output_store=None,  # in-memory only; example 03 wires a Parquet store
        initial_state=default_normal_subject_initial_state(params),
        horizon=sim.n_control_points,
        action_dim=1,
        aux={"meal_schedule": MealSchedule.empty()},
        sim_params=params,
    )

    # 2. Generate the corpus. The runner yields one trajectory per call to
    # the policy; the simulator integrates the resulting open-loop
    # schedule end-to-end. The 50/50 random+heuristic mix gives a useful
    # spread of rho values for the filter comparison.
    print(f"Generating N={_N_TRAJECTORIES} trajectories under policy_mix={_POLICY_MIX} ...")
    key = jax.random.key(_SEED)
    trajectories, metadata = runner.generate_trajectories(
        task="glucose_insulin",
        n=_N_TRAJECTORIES,
        policy_mix=_POLICY_MIX,
        key=key,
        spec_key=_SPEC_KEY,
    )
    rhos = np.array([m["robustness"] for m in metadata], dtype=np.float64)
    policies = [m["policy"] for m in metadata]
    print(f"  generated {len(trajectories)} trajectories.")
    print(
        f"  policy split: random={policies.count('random')} heuristic={policies.count('heuristic')}"
    )
    print()

    # 3. Per-policy summary so the reader can sanity-check the heuristic
    # really does outperform random on this spec.
    print("Robustness by policy")
    for p in sorted(set(policies)):
        sub = rhos[np.array([pp == p for pp in policies])]
        print(_summary(sub, f"policy={p}"))
    print(_summary(rhos, "all (pre-filter)"))
    print()

    # 4. Apply each filter. We catch FilterError defensively because all
    # three filters raise if the kept subset is below `min_kept` — the
    # contract is "no silent fallback" per paper/theory.md FM2.
    print("Filter comparison")
    spec_filter_pairs = [
        ("HardFilter(threshold=0.0)", HardFilter(rho_threshold=0.0, min_kept=1)),
        ("QuantileFilter(top=25%)", QuantileFilter(top_k_pct=25.0, min_kept=1)),
        (
            "ContinuousWeightedFilter()",
            ContinuousWeightedFilter(min_kept=1),
        ),
    ]

    # The filter Protocol returns (kept_trajectories, weights) but does not
    # surface the kept indices. Both HardFilter and QuantileFilter compute
    # them internally from rho; we re-derive them here to recover the
    # per-policy rho summary for the kept subset.
    for label, filt in spec_filter_pairs:
        try:
            _kept_traj, weights = filt.filter(trajectories, rhos)
        except FilterError as e:
            print(f"  {label:<28s} FilterError: {e}")
            continue
        weights_np = np.asarray(weights)
        if filt.name == "hard":
            kept_rhos = rhos[rhos > filt.rho_threshold]
        elif filt.name == "quantile":
            n_keep = int(np.ceil(filt.top_k_pct / 100.0 * rhos.size))
            kept_rhos = np.sort(rhos)[-n_keep:]
        else:  # continuous: kept set is the full corpus
            kept_rhos = rhos
        print(_summary(kept_rhos, label))
        print(
            f"      weights: min={weights_np.min():.3f}  "
            f"max={weights_np.max():.3f}  "
            f"mean={weights_np.mean():.3f}  sum={weights_np.sum():.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

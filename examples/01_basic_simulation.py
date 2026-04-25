"""Example 01 — Glucose-insulin simulation + STL robustness.

Three open-loop schedules (zero infusion, constant 1.5 U/h, fasting
baseline) on the Bergman 1979 + Dalla Man 2007 model, each scored under
the registered ADA 2024 Time-in-Range spec. Run from the repo root:

    uv run python examples/01_basic_simulation.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from stl_seed.specs import REGISTRY
from stl_seed.stl.evaluator import evaluate_robustness
from stl_seed.tasks._trajectory import Trajectory
from stl_seed.tasks.glucose_insulin import (
    BergmanParams,
    GlucoseInsulinSimulator,
    default_normal_subject_initial_state,
    single_meal_schedule,
)


def _simulate_open_loop(
    sim: GlucoseInsulinSimulator,
    params: BergmanParams,
    initial_state: jax.Array,
    meals: object,
    control_sequence: jax.Array,
    key: jax.Array,
) -> Trajectory:
    """Run one open-loop rollout and wrap into a canonical Trajectory pytree.

    Glucose-insulin's simulator returns ``(states, times, meta)`` rather
    than the canonical ``Trajectory`` (the runner adapter wraps it). We
    do the same wrapping here so the STL evaluator can consume the
    output via the duck-typed Trajectory protocol.
    """
    states, times, meta = sim.simulate(initial_state, control_sequence, meals, params, key)
    return Trajectory(
        states=states,
        actions=control_sequence.reshape(-1, 1),
        times=times,
        meta=meta,
    )


def _report(label: str, traj: Trajectory, spec) -> None:
    rho = float(evaluate_robustness(spec, traj))
    G = np.asarray(traj.states[:, 0])
    verdict = "SATISFIES" if rho >= 0 else "VIOLATES"
    print(f"  {label}")
    print(f"    glucose range : [{G.min():.1f}, {G.max():.1f}] mg/dL")
    print(f"    NaN-replaced  : {int(traj.meta.n_nan_replacements)} / {G.size * 3} entries")
    print(f"    robustness rho: {rho:+.3f} mg/dL  ({verdict} with margin {abs(rho):.3f})")


def main() -> int:
    # 1. Simulator + parameters (literature defaults: healthy adult subject).
    sim = GlucoseInsulinSimulator()
    params = BergmanParams()
    initial_state = default_normal_subject_initial_state(params)

    # A 50 g carbohydrate meal at t=10 min (Dalla Man 2007 Fig. 4 reference).
    meals = single_meal_schedule(onset_min=10.0, carb_grams=50.0)
    spec = REGISTRY["glucose_insulin.tir.easy"]

    print("Bergman 1979 + Dalla Man 2007 (single-meal challenge)")
    print(f"  horizon         : {sim.horizon_min} min, {sim.n_control_points} control points")
    print("  action bounds   : [0.0, 5.0] U/h insulin infusion")
    print(f"  STL spec        : {spec.name}")
    print(f"  formula         : {spec.formula_text}")
    print()

    H = sim.n_control_points
    key = jax.random.key(0)

    # 2a. Zero-infusion baseline. With a 50 g meal and no insulin, the
    # patient hyperglycaemias and the spec is violated.
    print("Open-loop schedule A: zero infusion (no exogenous insulin)")
    u_zero = jnp.zeros((H,))
    traj_zero = _simulate_open_loop(sim, params, initial_state, meals, u_zero, key)
    _report("u = 0 U/h", traj_zero, spec)
    print()

    # 2b. Constant low-rate infusion. A flat 1.5 U/h schedule (toward the
    # mid-range of clinically realistic basal needs in the artificial
    # pancreas literature) holds glucose closer to the band.
    print("Open-loop schedule B: constant 1.5 U/h infusion")
    u_const = jnp.full((H,), 1.5)
    traj_const = _simulate_open_loop(sim, params, initial_state, meals, u_const, key)
    _report("u = 1.5 U/h", traj_const, spec)
    print()

    # 3. The fasting / no-meal regime is a useful sanity check: glucose
    # should stay near basal Gb = 90 mg/dL and the spec is trivially met.
    print("Open-loop schedule C: zero infusion AND no meal (fasting baseline)")
    from stl_seed.tasks.glucose_insulin import MealSchedule

    traj_fast = _simulate_open_loop(sim, params, initial_state, MealSchedule.empty(), u_zero, key)
    _report("u = 0 U/h, no meal", traj_fast, spec)
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Task families for stl-seed.

Each task family is a (simulator, action_spec, observation_spec) bundle that
the SERA-style soft-filtered SFT pipeline can roll out against an LLM policy.
The simulator is a pure JAX/Diffrax ODE integrator; the action and
observation specs are NumPy/JAX dataclasses defining what the policy emits
and what the STL evaluator sees.

Currently exposed:
    glucose_insulin: Bergman 1979 minimal model + Dalla Man 2007 meal
                     disturbance, controlled by an exogenous insulin
                     infusion schedule u_{1:H} (12 control points over 2 h).
"""

from __future__ import annotations

from stl_seed.tasks.glucose_insulin import (
    U_INSULIN_MAX_U_PER_H,
    U_INSULIN_MIN_U_PER_H,
    BergmanParams,
    GlucoseInsulinSimulator,
    MealSchedule,
    TrajectoryMeta,
    default_normal_subject_initial_state,
    single_meal_schedule,
)

__all__ = [
    "BergmanParams",
    "GlucoseInsulinSimulator",
    "MealSchedule",
    "TrajectoryMeta",
    "U_INSULIN_MAX_U_PER_H",
    "U_INSULIN_MIN_U_PER_H",
    "default_normal_subject_initial_state",
    "single_meal_schedule",
]

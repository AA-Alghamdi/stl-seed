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
    bio_ode:         Repressilator (Elowitz & Leibler 2000), Toggle switch
                     (Gardner et al. 2000), and MAPK cascade (Huang &
                     Ferrell 1996), controlled by per-gene inducer
                     fractions / inducer levels / stimulus intensity.
"""

from __future__ import annotations

from stl_seed.tasks._trajectory import Trajectory, TrajectoryMeta
from stl_seed.tasks.bio_ode import (
    MAPKSimulator,
    RepressilatorSimulator,
    Simulator,
    ToggleSimulator,
    default_mapk_initial_state,
    default_repressilator_initial_state,
    default_toggle_initial_state,
)
from stl_seed.tasks.bio_ode_params import (
    MAPKParams,
    RepressilatorParams,
    ToggleParams,
)
from stl_seed.tasks.glucose_insulin import (
    U_INSULIN_MAX_U_PER_H,
    U_INSULIN_MIN_U_PER_H,
    BergmanParams,
    GlucoseInsulinSimulator,
    MealSchedule,
    default_normal_subject_initial_state,
    single_meal_schedule,
)

__all__ = [
    "BergmanParams",
    "GlucoseInsulinSimulator",
    "MAPKParams",
    "MAPKSimulator",
    "MealSchedule",
    "RepressilatorParams",
    "RepressilatorSimulator",
    "Simulator",
    "ToggleParams",
    "ToggleSimulator",
    "Trajectory",
    "TrajectoryMeta",
    "U_INSULIN_MAX_U_PER_H",
    "U_INSULIN_MIN_U_PER_H",
    "default_mapk_initial_state",
    "default_normal_subject_initial_state",
    "default_repressilator_initial_state",
    "default_toggle_initial_state",
    "single_meal_schedule",
]

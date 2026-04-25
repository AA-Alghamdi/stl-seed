"""Unit tests for the glucose-insulin task family.

Each test pre-registers a physiological expectation derivable from the
literature, then asserts the simulator reproduces it. References are inline.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from stl_seed.tasks.glucose_insulin import (
    U_INSULIN_MAX_U_PER_H,
    BergmanParams,
    GlucoseInsulinSimulator,
    MealSchedule,
    default_normal_subject_initial_state,
    single_meal_schedule,
)


@pytest.fixture
def params() -> BergmanParams:
    return BergmanParams()


@pytest.fixture
def sim() -> GlucoseInsulinSimulator:
    # 121 save points over 120 min => 1-min resolution
    return GlucoseInsulinSimulator()


@pytest.fixture
def key() -> jax.Array:
    return jax.random.key(0)


# -----------------------------------------------------------------------------


def test_baseline_glucose_stable(
    sim: GlucoseInsulinSimulator, params: BergmanParams, key: jax.Array
) -> None:
    """With no insulin infusion above basal and no meal, plasma glucose at
    fasting steady state must remain near Gb (90 mg/dL) over the 2 h horizon.

    Tolerance: +/- 10 mg/dL is the standard "fasting normal" band
    (WHO: 70-99 mg/dL fasting normal). The Bergman 1979 model is
    autonomous-stable at (Gb, 0, Ib) by construction; any drift outside
    +/- 10 indicates a sign error or units bug in the vector field.
    """
    y0 = default_normal_subject_initial_state(params)
    u = jnp.zeros((sim.n_control_points,))  # no exogenous insulin
    meals = MealSchedule.empty()

    ys, _, meta = sim.simulate(y0, u, meals, params, key)
    G = ys[:, 0]

    assert meta.n_nan_replacements == 0, "no NaN expected at fasting baseline"
    assert jnp.all(jnp.abs(G - params.Gb) <= 10.0), (
        f"glucose drifted: G range = ({float(jnp.min(G)):.2f}, "
        f"{float(jnp.max(G)):.2f}) mg/dL, expected within +/- 10 of Gb=90"
    )


def test_meal_glucose_rise(
    sim: GlucoseInsulinSimulator, params: BergmanParams, key: jax.Array
) -> None:
    """A 75 g carbohydrate oral load (the standard OGTT dose; ADA Criteria
    for the Diagnosis of Diabetes Mellitus, Diabetes Care 2010) with no
    exogenous insulin must drive plasma glucose above 180 mg/dL within 60 min.

    OGTT reference: in published normal-subject curves [DallaMan 2007 Fig. 6],
    a 75 g oral load with intact endogenous insulin response peaks ~150 mg/dL
    at 60 min. Here we have ZEROED endogenous secretion (the simulator
    reduction documented in the docstring), so the response is exaggerated:
    we expect the peak to clearly exceed 180 mg/dL by 60 min. This is the
    correct test of the meal-Ra term given our reduction.
    """
    y0 = default_normal_subject_initial_state(params)
    u = jnp.zeros((sim.n_control_points,))
    meals = single_meal_schedule(onset_min=0.0, carb_grams=75.0)

    ys, ts, meta = sim.simulate(y0, u, meals, params, key)
    G = ys[:, 0]

    assert meta.n_nan_replacements == 0
    G_at_or_before_60 = G[ts <= 60.0]
    assert jnp.max(G_at_or_before_60) > 180.0, (
        f"meal response too small: max G in [0, 60] min = "
        f"{float(jnp.max(G_at_or_before_60)):.2f} mg/dL, expected > 180"
    )


def test_insulin_drop(sim: GlucoseInsulinSimulator, params: BergmanParams, key: jax.Array) -> None:
    """A steady infusion at the maximum allowed rate (5 U/h) with no meal
    must drive plasma glucose below 80 mg/dL within 90 min for a normal
    subject.

    Justification: the Bergman insulin-action gain SI = p3 / p2
    = 4.92e-6 / 0.0123 ~ 4e-4 (mg/dL)^-1 (microU/mL)^-1 min^-1. A 5 U/h
    infusion drives steady-state I ~ 5 * 1e6 / (60 * V_I_L * 1000 * n)
    + Ib = 5 * 1e3 / (60 * 3.5 * 0.2659) + 7 ~ 96 microU/mL, which yields
    steady-state X ~ p3 / p2 * (I - Ib) ~ 4e-4 * 89 ~ 0.036 1/min. At that
    X, dG/dt = -p1 * (G - Gb) - 0.036 * G evaluated at G = 90 gives
    dG/dt ~ -3.2 mg/dL/min initially — easily clearing 10 mg/dL within 90 min.
    """
    y0 = default_normal_subject_initial_state(params)
    u = jnp.full((sim.n_control_points,), U_INSULIN_MAX_U_PER_H)
    meals = MealSchedule.empty()

    ys, ts, meta = sim.simulate(y0, u, meals, params, key)
    G = ys[:, 0]

    assert meta.n_nan_replacements == 0
    G_at_or_before_90 = G[ts <= 90.0]
    assert jnp.min(G_at_or_before_90) < 80.0, (
        f"insulin infusion failed to lower G: min G in [0, 90] min = "
        f"{float(jnp.min(G_at_or_before_90)):.2f} mg/dL, expected < 80"
    )


def test_oscillation_stable(
    sim: GlucoseInsulinSimulator, params: BergmanParams, key: jax.Array
) -> None:
    """Periodic insulin pulses (alternating max/zero per 10-min interval)
    must yield a bounded, finite trajectory — no runaway, no oscillatory
    blow-up.

    "Quasi-stable cycle" is operationalized as: trajectory stays inside a
    physiologically plausible envelope (G in [20, 250] mg/dL; I in [0, 500]
    microU/mL) and there are no NaN/Inf events. This rules out any
    integrator instability under chattering control.
    """
    y0 = default_normal_subject_initial_state(params)
    H = sim.n_control_points
    u = jnp.where(
        jnp.arange(H) % 2 == 0,
        U_INSULIN_MAX_U_PER_H,
        0.0,
    )
    meals = MealSchedule.empty()

    ys, _, meta = sim.simulate(y0, u, meals, params, key)
    G, _, Ins = ys[:, 0], ys[:, 1], ys[:, 2]

    assert meta.n_nan_replacements == 0
    assert jnp.all(jnp.isfinite(ys)), "non-finite values in trajectory"
    assert jnp.all(G > 20.0) and jnp.all(G < 250.0), (
        f"G escaped envelope: range = ({float(jnp.min(G)):.2f}, {float(jnp.max(G)):.2f}) mg/dL"
    )
    assert jnp.all(Ins >= 0.0) and jnp.all(Ins < 500.0), (
        f"I escaped envelope: range = ({float(jnp.min(Ins)):.2f}, "
        f"{float(jnp.max(Ins)):.2f}) microU/mL"
    )


def test_no_nan(sim: GlucoseInsulinSimulator, params: BergmanParams, key: jax.Array) -> None:
    """Across a battery of random control sequences and meal schedules, no
    simulation should produce NaN/Inf values that escape the sentinel guard.
    """
    keys = jax.random.split(key, 8)
    y0 = default_normal_subject_initial_state(params)

    for k in keys:
        ku, km = jax.random.split(k)
        u = jax.random.uniform(
            ku,
            shape=(sim.n_control_points,),
            minval=0.0,
            maxval=U_INSULIN_MAX_U_PER_H,
        )
        # Random meal: 0-100 g of carb between 0 and 60 min
        onset = float(jax.random.uniform(km, minval=0.0, maxval=60.0))
        carbs = float(jax.random.uniform(km, minval=0.0, maxval=100.0))
        meals = single_meal_schedule(onset_min=onset, carb_grams=carbs)

        ys, _, meta = sim.simulate(y0, u, meals, params, k)
        assert jnp.all(jnp.isfinite(ys)), "non-finite in trajectory after sentinel"
        # The sentinel guard means n_nan_replacements may be > 0 in pathological
        # cases, but the returned ys must be finite either way.
        _ = meta.n_nan_replacements


def test_jit_works(sim: GlucoseInsulinSimulator, params: BergmanParams, key: jax.Array) -> None:
    """The simulator's `simulate` method must be JIT-compilable end to end
    (including the meal schedule, which is a pytree of JAX arrays).
    """
    y0 = default_normal_subject_initial_state(params)
    u = jnp.full((sim.n_control_points,), 1.0)
    meals = single_meal_schedule(onset_min=10.0, carb_grams=50.0)

    @jax.jit
    def run(
        y0_: jax.Array,
        u_: jax.Array,
        meals_: MealSchedule,
        params_: BergmanParams,
        key_: jax.Array,
    ) -> jax.Array:
        ys, _, _ = sim.simulate(y0_, u_, meals_, params_, key_)
        return ys

    ys = run(y0, u, meals, params, key)
    assert ys.shape == (sim.n_save_points, 3)
    assert jnp.all(jnp.isfinite(ys))

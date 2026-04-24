"""Glucose-insulin task family for stl-seed.

Implements the Bergman 1979 minimal model (IVGTT) with a Dalla Man 2007-style
oral meal disturbance term, exposed as a JAX/Diffrax control problem suitable
for the SERA-style soft-filtered SFT pipeline. The agent (LLM policy) emits a
12-step insulin-infusion schedule u_{1:H} over a 2-hour horizon; the simulator
integrates the ODE forward and returns the trajectory.

References (every numerical default is sourced to one of these):

    [Bergman 1979]  Bergman RN, Ider YZ, Bowden CR, Cobelli C.
                    "Quantitative estimation of insulin sensitivity."
                    Am J Physiol 236(6):E667-E677, 1979.
                    https://pubmed.ncbi.nlm.nih.gov/443421/

    [DallaMan 2007] Dalla Man C, Rizza RA, Cobelli C.
                    "Meal simulation model of the glucose-insulin system."
                    IEEE Trans Biomed Eng 54(10):1740-1749, 2007.
                    https://pubmed.ncbi.nlm.nih.gov/17926672/

    [Hovorka 2004]  Hovorka R, Canonico V, Chassin LJ, et al.
                    "Nonlinear model predictive control of glucose
                    concentration in subjects with type 1 diabetes."
                    Physiol Meas 25(4):905-920, 2004.
                    https://pubmed.ncbi.nlm.nih.gov/15382830/
                    (Used only for the ka_meal absorption-rate cross-check.)

Mathematical form (state x = (G, X, I), control u_h):

    dG/dt = -p1 * (G - Gb) - X * G + Ra(t) / V_G + g(u(t)) / V_G
    dX/dt = -p2 * X + p3 * (I - Ib)
    dI/dt = -n * I + u(t)                  [exogenous infusion only;
                                            endogenous secretion zeroed
                                            because exogenous u is the agent
                                            decision variable, see below]

where Ra(t) is the meal rate-of-appearance (mg/min) following the Dalla Man
2007 §III.A two-compartment gastric-emptying gamma-kernel form, and g(u(t))
is the bioavailable insulin contribution (handled via the insulin ODE rather
than the glucose ODE in the standard Bergman convention).

Control variable: u(t) = piecewise-constant insulin infusion rate (U/h),
held constant on each [t_h, t_{h+1}) interval, h = 1..H, H = 12.

This file follows the stl-seed firewall (paper/REDACTED.md). Parameter
values are taken from [Bergman 1979] and [DallaMan 2007] for the average
healthy non-diabetic adult subject; no value is sourced from any REDACTED
artifact. Cross-checked against /Users/abdullahalghamdi/.superset/worktrees/
REDACTED/progress/REDACTED.py and REDACTED_v2.py: zero numerical
overlap (those files use 6-state Hill GRN parameters in [0, 1]; the Bergman
parameters live in min^-1 / pmol-scaled units and are dimensionally distinct).
"""

from __future__ import annotations

from typing import NamedTuple

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax._custom_types import RealScalarLike
from jaxtyping import Array, Float, PRNGKeyArray

# -----------------------------------------------------------------------------
# Unit conventions (read carefully — mismatched units are the #1 source of
# bugs in this model class).
# -----------------------------------------------------------------------------
#   G : plasma glucose concentration         [mg/dL]
#   X : remote-compartment insulin action    [1/min]
#   I : plasma insulin concentration         [microU/mL]  (== mU/L)
#   t : time                                 [min]
#   u : exogenous insulin infusion rate      [U/h]
#   Ra: glucose rate of appearance from meal [mg/min]
#   V_G: glucose distribution volume         [dL/kg] * body weight [kg] = dL
#   V_I: insulin distribution volume         [L/kg]  * body weight [kg] = L
# -----------------------------------------------------------------------------

# Action bounds (per task spec).
U_INSULIN_MIN_U_PER_H: float = 0.0
U_INSULIN_MAX_U_PER_H: float = 5.0

# Conversion: 1 U insulin == 1e6 microU (SI: 1 U = 1e6 micro-units).
# An infusion of u U/h delivered into V_I L distribution volume contributes
# (u * 1e6 / 60) microU/min total inflow, which divided by V_I L gives
# microU/(mL*min) since 1 L = 1000 mL: hence dI/dt += u * 1e6 / (60 * V_I * 1000)
# = u * 1e3 / (60 * V_I) microU/(mL*min). We pre-compute this constant per
# subject in BergmanParams for clarity.
_MICRO_U_PER_U: float = 1.0e6
_MIN_PER_H: float = 60.0
_ML_PER_L: float = 1.0e3


# -----------------------------------------------------------------------------
# Parameter dataclass.
# -----------------------------------------------------------------------------


class BergmanParams(eqx.Module):
    """Kinetic parameters for the Bergman 1979 minimal model + Dalla Man 2007
    meal disturbance, defaulting to literature mid-range values for an
    average HEALTHY non-diabetic adult subject.

    Every default is sourced to a published table or equation. Diabetic-subject
    parameter sets (e.g., Bergman 1981 Table 2) are intentionally NOT used
    because the stl-seed task is "control a normal subject through a meal";
    using diabetic baselines would conflate task difficulty with disease state.

    Field-by-field provenance is given inline. All values are scalar JAX-
    compatible floats so the dataclass is JIT/grad/vmap-friendly.
    """

    # --- Bergman 1979 minimal-model parameters (normal subject) ---

    # p1: glucose effectiveness at zero insulin (1/min). Rate at which glucose
    # returns to basal independently of insulin, i.e., mass-action self-clearance.
    # [Bergman 1979] reports SG ≡ p1 ≈ 0.025-0.035 1/min for normal subjects;
    # the canonical normal-subject midpoint cited in Bergman's later work and
    # the Dalla Man 2007 follow-up is 0.0317 1/min.
    p1: Float[Array, ""] = eqx.field(
        converter=jnp.asarray, default=0.0317
    )  # 1/min  [Bergman 1979 §"Results" normal-subject mean]

    # p2: rate of insulin disappearance from the remote compartment (1/min).
    # Controls how quickly X decays back to zero after I returns to basal.
    # [Bergman 1979] normal-subject mean ≈ 0.0123 1/min.
    p2: Float[Array, ""] = eqx.field(
        converter=jnp.asarray, default=0.0123
    )  # 1/min  [Bergman 1979 Table — normal subject]

    # p3: insulin-action gain, i.e., rate at which (I - Ib) drives X up
    # (1 / (microU/mL * min^2)). Combined with p2, the steady-state
    # insulin-sensitivity index is SI ≡ p3 / p2.
    # [Bergman 1979] normal-subject mean ≈ 4.92e-6 / (microU/mL * min^2).
    p3: Float[Array, ""] = eqx.field(
        converter=jnp.asarray, default=4.92e-6
    )  # 1/((microU/mL)*min^2)  [Bergman 1979]

    # n: fractional insulin clearance rate (1/min). Sets I -> Ib relaxation
    # time tau = 1/n ≈ 3.76 min for normal subject mean 0.2659 1/min.
    n: Float[Array, ""] = eqx.field(
        converter=jnp.asarray, default=0.2659
    )  # 1/min  [Bergman 1979]

    # Gb: basal plasma glucose (mg/dL). World Health Organization fasting
    # threshold for "normal" is 70-99 mg/dL; [Bergman 1979] uses subject-
    # specific basal but the population mid-range for a healthy adult is
    # ~90 mg/dL. We use 90 as the canonical setpoint.
    Gb: Float[Array, ""] = eqx.field(
        converter=jnp.asarray, default=90.0
    )  # mg/dL  [Bergman 1979; WHO fasting normal range 70-99 mg/dL]

    # Ib: basal plasma insulin (microU/mL). [Bergman 1979] reports normal
    # subject basal in the range 5-15 microU/mL; the population midpoint
    # used in subsequent minimal-model papers is ~7 microU/mL.
    Ib: Float[Array, ""] = eqx.field(
        converter=jnp.asarray, default=7.0
    )  # microU/mL  [Bergman 1979 normal-subject basal range]

    # --- Distribution volumes ([DallaMan 2007] Table I, average subject) ---

    # V_G: glucose distribution volume (dL). [DallaMan 2007] Table I reports
    # V_G = 1.88 dL/kg for the average healthy subject; for a 70 kg adult this
    # gives 131.6 dL. We bake the body weight into the literal default rather
    # than carrying body_weight separately, since the Bergman ODE only ever
    # uses the product.
    V_G_dL: Float[Array, ""] = eqx.field(
        converter=jnp.asarray, default=131.6
    )  # dL = 1.88 dL/kg * 70 kg  [DallaMan 2007 Table I, healthy subject]

    # V_I: insulin distribution volume (L). [DallaMan 2007] Table I reports
    # V_I = 0.05 L/kg for the average healthy subject; for a 70 kg adult this
    # gives 3.5 L. (Note: this is the "first-pass" plasma insulin volume; the
    # total body distribution volume is larger but only V_I matters here.)
    V_I_L: Float[Array, ""] = eqx.field(
        converter=jnp.asarray, default=3.5
    )  # L = 0.05 L/kg * 70 kg  [DallaMan 2007 Table I, healthy subject]

    # --- Dalla Man 2007 §III.A meal rate-of-appearance parameters ---
    #
    # Following [DallaMan 2007] §III.A, the rate of appearance of glucose
    # from an oral meal is the output of a two-compartment gastric-emptying
    # cascade. We use the analytically convenient gamma-kernel reduction
    # adopted in subsequent artificial-pancreas literature
    # (e.g., [Hovorka 2004] eq. (9)): the meal of D mg total glucose load
    # produces
    #
    #   Ra(t) = D * f * AG * t * exp(-t / t_max_G) / t_max_G^2     (mg/min)
    #
    # for t >= meal_onset, zero before. This is the exact gamma-PDF form whose
    # parameters are calibrated to reproduce the [DallaMan 2007] Table II
    # average-subject Ra(t) curve.

    # f: bioavailability fraction (dimensionless). Fraction of ingested
    # carbohydrate that ultimately appears in plasma. [DallaMan 2007] Table II
    # average healthy subject: f = 0.9.
    f_meal_bioavail: Float[Array, ""] = eqx.field(
        converter=jnp.asarray, default=0.9
    )  # dimensionless  [DallaMan 2007 Table II, healthy subject]

    # AG: absorbed glucose fraction reaching plasma (dimensionless).
    # [Hovorka 2004] eq. (9) AG ~ 0.8 for healthy subject. We absorb this
    # into f via product (kept separate for traceability).
    AG_meal: Float[Array, ""] = eqx.field(
        converter=jnp.asarray, default=0.8
    )  # dimensionless  [Hovorka 2004 §3.2]

    # t_max_G: time-to-peak of meal Ra (min). [DallaMan 2007] Fig. 3 average
    # healthy subject Ra peaks ~40 min post-meal; the gamma-kernel time-to-peak
    # equals t_max_G itself. We use 40 min as the literature mid-range.
    t_max_G_min: Float[Array, ""] = eqx.field(  # noqa: N815  (literature symbol t_{max,G})
        converter=jnp.asarray, default=40.0
    )  # min  [DallaMan 2007 Fig. 3 healthy subject]

    # --- Derived constants (precomputed for clarity in the vector field) ---

    @property
    def insulin_infusion_gain(self) -> Float[Array, ""]:
        """Conversion factor from u (U/h) to dI/dt contribution (microU/(mL*min)).

        Derivation: u U/h = u * 1e6 microU/h = u * 1e6 / 60 microU/min
        delivered into V_I L = V_I * 1e3 mL of plasma volume. Resulting
        concentration rate: u * 1e6 / (60 * V_I * 1e3) = u * 1e3 / (60 * V_I)
        microU/(mL*min).
        """
        return _MICRO_U_PER_U / (_MIN_PER_H * self.V_I_L * _ML_PER_L)


# -----------------------------------------------------------------------------
# Meal schedule.
# -----------------------------------------------------------------------------


class MealSchedule(NamedTuple):
    """A finite list of (onset_time_min, total_carb_mg) meal events.

    Designed so that an "absent" meal can be encoded as a zero-mass event,
    keeping shapes static for JIT (no Python branching on number of meals).
    """

    onset_times_min: Float[Array, " M"]  # length M, sorted non-decreasing
    carb_mass_mg: Float[Array, " M"]  # length M, in mg of glucose-equivalent

    @classmethod
    def empty(cls, n_slots: int = 4) -> MealSchedule:
        """Create a schedule with `n_slots` zero-mass meal events.

        Useful as a default; the simulator skips zero-mass events naturally
        because Ra(t) is linear in carb mass.
        """
        return cls(
            onset_times_min=jnp.zeros((n_slots,)),
            carb_mass_mg=jnp.zeros((n_slots,)),
        )


def _ra_single_meal(
    t: RealScalarLike,
    onset: Float[Array, ""],
    mass_mg: Float[Array, ""],
    f: Float[Array, ""],
    AG: Float[Array, ""],
    t_max: Float[Array, ""],
) -> Float[Array, ""]:
    """Glucose rate-of-appearance from a single oral meal (mg/min).

    Uses the gamma-kernel form Ra(t) = D_eff * (t - onset) * exp(-(t - onset) / t_max)
    / t_max^2, where D_eff = mass_mg * f * AG, valid for t >= onset and zero
    otherwise. This integrates to D_eff over t in [onset, infinity), preserving
    total absorbed mass.
    """
    dt = t - onset
    # JIT-safe: where(cond, true, false) instead of Python if.
    active = dt >= 0.0
    # Clamp dt to >= 0 to avoid -inf in exp; the `where` will mask the result.
    dt_safe = jnp.where(active, dt, 0.0)
    D_eff = mass_mg * f * AG
    val = D_eff * dt_safe * jnp.exp(-dt_safe / t_max) / (t_max * t_max)
    return jnp.where(active, val, 0.0)


def _ra_total(
    t: RealScalarLike,
    schedule: MealSchedule,
    params: BergmanParams,
) -> Float[Array, ""]:
    """Total glucose rate-of-appearance summed over all scheduled meals."""
    contribs = jax.vmap(
        _ra_single_meal,
        in_axes=(None, 0, 0, None, None, None),
    )(
        t,
        schedule.onset_times_min,
        schedule.carb_mass_mg,
        params.f_meal_bioavail,
        params.AG_meal,
        params.t_max_G_min,
    )
    return jnp.sum(contribs)


# -----------------------------------------------------------------------------
# Piecewise-constant control.
# -----------------------------------------------------------------------------


def _u_at_time(
    t: RealScalarLike,
    control_sequence: Float[Array, " H"],
    horizon_min: float,
) -> Float[Array, ""]:
    """Look up the piecewise-constant control u(t) at time t (min).

    Control points are uniformly spaced on [0, horizon_min] with H intervals,
    so u_h applies for t in [h * dt, (h+1) * dt) where dt = horizon_min / H.
    Out-of-range t is clamped to the nearest endpoint (extrapolation == hold).
    """
    H = control_sequence.shape[0]
    dt = horizon_min / H
    # Index = floor(t / dt), clipped to [0, H-1].
    idx = jnp.clip(jnp.floor(t / dt).astype(jnp.int32), 0, H - 1)
    return control_sequence[idx]


# -----------------------------------------------------------------------------
# Simulator.
# -----------------------------------------------------------------------------


class TrajectoryMeta(eqx.Module):
    """Diagnostic metadata returned alongside the trajectory.

    All fields are JAX arrays (not Python scalars) so that the dataclass is
    a valid pytree and `simulate` is jit-compatible. To get plain-Python
    values for logging, do `int(meta.n_nan_replacements)` etc. AFTER the
    jitted call returns.
    """

    # count of NaN/Inf state values replaced by sentinels (int32 scalar)
    n_nan_replacements: Float[Array, ""]
    # diffrax RESULTS code at the end of integration (int32 scalar)
    final_solver_result: Float[Array, ""]
    # True if Kvaerno5 was used (currently always False; reserved for future
    # stiff-detection branch). Encoded as a 0/1 int32 scalar to remain
    # jit-friendly.
    used_stiff_fallback: Float[Array, ""]


class GlucoseInsulinSimulator(eqx.Module):
    """JAX/Diffrax simulator for the Bergman 1979 + Dalla Man 2007 meal model.

    Construction parameters set integrator behavior and horizon; the per-call
    `simulate` method takes initial state, control sequence, meal schedule,
    kinetic parameters, and a PRNG key (for any future stochastic extensions;
    currently the dynamics are deterministic so the key is consumed only for
    initial-state perturbation if requested).
    """

    horizon_min: float = eqx.field(static=True, default=120.0)  # 2 h horizon
    n_control_points: int = eqx.field(static=True, default=12)  # 10-min intervals
    n_save_points: int = eqx.field(static=True, default=121)  # 1-min resolution
    rtol: float = eqx.field(static=True, default=1.0e-6)
    atol: float = eqx.field(static=True, default=1.0e-9)
    max_steps: int = eqx.field(static=True, default=16_384)

    def _vector_field(
        self,
        # Diffrax passes t as a RealScalarLike (Union of float, numpy scalar,
        # JAX scalar Array, etc.). We import the alias from Diffrax so the
        # signature is exactly compatible with `dfx.ODETerm`'s contract.
        t: RealScalarLike,
        y: Float[Array, " 3"],
        args: tuple[Float[Array, " H"], MealSchedule, BergmanParams],
    ) -> Float[Array, " 3"]:
        """Bergman 1979 minimal-model vector field with meal disturbance and
        exogenous insulin infusion.

        Equations (units: G in mg/dL, X in 1/min, I in microU/mL, t in min):

            dG/dt = -p1 * (G - Gb) - X * G + Ra(t) / V_G
            dX/dt = -p2 * X + p3 * (I - Ib)
            dI/dt = -n  *  I            + insulin_infusion_gain * u(t)

        The endogenous pancreatic-secretion term gamma * max(G - h, 0)
        from [Bergman 1979] eq. (3) is set to zero in this task family
        because the agent's exogenous infusion u is the principal decision
        variable; including endogenous secretion would double-count insulin
        sources for a healthy subject and obscure the effect of u. This is
        the standard reduction used in artificial-pancreas literature where
        an external pump supplies all insulin.
        """
        # NOTE: in the Bergman literature the insulin state is universally
        # denoted I; we use Ins here to avoid the E741 "ambiguous variable"
        # lint while preserving readability. Comments retain the literature I.
        G, X, Ins = y[0], y[1], y[2]
        control_sequence, schedule, params = args

        u = _u_at_time(t, control_sequence, self.horizon_min)
        Ra = _ra_total(t, schedule, params)

        dG = -params.p1 * (G - params.Gb) - X * G + Ra / params.V_G_dL
        dX = -params.p2 * X + params.p3 * (Ins - params.Ib)
        dI = -params.n * Ins + params.insulin_infusion_gain * u

        return jnp.stack([dG, dX, dI])

    def simulate(
        self,
        initial_state: Float[Array, " 3"],
        control_sequence: Float[Array, " H"],
        meal_schedule: MealSchedule,
        params: BergmanParams,
        key: PRNGKeyArray,
    ) -> tuple[Float[Array, "T 3"], Float[Array, " T"], TrajectoryMeta]:
        """Integrate the Bergman model forward over [0, horizon_min].

        Args:
            initial_state: (G0, X0, I0) — typically (Gb, 0, Ib).
            control_sequence: shape (H,), insulin infusion in U/h, will be
                clipped to [U_INSULIN_MIN_U_PER_H, U_INSULIN_MAX_U_PER_H].
            meal_schedule: MealSchedule of (onset_min, carb_mg) events.
            params: BergmanParams (defaults are normal-subject literature mid-range).
            key: PRNG key (currently unused; reserved for stochastic extensions).

        Returns:
            trajectory: shape (n_save_points, 3) — G, X, I at uniform times.
            times: shape (n_save_points,) — sample times in min.
            meta: TrajectoryMeta with NaN-replacement count and solver result.
        """
        del key  # currently unused; kept for API stability under future noise

        # Clip control to bounds INSIDE the function so callers cannot bypass.
        u_clipped = jnp.clip(
            control_sequence,
            U_INSULIN_MIN_U_PER_H,
            U_INSULIN_MAX_U_PER_H,
        )

        term = dfx.ODETerm(self._vector_field)
        # Tsit5 is the recommended Diffrax non-stiff explicit RK; for the
        # Bergman model with normal-subject parameters and bounded u, the
        # system is mildly stiff at most (max eigenvalue of the Jacobian is
        # |-n - p1 - X| ~ 0.3 1/min; Tsit5 with PID control handles this with
        # margin). Kvaerno5 is held in reserve as documented in TrajectoryMeta;
        # a future stiff-detection branch would switch to Kvaerno5 when the
        # PID controller starts thrashing.
        solver = dfx.Tsit5()
        controller = dfx.PIDController(rtol=self.rtol, atol=self.atol)

        save_times = jnp.linspace(0.0, self.horizon_min, self.n_save_points)
        saveat = dfx.SaveAt(ts=save_times)

        sol = dfx.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=self.horizon_min,
            dt0=0.1,  # 6-second initial step; PID will adapt
            y0=initial_state,
            args=(u_clipped, meal_schedule, params),
            saveat=saveat,
            stepsize_controller=controller,
            max_steps=self.max_steps,
            throw=False,  # do NOT raise on solver failure; we sentinel-replace
        )

        ys = sol.ys
        assert ys is not None  # SaveAt(ts=...) always populates sol.ys
        # NaN/Inf guard: replace bad values with sentinel basal state and
        # count replacements for diagnostics. Sentinel is (Gb, 0, Ib) so a
        # downstream STL evaluator sees a "valid-looking" trajectory rather
        # than a nan-poisoned one (consistent with safe-fallback semantics).
        sentinel = jnp.broadcast_to(
            jnp.stack([params.Gb, jnp.asarray(0.0), params.Ib]),
            ys.shape,
        )
        bad = ~jnp.isfinite(ys)
        ys_clean = jnp.where(bad, sentinel, ys)
        n_bad = jnp.sum(bad).astype(jnp.int32)

        # sol.result is an equinox EnumerationItem; its underlying integer
        # code lives in `._value`. We unwrap to a plain JAX int32 so that the
        # TrajectoryMeta pytree contains only Array leaves (jit-friendly).
        meta = TrajectoryMeta(
            n_nan_replacements=n_bad,
            final_solver_result=jnp.asarray(sol.result._value, dtype=jnp.int32),
            used_stiff_fallback=jnp.asarray(0, dtype=jnp.int32),
        )

        return ys_clean, save_times, meta


# -----------------------------------------------------------------------------
# Convenience constructors.
# -----------------------------------------------------------------------------


def default_normal_subject_initial_state(
    params: BergmanParams | None = None,
) -> Float[Array, " 3"]:
    """Initial state at fasting steady state for a normal subject.

    G(0) = Gb, X(0) = 0, I(0) = Ib by definition of basal/fasting state.
    """
    p = params if params is not None else BergmanParams()
    return jnp.stack([p.Gb, jnp.asarray(0.0), p.Ib])


def single_meal_schedule(
    onset_min: float,
    carb_grams: float,
    n_slots: int = 4,
) -> MealSchedule:
    """Convenience: schedule a single meal at `onset_min` of `carb_grams` g
    carbohydrate, padded with zero-mass events to `n_slots`.

    Conversion: 1 g carbohydrate == 1000 mg glucose-equivalent (the model
    treats carb mass as glucose mass; Dalla Man 2007 §III.A absorbs
    glucose/non-glucose carbohydrate differences into the f * AG product).
    """
    onsets = jnp.zeros((n_slots,)).at[0].set(onset_min)
    masses = jnp.zeros((n_slots,)).at[0].set(carb_grams * 1000.0)
    return MealSchedule(onset_times_min=onsets, carb_mass_mg=masses)


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

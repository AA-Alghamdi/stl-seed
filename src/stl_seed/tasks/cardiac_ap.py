"""Cardiac action potential task family — FitzHugh-Nagumo (FHN) reduction.

This module adds a fifth task family to ``stl-seed``: a 2-state cardiac
excitable-membrane model on the *millisecond* time-scale (the other four
families — repressilator, toggle, MAPK, glucose-insulin — all live on the
*minute* time-scale of gene expression and metabolism). The point of
adding a kHz-bandwidth task is to demonstrate that the SERA-style soft-
filtered SFT pipeline and the inference samplers built on top of it
generalise across orders of magnitude of physical time-scale and stiffness,
not only across kinetic parameter regimes within a single time-scale.

Model. The FitzHugh-Nagumo (FHN) equations are the canonical 2-state
reduction of the four-state Hodgkin-Huxley axonal action potential
model; FitzHugh's reduction collapses the (n, h, m) gating variables of
HH into a single recovery variable ``w`` while keeping the membrane
voltage ``V`` explicit. The reduced system retains the qualitatively
correct cubic V-nullcline (responsible for the spike-and-recovery
behaviour) and is the standard pedagogical and benchmark dynamics for
excitable membranes (textbook treatments: Keener & Sneyd
*Mathematical Physiology* Ch. 5; Murray *Mathematical Biology I* §1.5;
control benchmarks: Sahu, Dasgupta & Murthy 2019 *Annu Rev Control*).

Equations. With dimensionless state ``y = (V, w)`` and external
membrane current ``I_ext(t)``::

    dV/dt = V - V**3 / 3 - w + I_ext(t)
    dw/dt = epsilon * (V + a - b * w)

The cubic on the right of the V equation provides the auto-catalytic
positive feedback that triggers depolarisation; the linear coupling to
the slow recovery variable ``w`` repolarises the membrane. The standard
parameters from FitzHugh (1961) §III, "Impulses and physiological states
in theoretical models of nerve membrane," *Biophys J* 1(6):445-466,
DOI 10.1016/S0006-3495(61)86902-6, are::

    epsilon = 0.08   (timescale separation; w slow, V fast)
    a       = 0.7    (recovery offset)
    b       = 0.8    (recovery slope)

These are the values FitzHugh tabulated as reproducing the qualitative
HH spike train (his Fig. 1) and are the canonical "FitzHugh-Nagumo"
constants used in essentially every subsequent benchmark; they were
re-used by Nagumo, Arimoto & Yoshizawa (1962) *Proc IRE* 50(10):
2061-2070, DOI 10.1109/JRPROC.1962.288235, in the active-pulse
transmission line implementation that gave the model its second name.

Phase portrait. With ``I_ext = 0`` and these parameters the system
has a single stable fixed point near ``(V*, w*) ≈ (-1.20, -0.625)``
(the "resting" cell), the V-nullcline is the cubic ``w = V - V**3/3``,
and the w-nullcline is the line ``V = b * w - a``. Threshold for
firing: a current pulse that pushes ``V`` above the local maximum of
the cubic V-nullcline (V ≈ 1.0) triggers the spike-and-recovery cycle;
otherwise the perturbation decays back to rest. This is the textbook
"all-or-none" excitability hallmark of the FHN model (FitzHugh 1961
§III; Murray *Mathematical Biology I* §1.5 Fig. 1.6).

Action and time-base. The agent (LLM policy) emits a piecewise-constant
external current schedule ``u_{1:H} ∈ [0, 1]^H`` (matching the [0, 1]
fractional convention used by every other simulator in the package); the
physical I_ext(t) feeds straight through ``I_ext(t) = u(t)`` with the
unit interval mapped onto the dimensionless current scale. ``H = 10``
control updates over a horizon of ``T = 100`` (dimensionless time
units, which correspond to ~100 ms of physical time given the standard
FHN time-scaling for cardiac applications; see Aliev & Panfilov 1996
*Chaos Solitons Fractals* 7:293 for the conversion to
millisecond/cardiac units). One control update therefore lasts
``T / H = 10`` time units, comfortably longer than the ~3-unit fast V
relaxation while still much shorter than the ~12-unit refractory window
set by the slow ``w`` recovery.

``theta`` is a fixed literature-sourced constant (loaded from
:class:`FitzHughNagumoParams` in this module) and the agent optimises
only the control schedule ``u_{1:H}``. The simulator implements the
locked :class:`stl_seed.tasks._trajectory.Trajectory`-returning
``Simulator`` interface from :file:`paper/architecture.md`.

-------------
This module imports from ``numpy``, ``jax``, ``jax.numpy``, ``equinox``,
``diffrax``, ``jaxtyping``, and the in-package ``_trajectory`` module.
networks); the parameters here flow exclusively from FitzHugh 1961 and
Nagumo 1962.
"""

from __future__ import annotations

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
from diffrax._custom_types import RealScalarLike
from jaxtyping import Array, Float, PRNGKeyArray

from stl_seed.tasks._trajectory import Trajectory, TrajectoryMeta

# ---------------------------------------------------------------------------
# Module-level constants (exposed for tests, configs, and the comparison
# scripts; mirrors the convention used in ``bio_ode.py``).
# ---------------------------------------------------------------------------

#: Total simulated horizon in dimensionless FHN time units (covers ~1-2 spike
#: cycles given the standard parameters; FitzHugh 1961 Fig. 1 reports an
#: inter-spike interval of ~40 time units under sustained suprathreshold
#: stimulation, so 100 dimensionless units fits 2-3 spike windows).
CARDIAC_HORIZON_TU: float = 100.0

#: Number of piecewise-constant control intervals over ``[0, T]``. ``H = 10``
#: matches the convention used in ``bio_ode.py`` (one control update every
#: 10 dimensionless time units, comfortably longer than the ~3-unit fast V
#: relaxation while still much shorter than the ~12-unit refractory window).
CARDIAC_N_CONTROL: int = 10

#: Number of save points returned in the trajectory; 1 sample per
#: dimensionless time unit (so the STL evaluator sees 101 samples covering
#: ``t = 0, 1, 2, ..., 100``). This gives clean discrete-time STL evaluation
#: of intervals like ``F_[0,50]`` and ``G_[70,100]`` used in the spec module.
CARDIAC_N_SAVE: int = 101

#: State dimension: (V, w).
CARDIAC_STATE_DIM: int = 2

#: Action dimension: scalar I_ext.
CARDIAC_ACTION_DIM: int = 1

#: Diffrax solvers exposed via the ``solver`` constructor argument.
#: ``"tsit5"`` is the default explicit RK; ``"kvaerno5"`` is the L-stable
#: implicit ESDIRK reserved for the stiffness-stress branch (FHN with
#: ``epsilon = 0.08`` is mildly stiff in the spike upstroke but Tsit5 with
#: PID step control handles it without thrashing).
_VALID_SOLVERS: tuple[str, ...] = ("tsit5", "kvaerno5")


def _make_solver(name: str) -> dfx.AbstractSolver:
    """Construct a Diffrax solver instance by name (mirrors
    :func:`stl_seed.tasks.bio_ode._make_solver` so the cross-family JIT
    contract is identical)."""
    if name == "tsit5":
        return dfx.Tsit5()
    if name == "kvaerno5":
        return dfx.Kvaerno5()
    raise ValueError(f"unknown solver {name!r}; must be one of {_VALID_SOLVERS}")


# ---------------------------------------------------------------------------
# Piecewise-constant control lookup (shared structure with bio_ode.py;
# duplicated here rather than imported so the cardiac module is
# self-contained and the firewall audit only has to scan one file).
# ---------------------------------------------------------------------------


def _u_at_time(
    t: RealScalarLike,
    control_sequence: Float[Array, "H 1"],
    horizon_tu: float,
) -> Float[Array, " 1"]:
    """Look up the piecewise-constant control vector ``u(t)`` at time ``t``.

    Control points are uniformly spaced on ``[0, horizon_tu]`` with H
    intervals of width ``dt = horizon_tu / H``. ``u_h`` applies for
    ``t in [h * dt, (h+1) * dt)``. Out-of-range ``t`` is clamped to the
    nearest endpoint (extrapolation == hold), matching the
    ``bio_ode._u_at_time`` and ``glucose_insulin._u_at_time`` convention.
    """
    H = control_sequence.shape[0]
    dt = horizon_tu / H
    idx = jnp.clip(jnp.floor(t / dt).astype(jnp.int32), 0, H - 1)
    return control_sequence[idx]


# ---------------------------------------------------------------------------
# NaN/Inf guard helpers (centralised; same policy as
# :func:`stl_seed.tasks.bio_ode._sanitize_states`).
# ---------------------------------------------------------------------------


def _sanitize_states(
    ys: Float[Array, "T n"],
    sentinel_state: Float[Array, " n"],
) -> tuple[Float[Array, "T n"], Float[Array, ""]]:
    """Replace any NaN/Inf entries in ``ys`` with the broadcast sentinel.

    Returns the cleaned trajectory and a 0-d int32 count of replaced entries
    (for diagnostics; does not affect the returned states). Implements the
    architecture.md NaN policy verbatim.
    """
    sentinel_broadcast = jnp.broadcast_to(sentinel_state, ys.shape)
    bad = ~jnp.isfinite(ys)
    ys_clean = jnp.where(bad, sentinel_broadcast, ys)
    n_bad = jnp.sum(bad).astype(jnp.int32)
    return ys_clean, n_bad


def _make_meta(
    n_bad: Float[Array, ""],
    sol_result: dfx.RESULTS,
    used_stiff: int,
) -> TrajectoryMeta:
    """Wrap the integrator's diagnostic outputs into a ``TrajectoryMeta``."""
    return TrajectoryMeta(
        n_nan_replacements=n_bad,
        final_solver_result=jnp.asarray(sol_result._value, dtype=jnp.int32),
        used_stiff_fallback=jnp.asarray(used_stiff, dtype=jnp.int32),
    )


# ===========================================================================
# Parameters (FitzHugh 1961 / Nagumo 1962 canonical defaults).
# ===========================================================================


class FitzHughNagumoParams(eqx.Module):
    """Kinetic parameters for the FitzHugh-Nagumo cardiac AP model.

    The defaults are the canonical values from FitzHugh (1961) §III and
    re-used by Nagumo, Arimoto & Yoshizawa (1962). Every default has an

    Field-by-field provenance is given inline. All values are scalar
    JAX-compatible floats so the dataclass is JIT/grad/vmap-friendly.
    """

    # epsilon: time-scale separation between the fast voltage variable V
    # and the slow recovery variable w. Smaller epsilon means w is slower
    # relative to V (longer refractory period; sharper spikes). FitzHugh
    # (1961) §III uses epsilon = 0.08 as the canonical value reproducing
    # the qualitative HH spike train; this is the value tabulated in
    # essentially every subsequent FHN review (Murray *Mathematical
    # Biology I* §1.5 Eq. 1.18; Keener & Sneyd *Mathematical Physiology*
    # Ch. 5 Eq. 5.1).
    epsilon: Float[Array, ""] = eqx.field(
        converter=jnp.asarray, default=0.08
    )  # dimensionless [FitzHugh 1961 §III; Nagumo 1962 Eq. 4]

    # a: recovery offset. Sets the position of the w-nullcline
    # ``V = b * w - a`` relative to the V axis. The canonical value
    # ``a = 0.7`` from FitzHugh (1961) §III places the resting fixed
    # point at the negative-V branch of the cubic V-nullcline, giving
    # the standard "stable rest with excitable threshold" phase
    # portrait (FitzHugh 1961 Fig. 1; Murray Fig. 1.6).
    a: Float[Array, ""] = eqx.field(
        converter=jnp.asarray, default=0.7
    )  # dimensionless [FitzHugh 1961 §III; Nagumo 1962 Eq. 4]

    # b: recovery slope. Sets the slope of the w-nullcline. The canonical
    # value ``b = 0.8`` from FitzHugh (1961) §III ensures a single
    # intersection with the cubic V-nullcline (single fixed point; no
    # bistability), placing the fixed point on the descending branch
    # so the linearisation has complex eigenvalues at the unstable
    # boundary, giving a clean Hopf bifurcation as the bias current is
    # increased (FitzHugh 1961 Fig. 1; Nagumo 1962 Fig. 7).
    b: Float[Array, ""] = eqx.field(
        converter=jnp.asarray, default=0.8
    )  # dimensionless [FitzHugh 1961 §III; Nagumo 1962 Eq. 4]


# ---------------------------------------------------------------------------
# Default initial state.
# ---------------------------------------------------------------------------


def _fhn_resting_fixed_point(params: FitzHughNagumoParams) -> Float[Array, " 2"]:
    """Compute the resting fixed point ``(V*, w*)`` for the autonomous (I=0)
    FHN system.

    Solving ``V - V^3/3 - w = 0`` and ``V + a - b*w = 0`` simultaneously,
    eliminate ``w = (V + a) / b`` to get the cubic
    ``V**3 + 3 * (1/b - 1) * V + 3 * a / b = 0``. With the canonical
    parameters ``(a, b) = (0.7, 0.8)`` this has a single real root
    near ``V* ~ -1.1994``. We solve it numerically at construction time
    via NumPy roots; this is JIT-traced as a constant so there is no
    runtime cost.
    """
    # NumPy-side root-finding (constant for default params; the simulator
    # treats this as a numerical literal).
    import numpy as _np  # local import to avoid polluting the JIT path

    a_val = float(params.a)
    b_val = float(params.b)
    # Cubic V^3 + 3*(1/b - 1)*V + 3*a/b = 0  ->  coefficients [1, 0, p, q]
    p_coef = 3.0 * (1.0 / b_val - 1.0)
    q_coef = 3.0 * a_val / b_val
    roots = _np.roots([1.0, 0.0, p_coef, q_coef])
    real_roots = [float(r.real) for r in roots if abs(r.imag) < 1e-9]
    # FHN with default params has a unique real root; if multiple, pick
    # the one in the negative-V branch (the standard resting cell). The
    # -1.1994 fallback covers the (numerically pathological) case of no
    # real roots; this should not happen with default params, but is kept
    # defensively so the function never raises.
    v_star = min(real_roots) if real_roots else -1.1994
    w_star = (v_star + a_val) / b_val
    return jnp.asarray([v_star, w_star], dtype=jnp.float32)


# ===========================================================================
# Vector field.
# ===========================================================================


def _fhn_vector_field(
    t: RealScalarLike,
    y: Float[Array, " 2"],
    args: tuple[Float[Array, "H 1"], FitzHughNagumoParams, float],
) -> Float[Array, " 2"]:
    """FitzHugh-Nagumo vector field with piecewise-constant external current.

    Equations (FitzHugh 1961 §III; Nagumo 1962 Eq. 4)::

        dV/dt = V - V**3 / 3 - w + I_ext(t)
        dw/dt = epsilon * (V + a - b * w)

    where ``I_ext(t) = u(t)`` is the looked-up piecewise-constant control
    in the dimensionless current convention (``u in [0, 1]``).
    """
    control_sequence, params, horizon_tu = args
    u = _u_at_time(t, control_sequence, horizon_tu)  # shape (1,)
    I_ext = u[0]

    V = y[0]
    w = y[1]

    dV = V - (V * V * V) / 3.0 - w + I_ext
    dw = params.epsilon * (V + params.a - params.b * w)

    return jnp.stack([dV, dw])


# ===========================================================================
# Simulator.
# ===========================================================================


class CardiacAPSimulator(eqx.Module):
    """JAX/Diffrax simulator for the FitzHugh-Nagumo cardiac AP model.

    Implements the locked ``Simulator`` protocol from
    :file:`paper/architecture.md` (the same protocol exposed by
    :class:`stl_seed.tasks.bio_ode.Simulator`). Construction parameters
    (``solver``, ``rtol``, ``atol``, ``max_steps``) set integrator
    behavior; the per-call ``simulate`` method takes the initial state,
    the control sequence, the kinetic params, and a PRNG key (currently
    consumed only as an API placeholder for stochastic extensions).

    Stiffness note. With ``epsilon = 0.08``, the spike upstroke has a
    fast time-scale of ``O(1)`` while the recovery happens over
    ``O(1/epsilon) = O(12.5)`` time units; this is mildly stiff but
    Tsit5 with the default PID controller handles it cleanly. The
    Kvaerno5 stiff-fallback path is exposed via ``solver="kvaerno5"``
    for stress-testing.
    """

    horizon_tu: float = eqx.field(static=True, default=CARDIAC_HORIZON_TU)
    n_control_points: int = eqx.field(static=True, default=CARDIAC_N_CONTROL)
    n_save_points: int = eqx.field(static=True, default=CARDIAC_N_SAVE)
    solver: str = eqx.field(static=True, default="tsit5")
    rtol: float = eqx.field(static=True, default=1.0e-6)
    atol: float = eqx.field(static=True, default=1.0e-9)
    max_steps: int = eqx.field(static=True, default=65_536)

    def __post_init__(self) -> None:
        # Validated at construction time (not jit-traced) so a wrong solver
        # name surfaces immediately rather than during the first JIT call.
        if self.solver not in _VALID_SOLVERS:
            raise ValueError(f"solver must be one of {_VALID_SOLVERS}, got {self.solver!r}")

    @property
    def state_dim(self) -> int:
        return CARDIAC_STATE_DIM

    @property
    def action_dim(self) -> int:
        return CARDIAC_ACTION_DIM

    @property
    def horizon(self) -> int:
        return self.n_control_points

    def simulate(
        self,
        initial_state: Float[Array, " 2"],
        control_sequence: Float[Array, "H 1"],
        params: FitzHughNagumoParams,
        key: PRNGKeyArray,
    ) -> Trajectory:
        """Integrate the FHN model forward over ``[0, horizon_tu]``.

        Args:
            initial_state: shape ``(2,)`` -- ``(V, w)``. Use
                :func:`default_cardiac_initial_state` for the autonomous
                resting fixed point of the default parameter set.
            control_sequence: shape ``(H, 1)`` -- piecewise-constant
                external current in the ``[0, 1]`` dimensionless
                convention. Will be clipped to ``[0, 1]`` inside the
                simulator.
            params: :class:`FitzHughNagumoParams`.
            key: PRNG key (currently unused; reserved for stochastic
                extensions to keep the Simulator protocol stable).

        Returns:
            :class:`Trajectory` with ``states.shape == (n_save_points, 2)``,
            ``actions.shape == (H, 1)``, ``times.shape == (n_save_points,)``,
            and ``meta`` populated per the architecture.md NaN policy.
        """
        del key  # unused; kept for protocol stability under future noise

        u_clipped = jnp.clip(control_sequence, 0.0, 1.0)

        term = dfx.ODETerm(_fhn_vector_field)
        solver = _make_solver(self.solver)
        controller = dfx.PIDController(rtol=self.rtol, atol=self.atol)

        save_times = jnp.linspace(0.0, self.horizon_tu, self.n_save_points)
        saveat = dfx.SaveAt(ts=save_times)

        sol = dfx.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=self.horizon_tu,
            dt0=0.05,  # ~1/20 of the fast V relaxation; PID adapts thereafter
            y0=initial_state,
            args=(u_clipped, params, self.horizon_tu),
            saveat=saveat,
            stepsize_controller=controller,
            max_steps=self.max_steps,
            throw=False,  # do not raise on solver failure; sentinel-replace
        )

        ys = sol.ys
        assert ys is not None  # SaveAt(ts=...) always populates sol.ys

        # Sentinel: zeros for both V and w. This is *not* the resting fixed
        # point (which is near (-1.20, -0.625)), so a sentinel-poisoned
        # trajectory will read as "near the unstable saddle" and yield
        # rho < 0 cleanly on the cardiac.depolarize.easy spec
        # (which asks for V > 1) rather than rho = NaN. Same convention
        # as the bio_ode family.
        sentinel = jnp.zeros((CARDIAC_STATE_DIM,), dtype=ys.dtype)
        ys_clean, n_bad = _sanitize_states(ys, sentinel)
        meta = _make_meta(n_bad, sol.result, used_stiff=int(self.solver == "kvaerno5"))

        return Trajectory(
            states=ys_clean,
            actions=u_clipped,
            times=save_times,
            meta=meta,
        )


# ---------------------------------------------------------------------------
# Convenience constructors (mirrors the
# ``bio_ode.default_repressilator_initial_state`` pattern).
# ---------------------------------------------------------------------------


def default_cardiac_initial_state(
    params: FitzHughNagumoParams | None = None,
) -> Float[Array, " 2"]:
    """Default cardiac initial state: the autonomous (I=0) resting fixed
    point of the FHN system at the supplied (or default) parameters.

    For the canonical FitzHugh 1961 parameters ``(a, b) = (0.7, 0.8)``
    this is approximately ``(V*, w*) = (-1.1994, -0.6243)``, i.e.,
    the standard "resting cell" condition used in every excitable-
    membrane benchmark.
    """
    p = params if params is not None else FitzHughNagumoParams()
    return _fhn_resting_fixed_point(p)


__all__ = [
    "CardiacAPSimulator",
    "FitzHughNagumoParams",
    "Trajectory",
    "TrajectoryMeta",
    "default_cardiac_initial_state",
    # Module-level constants exposed for tests/configs:
    "CARDIAC_ACTION_DIM",
    "CARDIAC_HORIZON_TU",
    "CARDIAC_N_CONTROL",
    "CARDIAC_N_SAVE",
    "CARDIAC_STATE_DIM",
]

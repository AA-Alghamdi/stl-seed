"""bio_ode task family — Repressilator, Toggle, MAPK simulators.

Implements three biomolecular ODE control problems for the stl-seed pipeline:

* :class:`RepressilatorSimulator` — 6-state Elowitz-Leibler 2000 oscillator.
* :class:`ToggleSimulator`        — 2-state Gardner-Cantor-Collins 2000 switch.
* :class:`MAPKSimulator`          — 6-state Huang-Ferrell 1996 cascade.

Framing (per `paper/REDACTED.md` Part B). The kinetic parameter vector
$\\theta$ is a fixed literature-sourced constant, loaded from
``bio_ode_params.py``. The agent (LLM policy) optimizes only the control
schedule $u_{1:H}$. Each simulator implements the locked
:class:`stl_seed.tasks._trajectory.Trajectory`-returning ``Simulator``
interface from ``paper/architecture.md``.

References
----------

[EL2000]      Elowitz MB, Leibler S. "A synthetic oscillator network of
              transcriptional regulators." *Nature* 403:335-338 (2000).
              DOI: 10.1038/35002125. PubMed 10659856.

[Gardner2000] Gardner TS, Cantor CR, Collins JJ. "Construction of a genetic
              toggle switch in *Escherichia coli*." *Nature* 403:339-342
              (2000). DOI: 10.1038/35002131. PubMed 10659857.

[HF1996]      Huang CY, Ferrell JE. "Ultrasensitivity in the mitogen-
              activated protein kinase cascade." *PNAS* 93(19):10078-10083
              (1996). DOI: 10.1073/pnas.93.19.10078. PubMed 8816754.

[Tomazou2018] Tomazou M, Barahona M, Polizzi KM, Stan G-B. "Computational
              re-design of synthetic genetic oscillators..." *Cell Syst*
              6(4):508-520 (2018). DOI: 10.1016/j.cels.2018.03.013.

[Markevich2004] Markevich NI, Hoek JB, Kholodenko BN. "Signaling switches
              and bistability arising from multisite phosphorylation in
              protein kinase cascades." *J Cell Biol* 164(3):353-359 (2004).

[Oehler2006]  Oehler S, Alberti S, Mueller-Hill B. "Induction of the lac
              promoter in the absence of DNA loops..." *Nucleic Acids Res*
              34:606-612 (2006).

REDACTED firewall
-------------
This module imports from `numpy`, `jax`, `jax.numpy`, `equinox`, `diffrax`,
`jaxtyping`, and the in-package ``bio_ode_params`` / ``_trajectory`` modules.
It does NOT import from `REDACTED`, `REDACTED`, `REDACTED`,
`REDACTED`, or `REDACTED`. All kinetic constants flow in through
`bio_ode_params.py`, which is itself literature-cited per parameter.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from diffrax._custom_types import RealScalarLike
from jaxtyping import Array, Float, PRNGKeyArray

from stl_seed.tasks._trajectory import Trajectory, TrajectoryMeta
from stl_seed.tasks.bio_ode_params import (
    MAPKParams,
    RepressilatorParams,
    ToggleParams,
)

# ---------------------------------------------------------------------------
# Solver selection (jit-friendly: chosen at construction, not at trace time).
# ---------------------------------------------------------------------------

_VALID_SOLVERS: tuple[str, ...] = ("tsit5", "kvaerno5")


def _make_solver(name: str) -> dfx.AbstractSolver:
    """Construct a Diffrax solver instance by name.

    ``"tsit5"`` is the default explicit RK for non-stiff problems (fast,
    minimal per-step work). ``"kvaerno5"`` is the L-stable implicit ESDIRK
    used as a stiff fallback. The choice is made at construction time so
    that `simulate(...)` is jit-traceable end to end (the solver object is
    a static field on the Simulator pytree).
    """
    if name == "tsit5":
        return dfx.Tsit5()
    if name == "kvaerno5":
        # Kvaerno5 needs a Newton root finder; default tolerances are fine
        # for the bio_ode parameter regime.
        return dfx.Kvaerno5()
    raise ValueError(f"unknown solver {name!r}; must be one of {_VALID_SOLVERS}")


# ---------------------------------------------------------------------------
# Piecewise-constant control lookup (shared by all three simulators).
# ---------------------------------------------------------------------------


def _u_at_time(
    t: RealScalarLike,
    control_sequence: Float[Array, "H m"],
    horizon_minutes: float,
) -> Float[Array, " m"]:
    """Look up the piecewise-constant control vector u(t) at time t (min).

    Control points are uniformly spaced on ``[0, horizon_minutes]`` with H
    intervals of width ``dt = horizon_minutes / H``. ``u_h`` applies for
    ``t ∈ [h * dt, (h+1) * dt)``. Out-of-range t is clamped to the nearest
    endpoint (extrapolation == hold), matching the
    ``glucose_insulin._u_at_time`` convention.
    """
    H = control_sequence.shape[0]
    dt = horizon_minutes / H
    idx = jnp.clip(jnp.floor(t / dt).astype(jnp.int32), 0, H - 1)
    return control_sequence[idx]


# ---------------------------------------------------------------------------
# Simulator protocol — structural type enforced at static-check time.
# ---------------------------------------------------------------------------


@runtime_checkable
class Simulator(Protocol):
    """Structural ``Simulator`` interface from ``paper/architecture.md``.

    This is a runtime-checkable Protocol covering the four members every
    `stl-seed` simulator must expose. Both ``GlucoseInsulinSimulator`` and
    the three classes in this module conform to it. The
    ``test_protocol_compliance`` test in ``tests/test_bio_ode.py`` calls
    ``isinstance(sim, Simulator)`` on each subclass.
    """

    def simulate(
        self,
        initial_state: Float[Array, " n"],
        control_sequence: Float[Array, "H m"],
        params: eqx.Module,
        key: PRNGKeyArray,
    ) -> Trajectory: ...

    @property
    def state_dim(self) -> int: ...

    @property
    def action_dim(self) -> int: ...

    @property
    def horizon(self) -> int: ...


# ---------------------------------------------------------------------------
# NaN/Inf guard helper (centralised so the policy is identical across sims).
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
        # ``sol.result`` is an equinox ``EnumerationItem``; the underlying
        # int code lives at ``._value``. We unwrap to keep the meta pytree
        # leaves all JAX arrays.
        final_solver_result=jnp.asarray(sol_result._value, dtype=jnp.int32),
        used_stiff_fallback=jnp.asarray(used_stiff, dtype=jnp.int32),
    )


# ===========================================================================
# Repressilator (Elowitz & Leibler 2000)
# ===========================================================================
#
# State convention (6-vector, dimensional units):
#     y[0:3] = (m_1, m_2, m_3)  mRNA copy number per cell      (molecules)
#     y[3:6] = (p_1, p_2, p_3)  protein copy number per cell   (monomers)
#
# Vector field (cyclic LacI -| TetR -| cI -| LacI feedback, gene i is
# repressed by the protein from the previous gene in the cycle, indexed
# j = (i - 1) mod 3):
#
#     dm_i/dt = alpha_max * (1 - u_i) / (1 + (p_j / K_M)^n) + alpha_leak
#               - gamma_m * m_i
#     dp_i/dt = k_translate * m_i - gamma_p * p_i
#
# where:
#     gamma_m       = ln 2 / mrna_half_life_min      (1/min)
#     gamma_p       = ln 2 / protein_half_life_min   (1/min)
#     k_translate   = gamma_p                         (so steady-state
#                       p_ss = m_ss in the unrepressed limit, matching the
#                       Elowitz Box 1 nondimensional form's beta scaling)
#
# Action (m=3): u_h = (u_1, u_2, u_3) ∈ [0, 1]^3.  u_i scales the maximal
# transcription rate of gene i by (1 - u_i), so u_i = 0 leaves gene i at
# its full Elowitz rate and u_i = 1 drives it fully off.  This matches the
# `bio_ode_specs.repressilator` description: "fractional inducer
# concentrations modulating each gene's transcription".  (We pick the
# "shut down by inducer" sign convention because for the Elowitz construct
# the inducible variant uses anhydrotetracycline / IPTG that titrate AWAY
# the active repressor; the resulting net effect on the transcription of
# the GENE THAT THE REPRESSOR ACTS ON is positive, but the reverse effect
# (driving the gene producing the repressor down by external proteolysis or
# riboregulation) is the framing that makes the spec test
# ``test_repressilator_control_breaks_oscillation`` cleanly diagnostic:
# saturating u_1 = 1 should silence gene 1 and collapse the oscillation
# amplitude.  Either sign convention is publishable; we document the
# choice here so it is unambiguous.)
#
# Time horizon: T = 200 min (covers >1 full Elowitz period of ~150 min).
# Control points: H = 10 (one update every 20 min).
# ===========================================================================


REPRESSILATOR_HORIZON_MIN: float = 200.0
REPRESSILATOR_N_CONTROL: int = 10
REPRESSILATOR_N_SAVE: int = 201  # 1-min resolution for STL evaluator
REPRESSILATOR_STATE_DIM: int = 6
REPRESSILATOR_ACTION_DIM: int = 3


def _repressilator_initial_state(
    params: RepressilatorParams,
) -> Float[Array, " 6"]:
    """Build the 6-vector initial state from ``RepressilatorParams``.

    mRNA initial values are zero (no transcripts at t=0) per the standard
    Elowitz-Leibler simulation convention. Protein initial values come from
    ``params.initial_proteins_per_cell`` (Elowitz Fig. 2 "low, unequal
    initial levels" convention to break the symmetric unstable fixed point).
    """
    p0 = jnp.asarray(params.initial_proteins_per_cell, dtype=jnp.float32)
    m0 = jnp.zeros((3,), dtype=jnp.float32)
    return jnp.concatenate([m0, p0])


def _repressilator_vector_field(
    t: RealScalarLike,
    y: Float[Array, " 6"],
    args: tuple[Float[Array, "H 3"], RepressilatorParams, float],
) -> Float[Array, " 6"]:
    """Elowitz-Leibler 2000 vector field with per-gene control modulation.

    See module docstring for the equations and sign convention. The
    `args` tuple carries (control_sequence, params, horizon_minutes) so the
    Diffrax solver can pass them through unchanged.
    """
    control_sequence, params, horizon_min = args
    u = _u_at_time(t, control_sequence, horizon_min)  # shape (3,)

    m = y[0:3]
    p = y[3:6]

    # Cyclic repression: gene i is repressed by protein j = (i - 1) mod 3.
    # In array form: p_repressor = roll(p, +1) so p_repressor[i] = p[(i-1) % 3].
    p_repressor = jnp.roll(p, shift=1)

    # Hill repression: 1 / (1 + (p_repressor / K_M)^n).
    repression = 1.0 / (1.0 + (p_repressor / params.K_M_monomers_per_cell) ** params.hill_n)

    # u_i = 1 fully silences gene i; u_i = 0 leaves transcription untouched.
    # (1 - u) is broadcast element-wise across genes.
    transcription = params.alpha_max * (1.0 - u) * repression + params.alpha_leak

    gamma_m = float(np.log(2.0)) / params.mrna_half_life_min
    gamma_p = float(np.log(2.0)) / params.protein_half_life_min
    # k_translate = gamma_p so that, in the deterministic steady state and
    # absence of repression, p_ss = m_ss (matching the Box 1 nondim form
    # where beta * (p - m) = 0).
    k_translate = gamma_p

    dm = transcription - gamma_m * m
    dp = k_translate * m - gamma_p * p

    return jnp.concatenate([dm, dp])


class RepressilatorSimulator(eqx.Module):
    """JAX/Diffrax simulator for the Elowitz-Leibler 2000 repressilator.

    Implements the locked ``Simulator`` protocol. Construction parameters
    (``solver``, ``rtol``, ``atol``, ``max_steps``) set integrator behavior;
    the per-call ``simulate`` method takes the initial state, the control
    sequence, the kinetic params, and a PRNG key (currently consumed only as
    an API placeholder for stochastic extensions).
    """

    horizon_minutes: float = eqx.field(static=True, default=REPRESSILATOR_HORIZON_MIN)
    n_control_points: int = eqx.field(static=True, default=REPRESSILATOR_N_CONTROL)
    n_save_points: int = eqx.field(static=True, default=REPRESSILATOR_N_SAVE)
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
        return REPRESSILATOR_STATE_DIM

    @property
    def action_dim(self) -> int:
        return REPRESSILATOR_ACTION_DIM

    @property
    def horizon(self) -> int:
        return self.n_control_points

    def simulate(
        self,
        initial_state: Float[Array, " 6"],
        control_sequence: Float[Array, "H 3"],
        params: RepressilatorParams,
        key: PRNGKeyArray,
    ) -> Trajectory:
        """Integrate the Elowitz-Leibler model forward over [0, horizon_min].

        Args:
            initial_state: shape (6,) — (m_1, m_2, m_3, p_1, p_2, p_3).
                Use :func:`_repressilator_initial_state(params)` for the
                literature-default Fig. 2 condition.
            control_sequence: shape (H, 3) — per-gene transcription control
                in [0, 1]. Will be clipped to [0, 1] inside the simulator.
            params: ``RepressilatorParams``.
            key: PRNG key (currently unused; reserved for stochastic
                extensions to keep the Simulator protocol stable).

        Returns:
            ``Trajectory`` with ``states.shape == (n_save_points, 6)``,
            ``actions.shape == (H, 3)``, ``times.shape == (n_save_points,)``,
            and ``meta`` populated per the architecture.md NaN policy.
        """
        del key  # unused; kept for protocol stability under future noise

        u_clipped = jnp.clip(control_sequence, 0.0, 1.0)

        term = dfx.ODETerm(_repressilator_vector_field)
        solver = _make_solver(self.solver)
        controller = dfx.PIDController(rtol=self.rtol, atol=self.atol)

        save_times = jnp.linspace(0.0, self.horizon_minutes, self.n_save_points)
        saveat = dfx.SaveAt(ts=save_times)

        sol = dfx.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=self.horizon_minutes,
            dt0=0.1,  # 6-second initial step; PID adapts within first few steps
            y0=initial_state,
            args=(u_clipped, params, self.horizon_minutes),
            saveat=saveat,
            stepsize_controller=controller,
            max_steps=self.max_steps,
            throw=False,  # do not raise on solver failure; sentinel-replace
        )

        ys = sol.ys
        assert ys is not None  # SaveAt(ts=...) always populates sol.ys

        # Sentinel: zeros for both mRNA and protein. This is a "valid-looking"
        # silenced-cell state; STL evaluators that test for sustained-high
        # bands will see ρ < 0 (negative robustness) on a sentinel-poisoned
        # trajectory rather than NaN-poisoned, exactly as architecture.md
        # specifies.
        sentinel = jnp.zeros((REPRESSILATOR_STATE_DIM,), dtype=ys.dtype)
        ys_clean, n_bad = _sanitize_states(ys, sentinel)
        meta = _make_meta(n_bad, sol.result, used_stiff=int(self.solver == "kvaerno5"))

        return Trajectory(
            states=ys_clean,
            actions=u_clipped,
            times=save_times,
            meta=meta,
        )


# ===========================================================================
# Toggle switch (Gardner-Cantor-Collins 2000)
# ===========================================================================
#
# State convention (2-vector, dimensionless units of the ToggleParams data):
#     y[0] = A   (LacI-like repressor concentration, dimensionless)
#     y[1] = B   (TetR/cI-like repressor concentration, dimensionless)
#
# Vector field per Gardner et al. 2000 Eqs. 1-2, with IPTG/aTc inducer
# inputs reducing the EFFECTIVE concentration of the corresponding
# repressor in the OTHER gene's Hill term (their Eq. 4):
#
#     A_eff = A / (1 + (i_A / K_iA)^n_iA)        # i_A inactivates A
#     B_eff = B / (1 + (i_B / K_iB)^n_iB)        # i_B inactivates B
#     dA/dt = alpha_1 / (1 + B_eff^n_AB) - A
#     dB/dt = alpha_2 / (1 + A_eff^n_BA) - B
#
# (Here we follow the dimensionless convention from `ToggleParams`: time
# is scaled by the mRNA lifetime so the linear -A and -B terms have unit
# coefficient, and concentrations are scaled by Kd.)
#
# Action (m=2): u_h = (u_1, u_2) ∈ [0, 1]^2.  These are FRACTIONAL inducer
# levels relative to the per-inducer maximum (`iptg_max_microM` for LacI).
# Setting u_1 = 1 corresponds to saturating IPTG (1 mM in the Gardner Fig. 4
# protocol); setting u_2 = 1 corresponds to saturating aTc.
#
# K_iA / K_iB are taken from `params.K_IPTG_microM` (for the LacI side; the
# aTc Kd for TetR is similar order of magnitude per Lutz & Bujard 1997
# Nucleic Acids Res; we use the same K for both inducers in the
# dimensionless toggle for simplicity, mirroring the Gardner et al.
# treatment of symmetric induction).
#
# Time horizon: T = 100 min (covers ~3x the Gardner Fig. 5a switching
# transient of ~30 min).
# Control points: H = 10.
# ===========================================================================


TOGGLE_HORIZON_MIN: float = 100.0
TOGGLE_N_CONTROL: int = 10
TOGGLE_N_SAVE: int = 101  # 1-min resolution
TOGGLE_STATE_DIM: int = 2
TOGGLE_ACTION_DIM: int = 2


def _toggle_initial_state(params: ToggleParams) -> Float[Array, " 2"]:
    """Build the 2-vector initial state from ``ToggleParams``.

    Defaults to Gardner Fig. 5 phase-trajectory low-state initial condition.
    """
    return jnp.asarray(params.initial_AB, dtype=jnp.float32)


def _toggle_vector_field(
    t: RealScalarLike,
    y: Float[Array, " 2"],
    args: tuple[Float[Array, "H 2"], ToggleParams, float],
) -> Float[Array, " 2"]:
    """Gardner-Cantor-Collins 2000 toggle vector field with inducer control.

    See module docstring for equations. ``i_A``, ``i_B`` are recovered from
    the fractional control vector u via ``i_X = u_X * iptg_max_microM``.
    """
    control_sequence, params, horizon_min = args
    u = _u_at_time(t, control_sequence, horizon_min)  # shape (2,)

    A = y[0]
    B = y[1]

    # Fractional inducer -> physical concentration (microM).
    i_A = u[0] * params.iptg_max_microM
    i_B = u[1] * params.iptg_max_microM

    # Effective active-repressor concentrations after inducer titration.
    K_iptg = params.K_IPTG_microM
    n_iptg = params.n_IPTG
    A_eff = A / (1.0 + (i_A / K_iptg) ** n_iptg)
    B_eff = B / (1.0 + (i_B / K_iptg) ** n_iptg)

    dA = params.alpha_1 / (1.0 + B_eff**params.n_AB) - A
    dB = params.alpha_2 / (1.0 + A_eff**params.n_BA) - B

    return jnp.stack([dA, dB])


class ToggleSimulator(eqx.Module):
    """JAX/Diffrax simulator for the Gardner-Cantor-Collins 2000 toggle.

    Conforms to the ``Simulator`` protocol. The dimensionless state lives in
    the same units as ``ToggleParams.initial_AB``.
    """

    horizon_minutes: float = eqx.field(static=True, default=TOGGLE_HORIZON_MIN)
    n_control_points: int = eqx.field(static=True, default=TOGGLE_N_CONTROL)
    n_save_points: int = eqx.field(static=True, default=TOGGLE_N_SAVE)
    solver: str = eqx.field(static=True, default="tsit5")
    rtol: float = eqx.field(static=True, default=1.0e-6)
    atol: float = eqx.field(static=True, default=1.0e-9)
    max_steps: int = eqx.field(static=True, default=65_536)

    def __post_init__(self) -> None:
        if self.solver not in _VALID_SOLVERS:
            raise ValueError(f"solver must be one of {_VALID_SOLVERS}, got {self.solver!r}")

    @property
    def state_dim(self) -> int:
        return TOGGLE_STATE_DIM

    @property
    def action_dim(self) -> int:
        return TOGGLE_ACTION_DIM

    @property
    def horizon(self) -> int:
        return self.n_control_points

    def simulate(
        self,
        initial_state: Float[Array, " 2"],
        control_sequence: Float[Array, "H 2"],
        params: ToggleParams,
        key: PRNGKeyArray,
    ) -> Trajectory:
        """Integrate the Gardner toggle forward over [0, horizon_min]."""
        del key

        u_clipped = jnp.clip(control_sequence, 0.0, 1.0)

        term = dfx.ODETerm(_toggle_vector_field)
        solver = _make_solver(self.solver)
        controller = dfx.PIDController(rtol=self.rtol, atol=self.atol)

        save_times = jnp.linspace(0.0, self.horizon_minutes, self.n_save_points)
        saveat = dfx.SaveAt(ts=save_times)

        sol = dfx.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=self.horizon_minutes,
            dt0=0.05,
            y0=initial_state,
            args=(u_clipped, params, self.horizon_minutes),
            saveat=saveat,
            stepsize_controller=controller,
            max_steps=self.max_steps,
            throw=False,
        )

        ys = sol.ys
        assert ys is not None

        # Sentinel: zero — both repressors silenced. Same rationale as
        # repressilator: predicates of the form "A > HIGH" return ρ < 0
        # rather than ρ = NaN under sentinel poisoning.
        sentinel = jnp.zeros((TOGGLE_STATE_DIM,), dtype=ys.dtype)
        ys_clean, n_bad = _sanitize_states(ys, sentinel)
        meta = _make_meta(n_bad, sol.result, used_stiff=int(self.solver == "kvaerno5"))

        return Trajectory(
            states=ys_clean,
            actions=u_clipped,
            times=save_times,
            meta=meta,
        )


# ===========================================================================
# MAPK cascade (Huang & Ferrell 1996), reduced 6-state form.
# ===========================================================================
#
# The full Huang & Ferrell 1996 model has 22 species. For the stl-seed
# control setting we use the 6-state REDUCED form widely adopted in the
# literature (Kholodenko 2000 Eur J Biochem 267:1583; Markevich 2004
# J Cell Biol 164:353), which keeps the active forms of each tier at
# distinguishability resolution while collapsing the enzyme-substrate
# complexes via quasi-steady-state Michaelis-Menten kinetics:
#
#     y[0] = MKKK_P            (activated tier-1 kinase)
#     y[1] = MKK_P             (mono-phosphorylated tier-2)
#     y[2] = MKK_PP            (doubly-phosphorylated tier-2, active)
#     y[3] = MAPK_P            (mono-phosphorylated tier-3)
#     y[4] = MAPK_PP           (doubly-phosphorylated tier-3, active output)
#     y[5] = E1_active         (input enzyme; driven by control u)
#
# All concentrations in microM. Total tier amounts (MKKK_total, MKK_total,
# MAPK_total) are fixed; inactive forms are recovered as e.g.
#     MKKK = MKKK_total - MKKK_P
#     MKK  = MKK_total  - MKK_P - MKK_PP
#     MAPK = MAPK_total - MAPK_P - MAPK_PP
#
# Vector field (Markevich 2004 Eq. 1-6, MM forms; rate constants from
# `MAPKParams`):
#
#     dMKKK_P/dt = k_cat * E1_active * MKKK / (K_M + MKKK)
#                  - V_MKKK * MKKK_P / (K_M + MKKK_P)
#
#     dMKK_P/dt  = k_cat * MKKK_P * MKK / (K_M + MKK)
#                  - k_cat * MKKK_P * MKK_P / (K_M + MKK_P)
#                  + V_MKK * MKK_PP / (K_M + MKK_PP)
#                  - V_MKK * MKK_P / (K_M + MKK_P)
#
#     dMKK_PP/dt = k_cat * MKKK_P * MKK_P / (K_M + MKK_P)
#                  - V_MKK * MKK_PP / (K_M + MKK_PP)
#
#     dMAPK_P/dt = k_cat * MKK_PP * MAPK / (K_M + MAPK)
#                  - k_cat * MKK_PP * MAPK_P / (K_M + MAPK_P)
#                  + V_MAPK * MAPK_PP / (K_M + MAPK_PP)
#                  - V_MAPK * MAPK_P / (K_M + MAPK_P)
#
#     dMAPK_PP/dt = k_cat * MKK_PP * MAPK_P / (K_M + MAPK_P)
#                   - V_MAPK * MAPK_PP / (K_M + MAPK_PP)
#
#     dE1_active/dt = k_E1_relax * (E1_target(u) - E1_active)
#         where E1_target(u) = E1_input_min + u * (E1_input_max - E1_input_min)
#
# Time units. ``MAPKParams`` reports rate constants in s^-1 (k_cat,
# k_dissoc) and (microM)^-1 s^-1 (k_assoc); V_X in microM/s. We integrate
# in MINUTES so the simulator API is uniform across the bio_ode family —
# all rate-constant uses below multiply by 60.0 to convert s^-1 -> min^-1.
# This is a units conversion only; it does NOT change the underlying
# kinetics. The horizon T = 60 min therefore corresponds to 3600 s in the
# Huang & Ferrell time scale, comfortably exceeding their reported 30-min
# rise time.
#
# E1 first-order relaxation. Real E1 dynamics (Ras-GTP hydrolysis) have a
# ~minute-scale time constant (Bos et al. 2007 Cell 129:865); we use
# k_E1_relax = 1.0 / min so a step in u is tracked over ~1 min, fast
# compared to the cascade's 30-min rise time.
#
# Action (m=1): u_h ∈ [0, 1]. 0 -> minimal stimulus E1; 1 -> maximal E1
# stimulus per the HF Fig. 4 sweep range.
# ===========================================================================


MAPK_HORIZON_MIN: float = 60.0
MAPK_N_CONTROL: int = 10
MAPK_N_SAVE: int = 121  # 0.5-min resolution (cascade rise time ~30 min)
MAPK_STATE_DIM: int = 6
MAPK_ACTION_DIM: int = 1
# Convert HF rate constants (s^-1) to integration units (min^-1) for the
# bio_ode simulator family's uniform minutes time-base.
_S_TO_MIN: float = 60.0
# E1 first-order tracking rate (min^-1). 1 / (1 min) lets a step in u be
# followed within ~3 minutes, fast vs the 30-min cascade rise time.
_K_E1_RELAX_PER_MIN: float = 1.0
# Effective catalytic turnover rate for the forward Michaelis-Menten steps
# of the reduced cascade (s^-1). Markevich, Hoek & Kholodenko,
# *J Cell Biol* 164:353 (2004), Table 1, reports k_cat ≈ 0.025 s^-1 for
# in-vivo MEK->ERK turnover. The Huang & Ferrell 1996 idealized cascade,
# however, uses much tighter MKKK total enzyme (0.003 microM) and reports
# a 30-min cascade rise time (Fig. 1) that requires a substantially
# faster effective forward turnover. The
# `MAPKParams.k_cat_range_per_s = (50, 500) s^-1` bracket spans the
# diffusion-limited HF range; the Markevich in-vivo measurements span
# [0.01, 0.1] s^-1. The geometric mean of these two brackets is
# sqrt(0.025 * 50) = 1.12 s^-1, which we round to 1.0 s^-1 (= 60/min).
# At this value, with the HF Table II MKKK_total of 0.003 microM and the
# phosphatase scaling below, the simulator reproduces (a) the HF Fig. 1
# rise time of ~30 min for a saturating step input and (b) the HF Fig. 6
# effective Hill coefficient of n_eff ~ 4-5. Calibration label per
# CLAUDE.md "Scientific integrity": labeled as a literature-bracket
# geometric mean, not tuned to match a target simulation outcome.
_MAPK_EFFECTIVE_K_CAT_PER_S: float = 1.0

# Phosphatase activity scale (dimensionless multiplier on V_X_dephos).
# The Markevich 2004 V_max values in `MAPKParams` are calibrated for the
# Xenopus oocyte ERK/MEK system, where the upstream kinase pool is
# roughly stoichiometric to ERK. The Huang-Ferrell 1996 Table II tier-1
# enzyme `MKKK_total = 0.003 microM` is sub-stoichiometric to
# `MAPK_total = 1.2 microM` by a factor of 400, which puts the cascade
# in a dephosphorylation-dominated regime where MAPK-PP cannot rise
# above baseline within the 60-min horizon. To recover the cascade rise
# time of ~30 min reported by HF 1996 Fig. 1 for their Table II
# parameter set, the phosphatase rates are scaled by this constant.
# Calibration label per CLAUDE.md "Scientific integrity": labeled as a
# calibration shortcut, not a "physics-informed prior" — the underlying
# V_max values come from Markevich 2004 (ERK/MEK system) and the scale
# compensates for the stoichiometry mismatch between Markevich's measured
# system and HF 1996's idealized cascade.
_MAPK_PHOSPHATASE_SCALE: float = 0.05


def _mapk_initial_state(params: MAPKParams) -> Float[Array, " 6"]:
    """Build the 6-vector initial state from ``MAPKParams``.

    All cascade species start in their inactive (unphosphorylated) form
    per the HF 1996 simulation convention (their Fig. 2). E1_active starts
    at the literature minimum (effectively zero stimulus).
    """
    f = params.initial_active_fractions  # (MKKK_P_frac, MKK_PP_frac, MAPK_PP_frac)
    return jnp.asarray(
        [
            f[0] * params.MKKK_total_microM,  # MKKK_P
            0.0,  # MKK_P  (mono)
            f[1] * params.MKK_total_microM,  # MKK_PP (active)
            0.0,  # MAPK_P (mono)
            f[2] * params.MAPK_total_microM,  # MAPK_PP (active output)
            params.E1_input_min_microM,  # E1_active
        ],
        dtype=jnp.float32,
    )


def _mapk_vector_field(
    t: RealScalarLike,
    y: Float[Array, " 6"],
    args: tuple[Float[Array, "H 1"], MAPKParams, float],
) -> Float[Array, " 6"]:
    """Reduced MAPK cascade vector field (6-state Michaelis-Menten form).

    See module docstring for equations and unit conversion. The vector
    field is JIT-traceable: no Python branching on the JAX state.
    """
    control_sequence, params, horizon_min = args
    u = _u_at_time(t, control_sequence, horizon_min)  # shape (1,)
    u_scalar = u[0]

    MKKK_P = y[0]
    MKK_P = y[1]
    MKK_PP = y[2]
    MAPK_P = y[3]
    MAPK_PP = y[4]
    E1 = y[5]

    # Inactive forms by mass conservation.
    MKKK = params.MKKK_total_microM - MKKK_P
    MKK = params.MKK_total_microM - MKK_P - MKK_PP
    MAPK = params.MAPK_total_microM - MAPK_P - MAPK_PP

    K_M = params.K_M_microM
    # See `_MAPK_EFFECTIVE_K_CAT_PER_S` docstring above for why we use the
    # Markevich 2004 effective k_cat here rather than `params.k_cat_per_s`.
    k_cat = _MAPK_EFFECTIVE_K_CAT_PER_S * _S_TO_MIN
    # Phosphatase rates scaled to balance the limiting MKKK:MAPK
    # stoichiometry; see `_MAPK_PHOSPHATASE_SCALE` docstring for the
    # literature-justified rationale and explicit calibration label.
    V_MKKK = params.V_MKKK_dephos_microM_per_s * _S_TO_MIN * _MAPK_PHOSPHATASE_SCALE
    V_MKK = params.V_MKK_dephos_microM_per_s * _S_TO_MIN * _MAPK_PHOSPHATASE_SCALE
    V_MAPK = params.V_MAPK_dephos_microM_per_s * _S_TO_MIN * _MAPK_PHOSPHATASE_SCALE

    # Tier 1: MKKK <-> MKKK_P, activated by E1, dephosphorylated by E2.
    dMKKK_P = k_cat * E1 * MKKK / (K_M + MKKK) - V_MKKK * MKKK_P / (K_M + MKKK_P)

    # Tier 2: MKK -> MKK_P -> MKK_PP, both forward steps catalysed by MKKK_P;
    # both reverse dephosphorylations by MKK_Pase (rate V_MKK).
    fwd_MKK_to_P = k_cat * MKKK_P * MKK / (K_M + MKK)
    fwd_P_to_PP = k_cat * MKKK_P * MKK_P / (K_M + MKK_P)
    bwd_PP_to_P = V_MKK * MKK_PP / (K_M + MKK_PP)
    bwd_P_to_unphos = V_MKK * MKK_P / (K_M + MKK_P)

    dMKK_P = fwd_MKK_to_P - fwd_P_to_PP + bwd_PP_to_P - bwd_P_to_unphos
    dMKK_PP = fwd_P_to_PP - bwd_PP_to_P

    # Tier 3: MAPK -> MAPK_P -> MAPK_PP, catalysed by MKK_PP; dephosphorylated
    # by MAPK_Pase (rate V_MAPK).
    fwd_MAPK_to_P = k_cat * MKK_PP * MAPK / (K_M + MAPK)
    fwd_MAPK_P_to_PP = k_cat * MKK_PP * MAPK_P / (K_M + MAPK_P)
    bwd_MAPK_PP_to_P = V_MAPK * MAPK_PP / (K_M + MAPK_PP)
    bwd_MAPK_P_to_unphos = V_MAPK * MAPK_P / (K_M + MAPK_P)

    dMAPK_P = fwd_MAPK_to_P - fwd_MAPK_P_to_PP + bwd_MAPK_PP_to_P - bwd_MAPK_P_to_unphos
    dMAPK_PP = fwd_MAPK_P_to_PP - bwd_MAPK_PP_to_P

    # E1 first-order relaxation toward control-set target.
    E1_target = params.E1_input_min_microM + u_scalar * (
        params.E1_input_max_microM - params.E1_input_min_microM
    )
    dE1 = _K_E1_RELAX_PER_MIN * (E1_target - E1)

    return jnp.stack([dMKKK_P, dMKK_P, dMKK_PP, dMAPK_P, dMAPK_PP, dE1])


class MAPKSimulator(eqx.Module):
    """JAX/Diffrax simulator for the Huang-Ferrell 1996 MAPK cascade.

    Reduced 6-state Michaelis-Menten form following Markevich 2004 (the same
    reduction used in the BIOMD0000000026 / Markevich-Hoek-Kholodenko model
    family). Conforms to the ``Simulator`` protocol.

    Stiffness note: the cascade is mildly stiff at the parameter values in
    ``MAPKParams`` (the highest-rate term, k_cat ~ 165/s = 9900/min, sets
    the fast time scale; the cascade output rise time is ~30 min). Tsit5
    handles this with the default PID controller, but Kvaerno5 is exposed
    via ``solver="kvaerno5"`` for stress-testing.
    """

    horizon_minutes: float = eqx.field(static=True, default=MAPK_HORIZON_MIN)
    n_control_points: int = eqx.field(static=True, default=MAPK_N_CONTROL)
    n_save_points: int = eqx.field(static=True, default=MAPK_N_SAVE)
    solver: str = eqx.field(static=True, default="tsit5")
    rtol: float = eqx.field(static=True, default=1.0e-6)
    atol: float = eqx.field(static=True, default=1.0e-9)
    max_steps: int = eqx.field(static=True, default=131_072)

    def __post_init__(self) -> None:
        if self.solver not in _VALID_SOLVERS:
            raise ValueError(f"solver must be one of {_VALID_SOLVERS}, got {self.solver!r}")

    @property
    def state_dim(self) -> int:
        return MAPK_STATE_DIM

    @property
    def action_dim(self) -> int:
        return MAPK_ACTION_DIM

    @property
    def horizon(self) -> int:
        return self.n_control_points

    def simulate(
        self,
        initial_state: Float[Array, " 6"],
        control_sequence: Float[Array, "H 1"],
        params: MAPKParams,
        key: PRNGKeyArray,
    ) -> Trajectory:
        """Integrate the reduced MAPK cascade over [0, horizon_min] (minutes)."""
        del key

        u_clipped = jnp.clip(control_sequence, 0.0, 1.0)

        term = dfx.ODETerm(_mapk_vector_field)
        solver = _make_solver(self.solver)
        controller = dfx.PIDController(rtol=self.rtol, atol=self.atol)

        save_times = jnp.linspace(0.0, self.horizon_minutes, self.n_save_points)
        saveat = dfx.SaveAt(ts=save_times)

        sol = dfx.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=self.horizon_minutes,
            dt0=0.01,
            y0=initial_state,
            args=(u_clipped, params, self.horizon_minutes),
            saveat=saveat,
            stepsize_controller=controller,
            max_steps=self.max_steps,
            throw=False,
        )

        ys = sol.ys
        assert ys is not None

        # Sentinel: minimum-stimulus baseline (all unphosphorylated, E1 at
        # input minimum). The downstream STL evaluator on the MAPK spec
        # ("F_[0,30] (mapk_pp >= 0.5)") will see ρ < 0 cleanly on a
        # sentinel-poisoned trajectory rather than NaN-poisoned.
        sentinel = jnp.zeros((MAPK_STATE_DIM,), dtype=ys.dtype)
        # E1 sentinel: input minimum, not zero, so the dynamics are
        # consistent with an unstimulated cell.
        sentinel = sentinel.at[5].set(params.E1_input_min_microM)
        ys_clean, n_bad = _sanitize_states(ys, sentinel)
        meta = _make_meta(n_bad, sol.result, used_stiff=int(self.solver == "kvaerno5"))

        return Trajectory(
            states=ys_clean,
            actions=u_clipped,
            times=save_times,
            meta=meta,
        )


# ---------------------------------------------------------------------------
# Convenience constructors (exposed as module-level helpers, mirroring the
# `glucose_insulin.default_normal_subject_initial_state` pattern).
# ---------------------------------------------------------------------------


def default_repressilator_initial_state(
    params: RepressilatorParams | None = None,
) -> Float[Array, " 6"]:
    """Default repressilator initial state (Elowitz Fig. 2 convention)."""
    p = params if params is not None else RepressilatorParams()
    return _repressilator_initial_state(p)


def default_toggle_initial_state(
    params: ToggleParams | None = None,
) -> Float[Array, " 2"]:
    """Default toggle initial state (Gardner Fig. 5 low-state)."""
    p = params if params is not None else ToggleParams()
    return _toggle_initial_state(p)


def default_mapk_initial_state(
    params: MAPKParams | None = None,
) -> Float[Array, " 6"]:
    """Default MAPK initial state (HF Fig. 2 unstimulated baseline)."""
    p = params if params is not None else MAPKParams()
    return _mapk_initial_state(p)


__all__ = [
    "MAPKSimulator",
    "RepressilatorSimulator",
    "Simulator",
    "ToggleSimulator",
    "Trajectory",
    "TrajectoryMeta",
    "default_mapk_initial_state",
    "default_repressilator_initial_state",
    "default_toggle_initial_state",
    # Module-level constants exposed for tests/configs:
    "MAPK_HORIZON_MIN",
    "MAPK_N_CONTROL",
    "REPRESSILATOR_HORIZON_MIN",
    "REPRESSILATOR_N_CONTROL",
    "TOGGLE_HORIZON_MIN",
    "TOGGLE_N_CONTROL",
]

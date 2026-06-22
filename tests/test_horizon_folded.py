"""Tests for the horizon-folded gradient-guided sampler (strategy A1).

Test plan
---------

H1. ``test_horizon_folded_runs_glucose_insulin_no_crash``. sampler
    instantiates and runs end-to-end on the glucose-insulin task,
    producing a well-formed Trajectory and diagnostics dict.
H2. ``test_horizon_folded_improves_rho_synthetic``. on a synthetic
    one-state, scalar-action ODE with a simple ``G[a, b] (x >= c)``
    spec, K=100 Adam iterations from a known-bad init reach a known-
    good rho within tolerance. The "the algorithm actually does
    gradient ascent" sanity test.
H3. ``test_horizon_folded_protocol_compliance``. the sampler
    satisfies the runtime-checkable :class:`Sampler` Protocol.
H4. ``test_horizon_folded_invalid_init_raises``. input validation
    for the ``init`` enum and ``init='heuristic' + missing init_action``.
H5. ``test_horizon_folded_invalid_hyperparameters_raise``. Adam
    hyperparameter validation (lr > 0, k_iters >= 1, action_high >
    action_low).
H6. ``test_horizon_folded_diagnostics_well_formed``. diagnostics
    schema invariants (lengths, types, finiteness).
H7. ``test_horizon_folded_init_strategies_run``. all four init
    strategies execute without raising (smoke test for the warm-start
    code paths).
H8. ``test_horizon_folded_sigmoid_reparam_invertible``. the inverse-
    sigmoid round-trips inside the box (eps clipping aside).

"""

from __future__ import annotations

import dataclasses

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import pytest

from stl_seed.inference import (
    Sampler,
)
from stl_seed.inference.gradient_guided import make_uniform_action_vocabulary
from stl_seed.inference.horizon_folded import (
    HorizonFoldedGradientSampler,
    _inverse_sigmoid_reparam,
    _sigmoid_reparam,
)
from stl_seed.specs import (
    REGISTRY,
    Always,
    Interval,
    Predicate,
    STLSpec,
)
from stl_seed.tasks._trajectory import Trajectory, TrajectoryMeta
from stl_seed.tasks.bio_ode_params import RepressilatorParams
from stl_seed.tasks.glucose_insulin import (
    BergmanParams,
    GlucoseInsulinSimulator,
    default_normal_subject_initial_state,
)

# ---------------------------------------------------------------------------
# Shared LLM proxies.
# ---------------------------------------------------------------------------


def _uniform_llm(K: int):
    """Flat (no-info) LLM. entropy = log K."""

    def llm(state, history, key):
        return jnp.zeros(K)

    return llm


# ---------------------------------------------------------------------------
# H1. Smoke test on glucose-insulin.
# ---------------------------------------------------------------------------


def test_horizon_folded_runs_glucose_insulin_no_crash() -> None:
    """The sampler runs end-to-end on the glucose-insulin task."""
    sim = GlucoseInsulinSimulator()
    params = BergmanParams()
    spec = REGISTRY["glucose_insulin.tir.easy"]
    V = make_uniform_action_vocabulary(0.0, 5.0, k_per_dim=5)
    x0 = default_normal_subject_initial_state(params)

    sampler = HorizonFoldedGradientSampler(
        llm=_uniform_llm(int(V.shape[0])),
        simulator=sim,
        spec=spec,
        action_vocabulary=V,
        sim_params=params,
        horizon=sim.n_control_points,
        action_low=0.0,
        action_high=5.0,
        lr=1e-2,
        k_iters=20,  # short run, just a smoke test
        init="zeros",
    )
    traj, diag = sampler.sample(x0, jax.random.key(0))

    assert isinstance(traj, Trajectory)
    assert traj.states.shape == (sim.n_save_points, 3)
    assert traj.actions.shape == (sim.n_control_points, 1)
    assert traj.times.shape == (sim.n_save_points,)
    assert "final_rho" in diag
    assert np.isfinite(diag["final_rho"])
    assert diag["sampler"] == "horizon_folded_gradient"
    assert diag["init_strategy"] == "zeros"
    assert diag["n_iters"] == 20
    # rho_at_iter has init + (k_iters) entries when no early-stop:
    assert len(diag["rho_at_iter"]) >= 1
    assert len(diag["rho_at_iter"]) == len(diag["grad_norm_at_iter"])


# ---------------------------------------------------------------------------
# H2. Synthetic 1-D ODE with known-optimal solution.
# ---------------------------------------------------------------------------
#
# We construct a deliberately trivial test problem so the optimiser's
# behaviour is unambiguous:
#
#   ODE:    dx/dt = u(t)             (single state, scalar control)
#   IC:     x(0) = 0
#   action: u(t) in [0, 1], piecewise-constant with H = 5 control steps,
#           horizon T = 5.0 (so each step lasts 1.0 time-unit).
#   Spec:   G_[3.5, 5.0] (x >= 4.0) . the trajectory must hold above 4.0
#                                      across the back half of the horizon.
#
# Optimal: ``u(t) = 1`` for all t -> ``x(t) = t`` -> ``x(3.5) = 3.5``,
# ``x(5.0) = 5.0``. Robustness ``rho = min(x(t) - 4.0 : t in [3.5, 5.0])
# = -0.5``. So even the optimal action sequence does not satisfy this
# spec across the FULL window. the test is on whether the optimiser
# can reach the optimal rho ``-0.5``, NOT on whether rho > 0.
#
# An equivalent way to phrase the success criterion: the optimiser
# starting from zeros (rho = -4.0, since x stays at 0) must reach rho
# very close to -0.5 within 100 iterations.


class _LinearIntegratorSimulator(eqx.Module):
    """Trivial single-state simulator: dx/dt = u(t).

    Used by H2 so the test isolates the optimiser's behaviour from the
    bio_ode / glucose-insulin simulator complexity. The vector field is
    a single line; the "kinetic params" are the empty pytree (None).
    """

    horizon_minutes: float = eqx.field(static=True, default=5.0)
    n_control_points: int = eqx.field(static=True, default=5)
    n_save_points: int = eqx.field(static=True, default=51)

    @property
    def state_dim(self) -> int:
        return 1

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def horizon(self) -> int:
        return self.n_control_points

    def simulate(
        self,
        initial_state: jt.Float[jt.Array, " 1"],
        control_sequence: jt.Float[jt.Array, "H 1"],
        params,
        key,
    ) -> Trajectory:
        del key, params
        H = control_sequence.shape[0]
        dt = self.horizon_minutes / H

        def vf(t, y, args):
            ctrl = args
            idx = jnp.clip(jnp.floor(t / dt).astype(jnp.int32), 0, H - 1)
            return ctrl[idx]

        save_times = jnp.linspace(0.0, self.horizon_minutes, self.n_save_points)
        sol = dfx.diffeqsolve(
            dfx.ODETerm(vf),
            dfx.Tsit5(),
            t0=0.0,
            t1=self.horizon_minutes,
            dt0=0.01,
            y0=initial_state,
            args=control_sequence,
            saveat=dfx.SaveAt(ts=save_times),
            stepsize_controller=dfx.PIDController(rtol=1e-6, atol=1e-9),
            max_steps=4096,
            throw=False,
        )
        ys = sol.ys
        assert ys is not None
        meta = TrajectoryMeta(
            n_nan_replacements=jnp.asarray(0, dtype=jnp.int32),
            final_solver_result=jnp.asarray(0, dtype=jnp.int32),
            used_stiff_fallback=jnp.asarray(0, dtype=jnp.int32),
        )
        return Trajectory(
            states=ys,
            actions=control_sequence,
            times=save_times,
            meta=meta,
        )


def _make_synthetic_spec() -> STLSpec:
    """Build the H2 spec via the same _gt convention used by bio_ode_specs."""

    def _gt(channel: int, threshold: float) -> Predicate:
        return Predicate(
            f"x>{threshold}",
            fn=lambda traj, t, c=channel, th=threshold: float(traj[t, c]) - th,
        )

    return STLSpec(
        name="synthetic.linear.test_h2",
        formula=Always(_gt(0, 4.0), interval=Interval(3.5, 5.0)),
        signal_dim=1,
        horizon_minutes=5.0,
        description="Synthetic test spec for horizon-folded sampler unit test.",
        citations=("local test fixture",),
        formula_text="G_[3.5, 5.0] (x >= 4.0)",
    )


def test_horizon_folded_improves_rho_synthetic() -> None:
    """K=100 Adam iters from u=0 reach the optimal rho on the synthetic ODE.

    Optimal action sequence is ``u_t = 1`` for all t (the box maximum),
    giving ``x(t) = t`` and ``rho = min_{t in [3.5, 5]} (t - 4.0) = -0.5``.
    We assert the optimiser reaches within 0.1 rho units of the optimum
    (i.e. ``rho >= -0.6``), starting from the all-zeros init where
    ``rho = -4.0``.
    """
    sim = _LinearIntegratorSimulator()
    spec = _make_synthetic_spec()
    V = make_uniform_action_vocabulary(0.0, 1.0, k_per_dim=2)
    x0 = jnp.zeros((1,), dtype=jnp.float32)

    sampler = HorizonFoldedGradientSampler(
        llm=_uniform_llm(int(V.shape[0])),
        simulator=sim,
        spec=spec,
        action_vocabulary=V,
        sim_params=None,
        horizon=sim.n_control_points,
        action_low=0.0,
        action_high=1.0,
        lr=5e-2,  # larger lr is fine on this simple landscape
        k_iters=100,
        init="zeros",
    )
    _, diag = sampler.sample(x0, jax.random.key(0))

    init_rho = diag["init_rho"]
    final_rho = diag["final_rho"]
    # init_rho is ~-2.25: sigmoid(0)=0.5 so init action is u=0.5,
    # x(t) = 0.5 * t, rho = min_{t in [3.5, 5]} (0.5 * t - 4) = 0.5 * 3.5 - 4 = -2.25.
    assert init_rho < -2.0, (
        f"unexpected init_rho={init_rho}; expected ~-2.25 from sigmoid(0)=0.5 -> u=0.5"
    )
    # After K=100 Adam steps with lr=5e-2 the latent saturates the sigmoid
    # near +1 and rho should be very close to the -0.5 optimum.
    assert final_rho > init_rho + 1.0, (
        f"horizon-folded gradient ascent did not improve rho: "
        f"init={init_rho:.3f}, final={final_rho:.3f}"
    )
    # Tighter check: within 0.1 of the analytical optimum -0.5.
    assert final_rho >= -0.6, (
        f"horizon-folded did not reach optimum rho ~ -0.5; got {final_rho:.3f}. "
        f"Trajectory of rho across iters: first 5 = {diag['rho_at_iter'][:5]}, "
        f"last 5 = {diag['rho_at_iter'][-5:]}"
    )


# ---------------------------------------------------------------------------
# H3. Sampler protocol compliance.
# ---------------------------------------------------------------------------


def test_horizon_folded_protocol_compliance() -> None:
    """HorizonFoldedGradientSampler satisfies the Sampler Protocol."""
    sim = GlucoseInsulinSimulator()
    params = BergmanParams()
    spec = REGISTRY["glucose_insulin.tir.easy"]
    V = make_uniform_action_vocabulary(0.0, 5.0, k_per_dim=5)
    sampler = HorizonFoldedGradientSampler(
        llm=_uniform_llm(int(V.shape[0])),
        simulator=sim,
        spec=spec,
        action_vocabulary=V,
        sim_params=params,
        horizon=sim.n_control_points,
        action_low=0.0,
        action_high=5.0,
        lr=1e-2,
        k_iters=5,
    )
    assert isinstance(sampler, Sampler)


# ---------------------------------------------------------------------------
# H4. Init strategy validation.
# ---------------------------------------------------------------------------


def test_horizon_folded_invalid_init_raises() -> None:
    sim = GlucoseInsulinSimulator()
    params = BergmanParams()
    spec = REGISTRY["glucose_insulin.tir.easy"]
    V = make_uniform_action_vocabulary(0.0, 5.0, k_per_dim=5)

    with pytest.raises(ValueError, match="init must be one of"):
        HorizonFoldedGradientSampler(
            llm=_uniform_llm(int(V.shape[0])),
            simulator=sim,
            spec=spec,
            action_vocabulary=V,
            sim_params=params,
            horizon=sim.n_control_points,
            init="totally_made_up",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="requires init_action"):
        HorizonFoldedGradientSampler(
            llm=_uniform_llm(int(V.shape[0])),
            simulator=sim,
            spec=spec,
            action_vocabulary=V,
            sim_params=params,
            horizon=sim.n_control_points,
            init="heuristic",  # missing init_action
        )

    with pytest.raises(ValueError, match="init_action shape"):
        HorizonFoldedGradientSampler(
            llm=_uniform_llm(int(V.shape[0])),
            simulator=sim,
            spec=spec,
            action_vocabulary=V,
            sim_params=params,
            horizon=sim.n_control_points,
            init="heuristic",
            init_action=jnp.zeros((sim.n_control_points + 1, 1)),  # wrong H
        )


# ---------------------------------------------------------------------------
# H5. Hyperparameter validation.
# ---------------------------------------------------------------------------


def test_horizon_folded_invalid_hyperparameters_raise() -> None:
    sim = GlucoseInsulinSimulator()
    params = BergmanParams()
    spec = REGISTRY["glucose_insulin.tir.easy"]
    V = make_uniform_action_vocabulary(0.0, 5.0, k_per_dim=5)
    llm = _uniform_llm(int(V.shape[0]))

    with pytest.raises(ValueError, match="lr must be > 0"):
        HorizonFoldedGradientSampler(
            llm=llm,
            simulator=sim,
            spec=spec,
            action_vocabulary=V,
            sim_params=params,
            horizon=sim.n_control_points,
            lr=0.0,
        )

    with pytest.raises(ValueError, match="k_iters must be >= 1"):
        HorizonFoldedGradientSampler(
            llm=llm,
            simulator=sim,
            spec=spec,
            action_vocabulary=V,
            sim_params=params,
            horizon=sim.n_control_points,
            k_iters=0,
        )

    with pytest.raises(ValueError, match="action_high"):
        HorizonFoldedGradientSampler(
            llm=llm,
            simulator=sim,
            spec=spec,
            action_vocabulary=V,
            sim_params=params,
            horizon=sim.n_control_points,
            action_low=5.0,
            action_high=2.0,  # high < low
        )

    with pytest.raises(ValueError, match="must be 2-d"):
        HorizonFoldedGradientSampler(
            llm=llm,
            simulator=sim,
            spec=spec,
            action_vocabulary=jnp.zeros((5,)),  # wrong dim
            sim_params=params,
            horizon=sim.n_control_points,
        )

    with pytest.raises(ValueError, match="horizon must be >= 1"):
        HorizonFoldedGradientSampler(
            llm=llm,
            simulator=sim,
            spec=spec,
            action_vocabulary=V,
            sim_params=params,
            horizon=0,
        )


# ---------------------------------------------------------------------------
# H6. Diagnostics schema.
# ---------------------------------------------------------------------------


def test_horizon_folded_diagnostics_well_formed() -> None:
    sim = GlucoseInsulinSimulator()
    params = BergmanParams()
    spec = REGISTRY["glucose_insulin.tir.easy"]
    V = make_uniform_action_vocabulary(0.0, 5.0, k_per_dim=5)
    x0 = default_normal_subject_initial_state(params)

    sampler = HorizonFoldedGradientSampler(
        llm=_uniform_llm(int(V.shape[0])),
        simulator=sim,
        spec=spec,
        action_vocabulary=V,
        sim_params=params,
        horizon=sim.n_control_points,
        action_low=0.0,
        action_high=5.0,
        k_iters=10,
        init="zeros",
    )
    _, diag = sampler.sample(x0, jax.random.key(7))

    # Required keys.
    for key in (
        "sampler",
        "rho_at_iter",
        "grad_norm_at_iter",
        "best_rho",
        "best_iter",
        "final_rho",
        "init_rho",
        "init_strategy",
        "fallback_used",
        "n_iters",
        "n_steps",
    ):
        assert key in diag, f"missing diagnostic key: {key!r}"

    # rho_at_iter and grad_norm_at_iter are paired sequences.
    assert len(diag["rho_at_iter"]) == len(diag["grad_norm_at_iter"])
    # best_iter is a non-negative int within range.
    assert 0 <= diag["best_iter"] <= diag["n_iters"]
    # rho values are finite floats.
    assert np.isfinite(diag["best_rho"])
    assert np.isfinite(diag["final_rho"])
    assert np.isfinite(diag["init_rho"])
    # n_steps mirrors n_iters (harness convention).
    assert diag["n_steps"] == diag["n_iters"]
    # init_strategy mirrored back.
    assert diag["init_strategy"] == "zeros"


# ---------------------------------------------------------------------------
# H7. All four init strategies execute.
# ---------------------------------------------------------------------------


def test_horizon_folded_init_strategies_run() -> None:
    """Each init strategy executes a short sampler call without raising."""
    sim = GlucoseInsulinSimulator()
    params = BergmanParams()
    spec = REGISTRY["glucose_insulin.tir.easy"]
    V = make_uniform_action_vocabulary(0.0, 5.0, k_per_dim=5)
    x0 = default_normal_subject_initial_state(params)
    H = sim.n_control_points

    for init in ("zeros", "llm", "random"):
        sampler = HorizonFoldedGradientSampler(
            llm=_uniform_llm(int(V.shape[0])),
            simulator=sim,
            spec=spec,
            action_vocabulary=V,
            sim_params=params,
            horizon=H,
            action_low=0.0,
            action_high=5.0,
            k_iters=3,
            init=init,
        )
        _, diag = sampler.sample(x0, jax.random.key(0))
        assert diag["init_strategy"] == init
        assert np.isfinite(diag["init_rho"])

    # heuristic warm-start: hand in a constant-mid action sequence.
    sampler = HorizonFoldedGradientSampler(
        llm=_uniform_llm(int(V.shape[0])),
        simulator=sim,
        spec=spec,
        action_vocabulary=V,
        sim_params=params,
        horizon=H,
        action_low=0.0,
        action_high=5.0,
        k_iters=3,
        init="heuristic",
        init_action=jnp.full((H, 1), 2.5, dtype=jnp.float32),
    )
    _, diag = sampler.sample(x0, jax.random.key(0))
    assert diag["init_strategy"] == "heuristic"


# ---------------------------------------------------------------------------
# H8. Sigmoid reparam round-trip.
# ---------------------------------------------------------------------------


def test_horizon_folded_sigmoid_reparam_invertible() -> None:
    """``inverse_sigmoid_reparam(sigmoid_reparam(z)) == z`` inside the box.

    For any ``z`` whose post-sigmoid mapping lies away from the
    ``[low + eps, high - eps]`` clipping band, the round-trip should be
    numerically identity.
    """
    H, m = 8, 3
    low = jnp.zeros((m,), dtype=jnp.float32)
    high = jnp.ones((m,), dtype=jnp.float32)
    z = jax.random.normal(jax.random.key(0), (H, m), dtype=jnp.float32)
    u = _sigmoid_reparam(z, low, high)
    z_back = _inverse_sigmoid_reparam(u, low, high, eps=1e-6)
    np.testing.assert_allclose(np.asarray(z), np.asarray(z_back), atol=1e-3, rtol=1e-3)
    # And the forward map is in [low + eps, high - eps] for finite z.
    assert bool(jnp.all((u > low) & (u < high)))


# ---------------------------------------------------------------------------
# H9. Bio_ode repressilator regression / progress check.
# ---------------------------------------------------------------------------
#
# This is the "does horizon-folding fix the repressilator failure?" probe,
# kept *separate* from the strict synthetic test so a regression in this
# (much harder) task does not block the synthetic correctness test from
# passing. We mark it slow but NOT xfail: if horizon-folding meaningfully
# improves rho over the per-step gradient sampler's -250 floor, this test
# should pass.


@pytest.mark.slow
def test_horizon_folded_repressilator_improves_over_floor() -> None:
    """Pre-registered: horizon-folding from zeros init reaches rho > -200
    on the bio_ode.repressilator.easy spec, beating the -250 saturation
    floor that the per-step gradient sampler hits.

    The pre-registered failure analysis in
    ``paper/cross_task_validation.md`` predicts the per-step myopic
    extrapolation is the bottleneck. Horizon-folding eliminates that
    bottleneck (the gradient sees the FULL trajectory, not a partial
    extrapolation), so the prediction is that rho should escape the
    -250 floor. The threshold ``rho > -200`` is conservative: 50 rho
    units above the saturation floor and 200 units short of the
    constant ``u=(0,0,1)`` analytical satisfier (rho ~ +25), so this is
    a "the optimiser is doing some work" sanity check, not a
    "horizon-folding solves the problem" claim.

    If this test passes, the unified eval phase should see horizon-
    folding matching or beating the +25 known-satisfier on at least
    some seeds. If it fails, that itself is informative: it means the
    failure is not purely about myopia. it is about the rho landscape
    being too non-convex for first-order methods even with full-horizon
    gradients.
    """
    from stl_seed.tasks.bio_ode import (
        REPRESSILATOR_ACTION_DIM,
        RepressilatorSimulator,
        _repressilator_initial_state,
    )

    sim = RepressilatorSimulator()
    params = RepressilatorParams()
    x0 = _repressilator_initial_state(params)
    spec = REGISTRY["bio_ode.repressilator.easy"]
    V = make_uniform_action_vocabulary(
        [0.0] * REPRESSILATOR_ACTION_DIM,
        [1.0] * REPRESSILATOR_ACTION_DIM,
        k_per_dim=2,
    )
    sampler = HorizonFoldedGradientSampler(
        llm=_uniform_llm(int(V.shape[0])),
        simulator=sim,
        spec=spec,
        action_vocabulary=V,
        sim_params=params,
        horizon=sim.n_control_points,
        action_low=0.0,
        action_high=1.0,
        lr=1e-2,
        k_iters=100,
        init="zeros",
    )
    _, diag = sampler.sample(x0, jax.random.key(0))
    final_rho = diag["final_rho"]
    init_rho = diag["init_rho"]
    # The saturation floor reported in cross_task_validation.md is -250.
    # Even -200 is a meaningful escape (50 rho units), and reaching > 0
    # would be a publishable success.
    assert final_rho > -200.0, (
        f"horizon-folded did not escape the -250 floor on repressilator. "
        f"init_rho={init_rho:.3f}, final_rho={final_rho:.3f}, "
        f"best_iter={diag['best_iter']}, "
        f"first 3 rho_at_iter={diag['rho_at_iter'][:3]}, "
        f"last 3 rho_at_iter={diag['rho_at_iter'][-3:]}"
    )


# ---------------------------------------------------------------------------
# Side helper kept here so test_horizon_folded_diagnostics_well_formed
# can introspect the diagnostics dataclass independently.
# ---------------------------------------------------------------------------


def test_horizon_folded_diagnostics_dataclass_fields() -> None:
    """The HorizonFoldedDiagnostics dataclass exposes the documented fields."""
    from stl_seed.inference.horizon_folded import HorizonFoldedDiagnostics

    fields = {f.name for f in dataclasses.fields(HorizonFoldedDiagnostics)}
    assert {
        "rho_at_iter",
        "grad_norm_at_iter",
        "best_rho",
        "best_iter",
        "final_rho",
        "init_rho",
        "init_strategy",
        "fallback_used",
        "n_iters",
    } <= fields

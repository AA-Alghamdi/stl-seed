"""Tests for ``stl_seed.analysis``: TrajectoryAdversary, gold scorers, decomposition."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stl_seed.analysis import (
    AdversaryResult,
    GoodhartGapResult,
    PerPolicyGap,
    TrajectoryAdversary,
    bio_ode_gold_score,
    bio_ode_repressilator_gold,
    bio_ode_toggle_gold,
    glucose_insulin_gold_score,
    measure_goodhart_gap,
)
from stl_seed.analysis.decomposition import (
    measure_from_arrays,
    random_policy,
)
from stl_seed.analysis.gold_scorers import (
    REPRESSILATOR_PHYSIO_HIGH_NM,
    get_gold_scorer,
)
from stl_seed.specs import REGISTRY
from stl_seed.stl.evaluator import evaluate_robustness
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
from stl_seed.tasks._trajectory import Trajectory, TrajectoryMeta

# ---------------------------------------------------------------------------
# Fixtures: simulators, specs, x0, meal schedules.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gi_setup():
    sim = GlucoseInsulinSimulator()
    params = BergmanParams()
    x0 = default_normal_subject_initial_state(params)
    meal = single_meal_schedule(onset_min=15.0, carb_grams=60.0)
    spec = REGISTRY["glucose_insulin.tir.easy"]
    return sim, params, x0, meal, spec


@pytest.fixture(scope="module")
def repress_setup():
    sim = RepressilatorSimulator()
    params = RepressilatorParams()
    x0 = default_repressilator_initial_state(params)
    spec = REGISTRY["bio_ode.repressilator.easy"]
    return sim, params, x0, spec


@pytest.fixture(scope="module")
def toggle_setup():
    sim = ToggleSimulator()
    params = ToggleParams()
    x0 = default_toggle_initial_state(params)
    spec = REGISTRY["bio_ode.toggle.medium"]
    return sim, params, x0, spec


def _gi_simulate(
    sim: GlucoseInsulinSimulator,
    x0,
    u_flat,
    meal,
    params,
    key,
) -> Trajectory:
    states, times, meta = sim.simulate(x0, u_flat, meal, params, key)
    return Trajectory(states=states, actions=u_flat[:, None], times=times, meta=meta)


# ---------------------------------------------------------------------------
# 1. Gold scorer sanity tests.
# ---------------------------------------------------------------------------


def test_gold_score_glucose_smooth_better(gi_setup):
    """Smooth (constant) insulin schedule scores higher than choppy bang-bang."""
    sim, params, x0, meal, _ = gi_setup
    key = jax.random.PRNGKey(0)

    smooth_u = jnp.ones((12,)) * 1.5  # constant 1.5 U/h infusion
    chop_u = jnp.array([0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0])

    smooth_traj = _gi_simulate(sim, x0, smooth_u, meal, params, key)
    chop_traj = _gi_simulate(sim, x0, chop_u, meal, params, key)

    g_smooth = float(glucose_insulin_gold_score(smooth_traj))
    g_chop = float(glucose_insulin_gold_score(chop_traj))

    # Bang-bang should be punished by the jerk penalty AND by glucose
    # variability driven by the alternating insulin pulses; constant
    # infusion should score strictly higher.
    assert g_smooth > g_chop, f"smooth ({g_smooth:.3f}) should beat chop ({g_chop:.3f})"


def test_gold_score_bio_realistic_better(repress_setup):
    """Trajectories within the physiological envelope outscore extreme ones."""
    sim, params, x0, _ = repress_setup
    key = jax.random.PRNGKey(0)

    # Mild control: steady-state Elowitz oscillation.
    mild_u = jnp.zeros((10, 3))
    mild_out = sim.simulate(x0, mild_u, params, key)
    g_mild = float(bio_ode_repressilator_gold(mild_out))

    # Construct a synthetic supra-physiological trajectory with the same
    # shape but proteins pinned at 100x the physiological upper bound.
    extreme_states = jnp.full_like(mild_out.states, 100.0 * REPRESSILATOR_PHYSIO_HIGH_NM)
    extreme_traj = Trajectory(
        states=extreme_states,
        actions=mild_out.actions,
        times=mild_out.times,
        meta=mild_out.meta,
    )
    g_extreme = float(bio_ode_repressilator_gold(extreme_traj))

    # The realism penalty should make the supra-physiological trajectory
    # score worse, even though it nominally satisfies the "p_1 high" clause.
    assert g_mild > g_extreme, f"mild ({g_mild:.3f}) should beat extreme ({g_extreme:.3f})"


def test_gold_dispatch_keys():
    """``get_gold_scorer`` accepts the documented keys and rejects unknowns."""
    assert get_gold_scorer("glucose_insulin") is glucose_insulin_gold_score
    assert get_gold_scorer("bio_ode.repressilator") is bio_ode_repressilator_gold
    assert get_gold_scorer("bio_ode.toggle") is bio_ode_toggle_gold
    with pytest.raises(KeyError):
        get_gold_scorer("nonexistent_family")


def test_bio_ode_gold_dispatch_subdomain(repress_setup):
    """``bio_ode_gold_score`` dispatch matches the per-subdomain functions."""
    sim, params, x0, _ = repress_setup
    key = jax.random.PRNGKey(0)
    u = jnp.zeros((10, 3))
    out = sim.simulate(x0, u, params, key)

    g_dispatch = float(bio_ode_gold_score(out, subdomain="repressilator"))
    g_direct = float(bio_ode_repressilator_gold(out))
    assert np.isclose(g_dispatch, g_direct, rtol=1e-6)

    with pytest.raises(ValueError):
        bio_ode_gold_score(out, subdomain="not_a_real_subdomain")


# ---------------------------------------------------------------------------
# 2. Adversary correctness: should drive spec_rho up and gold down.
# ---------------------------------------------------------------------------


def test_adversary_finds_high_spec_low_gold(gi_setup):
    """The adversary should improve spec_rho - lambda*gold over iterations.

    A successful adversary run on glucose_insulin.tir.easy starts from a
    random init (modest spec_rho, modest gold), then drives the loss DOWN
    monotonically (modulo Adam's small overshoots). At convergence the
    final spec_rho should be >= the initial spec_rho, AND the final gold
    should be <= the initial gold.
    """
    sim, params, x0, meal, spec = gi_setup
    adv = TrajectoryAdversary(
        simulator=sim,
        spec=spec,
        gold_score=glucose_insulin_gold_score,
        params=params,
        lambda_satisfaction=0.5,
        learning_rate=0.3,
        max_iters=40,
        project_actions=True,
        action_min=jnp.asarray([0.0]),
        action_max=jnp.asarray([5.0]),
        simulator_aux=(meal,),
    )
    result = adv.find_adversary(
        initial_state=x0,
        action_dim=1,
        horizon=12,
        key=jax.random.PRNGKey(7),
        n_restarts=2,
    )
    assert isinstance(result, AdversaryResult)
    # Spec satisfaction should be non-negative (proxy spec satisfied).
    assert result.best_spec_rho > 0.0, "adversary failed to find a spec-satisfying trajectory"
    # The adversary should have visited at least one iterate strictly
    # better than the init under the joint loss ``-rho + lambda*gold``.
    losses = np.asarray(result.iter_history[:, 2])
    init_loss = float(losses[0])
    best_loss = float(np.min(losses))
    assert best_loss <= init_loss - 1e-3, (
        f"adversary did not improve: best_loss={best_loss:.3f}, init={init_loss:.3f}"
    )
    # And the WINNING restart's gold should be lower than the init gold,
    # confirming the adversary exploited the spec-completeness gap.
    init_gold = float(result.iter_history[0, 1])
    final_gold = float(result.iter_history[-1, 1])
    assert final_gold < init_gold, (
        f"gold did not decrease: init={init_gold:.3f}, final={final_gold:.3f}"
    )


def test_adversary_synthetic_high_spec_low_gold():
    """On a controlled synthetic problem, the adversary should recover the
    known-bad action sequence.

    Construction. Use the glucose-insulin simulator. Define a 'gold' that
    penalizes the L2 norm of the action sequence (intuitively: low-action
    trajectories are gold-good). The TIR proxy spec is satisfied for a
    band of action values; the adversary should land on the high end of
    that band (max action -> low gold under our synthetic gold) rather
    than the low end.
    """
    sim = GlucoseInsulinSimulator()
    params = BergmanParams()
    x0 = default_normal_subject_initial_state(params)
    meal = single_meal_schedule(onset_min=15.0, carb_grams=60.0)
    spec = REGISTRY["glucose_insulin.tir.easy"]

    def synthetic_gold(traj: Trajectory):
        # Higher = better. Penalize large insulin doses.
        return -jnp.mean(traj.actions[:, 0] ** 2)

    adv = TrajectoryAdversary(
        simulator=sim,
        spec=spec,
        gold_score=synthetic_gold,
        params=params,
        lambda_satisfaction=0.05,  # weak spec constraint to let gold fight
        learning_rate=0.3,
        max_iters=40,
        project_actions=True,
        action_min=jnp.asarray([0.0]),
        action_max=jnp.asarray([5.0]),
        simulator_aux=(meal,),
    )
    result = adv.find_adversary(
        initial_state=x0,
        action_dim=1,
        horizon=12,
        key=jax.random.PRNGKey(13),
        n_restarts=2,
    )
    init_gold = float(result.iter_history[0, 1])
    final_gold = float(result.iter_history[-1, 1])
    assert final_gold < init_gold, (
        f"adversary failed to exploit synthetic gold: {init_gold:.3f} -> {final_gold:.3f}"
    )


# ---------------------------------------------------------------------------
# 3. Decomposition: measure population statistics.
# ---------------------------------------------------------------------------


def test_decomposition_measure_basic(gi_setup):
    """``measure_goodhart_gap`` returns finite per-policy stats and a sane range."""
    sim, params, x0, meal, spec = gi_setup
    pol1 = random_policy(horizon=12, action_dim=1, action_min=0.0, action_max=5.0)
    pol2 = random_policy(horizon=12, action_dim=1, action_min=1.0, action_max=2.0)

    result = measure_goodhart_gap(
        simulator=sim,
        proxy_spec=spec,
        gold_score=glucose_insulin_gold_score,
        policies={"random_full": pol1, "random_narrow": pol2},
        initial_state=x0,
        action_dim=1,
        horizon=12,
        params=params,
        key=jax.random.PRNGKey(2026),
        n_trajectories=30,  # small for speed
        simulator_aux=(meal,),
    )
    assert isinstance(result, GoodhartGapResult)
    assert set(result.per_policy.keys()) == {"random_full", "random_narrow"}
    for name, gap in result.per_policy.items():
        assert isinstance(gap, PerPolicyGap)
        assert gap.n_trajectories == 30
        assert gap.rho_values.shape == (30,)
        assert gap.gold_values.shape == (30,)
        assert -1.0 <= gap.pearson_r <= 1.0, f"{name} pearson out of range"
        assert -1.0 <= gap.spearman_r <= 1.0, f"{name} spearman out of range"
        assert np.isfinite(gap.regression_slope)
        assert np.isfinite(gap.top_decile_gap)
    lo, hi = result.cross_policy_pearson_range
    assert lo <= hi
    assert -1.0 <= lo <= 1.0
    assert -1.0 <= hi <= 1.0


def test_decomposition_known_correlation():
    """When proxy and gold are constructed to be perfectly anticorrelated,
    ``measure_from_arrays`` recovers ``r ~ -1`` and a large negative
    top-decile gap."""
    rho = np.linspace(-5, 5, 100)
    gold = -rho  # perfect negative correlation

    gap = measure_from_arrays(rho, gold, "synthetic_neg")
    assert np.isclose(gap.pearson_r, -1.0, atol=1e-6)
    assert np.isclose(gap.spearman_r, -1.0, atol=1e-6)
    # Top-decile of rho corresponds to bottom-decile of gold; the gap
    # should be substantially negative.
    assert gap.top_decile_gap < -1.0
    # Negative correlation -> flagged.
    assert gap.flagged


def test_decomposition_positive_correlation_unflagged():
    """Strong positive proxy/gold correlation is NOT flagged (FM2 OK)."""
    rho = np.linspace(-5, 5, 100)
    gold = rho + np.random.default_rng(0).normal(0, 0.1, size=100)
    gap = measure_from_arrays(rho, gold, "synthetic_pos")
    assert gap.spearman_r > 0.9
    assert not gap.flagged


# ---------------------------------------------------------------------------
# 4. JAX autodiff compatibility through simulator.
# ---------------------------------------------------------------------------


def test_jit_compatibility_through_simulator(gi_setup):
    """Confirm gradients flow through simulator + STL evaluator + gold.

    This is the load-bearing assumption for the adversary's gradient
    method. If autodiff fails through any link in the chain, the
    adversary regresses to random search. We verify the gradient at a
    random point exists, is finite, and has the expected shape.
    """
    sim, params, x0, meal, spec = gi_setup

    from stl_seed.stl.evaluator import compile_spec

    compiled = compile_spec(spec)

    def loss(z):
        u = 0.0 + (5.0 - 0.0) * jax.nn.sigmoid(z)  # shape (H,)
        states, times, meta = sim.simulate(x0, u, meal, params, jax.random.PRNGKey(0))
        traj = Trajectory(states=states, actions=u[:, None], times=times, meta=meta)
        rho = compiled(states, times)
        gold = glucose_insulin_gold_score(traj)
        return -rho + 0.5 * gold

    z = 0.5 * jax.random.normal(jax.random.PRNGKey(99), shape=(12,))
    grad = jax.grad(loss)(z)
    assert grad.shape == (12,)
    assert jnp.all(jnp.isfinite(grad)), "grads contain NaN/Inf - autodiff broken"
    # Magnitude check: with non-degenerate inputs, at least one entry has
    # appreciable gradient magnitude.
    assert float(jnp.max(jnp.abs(grad))) > 1e-4


# ---------------------------------------------------------------------------
# 5. Edge cases.
# ---------------------------------------------------------------------------


def test_adversary_zero_restarts_returns_sentinel(gi_setup):
    """Zero restarts is degenerate but should not crash; sentinel result returned."""
    sim, params, x0, meal, spec = gi_setup
    adv = TrajectoryAdversary(
        simulator=sim,
        spec=spec,
        gold_score=glucose_insulin_gold_score,
        params=params,
        max_iters=2,
        project_actions=True,
        action_min=jnp.asarray([0.0]),
        action_max=jnp.asarray([5.0]),
        simulator_aux=(meal,),
    )
    result = adv.find_adversary(
        initial_state=x0,
        action_dim=1,
        horizon=12,
        key=jax.random.PRNGKey(0),
        n_restarts=0,
    )
    assert isinstance(result, AdversaryResult)
    assert np.isnan(result.best_spec_rho)
    assert np.isnan(result.best_gold_score)


def test_smooth_indicator_gradient_finite():
    """The smooth indicator helper has finite, non-zero gradients
    in the transition zone (a load-bearing requirement for autodiff)."""
    from stl_seed.analysis.gold_scorers import _smooth_indicator

    x = jnp.array([0.0, 1.0, 2.0])
    g = jax.grad(lambda x: jnp.sum(_smooth_indicator(x, threshold=1.0)))(x)
    assert jnp.all(jnp.isfinite(g))
    # Largest gradient should be at the threshold itself.
    assert float(jnp.argmax(g)) == 1


def test_adversary_bio_ode_repressilator_runs(repress_setup):
    """End-to-end smoke for the bio_ode adversary path; ensures the
    multi-channel action box works and that the bio_ode simulator returns
    a Trajectory directly without the GI tuple-unpacking branch."""
    sim, params, x0, spec = repress_setup
    adv = TrajectoryAdversary(
        simulator=sim,
        spec=spec,
        gold_score=bio_ode_repressilator_gold,
        params=params,
        lambda_satisfaction=0.1,
        learning_rate=0.3,
        max_iters=15,
        project_actions=True,
        action_min=jnp.asarray([0.0, 0.0, 0.0]),
        action_max=jnp.asarray([1.0, 1.0, 1.0]),
    )
    result = adv.find_adversary(
        initial_state=x0,
        action_dim=3,
        horizon=10,
        key=jax.random.PRNGKey(123),
        n_restarts=2,
    )
    assert result.best_actions.shape == (10, 3)
    assert result.best_trajectory.states.shape[-1] == 6
    assert np.isfinite(result.best_spec_rho)
    assert np.isfinite(result.best_gold_score)


def test_per_policy_gap_with_constant_inputs():
    """Degenerate (constant rho or gold) cases give r=0, no NaN propagation."""
    constant_rho = np.ones(50)
    varying_gold = np.linspace(0, 1, 50)
    gap = measure_from_arrays(constant_rho, varying_gold, "edge")
    assert gap.pearson_r == 0.0  # zero variance in rho -> defined as 0
    assert gap.spearman_r == 0.0
    # Top-decile gap is well-defined even with constant rho.
    assert np.isfinite(gap.top_decile_gap)

"""Unit tests for the cardiac action potential task family (FitzHugh-Nagumo).

Each test pre-registers a physiological / phase-portrait expectation
derivable from the literature, then asserts the simulator and specs
reproduce it. References inline.

REDACTED firewall. None of these tests import REDACTED / REDACTED /
REDACTED / REDACTED / REDACTED. The FHN system is
not used in any REDACTED artifact.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stl_seed.inference import BeamSearchWarmstartSampler
from stl_seed.inference.gradient_guided import make_uniform_action_vocabulary
from stl_seed.specs import REGISTRY
from stl_seed.stl.evaluator import evaluate_robustness
from stl_seed.tasks._trajectory import Trajectory
from stl_seed.tasks.cardiac_ap import (
    CARDIAC_ACTION_DIM,
    CARDIAC_HORIZON_TU,
    CARDIAC_N_CONTROL,
    CARDIAC_N_SAVE,
    CARDIAC_STATE_DIM,
    CardiacAPSimulator,
    FitzHughNagumoParams,
    default_cardiac_initial_state,
)

# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def key() -> jax.Array:
    return jax.random.key(0)


@pytest.fixture
def fhn_params() -> FitzHughNagumoParams:
    return FitzHughNagumoParams()


@pytest.fixture
def fhn_sim() -> CardiacAPSimulator:
    return CardiacAPSimulator()


# ===========================================================================
# Resting-state tests (FitzHugh 1961 §III; phase-portrait)
# ===========================================================================


def test_fhn_resting_state(
    fhn_sim: CardiacAPSimulator,
    fhn_params: FitzHughNagumoParams,
    key: jax.Array,
) -> None:
    """With ``I_ext = 0`` everywhere, the FHN system started at the
    resting fixed point must remain there throughout the horizon.

    Justification: the autonomous FHN system (FitzHugh 1961 §III) has a
    single asymptotically stable fixed point at the intersection of the
    cubic V-nullcline ``w = V - V**3 / 3`` and the line w-nullcline
    ``V = b * w - a``. With the canonical parameters
    ``(a, b) = (0.7, 0.8)`` the fixed point is at ``V* ~ -1.20``,
    ``w* ~ -0.625`` (from the cubic root). Starting at this fixed point
    with zero input must therefore yield a constant trajectory; any drift
    indicates a sign error or an integrator-tolerance bug. We require
    ``|V - V*| < 0.01`` over the full horizon as a tight stability check.
    """
    y0 = default_cardiac_initial_state(fhn_params)
    u = jnp.zeros((fhn_sim.n_control_points, fhn_sim.action_dim))

    traj = fhn_sim.simulate(y0, u, fhn_params, key)
    assert traj.meta.n_nan_replacements == 0

    V = np.asarray(traj.states[:, 0])
    w = np.asarray(traj.states[:, 1])
    V_star = float(y0[0])
    w_star = float(y0[1])

    assert np.all(np.abs(V - V_star) < 0.01), (
        f"V drifted from resting V* = {V_star:.4f}: "
        f"V range = ({float(np.min(V)):.4f}, {float(np.max(V)):.4f})"
    )
    assert np.all(np.abs(w - w_star) < 0.01), (
        f"w drifted from resting w* = {w_star:.4f}: "
        f"w range = ({float(np.min(w)):.4f}, {float(np.max(w)):.4f})"
    )


def test_fhn_spike_on_pulse(
    fhn_sim: CardiacAPSimulator,
    fhn_params: FitzHughNagumoParams,
    key: jax.Array,
) -> None:
    """A sustained suprathreshold input ``u = 1`` triggers at least one
    full action potential (V crosses the cubic-V-nullcline local maximum
    at V = +1.0).

    Justification: the FHN excitability threshold is at the local
    maximum of the cubic V-nullcline, V = +1 (FitzHugh 1961 §III;
    Murray *Mathematical Biology I* §1.5 Fig. 1.6). A current ``I = 1``
    shifts the V-nullcline up by 1 dimensionless unit, eliminating the
    stable fixed point and forcing the system into a limit cycle whose
    V-amplitude reaches V ~ +2 on the depolarised branch. Starting at
    the resting fixed point with this input therefore drives V above
    +1 within the first few time units.
    """
    y0 = default_cardiac_initial_state(fhn_params)
    u = jnp.ones((fhn_sim.n_control_points, fhn_sim.action_dim))

    traj = fhn_sim.simulate(y0, u, fhn_params, key)
    assert traj.meta.n_nan_replacements == 0

    V = np.asarray(traj.states[:, 0])
    V_max = float(np.max(V))
    assert V_max > 1.0, (
        f"Sustained suprathreshold input failed to fire: V_max = "
        f"{V_max:.4f}, expected > 1.0 (FitzHugh 1961 §III firing threshold)"
    )
    # Should also reach the depolarised-branch peak near V ~ 2.0
    assert V_max > 1.5, (
        f"Spike amplitude too small: V_max = {V_max:.4f}, expected > 1.5 "
        f"(FHN limit-cycle peak ~ +2.0 under sustained drive; FitzHugh 1961 Fig. 1)"
    )


def test_fhn_refractory(
    fhn_sim: CardiacAPSimulator,
    fhn_params: FitzHughNagumoParams,
    key: jax.Array,
) -> None:
    """Two pulses delivered closer together than the refractory window
    cannot both elicit a spike: the second pulse, arriving while the
    recovery variable ``w`` is still elevated, fires a smaller (or no)
    spike compared to the first.

    Justification: the FHN slow-w time-constant is
    ``1 / (epsilon * b) = 1 / (0.08 * 0.8) ~ 15.6`` time units (Keener &
    Sneyd *Mathematical Physiology* Ch. 5). A second pulse arriving
    within ~10 time units of the first hits the cell while w is still
    above its resting value and the V-nullcline is therefore still
    shifted up; the cell is in its refractory period and the spike
    amplitude is suppressed.

    Test design. Compare the spike amplitude of two protocols with
    matched total current:
    * Protocol A: a single early pulse (burst from t=0..10), then idle.
    * Protocol B: that pulse followed by a second pulse delivered
      *immediately* (t=10..20), inside the refractory window.

    The peak V in the *second* pulse window of B must be less than the
    peak V in the *first* pulse window of A (the refractory effect).
    """
    y0 = default_cardiac_initial_state(fhn_params)

    # Single pulse on the first interval, then off for the rest.
    u_single = jnp.zeros((fhn_sim.n_control_points, fhn_sim.action_dim)).at[0, 0].set(1.0)
    traj_single = fhn_sim.simulate(y0, u_single, fhn_params, key)
    V_single = np.asarray(traj_single.states[:, 0])

    # Two pulses back-to-back (intervals 0 and 1), then off.
    u_twin = jnp.zeros((fhn_sim.n_control_points, fhn_sim.action_dim))
    u_twin = u_twin.at[0, 0].set(1.0).at[1, 0].set(1.0)
    traj_twin = fhn_sim.simulate(y0, u_twin, fhn_params, key)
    V_twin = np.asarray(traj_twin.states[:, 0])

    # Examine V on the second pulse's interval [10, 20] for the twin
    # protocol vs the same window for the single-pulse protocol.
    times = np.asarray(traj_single.times)
    in_second_window = (times >= 10.0) & (times <= 20.0)

    V_twin_in_window = V_twin[in_second_window]
    # V_single is used to compute the global first-spike peak below; we
    # don't slice it to the same window because the first spike's peak
    # in the single-pulse case lives in the [0, 10] window, not [10, 20].

    # In the single-pulse case, the second window contains the natural
    # decay of the first spike (w is large, V is hyperpolarising back).
    # In the twin case, the second pulse is added on top of this decay.
    # Because the cell is refractory, the EXTRA depolarisation from the
    # second pulse is partial: the twin's V in this window does not
    # reach the spike-peak amplitude that the FIRST pulse achieved.
    V_first_spike_peak = float(np.max(V_single))
    V_twin_second_window_peak = float(np.max(V_twin_in_window))

    assert V_twin_second_window_peak < V_first_spike_peak, (
        f"refractory test failed: twin-pulse window V max = "
        f"{V_twin_second_window_peak:.4f} should be < first-spike V max = "
        f"{V_first_spike_peak:.4f} (Keener-Sneyd FHN refractory window)"
    )


def test_fhn_no_nan(
    fhn_sim: CardiacAPSimulator,
    fhn_params: FitzHughNagumoParams,
    key: jax.Array,
) -> None:
    """Across a battery of random control sequences, no simulation
    should produce NaN/Inf values that escape the sentinel guard.
    """
    keys = jax.random.split(key, 16)
    y0 = default_cardiac_initial_state(fhn_params)

    for k in keys:
        u = jax.random.uniform(
            k,
            shape=(fhn_sim.n_control_points, fhn_sim.action_dim),
            minval=0.0,
            maxval=1.0,
        )
        traj = fhn_sim.simulate(y0, u, fhn_params, k)
        assert jnp.all(jnp.isfinite(traj.states)), "non-finite in trajectory after sentinel"


def test_fhn_jit_works(
    fhn_sim: CardiacAPSimulator,
    fhn_params: FitzHughNagumoParams,
    key: jax.Array,
) -> None:
    """The simulator's ``simulate`` method must be JIT-compilable end to end."""
    y0 = default_cardiac_initial_state(fhn_params)
    u = jnp.full((fhn_sim.n_control_points, fhn_sim.action_dim), 0.5)

    @jax.jit
    def run(y0_, u_, params_, key_):
        return fhn_sim.simulate(y0_, u_, params_, key_)

    traj = run(y0, u, fhn_params, key)
    assert traj.states.shape == (fhn_sim.n_save_points, CARDIAC_STATE_DIM)
    assert traj.actions.shape == (fhn_sim.n_control_points, CARDIAC_ACTION_DIM)
    assert traj.times.shape == (fhn_sim.n_save_points,)
    assert jnp.all(jnp.isfinite(traj.states))


def test_fhn_simulator_protocol_compliance(fhn_sim: CardiacAPSimulator) -> None:
    """``CardiacAPSimulator`` must conform to the ``Simulator`` Protocol.

    Verifies the four required members (``simulate``, ``state_dim``,
    ``action_dim``, ``horizon``) and their concrete values for the FHN
    system.
    """
    from stl_seed.tasks.bio_ode import Simulator as SimProtocol

    assert isinstance(fhn_sim, SimProtocol)
    assert fhn_sim.state_dim == CARDIAC_STATE_DIM == 2
    assert fhn_sim.action_dim == CARDIAC_ACTION_DIM == 1
    assert fhn_sim.horizon == fhn_sim.n_control_points == CARDIAC_N_CONTROL == 10


def test_fhn_module_constants_consistent() -> None:
    """The exported module constants must be self-consistent."""
    assert CARDIAC_HORIZON_TU == 100.0
    assert CARDIAC_N_CONTROL == 10
    assert CARDIAC_N_SAVE == 101  # 1 sample per dimensionless time unit
    assert CARDIAC_STATE_DIM == 2
    assert CARDIAC_ACTION_DIM == 1


# ===========================================================================
# Spec satisfiability tests
# ===========================================================================


def test_cardiac_specs_registered() -> None:
    """All three cardiac specs must be in the global REGISTRY."""
    for name in (
        "cardiac.depolarize.easy",
        "cardiac.train.medium",
        "cardiac.suppress_after_two.hard",
    ):
        assert name in REGISTRY, f"spec {name!r} not registered"


def test_cardiac_specs_satisfiable_easy(
    fhn_sim: CardiacAPSimulator,
    fhn_params: FitzHughNagumoParams,
    key: jax.Array,
) -> None:
    """``cardiac.depolarize.easy`` must be satisfiable by a constant-on
    policy ``u = 1``: the cell fires within the first half of the
    horizon, so the Eventually clause is satisfied.
    """
    y0 = default_cardiac_initial_state(fhn_params)
    u = jnp.ones((fhn_sim.n_control_points, fhn_sim.action_dim))
    traj = fhn_sim.simulate(y0, u, fhn_params, key)
    rho = float(evaluate_robustness(REGISTRY["cardiac.depolarize.easy"], traj))
    assert rho > 0.0, f"easy spec not satisfied by constant u=1 policy: rho = {rho:+.4f}"


def test_cardiac_specs_satisfiable_medium(
    fhn_sim: CardiacAPSimulator,
    fhn_params: FitzHughNagumoParams,
    key: jax.Array,
) -> None:
    """``cardiac.train.medium`` must be satisfiable by a constant-on
    policy ``u = 1``: under sustained suprathreshold drive the FHN
    system enters its limit cycle and fires periodically (FitzHugh
    1961 Fig. 1, period ~40 time units), so both the early-window
    and late-window Eventually clauses are satisfied.
    """
    y0 = default_cardiac_initial_state(fhn_params)
    u = jnp.ones((fhn_sim.n_control_points, fhn_sim.action_dim))
    traj = fhn_sim.simulate(y0, u, fhn_params, key)
    rho = float(evaluate_robustness(REGISTRY["cardiac.train.medium"], traj))
    assert rho > 0.0, f"medium spec not satisfied by constant u=1 policy: rho = {rho:+.4f}"


def test_cardiac_specs_satisfiable_hard(
    fhn_sim: CardiacAPSimulator,
    fhn_params: FitzHughNagumoParams,
    key: jax.Array,
) -> None:
    """``cardiac.suppress_after_two.hard`` must be satisfiable by a
    pulse-then-quiet policy: brief pulses at intervals 0 and 4 elicit
    one spike each, then zero current lets ``V`` return to (and stay at)
    rest over the back third [70, 100].

    The satisfying policy is structurally a "burst-and-quiet" protocol
    that delivers exactly two spikes inside the prescribed reach windows
    and then withdraws drive entirely so the cell can re-equilibrate.
    """
    y0 = default_cardiac_initial_state(fhn_params)
    # Pulse on at intervals 0 and 4; off elsewhere.
    u = jnp.zeros((fhn_sim.n_control_points, fhn_sim.action_dim))
    u = u.at[0, 0].set(1.0).at[4, 0].set(1.0)
    traj = fhn_sim.simulate(y0, u, fhn_params, key)
    rho = float(evaluate_robustness(REGISTRY["cardiac.suppress_after_two.hard"], traj))
    assert rho > 0.0, (
        f"hard spec not satisfied by burst-and-quiet policy: rho = {rho:+.4f}; "
        f"actions = {[float(x) for x in u.ravel()]}"
    )


def test_cardiac_easy_violated_by_zero_input(
    fhn_sim: CardiacAPSimulator,
    fhn_params: FitzHughNagumoParams,
    key: jax.Array,
) -> None:
    """Sanity: zero input never fires, so the easy spec must be
    violated by ``u = 0`` (the cell stays at rest with V ~ -1.20 and
    never crosses V = 1.0).
    """
    y0 = default_cardiac_initial_state(fhn_params)
    u = jnp.zeros((fhn_sim.n_control_points, fhn_sim.action_dim))
    traj = fhn_sim.simulate(y0, u, fhn_params, key)
    rho = float(evaluate_robustness(REGISTRY["cardiac.depolarize.easy"], traj))
    assert rho < 0.0, f"easy spec spuriously satisfied by zero-input policy: rho = {rho:+.4f}"


# ===========================================================================
# Beam-search warmstart sampler integration test
# ===========================================================================


def test_beam_search_solves_cardiac_easy() -> None:
    """Beam-search warmstart with a 5-level vocabulary on [0, 1] reaches
    rho > 0 on ``cardiac.depolarize.easy`` in at least 2 of 3 fixed seeds.

    This is the cardiac analogue of the headline beam-search test on the
    repressilator (``test_beam_search_recovers_repressilator_solution``):
    structural enumeration over a denser action lattice consistently
    reaches the satisfying region. With a 5-level vocabulary on the 1-D
    action box (K=5) and a model-predictive constant-extrapolation
    lookahead, even seed-to-seed variance in the LLM proxy should not
    break it -- the easy spec only requires firing once in [0, 50],
    which is satisfied by any constant-positive-current policy.
    """
    sim = CardiacAPSimulator()
    params = FitzHughNagumoParams()
    spec = REGISTRY["cardiac.depolarize.easy"]
    V = make_uniform_action_vocabulary(
        [0.0] * CARDIAC_ACTION_DIM,
        [1.0] * CARDIAC_ACTION_DIM,
        k_per_dim=5,
    )
    K = int(V.shape[0])
    x0 = default_cardiac_initial_state(params)

    # Flat-prior LLM (the beam sampler ignores logits but the protocol
    # requires one).
    def llm(state, history, key):
        return jnp.zeros(K, dtype=jnp.float32)

    sampler = BeamSearchWarmstartSampler(
        llm,
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
        beam_size=8,
        gradient_refine_iters=0,
        tail_strategy="repeat_candidate",
    )

    n_pass = 0
    rhos: list[float] = []
    for seed in (0, 1, 2):
        traj, diag = sampler.sample(x0, jax.random.key(seed))
        assert isinstance(traj, Trajectory)
        rho = float(diag["final_rho"])
        rhos.append(rho)
        if rho > 0.0:
            n_pass += 1
    assert n_pass >= 2, (
        f"beam search should reach rho > 0 on cardiac.depolarize.easy in >=2/3 "
        f"seeds; got {n_pass}/3, rhos = {[f'{r:+.3f}' for r in rhos]}"
    )

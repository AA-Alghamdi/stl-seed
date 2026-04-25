"""Unit tests for the bio_ode task family (Repressilator, Toggle, MAPK).

Each test pre-registers a physiological expectation derivable from the
literature, then asserts the simulator reproduces it. References inline.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stl_seed.tasks._trajectory import Trajectory, TrajectoryMeta
from stl_seed.tasks.bio_ode import (
    MAPK_HORIZON_MIN,
    MAPK_N_CONTROL,
    REPRESSILATOR_HORIZON_MIN,
    REPRESSILATOR_N_CONTROL,
    TOGGLE_HORIZON_MIN,
    TOGGLE_N_CONTROL,
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

# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def key() -> jax.Array:
    return jax.random.key(0)


@pytest.fixture
def rep_params() -> RepressilatorParams:
    return RepressilatorParams()


@pytest.fixture
def toggle_params() -> ToggleParams:
    return ToggleParams()


@pytest.fixture
def mapk_params() -> MAPKParams:
    return MAPKParams()


@pytest.fixture
def rep_sim() -> RepressilatorSimulator:
    return RepressilatorSimulator()


@pytest.fixture
def toggle_sim() -> ToggleSimulator:
    return ToggleSimulator()


@pytest.fixture
def mapk_sim() -> MAPKSimulator:
    return MAPKSimulator()


# ===========================================================================
# Repressilator tests (Elowitz & Leibler 2000)
# ===========================================================================


def test_repressilator_oscillates(
    rep_params: RepressilatorParams, key: jax.Array
) -> None:
    """With no control inputs (u=0), the repressilator must oscillate with
    a classical period inside the literature range.

    Elowitz & Leibler 2000 Fig. 3b reports period ~150 min for the nominal
    parameter set. Tomazou et al. 2018 *Cell Syst* 6:508 review reports a
    parameter-dependent period range of 50-300 min for repressilator-class
    oscillators (their Table 2). With our `RepressilatorParams` defaults
    (perturbed protein half-life = 5.5 min, mRNA half-life = 4 min, both
    inside the literature ranges per `bio_ode_params.py` comments), the
    expected period is on the faster end of this range. We test for
    period in [60, 220] min as a wide acceptance band that catches a
    correct oscillation while rejecting a non-oscillating or wrong-scale
    result.
    """
    # Long horizon to count multiple oscillations.
    sim_long = RepressilatorSimulator(
        horizon_minutes=600.0, n_save_points=601
    )
    y0 = default_repressilator_initial_state(rep_params)
    u = jnp.zeros((sim_long.n_control_points, sim_long.action_dim))

    traj = sim_long.simulate(y0, u, rep_params, key)
    assert traj.meta.n_nan_replacements == 0
    p1 = np.asarray(traj.states[:, 3])
    ts = np.asarray(traj.times)

    # Discard initial transient (first 100 min) before period analysis.
    settled = ts >= 100.0
    p1_settled = p1[settled]
    ts_settled = ts[settled]
    demean = p1_settled - p1_settled.mean()
    crossings = np.where(np.diff(np.sign(demean)) != 0)[0]
    assert len(crossings) >= 4, (
        f"too few zero-crossings ({len(crossings)}); "
        "system did not oscillate"
    )

    period = 2.0 * np.mean(np.diff(ts_settled[crossings]))
    assert 60.0 <= period <= 220.0, (
        f"repressilator period {period:.1f} min outside literature band "
        f"[60, 220] (Elowitz 2000 reports ~150 min nominal; "
        f"Tomazou 2018 reports parameter-dependent range 50-300 min)"
    )

    # Amplitude must be substantial (Elowitz Fig. 3b: peak/trough > 5x).
    amp = p1_settled.max() - p1_settled.min()
    assert amp > 100.0, (
        f"repressilator amplitude {amp:.1f} too small (expected > 100 nM)"
    )


def test_repressilator_control_breaks_oscillation(
    rep_sim: RepressilatorSimulator,
    rep_params: RepressilatorParams,
    key: jax.Array,
) -> None:
    """With u_1 = 1 throughout (saturating gene-1 silencer), the
    oscillation amplitude on gene 1's protein channel must drop measurably.

    Justification: in our control convention, u_i = 1 fully suppresses
    gene i's transcription (see `bio_ode.py` module docstring
    "sign convention" section). With gene 1 silenced, the cyclic feedback
    loop is broken: gene 2 (no longer repressed by p_1 cycling away) goes
    high, gene 3 (now repressed by p_2 high) goes low, gene 1 stays
    silenced. The oscillation collapses to a static fixed point on p_1.
    """
    y0 = default_repressilator_initial_state(rep_params)

    # Free oscillation
    u_free = jnp.zeros((rep_sim.n_control_points, rep_sim.action_dim))
    traj_free = rep_sim.simulate(y0, u_free, rep_params, key)
    p1_free = np.asarray(traj_free.states[:, 3])

    # Saturating gene-1 silencer
    u_silenced = jnp.zeros((rep_sim.n_control_points, rep_sim.action_dim))
    u_silenced = u_silenced.at[:, 0].set(1.0)
    traj_s = rep_sim.simulate(y0, u_silenced, rep_params, key)
    p1_s = np.asarray(traj_s.states[:, 3])

    amp_free = p1_free.max() - p1_free.min()
    amp_silenced = p1_s.max() - p1_s.min()

    assert traj_free.meta.n_nan_replacements == 0
    assert traj_s.meta.n_nan_replacements == 0
    assert amp_silenced < 0.5 * amp_free, (
        f"silencing did not collapse amplitude: free={amp_free:.1f}, "
        f"silenced={amp_silenced:.1f} (ratio "
        f"{amp_silenced / max(amp_free, 1e-9):.3f}, expected < 0.5)"
    )


# ===========================================================================
# Toggle tests (Gardner-Cantor-Collins 2000)
# ===========================================================================


def test_toggle_no_input_stays(
    toggle_sim: ToggleSimulator,
    toggle_params: ToggleParams,
    key: jax.Array,
) -> None:
    """Starting at the low stable state with u = 0, the toggle stays in
    the low state for the full horizon T.

    Gardner-Cantor-Collins 2000 Fig. 5a shows that, in the absence of an
    inducer pulse, the toggle is locked in whichever stable basin its
    initial condition resides in. Our default initial condition
    (A=0.05, B=7.5) is in the "B-high, A-low" basin (Gardner Fig. 5a,
    bottom-right of the phase diagram), and should remain there.
    """
    y0 = default_toggle_initial_state(toggle_params)
    u = jnp.zeros((toggle_sim.n_control_points, toggle_sim.action_dim))

    traj = toggle_sim.simulate(y0, u, toggle_params, key)
    assert traj.meta.n_nan_replacements == 0

    A_final = float(traj.states[-1, 0])
    B_final = float(traj.states[-1, 1])
    # Both endpoints in the same low/high configuration as the start.
    assert A_final < 1.0, (
        f"A escaped low-state basin: A_final = {A_final:.3f}, expected < 1.0"
    )
    assert B_final > 5.0, (
        f"B fell from high-state basin: B_final = {B_final:.3f}, "
        f"expected > 5.0"
    )


def test_toggle_bistable(
    toggle_sim: ToggleSimulator,
    toggle_params: ToggleParams,
    key: jax.Array,
) -> None:
    """Starting in the (A-low, B-high) state, an appropriate inducer pulse
    on u_1 (which inactivates A — wait, no: u_1 inactivates A, which is
    already low; we want to grow A by removing B). The correct flip is
    via u_2 (inactivates B, releasing A from repression).

    Specifically, we apply u_2 = 1 for the first 30 min (inactivating B,
    so the (A_eff)-on-(B-promoter) repression term goes to zero,
    releasing A's growth) then turn it off. Per Gardner-Cantor-Collins
    2000 Fig. 5a, this drives the system across the separatrix into the
    (A-high, B-low) stable state, where it remains after the pulse.
    """
    y0 = default_toggle_initial_state(toggle_params)
    # u_2 = 1 for first 30 min (3 of 10 control points), then off
    u_pulse = jnp.zeros((toggle_sim.n_control_points, toggle_sim.action_dim))
    u_pulse = u_pulse.at[:3, 1].set(1.0)

    traj = toggle_sim.simulate(y0, u_pulse, toggle_params, key)
    assert traj.meta.n_nan_replacements == 0

    A_final = float(traj.states[-1, 0])
    B_final = float(traj.states[-1, 1])
    # System should have flipped to (A-high, B-low).
    assert A_final > 50.0, (
        f"A failed to switch to high state: A_final = {A_final:.3f}, "
        f"expected > 50 (Gardner Fig. 5a high-state ~160)"
    )
    assert B_final < 1.0, (
        f"B failed to fall to low state: B_final = {B_final:.3f}, "
        f"expected < 1.0"
    )


# ===========================================================================
# MAPK tests (Huang-Ferrell 1996)
# ===========================================================================


def test_mapk_dose_response(
    mapk_sim: MAPKSimulator, mapk_params: MAPKParams, key: jax.Array
) -> None:
    """With constant u increasing 0 -> 1, terminal MAPK_PP shows
    ultrasensitive sigmoid with effective Hill coefficient > 1.

    Huang & Ferrell 1996 Fig. 6 reports n_eff ≈ 4-5 for the MAPK_PP
    response. We require n_eff > 1 here as a permissive "is the cascade
    actually ultrasensitive at all" test (the Hill fit is noisy when
    the dose-response sweep contains only a few transition points; a
    tight n_eff bound is reserved for the calibration spec test in
    `tests/test_bio_ode_specs.py`).
    """
    us = np.linspace(0.0, 1.0, 21)
    final_mapk_pp = []
    for u_val in us:
        y0 = default_mapk_initial_state(mapk_params)
        u_const = jnp.full(
            (mapk_sim.n_control_points, mapk_sim.action_dim), float(u_val)
        )
        traj = mapk_sim.simulate(y0, u_const, mapk_params, key)
        assert traj.meta.n_nan_replacements == 0
        final_mapk_pp.append(float(traj.states[-1, 4]))

    final = np.array(final_mapk_pp)
    # Monotone increasing in u (allow small numerical jitter).
    diffs = np.diff(final)
    assert (diffs >= -1e-3).all(), (
        f"MAPK dose-response not monotone: max negative step = "
        f"{float(diffs.min()):.4f}"
    )
    # Saturation: terminal value at u=1 should be near MAPK_total.
    assert final[-1] > 0.5 * mapk_params.MAPK_total_microM, (
        f"MAPK saturation too low: final at u=1 = {final[-1]:.3f}, "
        f"expected > {0.5 * mapk_params.MAPK_total_microM:.3f}"
    )

    # Hill exponent fit on the rising portion.
    y_norm = final / mapk_params.MAPK_total_microM
    mask = (y_norm > 0.05) & (y_norm < 0.95) & (us > 0.0)
    if mask.sum() >= 3:
        log_u = np.log(us[mask])
        log_y = np.log(y_norm[mask] / (1.0 - y_norm[mask]))
        n_eff, _ = np.polyfit(log_u, log_y, 1)
        # Permissive bound: cascade is ultrasensitive (n_eff > 1) — full
        # n_eff range bracket is (3, 6) per `MAPKParams.hill_n_effective_range`,
        # but the polyfit on a 21-point sweep is noisy.
        assert n_eff > 1.0, (
            f"MAPK is not ultrasensitive: n_eff = {n_eff:.2f}, expected > 1"
        )


# ===========================================================================
# Cross-cutting tests (NaN, JIT, protocol)
# ===========================================================================


def test_no_nan_repressilator(
    rep_sim: RepressilatorSimulator,
    rep_params: RepressilatorParams,
    key: jax.Array,
) -> None:
    """Random control sequences must never produce non-finite states."""
    keys = jax.random.split(key, 8)
    for k in keys:
        u = jax.random.uniform(
            k, shape=(rep_sim.n_control_points, rep_sim.action_dim),
            minval=0.0, maxval=1.0,
        )
        y0 = default_repressilator_initial_state(rep_params)
        traj = rep_sim.simulate(y0, u, rep_params, k)
        assert jnp.all(jnp.isfinite(traj.states)), (
            "non-finite state escaped sentinel guard (repressilator)"
        )


def test_no_nan_toggle(
    toggle_sim: ToggleSimulator,
    toggle_params: ToggleParams,
    key: jax.Array,
) -> None:
    """Random control sequences must never produce non-finite states."""
    keys = jax.random.split(key, 8)
    for k in keys:
        u = jax.random.uniform(
            k, shape=(toggle_sim.n_control_points, toggle_sim.action_dim),
            minval=0.0, maxval=1.0,
        )
        y0 = default_toggle_initial_state(toggle_params)
        traj = toggle_sim.simulate(y0, u, toggle_params, k)
        assert jnp.all(jnp.isfinite(traj.states)), (
            "non-finite state escaped sentinel guard (toggle)"
        )


def test_no_nan_mapk(
    mapk_sim: MAPKSimulator, mapk_params: MAPKParams, key: jax.Array
) -> None:
    """Random control sequences must never produce non-finite states."""
    keys = jax.random.split(key, 8)
    for k in keys:
        u = jax.random.uniform(
            k, shape=(mapk_sim.n_control_points, mapk_sim.action_dim),
            minval=0.0, maxval=1.0,
        )
        y0 = default_mapk_initial_state(mapk_params)
        traj = mapk_sim.simulate(y0, u, mapk_params, k)
        assert jnp.all(jnp.isfinite(traj.states)), (
            "non-finite state escaped sentinel guard (mapk)"
        )


def test_jit_works_repressilator(
    rep_sim: RepressilatorSimulator,
    rep_params: RepressilatorParams,
    key: jax.Array,
) -> None:
    """Repressilator simulator runs under jax.jit end to end."""
    y0 = default_repressilator_initial_state(rep_params)
    u = jnp.zeros((rep_sim.n_control_points, rep_sim.action_dim))

    # Params are static — they are literature-fixed constants, not traced
    # arrays, and the dataclass is not a JAX pytree. The simulator object
    # itself is also static (its eqx.Module fields are all static).
    @jax.jit
    def run(y0_, u_, key_, params_=rep_params):
        traj = rep_sim.simulate(y0_, u_, params_, key_)
        return traj.states

    states = run(y0, u, key)
    assert states.shape == (rep_sim.n_save_points, rep_sim.state_dim)
    assert jnp.all(jnp.isfinite(states))


def test_jit_works_toggle(
    toggle_sim: ToggleSimulator,
    toggle_params: ToggleParams,
    key: jax.Array,
) -> None:
    """Toggle simulator runs under jax.jit end to end."""
    y0 = default_toggle_initial_state(toggle_params)
    u = jnp.zeros((toggle_sim.n_control_points, toggle_sim.action_dim))

    @jax.jit
    def run(y0_, u_, key_, params_=toggle_params):
        traj = toggle_sim.simulate(y0_, u_, params_, key_)
        return traj.states

    states = run(y0, u, key)
    assert states.shape == (toggle_sim.n_save_points, toggle_sim.state_dim)
    assert jnp.all(jnp.isfinite(states))


def test_jit_works_mapk(
    mapk_sim: MAPKSimulator, mapk_params: MAPKParams, key: jax.Array
) -> None:
    """MAPK simulator runs under jax.jit end to end."""
    y0 = default_mapk_initial_state(mapk_params)
    u = jnp.zeros((mapk_sim.n_control_points, mapk_sim.action_dim))

    @jax.jit
    def run(y0_, u_, key_, params_=mapk_params):
        traj = mapk_sim.simulate(y0_, u_, params_, key_)
        return traj.states

    states = run(y0, u, key)
    assert states.shape == (mapk_sim.n_save_points, mapk_sim.state_dim)
    assert jnp.all(jnp.isfinite(states))


def test_protocol_compliance() -> None:
    """Each simulator implements the `Simulator` Protocol exposed in
    bio_ode.py, which mirrors `paper/architecture.md` "Simulator interface".

    Protocol checks: presence of `simulate` callable plus the three
    integer properties (`state_dim`, `action_dim`, `horizon`).
    """
    rep = RepressilatorSimulator()
    tog = ToggleSimulator()
    mapk = MAPKSimulator()
    for sim, expected_state, expected_action, expected_horizon in (
        (rep, 6, 3, REPRESSILATOR_N_CONTROL),
        (tog, 2, 2, TOGGLE_N_CONTROL),
        (mapk, 6, 1, MAPK_N_CONTROL),
    ):
        # Structural Protocol membership check.
        assert isinstance(sim, Simulator), (
            f"{type(sim).__name__} does not implement Simulator protocol"
        )
        assert callable(sim.simulate)
        assert isinstance(sim.state_dim, int)
        assert isinstance(sim.action_dim, int)
        assert isinstance(sim.horizon, int)
        assert sim.state_dim == expected_state
        assert sim.action_dim == expected_action
        assert sim.horizon == expected_horizon


def test_trajectory_shape_consistency(
    rep_sim: RepressilatorSimulator,
    toggle_sim: ToggleSimulator,
    mapk_sim: MAPKSimulator,
    rep_params: RepressilatorParams,
    toggle_params: ToggleParams,
    mapk_params: MAPKParams,
    key: jax.Array,
) -> None:
    """The returned `Trajectory` pytree has the shapes documented in
    `paper/architecture.md`: states (T, n), actions (H, m), times (T,).
    """
    for sim, params, y0_fn, expected_n, expected_m in (
        (rep_sim, rep_params, default_repressilator_initial_state, 6, 3),
        (toggle_sim, toggle_params, default_toggle_initial_state, 2, 2),
        (mapk_sim, mapk_params, default_mapk_initial_state, 6, 1),
    ):
        y0 = y0_fn(params)
        u = jnp.zeros((sim.n_control_points, sim.action_dim))
        traj = sim.simulate(y0, u, params, key)
        assert isinstance(traj, Trajectory)
        assert isinstance(traj.meta, TrajectoryMeta)
        assert traj.states.shape == (sim.n_save_points, expected_n)
        assert traj.actions.shape == (sim.n_control_points, expected_m)
        assert traj.times.shape == (sim.n_save_points,)
        # Times are monotone increasing.
        assert jnp.all(jnp.diff(traj.times) > 0.0)


def test_horizon_matches_specs() -> None:
    """The simulator horizons match the values declared in `bio_ode_specs.py`
    so spec evaluation aligns time intervals correctly.

    Pulled from the bio_ode_specs metadata to keep the cross-reference
    explicit (rather than re-hardcoding the literal numbers here).
    """
    from stl_seed.specs.bio_ode_specs import (
        MAPK_T,
        REPRESSILATOR_T,
        TOGGLE_T,
    )

    assert REPRESSILATOR_HORIZON_MIN == REPRESSILATOR_T
    assert TOGGLE_HORIZON_MIN == TOGGLE_T
    assert MAPK_HORIZON_MIN == MAPK_T


def test_solver_validation() -> None:
    """Constructor rejects unknown solver names with a clear ValueError."""
    with pytest.raises(ValueError, match="solver"):
        RepressilatorSimulator(solver="rk4")
    with pytest.raises(ValueError, match="solver"):
        ToggleSimulator(solver="euler")
    with pytest.raises(ValueError, match="solver"):
        MAPKSimulator(solver="dopri5")


def test_kvaerno5_solver_runs(
    rep_params: RepressilatorParams, key: jax.Array
) -> None:
    """The Kvaerno5 stiff fallback solver runs and produces a valid
    trajectory on the repressilator (which is non-stiff but still
    integrable by an implicit method).
    """
    sim = RepressilatorSimulator(solver="kvaerno5")
    y0 = default_repressilator_initial_state(rep_params)
    u = jnp.zeros((sim.n_control_points, sim.action_dim))
    traj = sim.simulate(y0, u, rep_params, key)
    assert jnp.all(jnp.isfinite(traj.states))
    # Stiff-fallback flag set.
    assert int(traj.meta.used_stiff_fallback) == 1

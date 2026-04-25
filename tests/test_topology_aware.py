"""Tests for ``TopologyAwareController`` and the repressilator heuristic dispatch.

These tests pin down the repressilator topology-aware feedback policy that
replaced the bang-bang controller in the ``bio_ode.repressilator`` slot of
``_HEURISTIC_DEFAULTS``. The bang-bang policy silenced the target gene
itself (which drives the target protein DOWN, the wrong direction); the
topology-aware policy silences the *upstream repressor* in the cyclic ring
(Elowitz & Leibler 2000 Nature 403:335 Fig. 1a), which drives the target
protein UP and satisfies ``bio_ode.repressilator.easy``.

topology dict is the wiring of the Elowitz-Leibler ring.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stl_seed.generation import (
    HeuristicPolicy,
    TopologyAwareController,
)
from stl_seed.generation.policies import PIDController
from stl_seed.specs import REGISTRY
from stl_seed.stl.evaluator import evaluate_robustness
from stl_seed.tasks.bio_ode import (
    MAPKSimulator,
    RepressilatorSimulator,
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

# Repressilator wiring (gene i is repressed by gene (i-1) mod 3).
_REPRESS_TOPOLOGY: dict[int, int] = {0: 2, 1: 0, 2: 1}


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------


def test_topology_aware_rejects_empty_topology() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        TopologyAwareController(topology={}, target_gene=0)


def test_topology_aware_rejects_unknown_target() -> None:
    with pytest.raises(KeyError, match="target_gene"):
        TopologyAwareController(topology=_REPRESS_TOPOLOGY, target_gene=99)


def test_topology_aware_rejects_bad_direction() -> None:
    with pytest.raises(ValueError, match="target_direction"):
        TopologyAwareController(
            topology=_REPRESS_TOPOLOGY,
            target_gene=0,
            target_direction="sideways",  # type: ignore[arg-type]
        )


def test_topology_aware_observation_indices_length_check() -> None:
    """observation_indices must have one entry per gene."""
    with pytest.raises(ValueError, match="observation_indices length"):
        TopologyAwareController(
            topology=_REPRESS_TOPOLOGY,
            target_gene=0,
            observation_indices=[3, 4],  # too short for 3 genes
        )


def test_topology_aware_default_action_dim_matches_genes() -> None:
    """With no explicit action_dim, the controller picks one channel per gene."""
    c = TopologyAwareController(topology=_REPRESS_TOPOLOGY, target_gene=0)
    assert c.action_dim == 3
    assert c._upstream_idx == 2  # gene 0 is repressed by gene 2


# ---------------------------------------------------------------------------
# Single-call action emission
# ---------------------------------------------------------------------------


def test_topology_aware_drive_high_silences_upstream_when_target_low() -> None:
    """target below threshold + direction='high' => u_upstream = 1, others 0."""
    c = TopologyAwareController(
        topology=_REPRESS_TOPOLOGY,
        target_gene=0,
        target_direction="high",
        threshold=137.5,
        observation_indices=[3, 4, 5],
    )
    state = jnp.array([0.0, 0.0, 0.0, 15.0, 5.0, 25.0])  # p1=15 < 137.5
    spec = REGISTRY["bio_ode.repressilator.easy"]
    a = np.asarray(c(state, spec, [], jax.random.key(0)))
    np.testing.assert_allclose(a, [0.0, 0.0, 1.0])  # silence gene 2 (upstream of 0)


def test_topology_aware_drive_high_releases_when_target_above() -> None:
    """target above threshold + direction='high' => all-zero (let it stay)."""
    c = TopologyAwareController(
        topology=_REPRESS_TOPOLOGY,
        target_gene=0,
        target_direction="high",
        threshold=137.5,
        observation_indices=[3, 4, 5],
    )
    state = jnp.array([0.0, 0.0, 0.0, 500.0, 5.0, 25.0])  # p1=500 > 137.5
    spec = REGISTRY["bio_ode.repressilator.easy"]
    a = np.asarray(c(state, spec, [], jax.random.key(0)))
    np.testing.assert_allclose(a, [0.0, 0.0, 0.0])


def test_topology_aware_drive_low_silences_target_when_above_threshold() -> None:
    """direction='low' silences the target gene's own transcription."""
    c = TopologyAwareController(
        topology=_REPRESS_TOPOLOGY,
        target_gene=1,
        target_direction="low",
        threshold=100.0,
        observation_indices=[3, 4, 5],
    )
    state = jnp.array([0.0, 0.0, 0.0, 5.0, 500.0, 25.0])  # p2=500 > 100
    spec = REGISTRY["bio_ode.repressilator.easy"]
    a = np.asarray(c(state, spec, [], jax.random.key(0)))
    np.testing.assert_allclose(a, [0.0, 1.0, 0.0])  # silence gene 1


# ---------------------------------------------------------------------------
# End-to-end: controller drives the repressilator into the spec-satisfying
# regime.
# ---------------------------------------------------------------------------


def _rollout_open_loop(controller, sim, params, init, spec):
    """Mimic ``TrajectoryRunner._rollout_one``: feed the *initial* state H times.

    History accumulates (state, action) pairs across the H calls so that
    stateful controllers (e.g. :class:`PIDController`'s integral term) see
    the same history the production runner shows them.
    """
    key = jax.random.key(0)
    actions: list = []
    history: list = []
    state = init
    for h in range(sim.n_control_points):
        a = controller(state, spec, history, jax.random.fold_in(key, h))
        a = jnp.asarray(a, dtype=jnp.float32)
        actions.append(a)
        history.append((state, a))
    control = jnp.stack(actions, axis=0)
    return sim.simulate(init, control, params, key)


def test_topology_aware_repressilator_satisfies() -> None:
    """End-to-end: with the default repressilator config, the simulated
    trajectory satisfies ``bio_ode.repressilator.easy`` with ρ > 0.

    This is the regression test for the A13 pilot fix: under the previous
    bang-bang policy ρ was -247 (catastrophic miss); under the topology-
    aware policy ρ should be ~ +25.
    """
    sim = RepressilatorSimulator()
    params = RepressilatorParams()
    init = default_repressilator_initial_state(params)
    spec = REGISTRY["bio_ode.repressilator.easy"]

    h = HeuristicPolicy("bio_ode.repressilator")
    traj = _rollout_open_loop(h, sim, params, init, spec)
    rho = float(evaluate_robustness(spec, traj))
    assert rho > 0, f"topology-aware heuristic failed to satisfy spec: rho={rho}"
    # The expected ρ under u=(0,0,1) constant is p_1_steady - 250 ~ 25.
    assert rho > 20.0, f"rho={rho} below expected ~+25"


def test_topology_aware_drives_target_high_via_simulator() -> None:
    """A different target-direction config (drive p_2 high by silencing gene 0)
    should also push the chosen target protein above threshold by t=horizon.

    This verifies that the controller is not hard-coded to gene 0 — the
    topology dispatch generalises.
    """
    sim = RepressilatorSimulator()
    params = RepressilatorParams()
    init = default_repressilator_initial_state(params)
    spec = REGISTRY["bio_ode.repressilator.easy"]  # spec object only used for protocol

    # Drive p_2 high: gene 1's upstream repressor is gene 0 (per topology[1] = 0).
    c = TopologyAwareController(
        topology=_REPRESS_TOPOLOGY,
        target_gene=1,
        target_direction="high",
        threshold=137.5,
        observation_indices=[3, 4, 5],
    )
    traj = _rollout_open_loop(c, sim, params, init, spec)
    p2_final = float(np.asarray(traj.states[-1, 4]))  # state[4] = p_2
    assert p2_final > 250.0, f"controller failed to drive p_2 high: p2_final={p2_final}"


# ---------------------------------------------------------------------------
# Toggle (Gardner-Cantor-Collins 2000) — topology-aware controller dispatch
# ---------------------------------------------------------------------------


def test_topology_aware_drives_toggle_high() -> None:
    """The HeuristicPolicy("bio_ode.toggle") should silence the upstream
    repressor (gene 1) of the target gene 0, driving x_1 to its
    asymptote (alpha_1 = 160 in dimensionless units) and x_2 to ~0.

    NOTE on spec satisfaction (post-2026-04-25 fix). The
    bio_ode.toggle.medium spec was corrected on 2026-04-25 to set
    ``x_1 >= 100`` (down from the unreachable ``x_1 >= 200``); the
    simulator's dimensionless dynamics cap x_1 at alpha_1 = 160, so
    the new HIGH=100 threshold is comfortably reachable. The
    controller drives x_1 to ~160 and x_2 to ~0, satisfying the spec
    with rho ~ +30 in the open-loop rollout. We still assert on the
    structural property (controller emits the right action shape and
    the simulated trajectory bottoms out at x_2 << LOW=30 with x_1
    saturated near alpha_1) because that is what this test exists to
    verify; the end-to-end spec satisfaction is exercised by
    ``test_inference.py::test_beam_search_solves_toggle``.
    """
    sim = ToggleSimulator()
    params = ToggleParams()
    init = default_toggle_initial_state(params)
    spec = REGISTRY["bio_ode.toggle.medium"]

    h = HeuristicPolicy("bio_ode.toggle")
    # Sanity check: the dispatch routes to the topology-aware controller.
    assert isinstance(h._impl, TopologyAwareController)
    # Action dim = 2 (one inducer per gene).
    assert h.action_dim == 2

    traj = _rollout_open_loop(h, sim, params, init, spec)
    # Controller silences gene 1 (the upstream repressor of gene 0):
    # u_1 should be ~1, u_0 ~0.
    actions_np = np.asarray(traj.actions)
    assert actions_np[:, 1].min() > 0.99, f"u_1 should be ~1, got {actions_np[:, 1]}"
    assert actions_np[:, 0].max() < 0.01, f"u_0 should be ~0, got {actions_np[:, 0]}"

    # Final state: x_1 saturated near its asymptote alpha_1 = 160; x_2 ~ 0.
    x1_final = float(np.asarray(traj.states[-1, 0]))
    x2_final = float(np.asarray(traj.states[-1, 1]))
    assert x1_final > 0.95 * params.alpha_1, (
        f"x_1 should approach alpha_1={params.alpha_1}, got {x1_final}"
    )
    # The bio_ode.toggle.medium spec requires x_2 < LOW = 30 nM in [60, 100].
    # That clause IS achievable; the controller drives it to ~0.008.
    assert x2_final < 30.0, f"x_2 should be << LOW=30, got {x2_final}"


# ---------------------------------------------------------------------------
# MAPK (Huang-Ferrell 1996) — PID controller dispatch
# ---------------------------------------------------------------------------


def test_pid_controller_observation_indices_default() -> None:
    """PIDController defaults to watching state[0] when observation_indices
    is not given (preserves backward compatibility with the glucose-insulin
    PID config)."""
    pid = PIDController(setpoint=110.0)
    assert pid.observation_indices == [0]
    assert pid._obs_idx == 0


def test_pid_controller_respects_observation_indices() -> None:
    """When observation_indices is given, the PID watches that state component."""
    pid = PIDController(
        setpoint=0.5,
        kp=1.0,
        ki=0.0,
        kd=0.0,
        action_clip=(0.0, 1.0),
        observation_indices=[4],
        error_sign=-1.0,
    )
    # state[4] = 0.0 (below setpoint); error = -1 * (0 - 0.5) = +0.5;
    # action = kp * error = 0.5. Clipped to [0, 1] so still 0.5.
    state = jnp.zeros((6,))
    a = pid(state, REGISTRY["bio_ode.mapk.hard"], [], jax.random.key(0))
    np.testing.assert_allclose(np.asarray(a), [0.5], atol=1e-6)


def test_pid_drives_mapk_setpoint() -> None:
    """The HeuristicPolicy("bio_ode.mapk") (PID on MAPK_PP at index 4) should
    raise the terminal kinase MAPK_PP above the 0.5 setpoint within the
    horizon.

    NOTE on spec satisfaction (post-2026-04-25 fix). The
    bio_ode.mapk.hard spec was corrected on 2026-04-25 to read state
    index 4 (MAPK_PP) using ABSOLUTE microM thresholds (peak >= 0.5,
    settle < 0.05, MKKK safety < 0.002975) instead of the broken
    fractional thresholds on the wrong state index. The PID
    controller below already observed index 4 at the textbook 0.5
    microM setpoint, so it remains structurally correct against the
    fixed spec. End-to-end spec satisfaction (the reach-then-settle
    pulse pattern) is exercised by
    ``test_inference.py::test_beam_search_solves_mapk``; this test
    just confirms the PID drives MAPK_PP above the activation gate.
    """
    sim = MAPKSimulator()
    params = MAPKParams()
    init = default_mapk_initial_state(params)
    spec = REGISTRY["bio_ode.mapk.hard"]

    h = HeuristicPolicy("bio_ode.mapk")
    # Sanity check: the dispatch routes to a PID controller.
    assert isinstance(h._impl, PIDController)
    assert h._impl._obs_idx == 4  # MAPK_PP in the 6-state simulator
    assert h.action_dim == 1

    traj = _rollout_open_loop(h, sim, params, init, spec)
    # MAPK_PP at the end of the horizon should exceed the 0.5 setpoint band.
    mapk_pp_peak = float(np.asarray(traj.states[:, 4]).max())
    assert mapk_pp_peak > 0.5, (
        f"PID failed to drive MAPK_PP above setpoint=0.5, peak={mapk_pp_peak}"
    )
    # Actions should monotonically grow as the integral term accumulates
    # under sustained below-setpoint observation.
    actions_np = np.asarray(traj.actions)[:, 0]
    assert actions_np[-1] > actions_np[0], (
        f"PID action did not grow over horizon: actions={actions_np}"
    )

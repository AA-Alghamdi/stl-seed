"""Tests for ``TopologyAwareController`` and the repressilator heuristic dispatch.

These tests pin down the repressilator topology-aware feedback policy that
replaced the bang-bang controller in the ``bio_ode.repressilator`` slot of
``_HEURISTIC_DEFAULTS``. The bang-bang policy silenced the target gene
itself (which drives the target protein DOWN, the wrong direction); the
topology-aware policy silences the *upstream repressor* in the cyclic ring
(Elowitz & Leibler 2000 Nature 403:335 Fig. 1a), which drives the target
protein UP and satisfies ``bio_ode.repressilator.easy``.

REDACTED firewall: nothing in this test imports any REDACTED artifact; the
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
from stl_seed.specs import REGISTRY
from stl_seed.stl.evaluator import evaluate_robustness
from stl_seed.tasks.bio_ode import (
    RepressilatorSimulator,
    default_repressilator_initial_state,
)
from stl_seed.tasks.bio_ode_params import RepressilatorParams

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
    """Mimic ``TrajectoryRunner._rollout_one``: feed the *initial* state H times."""
    key = jax.random.key(0)
    actions = []
    state = init
    for h in range(sim.n_control_points):
        a = controller(state, spec, [], jax.random.fold_in(key, h))
        actions.append(jnp.asarray(a, dtype=jnp.float32))
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

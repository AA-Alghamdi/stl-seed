"""Tests for the K-step rollout-tree gradient probing sampler.

Test plan
---------

T1. ``test_rollout_tree_protocol_compliance`` — RolloutTreeSampler conforms
    to the :class:`Sampler` Protocol; `sample()` returns a (Trajectory, dict)
    pair with the canonical keys.
T2. ``test_rollout_tree_runs_glucose_insulin_no_crash`` — end-to-end run on
    the glucose-insulin task family, all four continuation policies.
T3. ``test_rollout_tree_runs_repressilator_no_crash`` — end-to-end run on
    the repressilator (the failure case the algorithm exists to address).
T4. ``test_rollout_tree_better_than_standard_synthetic`` — on a synthetic
    delayed-reward problem where 1-step lookahead is bad and K-step
    lookahead is good, the rollout-tree sampler strictly beats vanilla
    sampling on mean rho across 6 seeds.
T5. ``test_rollout_tree_invalid_branch_k_raises`` — branch_k < 1 raises.
T6. ``test_rollout_tree_invalid_lookahead_h_raises`` — lookahead_h < 0 raises.
T7. ``test_rollout_tree_invalid_continuation_policy_raises`` — unknown policy
    string raises with a clear message.
T8. ``test_rollout_tree_heuristic_requires_callable`` — selecting
    ``"heuristic"`` without supplying ``heuristic_continuation`` raises.
T9. ``test_rollout_tree_branch_k_caps_at_vocabulary_size`` — requesting more
    branches than vocabulary items silently caps, doesn't raise.
T10. ``test_rollout_tree_diagnostics_well_formed`` — every diagnostic field
     has the expected shape / type / range.

REDACTED firewall. None of these tests import REDACTED / REDACTED /
REDACTED / REDACTED / REDACTED.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import pytest
from jaxtyping import Array, Float, PRNGKeyArray

from stl_seed.inference import LLMProposal, Sampler
from stl_seed.inference.gradient_guided import make_uniform_action_vocabulary
from stl_seed.inference.rollout_tree import (
    RolloutTreeSampler,
)
from stl_seed.specs import (
    REGISTRY,
    Always,
    And,
    Eventually,
    Interval,
    Predicate,
    STLSpec,
)
from stl_seed.tasks._trajectory import Trajectory, TrajectoryMeta
from stl_seed.tasks.bio_ode import (
    REPRESSILATOR_ACTION_DIM,
    RepressilatorSimulator,
    _repressilator_initial_state,
)
from stl_seed.tasks.bio_ode_params import RepressilatorParams
from stl_seed.tasks.glucose_insulin import (
    BergmanParams,
    GlucoseInsulinSimulator,
    default_normal_subject_initial_state,
)

# ---------------------------------------------------------------------------
# Synthetic LLM proxies (mirrors test_inference.py).
# ---------------------------------------------------------------------------


def _uniform_llm(K: int) -> LLMProposal:
    """Flat LLM (entropy = log K)."""

    def llm(state, history, key):
        return jnp.zeros(K)

    return llm


def _peaked_llm(target_idx: int, K: int, peak_logit: float = 5.0) -> LLMProposal:
    """Deterministic LLM peaked at ``target_idx``."""

    def llm(state, history, key):
        z = jnp.zeros(K)
        return z.at[target_idx].set(peak_logit)

    return llm


def _myopic_llm(K: int, myopic_idx: int = 0) -> LLMProposal:
    """LLM that strongly prefers ``myopic_idx`` — the locally-greedy choice
    on the synthetic delayed-reward problem below.

    Used to stress-test that the rollout-tree sampler can override the
    LLM's myopic prior in favour of the K-step-better candidate.
    """

    def llm(state, history, key):
        z = jnp.zeros(K)
        # Slight preference for ``myopic_idx``; rollout-tree should
        # override this when the K-step lookahead disagrees.
        return z.at[myopic_idx].set(2.0)

    return llm


# ---------------------------------------------------------------------------
# Synthetic delayed-reward simulator.
#
# A 1-d toy ODE where the state "reward potential" only develops when the
# control sequence sustains a specific action over multiple steps. This is
# the structural setting that makes 1-step gradient probing fail and
# K-step lookahead succeed.
#
# State y in R; control u in R^1.
# ODE: dy/dt = u - y/10 (first-order tracking with time constant 10).
# Horizon T = 10, n_control = 5 (one step per 2 time units).
#
# Spec: G_[8, 10] (y >= 0.8), i.e. the state must be sustained-high in the
# back third of the horizon. Achieving this requires u ~ 1 across most of
# the 5 control steps (the integrator's tracking time constant is 10 vs
# horizon 10, so a single high-u step is insufficient — the state decays
# back before the spec window).
#
# The 1-step probe at step t=0 sees only the immediate ramp-up and is
# myopic w.r.t. whether u is sustained; the K-step probe at step t=0 with
# lookahead_h=4 actually carries the candidate's effect through the
# horizon and ranks "u=1" highest.
# ---------------------------------------------------------------------------


_TOY_HORIZON_T = 10.0
_TOY_N_CONTROL = 5
_TOY_N_SAVE = 21  # 0.5-time-unit grid


class _ToyParams(eqx.Module):
    decay: Float[Array, ""] = eqx.field(default_factory=lambda: jnp.asarray(0.1))


def _toy_u_at_time(
    t: Float[Array, ""],
    control_sequence: Float[Array, "H 1"],
    horizon: float,
) -> Float[Array, " 1"]:
    H = control_sequence.shape[0]
    dt = horizon / H
    idx = jnp.clip(jnp.floor(t / dt).astype(jnp.int32), 0, H - 1)
    return control_sequence[idx]


def _toy_vector_field(
    t: Float[Array, ""],
    y: Float[Array, " 1"],
    args: tuple[Float[Array, "H 1"], _ToyParams, float],
) -> Float[Array, " 1"]:
    control, params, horizon = args
    u = _toy_u_at_time(t, control, horizon)
    # dy/dt = u - decay * y
    return u - params.decay * y


class _ToySimulator(eqx.Module):
    """Synthetic delayed-reward simulator for the K-step-vs-1-step test."""

    horizon_minutes: float = eqx.field(static=True, default=_TOY_HORIZON_T)
    n_control_points: int = eqx.field(static=True, default=_TOY_N_CONTROL)
    n_save_points: int = eqx.field(static=True, default=_TOY_N_SAVE)

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
        initial_state: Float[Array, " 1"],
        control_sequence: Float[Array, "H 1"],
        params: _ToyParams,
        key: PRNGKeyArray,
    ) -> Trajectory:
        del key
        import diffrax as dfx

        u_clipped = jnp.clip(control_sequence, 0.0, 1.0)
        term = dfx.ODETerm(_toy_vector_field)
        solver = dfx.Tsit5()
        controller = dfx.PIDController(rtol=1e-6, atol=1e-9)
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
            max_steps=8192,
            throw=False,
        )
        ys = sol.ys
        assert ys is not None
        meta = TrajectoryMeta(
            n_nan_replacements=jnp.asarray(0, dtype=jnp.int32),
            final_solver_result=jnp.asarray(sol.result._value, dtype=jnp.int32),
            used_stiff_fallback=jnp.asarray(0, dtype=jnp.int32),
        )
        return Trajectory(
            states=ys,
            actions=u_clipped,
            times=save_times,
            meta=meta,
        )


def _toy_spec() -> STLSpec:
    """G_[8, 10] (y >= 0.8) — sustained-high requirement.

    Constructed inline (not registered) so each test instance is fresh.
    The predicate uses the (channel, threshold) default-argument convention
    from :mod:`stl_seed.specs.bio_ode_specs._gt` so the STL evaluator can
    JIT-compile it (required by RolloutTreeSampler).
    """

    def _y_gt(name: str, channel: int, threshold: float) -> Predicate:
        return Predicate(
            name,
            fn=lambda traj, t, c=channel, th=threshold: float(traj[t, c]) - th,
        )

    return STLSpec(
        name="toy.delayed_reward",
        formula=Always(_y_gt("y_high", 0, 0.8), interval=Interval(8.0, _TOY_HORIZON_T)),
        signal_dim=1,
        horizon_minutes=_TOY_HORIZON_T,
        description="Sustained-high y over the back fifth of the horizon.",
        citations=("Synthetic test problem (no real-world reference).",),
        formula_text="G_[8,10] (y >= 0.8)",
        metadata={"subdomain": "toy", "difficulty": "synthetic"},
    )


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gi_setup():
    sim = GlucoseInsulinSimulator()
    params = BergmanParams()
    spec = REGISTRY["glucose_insulin.tir.easy"]
    V = make_uniform_action_vocabulary(0.0, 5.0, k_per_dim=5)
    x0 = default_normal_subject_initial_state(params)
    return sim, params, spec, V, x0


@pytest.fixture(scope="module")
def repressilator_setup():
    sim = RepressilatorSimulator()
    params = RepressilatorParams()
    x0 = _repressilator_initial_state(params)
    spec = REGISTRY["bio_ode.repressilator.easy"]
    V = make_uniform_action_vocabulary(
        [0.0] * REPRESSILATOR_ACTION_DIM,
        [1.0] * REPRESSILATOR_ACTION_DIM,
        k_per_dim=2,
    )
    return sim, params, spec, V, x0


@pytest.fixture(scope="module")
def toy_setup():
    sim = _ToySimulator()
    params = _ToyParams()
    spec = _toy_spec()
    V = jnp.asarray([[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=jnp.float32)
    x0 = jnp.asarray([0.0], dtype=jnp.float32)
    return sim, params, spec, V, x0


# ---------------------------------------------------------------------------
# T1. Protocol compliance.
# ---------------------------------------------------------------------------


def test_rollout_tree_protocol_compliance(gi_setup) -> None:
    sim, params, spec, V, x0 = gi_setup
    s = RolloutTreeSampler(
        _uniform_llm(int(V.shape[0])),
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
        branch_k=3,
        lookahead_h=2,
        continuation_policy="zero",
    )
    assert isinstance(s, Sampler)
    traj, diag = s.sample(x0, jax.random.key(0))
    assert isinstance(traj, Trajectory)
    assert diag["sampler"] == "rollout_tree"
    assert "final_rho" in diag
    assert np.isfinite(diag["final_rho"])
    # Per-step diagnostic lists must all have length H.
    H = sim.n_control_points
    for key in (
        "rho_stream_at_step",
        "projected_rho_at_step",
        "branch_rho_min_at_step",
        "branch_rho_max_at_step",
        "branch_rho_mean_at_step",
        "chosen_index_at_step",
        "would_pick_top_logit_at_step",
        "refine_grad_norm_at_step",
    ):
        assert len(diag[key]) == H, f"{key} length {len(diag[key])} != {H}"


# ---------------------------------------------------------------------------
# T2. End-to-end on glucose-insulin, all continuation policies.
# ---------------------------------------------------------------------------


def test_rollout_tree_runs_glucose_insulin_no_crash(gi_setup) -> None:
    sim, params, spec, V, x0 = gi_setup
    K = int(V.shape[0])

    # zero / random / llm continuation: no extra setup needed.
    for policy in ("zero", "random", "llm"):
        sampler = RolloutTreeSampler(
            _uniform_llm(K),
            sim,
            spec,
            V,
            params,
            horizon=sim.n_control_points,
            branch_k=3,
            lookahead_h=2,
            continuation_policy=policy,
        )
        traj, diag = sampler.sample(x0, jax.random.key(7))
        assert isinstance(traj, Trajectory)
        assert np.isfinite(diag["final_rho"])
        assert diag["continuation_policy"] == policy

    # Heuristic policy: supply a callable returning a constant block.
    def heuristic(state, history, L, key):
        # constant lookahead = midpoint vocabulary action
        return jnp.tile(jnp.asarray([[2.5]], dtype=jnp.float32), (L, 1))

    sampler = RolloutTreeSampler(
        _uniform_llm(K),
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
        branch_k=3,
        lookahead_h=2,
        continuation_policy="heuristic",
        heuristic_continuation=heuristic,
    )
    traj, diag = sampler.sample(x0, jax.random.key(8))
    assert isinstance(traj, Trajectory)
    assert np.isfinite(diag["final_rho"])
    assert diag["continuation_policy"] == "heuristic"


# ---------------------------------------------------------------------------
# T3. End-to-end on repressilator (the cross-task failure case).
# ---------------------------------------------------------------------------


def test_rollout_tree_runs_repressilator_no_crash(repressilator_setup) -> None:
    sim, params, spec, V, x0 = repressilator_setup
    K = int(V.shape[0])
    sampler = RolloutTreeSampler(
        _uniform_llm(K),
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
        branch_k=K,  # use all 8 corners
        lookahead_h=5,
        continuation_policy="zero",
    )
    traj, diag = sampler.sample(x0, jax.random.key(2026))
    assert isinstance(traj, Trajectory)
    assert traj.states.shape == (sim.n_save_points, 6)
    assert traj.actions.shape == (sim.n_control_points, 3)
    assert np.isfinite(diag["final_rho"])


# ---------------------------------------------------------------------------
# T4. Synthetic delayed-reward problem: K-step beats 1-step.
# ---------------------------------------------------------------------------


def test_rollout_tree_better_than_standard_synthetic(toy_setup) -> None:
    """Pre-registered: on the toy delayed-reward problem (sustained y high
    in [8, 10]), the K-step rollout-tree sampler with ``lookahead_h = 4``
    must achieve strictly higher mean rho than vanilla LLM sampling
    across 6 seeds.

    Setup. The toy ODE ``dy/dt = u - 0.1 * y`` with ``y(0) = 0`` and
    spec ``G_[8, 10] (y >= 0.8)`` requires sustained ``u ~ 1`` across
    the 5 control steps. The LLM is *biased toward the locally myopic
    choice* ``u = 0.0`` (logit +2 on vocabulary index 0 = action 0.0),
    so vanilla sampling almost always fails the spec.

    The K-step probe at step t = 0 with ``lookahead_h = 4``,
    ``continuation_policy = "zero"`` evaluates each of the 5 candidate
    first actions with the rest filled in zero. Even with a zero
    continuation, the candidate ``u = 1`` ramps the state up the
    fastest, and the leaf rho ranking points away from the LLM's
    myopic preference. Tree-search must therefore strictly beat
    vanilla.

    Falsification criterion: ``mean_tree <= mean_vanilla``.

    Note: the synthetic spec does NOT saturate, so any small mean lift
    is meaningful (rho is bounded between -0.8 and +0.2 on this
    problem).
    """
    sim, params, spec, V, x0 = toy_setup
    K = int(V.shape[0])  # K = 5

    vanilla = RolloutTreeSampler(
        _myopic_llm(K, myopic_idx=0),
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
        branch_k=1,  # branch_k=1 + lookahead_h=0 == argmax-LLM (tree degenerates)
        lookahead_h=0,
        continuation_policy="zero",
    )
    tree = RolloutTreeSampler(
        _myopic_llm(K, myopic_idx=0),
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
        branch_k=K,
        lookahead_h=4,
        continuation_policy="zero",
    )

    n_seeds = 6
    rhos_v: list[float] = []
    rhos_t: list[float] = []
    for s in range(n_seeds):
        key = jax.random.key(2000 + s)
        _, dv = vanilla.sample(x0, key)
        _, dt = tree.sample(x0, key)
        rhos_v.append(float(dv["final_rho"]))
        rhos_t.append(float(dt["final_rho"]))

    mean_v = float(np.mean(rhos_v))
    mean_t = float(np.mean(rhos_t))
    paired_wins = sum(t > v for t, v in zip(rhos_t, rhos_v, strict=True))
    assert mean_t > mean_v, (
        f"K-step rollout-tree did not beat vanilla on synthetic delayed-reward: "
        f"mean_vanilla = {mean_v:.4f}, mean_tree = {mean_t:.4f}, "
        f"paired wins = {paired_wins}/{n_seeds}, "
        f"per-seed vanilla = {rhos_v}, per-seed tree = {rhos_t}"
    )


# ---------------------------------------------------------------------------
# T5-T8. Constructor validation.
# ---------------------------------------------------------------------------


def test_rollout_tree_invalid_branch_k_raises(gi_setup) -> None:
    sim, params, spec, V, x0 = gi_setup
    with pytest.raises(ValueError, match="branch_k must be >= 1"):
        RolloutTreeSampler(
            _uniform_llm(int(V.shape[0])),
            sim,
            spec,
            V,
            params,
            horizon=sim.n_control_points,
            branch_k=0,
        )


def test_rollout_tree_invalid_lookahead_h_raises(gi_setup) -> None:
    sim, params, spec, V, x0 = gi_setup
    with pytest.raises(ValueError, match="lookahead_h must be >= 0"):
        RolloutTreeSampler(
            _uniform_llm(int(V.shape[0])),
            sim,
            spec,
            V,
            params,
            horizon=sim.n_control_points,
            branch_k=2,
            lookahead_h=-1,
        )


def test_rollout_tree_invalid_continuation_policy_raises(gi_setup) -> None:
    sim, params, spec, V, x0 = gi_setup
    with pytest.raises(ValueError, match="continuation_policy must be one of"):
        RolloutTreeSampler(
            _uniform_llm(int(V.shape[0])),
            sim,
            spec,
            V,
            params,
            horizon=sim.n_control_points,
            branch_k=2,
            lookahead_h=2,
            continuation_policy="bogus",  # type: ignore[arg-type]
        )


def test_rollout_tree_heuristic_requires_callable(gi_setup) -> None:
    sim, params, spec, V, x0 = gi_setup
    with pytest.raises(ValueError, match="requires a heuristic_continuation"):
        RolloutTreeSampler(
            _uniform_llm(int(V.shape[0])),
            sim,
            spec,
            V,
            params,
            horizon=sim.n_control_points,
            branch_k=2,
            lookahead_h=2,
            continuation_policy="heuristic",
        )


# ---------------------------------------------------------------------------
# T9. branch_k caps at vocabulary size.
# ---------------------------------------------------------------------------


def test_rollout_tree_branch_k_caps_at_vocabulary_size(gi_setup) -> None:
    sim, params, spec, V, x0 = gi_setup
    K = int(V.shape[0])
    # Request more branches than the vocabulary has — must silently cap.
    sampler = RolloutTreeSampler(
        _uniform_llm(K),
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
        branch_k=K + 100,
        lookahead_h=2,
    )
    assert sampler.branch_k == K
    traj, diag = sampler.sample(x0, jax.random.key(0))
    assert isinstance(traj, Trajectory)
    assert diag["branch_k"] == K


# ---------------------------------------------------------------------------
# T10. Diagnostics shape / type sanity.
# ---------------------------------------------------------------------------


def test_rollout_tree_diagnostics_well_formed(toy_setup) -> None:
    sim, params, spec, V, x0 = toy_setup
    K = int(V.shape[0])
    sampler = RolloutTreeSampler(
        _peaked_llm(K // 2, K=K),
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
        branch_k=3,
        lookahead_h=2,
        continuation_policy="zero",
        refine_iters=0,
    )
    traj, diag = sampler.sample(x0, jax.random.key(33))

    H = sim.n_control_points
    assert diag["n_steps"] == H
    assert diag["branch_k"] == 3
    assert diag["lookahead_h"] == 2
    assert diag["continuation_policy"] == "zero"
    assert diag["refine_iters"] == 0
    # branch min <= mean <= max at every step.
    for tt in range(H):
        assert (
            diag["branch_rho_min_at_step"][tt]
            <= diag["branch_rho_mean_at_step"][tt]
            <= diag["branch_rho_max_at_step"][tt]
        )
        # The chosen branch's projected rho must equal the per-step max
        # under sampling_temperature=0 (argmax over branches).
        assert diag["projected_rho_at_step"][tt] == pytest.approx(
            diag["branch_rho_max_at_step"][tt]
        )
    # Chosen indices are valid vocabulary indices.
    for idx in diag["chosen_index_at_step"]:
        assert 0 <= idx < K
    # Disagreement counter is consistent.
    n_disagree = diag["n_steps_disagree_with_llm"]
    n_agree = sum(1 for f in diag["would_pick_top_logit_at_step"] if f == 1)
    assert n_disagree + n_agree == H

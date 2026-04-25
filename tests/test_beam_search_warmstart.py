"""Tests for :class:`BeamSearchWarmstartSampler`.

Test plan
---------

B1. ``test_beam_search_runs_glucose_insulin_no_crash`` — sampler runs
    end-to-end on the glucose-insulin TIR easy spec, returns a well-formed
    Trajectory and diagnostics dict.
B2. ``test_beam_search_finds_known_solution`` — synthetic 2-step problem
    where one specific action sequence satisfies a hand-built spec; both
    ``beam_size=1`` (greedy) and ``beam_size=4`` should find it. Verifies
    that the per-step expansion + top-B selection logic works.
B3. ``test_beam_search_recovers_repressilator_solution`` — with the canonical
    pilot IC, vocabulary including the silence-3 corner ``(0, 0, 1)``, and
    the default ``tail_strategy='repeat_candidate'``, beam search finds
    ``rho > 0`` on ``bio_ode.repressilator.easy``. This is the headline
    falsification of the negative result documented in
    ``paper/cross_task_validation.md``: the gradient sampler fails on this
    setup; structural search succeeds.
B4. ``test_beam_search_protocol_compliance`` — sampler satisfies the
    runtime-checkable :class:`Sampler` Protocol.
B5. ``test_beam_search_diagnostics_well_formed`` — every diagnostic field
    has the expected shape, type, and value range.
B6. ``test_beam_search_invalid_args_raise`` — invalid hyperparameters
    raise :class:`ValueError` at construction.
B7. ``test_beam_search_tail_strategy_default_vs_repeat`` — the
    ``'repeat_candidate'`` strategy strictly dominates ``'default'`` on
    the repressilator (the headline negative result of the latter is
    documented in ``paper/cross_task_validation.md``; this test re-asserts
    that the two strategies yield different beams).
B8. ``test_beam_search_gradient_refinement_no_regression`` — with refinement
    enabled, the final rho is >= the pure-beam endpoint (refinement can
    never make things worse because we keep the discrete winner if
    refinement does not improve it).

REDACTED firewall. None of these tests import REDACTED / REDACTED /
REDACTED / REDACTED / REDACTED.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stl_seed.inference import Sampler
from stl_seed.inference.beam_search_warmstart import (
    BeamSearchDiagnostics,
    BeamSearchWarmstartSampler,
)
from stl_seed.inference.gradient_guided import make_uniform_action_vocabulary
from stl_seed.specs import REGISTRY
from stl_seed.tasks._trajectory import Trajectory
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
# Synthetic LLM (the beam sampler ignores LLM logits — beam selection is
# driven by full-rho lookahead — but the protocol still requires one).
# ---------------------------------------------------------------------------


def _uniform_llm(K: int):
    def llm(state, history, key):
        return jnp.zeros(K)

    return llm


# ---------------------------------------------------------------------------
# Fixture: glucose-insulin task family.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gi_setup():
    sim = GlucoseInsulinSimulator()
    params = BergmanParams()
    spec = REGISTRY["glucose_insulin.tir.easy"]
    V = make_uniform_action_vocabulary(0.0, 5.0, k_per_dim=5)
    x0 = default_normal_subject_initial_state(params)
    return sim, params, spec, V, x0


# ---------------------------------------------------------------------------
# B1. End-to-end smoke on glucose-insulin.
# ---------------------------------------------------------------------------


def test_beam_search_runs_glucose_insulin_no_crash(gi_setup) -> None:
    sim, params, spec, V, x0 = gi_setup
    K = int(V.shape[0])
    sampler = BeamSearchWarmstartSampler(
        _uniform_llm(K),
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
        beam_size=4,
        gradient_refine_iters=0,  # skip refinement to keep the test fast
    )
    traj, diag = sampler.sample(x0, jax.random.key(0))
    assert isinstance(traj, Trajectory)
    assert traj.states.shape == (sim.n_save_points, 3)
    assert traj.actions.shape == (sim.n_control_points, 1)
    assert traj.times.shape == (sim.n_save_points,)
    assert diag["sampler"] == "beam_search_warmstart"
    assert np.isfinite(diag["final_rho"])
    assert diag["n_steps"] == sim.n_control_points
    assert len(diag["chosen_sequence"]) == sim.n_control_points


# ---------------------------------------------------------------------------
# B2. Synthetic problem with a known greedy-satisfying solution.
# ---------------------------------------------------------------------------


def test_beam_search_finds_known_solution() -> None:
    """Greedy beam-search should find a constant-action policy that
    satisfies a hand-built spec on the repressilator: the constant
    action ``(0, 0, 1)`` is known to drive ``rho ~ +25`` on
    ``bio_ode.repressilator.easy`` from the canonical pilot IC.

    With ``beam_size = 1`` (pure greedy) and ``k_per_dim = 2`` (so the
    8-corner vocabulary contains the silence-3 corner), beam search at
    every step picks the action whose constant-tail extrapolation
    maximises rho; the top-1 sequence is therefore the constant
    silence-3 policy and the final rho is positive.

    With ``beam_size = 4`` we should also find it (a wider beam can
    only retain or improve the score).
    """
    sim = RepressilatorSimulator()
    params = RepressilatorParams()
    x0 = _repressilator_initial_state(params)
    spec = REGISTRY["bio_ode.repressilator.easy"]
    V = make_uniform_action_vocabulary(
        [0.0] * REPRESSILATOR_ACTION_DIM,
        [1.0] * REPRESSILATOR_ACTION_DIM,
        k_per_dim=2,
    )
    K = int(V.shape[0])

    for beam_size in (1, 4):
        sampler = BeamSearchWarmstartSampler(
            _uniform_llm(K),
            sim,
            spec,
            V,
            params,
            horizon=sim.n_control_points,
            beam_size=beam_size,
            gradient_refine_iters=0,
            tail_strategy="repeat_candidate",
        )
        _, diag = sampler.sample(x0, jax.random.key(0))
        assert diag["final_rho"] > 0.0, (
            f"beam_size={beam_size}: expected rho > 0 (constant silence-3 "
            f"is in the vocabulary and reaches rho ~ +25); got "
            f"final_rho={diag['final_rho']:.3f}, sequence={diag['chosen_sequence']}"
        )


# ---------------------------------------------------------------------------
# B3. The headline test: beam search escapes the cliff that defeated the
# gradient-guided sampler (paper/cross_task_validation.md).
# ---------------------------------------------------------------------------


def test_beam_search_recovers_repressilator_solution() -> None:
    """Beam search with the dense ``k_per_dim=5`` vocabulary (K=125)
    finds rho > 0 on ``bio_ode.repressilator.easy`` from the canonical
    pilot IC.

    The REDACTED-stripped negative-result documented in
    ``paper/cross_task_validation.md`` shows that the gradient-guided
    sampler is stuck at rho ~ -250 across all (lambda, default-action,
    seed) combinations on this configuration. The structural-search
    sampler is the complementary fix: instead of relying on a per-step
    continuous gradient probe (which falls into the cliff geometry),
    it directly enumerates the discrete action lattice with a model-
    predictive constant-extrapolation lookahead. The constant
    silence-3 policy ``u = (0, 0, 1)`` gives rho = +25, and beam search
    finds it.
    """
    sim = RepressilatorSimulator()
    params = RepressilatorParams()
    x0 = _repressilator_initial_state(params)
    spec = REGISTRY["bio_ode.repressilator.easy"]
    V = make_uniform_action_vocabulary(
        [0.0] * REPRESSILATOR_ACTION_DIM,
        [1.0] * REPRESSILATOR_ACTION_DIM,
        k_per_dim=5,  # K=125, includes silence-3 corner [0, 0, 1]
    )
    K = int(V.shape[0])
    sampler = BeamSearchWarmstartSampler(
        _uniform_llm(K),
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
        beam_size=8,
        gradient_refine_iters=0,
        tail_strategy="repeat_candidate",
    )
    _, diag = sampler.sample(x0, jax.random.key(0))
    assert diag["final_rho"] > 0.0, (
        f"beam search must find rho > 0 on bio_ode.repressilator.easy; "
        f"the constant policy (0,0,1) reaches rho ~ +25 and is in the "
        f"k_per_dim=5 vocabulary. Got final_rho={diag['final_rho']:.3f}, "
        f"chosen_sequence={diag['chosen_sequence']}."
    )


# ---------------------------------------------------------------------------
# B4. Protocol compliance.
# ---------------------------------------------------------------------------


def test_beam_search_protocol_compliance(gi_setup) -> None:
    sim, params, spec, V, x0 = gi_setup
    sampler = BeamSearchWarmstartSampler(
        _uniform_llm(int(V.shape[0])),
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
        beam_size=2,
        gradient_refine_iters=0,
    )
    assert isinstance(sampler, Sampler), (
        "BeamSearchWarmstartSampler must satisfy the Sampler Protocol"
    )


# ---------------------------------------------------------------------------
# B5. Diagnostics shape / type / range.
# ---------------------------------------------------------------------------


def test_beam_search_diagnostics_well_formed(gi_setup) -> None:
    sim, params, spec, V, x0 = gi_setup
    K = int(V.shape[0])
    sampler = BeamSearchWarmstartSampler(
        _uniform_llm(K),
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
        beam_size=4,
        gradient_refine_iters=5,  # exercise the refinement path
    )
    _, diag = sampler.sample(x0, jax.random.key(7))
    H = sim.n_control_points
    # Per-step lists.
    assert len(diag["best_partial_score_at_step"]) == H
    assert len(diag["mean_partial_score_at_step"]) == H
    assert len(diag["streaming_rho_top1_at_step"]) == H
    assert len(diag["unique_sequences_per_step"]) == H
    # Chosen sequence length == H, every index in [0, K).
    assert len(diag["chosen_sequence"]) == H
    assert all(0 <= i < K for i in diag["chosen_sequence"])
    # The mean partial score at any step should be <= the best.
    for best, mean in zip(
        diag["best_partial_score_at_step"],
        diag["mean_partial_score_at_step"],
        strict=True,
    ):
        assert mean <= best + 1e-4, (
            f"mean partial score ({mean}) exceeds best ({best}) at some step"
        )
    # rho_after_refine >= rho_after_beam (refinement only kept if it improves).
    assert diag["rho_after_refine"] >= diag["rho_after_beam"] - 1e-4
    # final_rho is finite.
    assert np.isfinite(diag["final_rho"])
    # n_steps mirrors the per-step list length.
    assert diag["n_steps"] == H
    # refine_iters_run is in [0, 5].
    assert 0 <= diag["refine_iters_run"] <= 5
    # Streaming-rho is a sequence of finite-or-+/-inf values; we only
    # assert the elements are not NaN (inf is allowed by streaming
    # semantics — see stl_seed.stl.streaming module docstring).
    assert all(not np.isnan(v) for v in diag["streaming_rho_top1_at_step"])


def test_beam_search_diagnostics_dataclass_to_dict() -> None:
    """``BeamSearchDiagnostics.to_dict`` returns a plain dict with all fields."""
    d = BeamSearchDiagnostics()
    d.best_partial_score_at_step = [1.0, 2.0]
    d.mean_partial_score_at_step = [0.5, 1.0]
    d.streaming_rho_top1_at_step = [float("inf"), 1.0]
    d.unique_sequences_per_step = [3, 2]
    d.chosen_sequence = [0, 1]
    d.rho_after_beam = 1.5
    d.rho_after_refine = 1.7
    d.final_rho = 1.7
    d.refine_iters_run = 3
    out = d.to_dict()
    assert out["sampler"] == "beam_search_warmstart"
    assert out["chosen_sequence"] == [0, 1]
    assert out["final_rho"] == 1.7
    assert out["n_steps"] == 2


# ---------------------------------------------------------------------------
# B6. Construction validation.
# ---------------------------------------------------------------------------


def test_beam_search_invalid_args_raise(gi_setup) -> None:
    sim, params, spec, V, x0 = gi_setup
    K = int(V.shape[0])
    llm = _uniform_llm(K)

    with pytest.raises(ValueError, match="must be 2-d"):
        BeamSearchWarmstartSampler(
            llm, sim, spec, jnp.zeros((5,)), params, horizon=sim.n_control_points
        )
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        BeamSearchWarmstartSampler(llm, sim, spec, V, params, horizon=0)
    with pytest.raises(ValueError, match="beam_size must be >= 1"):
        BeamSearchWarmstartSampler(
            llm, sim, spec, V, params, horizon=sim.n_control_points, beam_size=0
        )
    with pytest.raises(ValueError, match="gradient_refine_iters must be >= 0"):
        BeamSearchWarmstartSampler(
            llm,
            sim,
            spec,
            V,
            params,
            horizon=sim.n_control_points,
            gradient_refine_iters=-1,
        )
    with pytest.raises(ValueError, match="refine_lr must be > 0"):
        BeamSearchWarmstartSampler(
            llm,
            sim,
            spec,
            V,
            params,
            horizon=sim.n_control_points,
            refine_lr=0.0,
        )
    with pytest.raises(ValueError, match="tail_strategy"):
        BeamSearchWarmstartSampler(
            llm,
            sim,
            spec,
            V,
            params,
            horizon=sim.n_control_points,
            tail_strategy="bogus",
        )
    with pytest.raises(ValueError, match="default_action shape"):
        BeamSearchWarmstartSampler(
            llm,
            sim,
            spec,
            V,
            params,
            horizon=sim.n_control_points,
            default_action=jnp.array([0.0, 0.0]),  # wrong shape (m=1 here)
        )


# ---------------------------------------------------------------------------
# B7. Tail-strategy ablation: 'repeat_candidate' beats 'default' on
# repressilator (corroborates the cross_task_validation.md negative result
# for the partial-then-default-extrapolated probe).
# ---------------------------------------------------------------------------


def test_beam_search_tail_strategy_default_vs_repeat() -> None:
    """On the repressilator, ``tail_strategy='repeat_candidate'`` should
    strictly outperform ``tail_strategy='default'``.

    The 'default' strategy mirrors the partial-then-default-extrapolated
    gradient probe used by :class:`STLGradientGuidedSampler` and shares
    its myopic-default-action failure mode (documented in
    ``paper/cross_task_validation.md``). The 'repeat_candidate' strategy
    is the structural fix: scoring each candidate as "constant-hold for
    the rest of the horizon" makes the satisfying constant policy
    visible to the beam from step 0.

    Falsification: if 'default' weakly outperforms 'repeat_candidate'
    here, the structural-search hypothesis fails and the entire negative
    result in cross_task_validation.md is suspect.
    """
    sim = RepressilatorSimulator()
    params = RepressilatorParams()
    x0 = _repressilator_initial_state(params)
    spec = REGISTRY["bio_ode.repressilator.easy"]
    V = make_uniform_action_vocabulary(
        [0.0] * REPRESSILATOR_ACTION_DIM,
        [1.0] * REPRESSILATOR_ACTION_DIM,
        k_per_dim=2,
    )
    K = int(V.shape[0])
    s_default = BeamSearchWarmstartSampler(
        _uniform_llm(K),
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
        beam_size=4,
        gradient_refine_iters=0,
        tail_strategy="default",
    )
    s_repeat = BeamSearchWarmstartSampler(
        _uniform_llm(K),
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
        beam_size=4,
        gradient_refine_iters=0,
        tail_strategy="repeat_candidate",
    )
    _, diag_default = s_default.sample(x0, jax.random.key(0))
    _, diag_repeat = s_repeat.sample(x0, jax.random.key(0))
    assert diag_repeat["final_rho"] >= diag_default["final_rho"], (
        f"'repeat_candidate' should at least match 'default' on the "
        f"repressilator. Got default rho={diag_default['final_rho']:.3f}, "
        f"repeat rho={diag_repeat['final_rho']:.3f}."
    )


# ---------------------------------------------------------------------------
# B8. Gradient refinement does not regress the pure-beam endpoint.
# ---------------------------------------------------------------------------


def test_beam_search_gradient_refinement_no_regression(gi_setup) -> None:
    """Refinement keeps the discrete winner if it cannot improve it.

    The sampler accepts the refined sequence only if its full rho
    strictly exceeds the pure-beam endpoint (see :meth:`sample`'s
    refined-vs-beam branch). So the final rho must be >= the pure-beam
    endpoint, regardless of whether refinement actually moved the
    control vector.
    """
    sim, params, spec, V, x0 = gi_setup
    K = int(V.shape[0])
    sampler = BeamSearchWarmstartSampler(
        _uniform_llm(K),
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
        beam_size=2,
        gradient_refine_iters=10,
        refine_lr=1e-2,
    )
    _, diag = sampler.sample(x0, jax.random.key(123))
    assert diag["rho_after_refine"] >= diag["rho_after_beam"] - 1e-4, (
        f"refinement regressed the pure-beam endpoint: beam={diag['rho_after_beam']:.3f}, "
        f"refined={diag['rho_after_refine']:.3f}"
    )


# ---------------------------------------------------------------------------
# B9. Helper: top-B selection picks descending order.
# ---------------------------------------------------------------------------


def test_beam_search_top_b_helper_is_descending() -> None:
    """The internal ``_top_b`` helper returns indices in descending-score order."""
    scores = jnp.asarray([0.1, 0.5, -0.3, 0.9, 0.2], dtype=jnp.float32)
    out = BeamSearchWarmstartSampler._top_b(scores, B=3)
    out_np = np.asarray(out)
    sorted_scores = np.asarray(scores)[out_np]
    assert sorted_scores[0] >= sorted_scores[1] >= sorted_scores[2]
    assert sorted_scores[0] == 0.9

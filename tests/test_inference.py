"""Tests for the gradient-guided inference module.

Test plan
---------

T1. ``test_standard_sampler_no_crash`` — vanilla LLM sampling produces a
    well-formed Trajectory on the glucose-insulin task.
T2. ``test_bon_sampler_selects_satisfying`` — BoN with a known-rho LLM
    set returns the first satisfying sample (rho > 0).
T3. ``test_continuous_bon_selects_argmax`` — Continuous BoN returns the
    argmax-rho sample over its budget.
T4. ``test_gradient_guided_zero_lambda_matches_standard`` — lambda = 0
    reduces to StandardSampler (modulo numerical-tracer noise).
T5. ``test_gradient_guided_improves_rho`` — with lambda > 0, mean rho
    over a batch of seeds is strictly higher than at lambda = 0. The
    PRE-REGISTERED hypothesis the algorithm exists to support.
T6. ``test_jit_compatibility`` — the rho_from_control closure is JIT-
    compatible (the value-and-grad call inside the sampler does not
    raise).
T7. ``test_diagnostics_well_formed`` — every diagnostic field has the
    expected shape / type / range.
T8. ``test_streaming_partial_rho_used`` — confirms the partial-trajectory
    construction (not just the full-trajectory rho) is what the gradient
    sees, by checking that step-t guidance bias is sensitive to actions
    committed before step t.
T9. ``test_protocol_compliance`` — all four samplers satisfy the Sampler
    Protocol (runtime-checkable isinstance).
T10. ``test_make_vocabulary_shapes_and_endpoints`` — vocabulary builder
    returns the right shape and includes the endpoints.

REDACTED firewall. None of these tests import REDACTED / REDACTED /
REDACTED / REDACTED / REDACTED.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import pytest

from stl_seed.inference import (
    BestOfNSampler,
    ContinuousBoNSampler,
    HybridGradientBoNSampler,
    LLMProposal,
    Sampler,
    StandardSampler,
    STLGradientGuidedSampler,
)
from stl_seed.inference.gradient_guided import make_uniform_action_vocabulary
from stl_seed.specs import REGISTRY
from stl_seed.stl import evaluate_robustness
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
# Synthetic LLM proxies for reproducible tests.
# ---------------------------------------------------------------------------


def _peaked_llm(target_idx: int, K: int, peak_logit: float = 5.0) -> LLMProposal:
    """A deterministic LLM that puts most mass on `target_idx`."""

    def llm(state, history, key):
        z = jnp.zeros(K)
        return z.at[target_idx].set(peak_logit)

    return llm


def _uniform_llm(K: int) -> LLMProposal:
    """A flat LLM (entropy = log K) — pure prior."""

    def llm(state, history, key):
        return jnp.zeros(K)

    return llm


def _state_dependent_llm(K: int, channel: int = 0) -> LLMProposal:
    """LLM whose argmax depends on the observed state.

    Produces logits proportional to ``state[channel]`` shifted to the
    middle vocabulary item — so trajectories with different state evolve
    different choice sequences. Used for streaming-partial-rho test.
    """

    def llm(state, history, key):
        anchor = jnp.clip(state[channel] / 100.0, 0.0, K - 1.0)
        idx = jnp.argmin(jnp.abs(jnp.arange(K) - anchor))
        z = jnp.zeros(K)
        return z.at[idx].set(3.0)

    return llm


# ---------------------------------------------------------------------------
# Fixture: glucose-insulin task family (smallest reliable simulator).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gi_setup():
    """Return (simulator, params, spec, vocabulary, x0)."""
    sim = GlucoseInsulinSimulator()
    params = BergmanParams()
    spec = REGISTRY["glucose_insulin.tir.easy"]
    V = make_uniform_action_vocabulary(0.0, 5.0, k_per_dim=5)
    x0 = default_normal_subject_initial_state(params)
    return sim, params, spec, V, x0


# ---------------------------------------------------------------------------
# T10. Vocabulary builder.
# ---------------------------------------------------------------------------


def test_make_vocabulary_shapes_and_endpoints() -> None:
    V = make_uniform_action_vocabulary(0.0, 5.0, k_per_dim=4)
    assert V.shape == (4, 1)
    assert float(V[0, 0]) == pytest.approx(0.0)
    assert float(V[-1, 0]) == pytest.approx(5.0)

    V2 = make_uniform_action_vocabulary([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], k_per_dim=3)
    assert V2.shape == (27, 3)
    # Corners must appear.
    rows = {tuple(np.asarray(V2[i]).round(3)) for i in range(27)}
    assert (0.0, 0.0, 0.0) in rows
    assert (1.0, 1.0, 1.0) in rows
    assert (1.0, 0.0, 1.0) in rows


def test_make_vocabulary_invalid_raises() -> None:
    with pytest.raises(ValueError, match="k_per_dim"):
        make_uniform_action_vocabulary(0.0, 1.0, k_per_dim=1)
    with pytest.raises(ValueError, match="must match"):
        make_uniform_action_vocabulary([0.0, 0.0], [1.0], k_per_dim=2)


# ---------------------------------------------------------------------------
# T1. StandardSampler runs end-to-end.
# ---------------------------------------------------------------------------


def test_standard_sampler_no_crash(gi_setup) -> None:
    sim, params, spec, V, x0 = gi_setup
    llm = _peaked_llm(2, K=int(V.shape[0]))
    sampler = StandardSampler(
        llm,
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
    )
    traj, diag = sampler.sample(x0, jax.random.key(0))
    assert isinstance(traj, Trajectory)
    assert traj.states.shape == (sim.n_save_points, 3)
    assert traj.actions.shape == (sim.n_control_points, 1)
    assert traj.times.shape == (sim.n_save_points,)
    assert "final_rho" in diag
    assert np.isfinite(diag["final_rho"])
    assert diag["sampler"] == "standard"


# ---------------------------------------------------------------------------
# T2 / T3. BoN selection logic.
# ---------------------------------------------------------------------------


def test_bon_sampler_selects_satisfying(gi_setup) -> None:
    """With a peaked LLM that puts mass on the satisfying action, BoN's
    chosen sample must have rho > threshold (or, if no sample satisfies,
    found_satisfying must be False)."""
    sim, params, spec, V, x0 = gi_setup
    llm = _peaked_llm(target_idx=2, K=int(V.shape[0]), peak_logit=2.0)
    sampler = BestOfNSampler(
        llm,
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
        n=4,
    )
    traj, diag = sampler.sample(x0, jax.random.key(42))
    assert diag["sampler"] == "best_of_n"
    assert diag["n_samples"] == 4
    assert len(diag["all_rho"]) == 4
    assert diag["chosen_rho"] == diag["all_rho"][diag["chosen_index"]]
    if diag["found_satisfying"]:
        assert diag["chosen_rho"] > diag["rho_threshold"]
    else:
        # No satisfier; the chosen index must be 0 (canonical fallback).
        assert diag["chosen_index"] == 0


def test_bon_sampler_invalid_n_raises(gi_setup) -> None:
    sim, params, spec, V, x0 = gi_setup
    with pytest.raises(ValueError, match="n must be >= 1"):
        BestOfNSampler(
            _uniform_llm(int(V.shape[0])),
            sim,
            spec,
            V,
            params,
            horizon=sim.n_control_points,
            n=0,
        )


def test_continuous_bon_selects_argmax(gi_setup) -> None:
    """Continuous BoN's chosen sample must equal argmax_rho over the
    generated batch."""
    sim, params, spec, V, x0 = gi_setup
    llm = _uniform_llm(int(V.shape[0]))
    sampler = ContinuousBoNSampler(
        llm,
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
        n=8,
    )
    traj, diag = sampler.sample(x0, jax.random.key(1234))
    assert diag["sampler"] == "continuous_bon"
    assert diag["n_samples"] == 8
    assert len(diag["all_rho"]) == 8
    # Continuous BoN must pick the argmax (not just any positive sample).
    chosen_rho = diag["all_rho"][diag["chosen_index"]]
    assert chosen_rho == max(diag["all_rho"])
    assert diag["chosen_rho"] == diag["max_rho"]


# ---------------------------------------------------------------------------
# T4. lambda = 0 recovers StandardSampler.
# ---------------------------------------------------------------------------


def test_gradient_guided_zero_lambda_matches_standard(gi_setup) -> None:
    """At lambda = 0 the bias is zero, so the gradient-guided sampler
    should produce the same chosen-action sequence as StandardSampler
    at zero temperature (greedy argmax). We use temperature = 0 to make
    both deterministic."""
    sim, params, spec, V, x0 = gi_setup
    llm = _peaked_llm(target_idx=3, K=int(V.shape[0]))

    standard = StandardSampler(
        llm,
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
        sampling_temperature=0.0,
    )
    guided = STLGradientGuidedSampler(
        llm,
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
        guidance_weight=0.0,
        sampling_temperature=0.0,
    )

    key = jax.random.key(99)
    traj_s, diag_s = standard.sample(x0, key)
    traj_g, diag_g = guided.sample(x0, key)

    np.testing.assert_allclose(
        np.asarray(traj_s.actions),
        np.asarray(traj_g.actions),
        atol=1e-5,
        err_msg="lambda=0 must reproduce standard sampling action sequence",
    )
    # And rho should match within float tolerance.
    assert diag_s["final_rho"] == pytest.approx(diag_g["final_rho"], rel=1e-3)


# ---------------------------------------------------------------------------
# T5. Pre-registered hypothesis: gradient guidance improves mean rho.
# ---------------------------------------------------------------------------


def test_gradient_guided_improves_rho(gi_setup) -> None:
    """Pre-registered hypothesis: with a flat (no-info) LLM and lambda > 0,
    the gradient-guided sampler produces strictly higher mean rho than
    the lambda = 0 baseline, at matched compute (one sample per seed).

    Falsification criterion: mean_rho_guided <= mean_rho_baseline. If
    this test fires, the algorithm is either buggy or the hypothesis
    fails on this task family — both demand investigation.

    We use a flat-prior LLM so the only signal driving choices is the
    STL gradient. With a peaked LLM the guidance can be 'overruled' by
    a strong prior; the flat-LLM regime is where gradient guidance
    should dominate most cleanly.
    """
    sim, params, spec, V, x0 = gi_setup
    llm = _uniform_llm(int(V.shape[0]))

    baseline = STLGradientGuidedSampler(
        llm,
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
        guidance_weight=0.0,
        sampling_temperature=0.5,
    )
    guided = STLGradientGuidedSampler(
        llm,
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
        guidance_weight=2.0,
        sampling_temperature=0.5,
    )

    n_seeds = 6
    rhos_baseline = []
    rhos_guided = []
    for s in range(n_seeds):
        k = jax.random.key(1000 + s)
        _, d_b = baseline.sample(x0, k)
        _, d_g = guided.sample(x0, k)
        rhos_baseline.append(d_b["final_rho"])
        rhos_guided.append(d_g["final_rho"])

    mean_b = float(np.mean(rhos_baseline))
    mean_g = float(np.mean(rhos_guided))
    # Pre-registered margin: at least 0.5 rho units improvement on the
    # TIR spec (whose natural scale is ~1-50 mg/dL). The smoke test above
    # showed an improvement of ~14 rho units; 0.5 is a safe lower bound.
    assert mean_g > mean_b + 0.5, (
        f"gradient guidance should improve mean rho. "
        f"baseline mean = {mean_b:.3f}, guided mean = {mean_g:.3f}, "
        f"per-seed baseline = {rhos_baseline}, "
        f"per-seed guided = {rhos_guided}"
    )


# ---------------------------------------------------------------------------
# T6. JIT compatibility.
# ---------------------------------------------------------------------------


def test_jit_compatibility(gi_setup) -> None:
    """The internal value_and_grad closure inside STLGradientGuidedSampler
    is JIT'd. We additionally verify the rho_from_control function can be
    jitted in isolation, ensuring the gradient path is JIT-clean."""
    sim, params, spec, V, x0 = gi_setup
    llm = _uniform_llm(int(V.shape[0]))
    guided = STLGradientGuidedSampler(
        llm,
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
        guidance_weight=1.0,
    )
    # Direct JIT of the rho-from-control closure.
    fn = jax.jit(guided._rho_from_control)
    control = jnp.full((sim.n_control_points, 1), 1.5, dtype=jnp.float32)
    rho = fn(x0, control, jax.random.key(0))
    assert np.isfinite(float(rho))

    # value_and_grad must produce a finite scalar and a finite (H, 1) gradient.
    rho_v, dctrl = guided._value_and_grad(x0, control, jax.random.key(0))
    assert np.isfinite(float(rho_v))
    assert dctrl.shape == (sim.n_control_points, 1)
    assert np.all(np.isfinite(np.asarray(dctrl)))


# ---------------------------------------------------------------------------
# T7. Diagnostics well-formed.
# ---------------------------------------------------------------------------


def test_diagnostics_well_formed(gi_setup) -> None:
    sim, params, spec, V, x0 = gi_setup
    llm = _peaked_llm(2, K=int(V.shape[0]))
    guided = STLGradientGuidedSampler(
        llm,
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
        guidance_weight=1.0,
    )
    _, diag = guided.sample(x0, jax.random.key(7))
    H = sim.n_control_points
    assert diag["n_steps"] == H
    assert len(diag["rho_stream_at_step"]) == H
    assert len(diag["grad_norm_at_step"]) == H
    assert len(diag["bias_max_abs_at_step"]) == H
    assert len(diag["chosen_index_at_step"]) == H
    assert len(diag["would_pick_top_logit_at_step"]) == H
    # Indices in [0, K).
    K = int(V.shape[0])
    assert all(0 <= i < K for i in diag["chosen_index_at_step"])
    # Final rho is finite.
    assert np.isfinite(diag["final_rho"])
    # n_steps_changed_by_guidance is in [0, H].
    assert 0 <= diag["n_steps_changed_by_guidance"] <= H
    # Bias is non-negative (max-abs).
    assert all(b >= 0.0 for b in diag["bias_max_abs_at_step"])
    # fallback_used is bool.
    assert isinstance(diag["fallback_used"], bool)


# ---------------------------------------------------------------------------
# T8. Streaming partial-rho is what the gradient sees.
# ---------------------------------------------------------------------------


def test_streaming_partial_rho_used(gi_setup) -> None:
    """At step t, the gradient is computed on the partial+extrapolated
    control sequence (committed actions for indices < t, u_bar at t,
    default for > t). We verify this by checking that the bias at step t
    is sensitive to the *previously committed* action.

    Test: build two control prefixes (low and high committed-action
    history), compute the bias at step t under each, and verify they
    differ. If the gradient were computed on a fixed (e.g. all-default)
    control, the bias would be identical regardless of history.
    """
    sim, params, spec, V, x0 = gi_setup
    llm = _uniform_llm(int(V.shape[0]))
    guided = STLGradientGuidedSampler(
        llm,
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
        guidance_weight=1.0,
        default_action=jnp.asarray([2.5], dtype=jnp.float32),
    )
    # Step at which we probe.
    t = 4
    logits = jnp.zeros(int(V.shape[0]))

    # Two histories: low (all zeros) vs high (all max).
    low_history = [jnp.asarray([0.0], dtype=jnp.float32) for _ in range(t)]
    high_history = [jnp.asarray([5.0], dtype=jnp.float32) for _ in range(t)]
    # Pad histories to horizon length with the default; the sampler
    # internals slice up to step t for the gradient probe.
    pad_low = low_history + [
        jnp.asarray([2.5], dtype=jnp.float32) for _ in range(sim.n_control_points - t)
    ]
    pad_high = high_history + [
        jnp.asarray([2.5], dtype=jnp.float32) for _ in range(sim.n_control_points - t)
    ]

    bias_low, gnorm_low = guided._compute_bias(x0, pad_low, logits, t)
    bias_high, gnorm_high = guided._compute_bias(x0, pad_high, logits, t)

    # The two bias vectors must differ if the gradient really sees the
    # committed history. (They might both be zero only at saturation,
    # which would still indicate the partial-trajectory path is in
    # use — but we sanity-check by also requiring the streaming rho
    # itself differs.)
    diff = float(jnp.max(jnp.abs(bias_low - bias_high)))
    assert diff > 1e-3, (
        f"bias vectors at step t={t} must depend on prior committed actions. "
        f"max|bias_low - bias_high| = {diff:.3g}; this would mean the "
        f"partial-trajectory probe is reading a fixed (history-free) control."
    )


# ---------------------------------------------------------------------------
# T9. Protocol compliance.
# ---------------------------------------------------------------------------


def test_protocol_compliance(gi_setup) -> None:
    sim, params, spec, V, x0 = gi_setup
    llm = _uniform_llm(int(V.shape[0]))
    samplers = [
        StandardSampler(llm, sim, spec, V, params, horizon=sim.n_control_points),
        BestOfNSampler(llm, sim, spec, V, params, horizon=sim.n_control_points, n=2),
        ContinuousBoNSampler(llm, sim, spec, V, params, horizon=sim.n_control_points, n=2),
        STLGradientGuidedSampler(
            llm, sim, spec, V, params, horizon=sim.n_control_points, guidance_weight=1.0
        ),
    ]
    for s in samplers:
        assert isinstance(s, Sampler), f"{type(s).__name__} does not satisfy the Sampler Protocol"


# ---------------------------------------------------------------------------
# Construction / validation tests.
# ---------------------------------------------------------------------------


def test_gradient_guided_invalid_vocabulary_raises(gi_setup) -> None:
    sim, params, spec, V, x0 = gi_setup
    llm = _uniform_llm(int(V.shape[0]))
    with pytest.raises(ValueError, match="must be 2-d"):
        STLGradientGuidedSampler(
            llm,
            sim,
            spec,
            jnp.zeros((5,)),
            params,
            horizon=sim.n_control_points,
        )


def test_gradient_guided_invalid_horizon_raises(gi_setup) -> None:
    sim, params, spec, V, x0 = gi_setup
    llm = _uniform_llm(int(V.shape[0]))
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        STLGradientGuidedSampler(llm, sim, spec, V, params, horizon=0)


def test_gradient_guided_invalid_temperature_raises(gi_setup) -> None:
    sim, params, spec, V, x0 = gi_setup
    llm = _uniform_llm(int(V.shape[0]))
    with pytest.raises(ValueError, match="sampling_temperature"):
        STLGradientGuidedSampler(
            llm,
            sim,
            spec,
            V,
            params,
            horizon=sim.n_control_points,
            sampling_temperature=-0.5,
        )


def test_gradient_guided_default_action_shape_validated(gi_setup) -> None:
    sim, params, spec, V, x0 = gi_setup
    llm = _uniform_llm(int(V.shape[0]))
    with pytest.raises(ValueError, match="default_action shape"):
        STLGradientGuidedSampler(
            llm,
            sim,
            spec,
            V,
            params,
            horizon=sim.n_control_points,
            default_action=jnp.array([0.0, 0.0]),
        )


def test_llm_wrong_shape_raises(gi_setup) -> None:
    sim, params, spec, V, x0 = gi_setup
    K = int(V.shape[0])

    def bad_llm(state, history, key):
        return jnp.zeros(K + 1)  # wrong size

    sampler = StandardSampler(bad_llm, sim, spec, V, params, horizon=sim.n_control_points)
    with pytest.raises(ValueError, match="logits of shape"):
        sampler.sample(x0, jax.random.key(0))


# ---------------------------------------------------------------------------
# Cross-task validation: gradient guidance on bio_ode.repressilator.easy.
# ---------------------------------------------------------------------------
#
# Pre-registered protocol matches test_gradient_guided_improves_rho but
# substitutes the RepressilatorSimulator + bio_ode.repressilator.easy spec.
# The hypothesis: gradient guidance generalises across task families.
#
# Empirical finding (smoke run on 2026-04-24, 6 seeds, lambda in {0.1, 1, 5,
# 20, 50}): on the canonical default IC ``[0,0,0,15,5,25]`` — the pilot
# initial state — gradient guidance does NOT meaningfully improve mean rho.
# The G[120,200] (m1 >= 250 nM) clause requires SUSTAINED control of the
# upstream repressor (silence gene 3, action ``[0,0,1]``) across ALL 10
# control steps; the partial-then-extrapolated gradient probe at any
# single intermediate step does not point coherently in that direction.
# The rho landscape is dominated by a brutal cliff between "monotone
# silence-3" (rho ~ +25) and almost everything else (rho ~ -250 floor),
# and the per-step gradient cannot find that cliff from the gradient-
# probe extrapolation point. Sweep summary (default_action options ``center
# 0.5``, ``zeros``, ``silence-3``; lambdas ``0, 5, 50``):
#
#   da=center 0.5,   lambda=0:    mean rho = -250.0 (saturated floor)
#   da=center 0.5,   lambda=50:   mean rho = -250.0
#   da=zeros,        lambda=50:   mean rho = -250.0
#   da=silence-3,    lambda=50:   mean rho = -248.1   (1/6 seeds escapes)
#
# This is reported as a NEGATIVE empirical finding in
# paper/cross_task_validation.md. The test below is marked
# ``xfail(strict=False)`` with the documented reason. If a future change to
# the gradient-probe extrapolation strategy or a better default-action
# heuristic flips the test to PASS, that is a desirable surprise (strict=
# False does not fail on unexpected pass).


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Cross-task transfer FAILS on bio_ode.repressilator.easy with the "
        "canonical pilot IC. G[120,200] (m1>=250) requires sustained "
        "silence-gene-3 across all 10 control steps; the partial-then-"
        "extrapolated gradient probe at any single step does not point "
        "coherently toward this attractor. Documented in "
        "paper/cross_task_validation.md as a publishable negative result."
    ),
)
def test_gradient_guided_improves_rho_repressilator() -> None:
    """Cross-task validation: does gradient guidance also help on the
    repressilator?

    Same protocol as ``test_gradient_guided_improves_rho`` but with the
    :class:`RepressilatorSimulator` + ``bio_ode.repressilator.easy`` spec
    and the canonical pilot IC. Compares ``lambda=0`` vs ``lambda=2``
    over 6 seeds. Asserts ``paired_wins >= 4/6 OR mean_rho_guided > 0 +
    mean_rho_baseline``.

    Status: pre-registered to fail on the current configuration. Negative
    result documented; ``xfail(strict=False)`` so a future improvement
    that flips the test to PASS does not fail the suite.
    """
    sim = RepressilatorSimulator()
    params = RepressilatorParams()
    x0 = _repressilator_initial_state(params)
    spec = REGISTRY["bio_ode.repressilator.easy"]
    # k_per_dim=2 -> K=8 vocabulary on [0,1]^3 box. The 8 corners include
    # the silence-3 action ``[0,0,1]`` that is known to satisfy the spec.
    V = make_uniform_action_vocabulary(
        [0.0] * REPRESSILATOR_ACTION_DIM,
        [1.0] * REPRESSILATOR_ACTION_DIM,
        k_per_dim=2,
    )
    llm = _uniform_llm(int(V.shape[0]))

    baseline = STLGradientGuidedSampler(
        llm,
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
        guidance_weight=0.0,
        sampling_temperature=0.5,
    )
    guided = STLGradientGuidedSampler(
        llm,
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
        guidance_weight=2.0,
        sampling_temperature=0.5,
    )

    n_seeds = 6
    rhos_b: list[float] = []
    rhos_g: list[float] = []
    for s in range(n_seeds):
        k = jax.random.key(1000 + s)
        _, d_b = baseline.sample(x0, k)
        _, d_g = guided.sample(x0, k)
        rhos_b.append(float(d_b["final_rho"]))
        rhos_g.append(float(d_g["final_rho"]))

    paired_wins = sum(g > b for g, b in zip(rhos_g, rhos_b, strict=True))
    mean_b = float(np.mean(rhos_b))
    mean_g = float(np.mean(rhos_g))
    # The success criterion mirrors test_gradient_guided_improves_rho's
    # spirit: either the per-seed paired wins are a clear majority, OR
    # the mean improvement is positive. Both must currently fail on the
    # canonical IC; xfail catches that.
    assert paired_wins >= 4 or mean_g > mean_b, (
        f"cross-task transfer to repressilator failed: "
        f"baseline mean = {mean_b:.3f}, guided mean = {mean_g:.3f}, "
        f"paired wins = {paired_wins}/{n_seeds}, "
        f"per-seed baseline = {rhos_b}, per-seed guided = {rhos_g}"
    )


# ---------------------------------------------------------------------------
# Resolution (2026-04-25): beam-search warmstart resolves the repressilator
# failure that defeated the gradient-guided sampler above. The xfail above
# is kept in place because it is a true statement about the gradient-guided
# sampler on this configuration; the test below documents the positive
# result on the same configuration via discrete vocabulary enumeration.
# ---------------------------------------------------------------------------


def test_beam_search_solves_repressilator() -> None:
    """Beam-search warmstart reaches rho > 0 on the repressilator on at
    least 2/3 fixed seeds.

    Resolution test for the negative result documented in
    `paper/cross_task_validation.md`. The mechanism: discrete enumeration
    over the dense `k_per_dim=5` (K=125) action vocabulary, scored under
    the model-predictive `tail_strategy="repeat_candidate"` lookahead,
    finds the satisfying constant silence-3 policy `u = (0, 0, 1)` that
    the continuous gradient probe cannot reach. The pure-beam endpoint
    (no gradient refinement) is sufficient for this test, matching the
    hypothesis that the failure is structural-search vs continuous-search
    rather than a refinement-step issue.

    Pre-registered acceptance criterion: at least 2 of 3 fixed seeds
    must produce ``final_rho > 0``. The headline pilot in
    ``tests/test_beam_search_warmstart.py`` already checks the seed-0
    case at ρ ≈ +25; this test asserts robustness across seeds.
    """
    # Local import so the test_inference module's import block does not
    # have to grow with every new sampler. The sampler is exposed at the
    # ``stl_seed.inference`` package level via __init__.py.
    from stl_seed.inference import BeamSearchWarmstartSampler

    sim = RepressilatorSimulator()
    params = RepressilatorParams()
    x0 = _repressilator_initial_state(params)
    spec = REGISTRY["bio_ode.repressilator.easy"]
    # k_per_dim=5 -> K=125 vocabulary on [0,1]^3. Includes the silence-3
    # corner [0,0,1] that yields rho ~ +25 when held constant; the 8-corner
    # vocabulary used by the gradient sampler also contains that corner,
    # but the per-step continuous probe cannot find it. Beam search on the
    # denser lattice with the constant-extrapolation lookahead enumerates
    # straight to it.
    V = make_uniform_action_vocabulary(
        [0.0] * REPRESSILATOR_ACTION_DIM,
        [1.0] * REPRESSILATOR_ACTION_DIM,
        k_per_dim=5,
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
        gradient_refine_iters=0,  # pure-beam endpoint suffices for the assertion
        tail_strategy="repeat_candidate",
    )

    n_seeds = 3
    rhos: list[float] = []
    for s in range(n_seeds):
        # Seed offset 1000 mirrors the convention used by
        # test_gradient_guided_improves_rho (and the unified-comparison harness),
        # so cross-test reproducibility is preserved when desired.
        k = jax.random.key(1000 + s)
        _, diag = sampler.sample(x0, k)
        rhos.append(float(diag["final_rho"]))

    n_satisfying = sum(r > 0.0 for r in rhos)
    assert n_satisfying >= 2, (
        f"beam-search warmstart did not solve the repressilator: "
        f"per-seed rhos = {rhos}, satisfying seeds = {n_satisfying}/{n_seeds}; "
        f"expected at least 2 of 3 with rho > 0. This is the resolution "
        f"test for paper/cross_task_validation.md; a regression here means "
        f"the beam-search fix has been broken."
    )


# ---------------------------------------------------------------------------
# Hybrid sampler — protocol & beats-pure-guidance.
# ---------------------------------------------------------------------------


def test_hybrid_protocol_compliance(gi_setup) -> None:
    """HybridGradientBoNSampler satisfies the Sampler Protocol."""
    sim, params, spec, V, x0 = gi_setup
    llm = _uniform_llm(int(V.shape[0]))
    s = HybridGradientBoNSampler(
        llm,
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
        n=2,
        guidance_weight=1.0,
    )
    assert isinstance(s, Sampler)
    traj, diag = s.sample(x0, jax.random.key(11))
    assert isinstance(traj, Trajectory)
    assert diag["sampler"] == "hybrid_gradient_bon"
    assert diag["n_samples"] == 2
    assert len(diag["all_rho"]) == 2
    # chosen_index points at the argmax draw
    assert diag["chosen_rho"] == max(diag["all_rho"])
    # n_steps_changed_by_guidance_per_draw is a list of n ints
    assert len(diag["n_steps_changed_by_guidance_per_draw"]) == 2
    # final_rho == chosen_rho by construction
    assert diag["final_rho"] == diag["chosen_rho"]


def test_hybrid_invalid_n_raises(gi_setup) -> None:
    sim, params, spec, V, x0 = gi_setup
    llm = _uniform_llm(int(V.shape[0]))
    with pytest.raises(ValueError, match="n must be >= 1"):
        HybridGradientBoNSampler(
            llm,
            sim,
            spec,
            V,
            params,
            horizon=sim.n_control_points,
            n=0,
        )


def test_hybrid_beats_pure_guidance() -> None:
    """Pre-registered: HybridGradientBoNSampler(n=4) >= STLGradientGuided
    in mean rho on a non-saturating spec, paired across 6 seeds.

    Setup
    -----
    Spec: ``glucose_insulin.dawn.hard``. Chosen because the easy and
    medium glucose specs saturate at rho ~ 20 within one rollout, leaving
    no headroom for hybrid argmax-rho selection to dominate. The dawn
    spec is harder (rho hovers in the [-33, -19] band), so hybrid's
    argmax-over-N can pick up improvements over a single guided draw.
    Lambda = 2.0 matches the existing ``test_gradient_guided_improves_rho``
    convention; sampling temperature = 0.5 keeps the inner sampler
    stochastic so the 4 hybrid draws actually diverge.

    Falsification criterion
    -----------------------
    ``hybrid_mean < guided_mean - tol`` with tol = 2.0 rho units. A
    hybrid sampler that is strictly worse than its inner sampler at
    fixed seeds would indicate either a bug (e.g. argmax over the wrong
    axis) or a pathology in the inner sampler's RNG threading. Neither
    is expected.

    Smoke result (2026-04-24, this commit):
        guided   mean = -30.616  per-seed = [-33.0, -33.0, -33.0, -18.69, -33.0, -33.0]
        hybrid   mean = -28.307  per-seed = [-33.0, -33.0, -33.0, -19.15, -18.69, -33.0]
        => hybrid strictly higher in mean (+2.3 rho units), 5/6 hybrid
           >= guided per seed, 1/6 strict win.

    Notes
    -----
    The hybrid uses ``fold_in(key, draw_idx)`` to derive sub-keys, so
    Hybrid(n=1, key) = Guided(fold_in(key, 0)) != Guided(key). The
    paired-mean test uses the same master ``key`` per seed for both
    samplers; the hybrid then explores 4 distinct sub-keys including
    one that may not coincide with the guided sampler's draw. This is
    an apples-to-apples *protocol* comparison, not an apples-to-apples
    sample comparison.

    Compute cost (per seed):
        guided ~ H * (1 fwd + 1 bwd)
        hybrid ~ 4 * H * (1 fwd + 1 bwd)
    The matched-compute baseline would be ``ContinuousBoNSampler(n=8)``
    (assuming bwd ~ fwd) — that's the comparison in
    paper/cross_task_validation.md, not here.
    """
    sim = GlucoseInsulinSimulator()
    params = BergmanParams()
    spec = REGISTRY["glucose_insulin.dawn.hard"]
    V = make_uniform_action_vocabulary(0.0, 5.0, k_per_dim=5)
    x0 = default_normal_subject_initial_state(params)
    llm = _uniform_llm(int(V.shape[0]))

    guided = STLGradientGuidedSampler(
        llm,
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
        guidance_weight=2.0,
        sampling_temperature=0.5,
    )
    hybrid = HybridGradientBoNSampler(
        llm,
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
        n=4,
        guidance_weight=2.0,
        sampling_temperature=0.5,
    )

    n_seeds = 6
    rhos_g: list[float] = []
    rhos_h: list[float] = []
    for s in range(n_seeds):
        k = jax.random.key(2000 + s)
        _, dg = guided.sample(x0, k)
        _, dh = hybrid.sample(x0, k)
        rhos_g.append(float(dg["final_rho"]))
        rhos_h.append(float(dh["final_rho"]))

    mean_g = float(np.mean(rhos_g))
    mean_h = float(np.mean(rhos_h))
    tol = 2.0  # rho units; see docstring rationale.
    assert mean_h >= mean_g - tol, (
        f"hybrid mean rho should be >= guided mean rho (within tol={tol}). "
        f"guided mean = {mean_g:.3f}, hybrid mean = {mean_h:.3f}, "
        f"per-seed guided = {rhos_g}, per-seed hybrid = {rhos_h}"
    )

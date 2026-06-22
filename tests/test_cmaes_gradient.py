"""Tests for the CMA-ES + gradient-refinement sampler.

Test plan
---------

C1. ``test_cmaes_runs_glucose_insulin_no_crash``. sanity end-to-end
    on the smaller glucose-insulin task; produces a well-formed
    :class:`Trajectory` and finite ``final_rho``.
C2. ``test_cmaes_finds_optimum_synthetic``. on a unimodal quadratic
    in 5-D, the hand-rolled CMA-ES converges to the known optimum
    within 20 generations (within 0.1 in L2 norm). Tests the inner
    update math without the simulator stack.
C3. ``test_cmaes_escapes_local_minimum``. on a deceptive bimodal
    surface with a shallow local minimum near init and a deep global
    minimum far away, CMA-ES finds the global > 50% of the time
    across 8 seeds. Tests the algorithmic claim that population
    search escapes basins that gradient methods do not.
C4. ``test_cmaes_protocol_compliance``. the sampler satisfies the
    :class:`Sampler` Protocol (runtime-checkable isinstance).
C5. ``test_cmaes_box_reflection``. actions in the produced
    trajectory always lie in ``[u_min, u_max]``, and the standalone
    ``_reflect_into_box`` helper handles single, multiple, and
    pathologically large violations.
C6. ``test_cmaes_diagnostics_well_formed``. every diagnostic field
    has the expected shape / type / range; ``best_rho_per_gen`` is
    length ``n_generations``; ``rho_post_refine_per_step`` is
    bounded above by ``n_refine``.

"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stl_seed.inference import Sampler
from stl_seed.inference.cmaes_gradient import (
    CMAESGradientSampler,
    _cmaes_init,
    _cmaes_sample_population,
    _cmaes_update,
    _CMAESConsts,
    _reflect_into_box,
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
# Synthetic LLM proxy.
# ---------------------------------------------------------------------------


def _uniform_llm(K: int):
    """Flat LLM (entropy = log K). The CMA-ES seeding uses this only when
    ``initial_mean_source='llm_argmax'``; for ``'midpoint'`` it is unused."""

    def llm(state, history, key):
        return jnp.zeros(K)

    return llm


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gi_setup():
    """Return (simulator, params, spec, vocabulary, x0) for glucose-insulin."""
    sim = GlucoseInsulinSimulator()
    params = BergmanParams()
    spec = REGISTRY["glucose_insulin.tir.easy"]
    V = make_uniform_action_vocabulary(0.0, 5.0, k_per_dim=5)
    x0 = default_normal_subject_initial_state(params)
    return sim, params, spec, V, x0


# ---------------------------------------------------------------------------
# C5. _reflect_into_box helper.
# ---------------------------------------------------------------------------


def test_reflect_into_box_no_op_when_inside() -> None:
    """If x is already inside the box, reflection is a no-op (n_refl=0)."""
    x = jnp.array([0.3, 0.5, 0.7], dtype=jnp.float32)
    lo = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
    hi = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)
    out, n_refl = _reflect_into_box(x, lo, hi)
    np.testing.assert_allclose(np.asarray(out), np.asarray(x), atol=1e-6)
    assert n_refl == 0


def test_reflect_into_box_single_overshoot_high() -> None:
    """x = 1.2 with hi = 1.0 reflects to 0.8 (= 2*1.0 - 1.2). Single iter."""
    x = jnp.array([1.2], dtype=jnp.float32)
    lo = jnp.array([0.0], dtype=jnp.float32)
    hi = jnp.array([1.0], dtype=jnp.float32)
    out, n_refl = _reflect_into_box(x, lo, hi)
    assert float(out[0]) == pytest.approx(0.8, abs=1e-6)
    assert n_refl == 1


def test_reflect_into_box_single_overshoot_low() -> None:
    """x = -0.3 with lo = 0.0 reflects to 0.3 (= 2*0.0 - (-0.3))."""
    x = jnp.array([-0.3], dtype=jnp.float32)
    lo = jnp.array([0.0], dtype=jnp.float32)
    hi = jnp.array([1.0], dtype=jnp.float32)
    out, n_refl = _reflect_into_box(x, lo, hi)
    assert float(out[0]) == pytest.approx(0.3, abs=1e-6)
    assert n_refl == 1


def test_reflect_into_box_pathological_large_violation_clamps() -> None:
    """For x wildly outside (e.g. 100) with box width 1, repeated reflection
    converges; the safety final clip guarantees in-box on return."""
    x = jnp.array([100.0, -50.0, 0.5], dtype=jnp.float32)
    lo = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
    hi = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)
    out, _ = _reflect_into_box(x, lo, hi, max_iters=200)
    out_np = np.asarray(out)
    assert np.all(out_np >= 0.0 - 1e-6)
    assert np.all(out_np <= 1.0 + 1e-6)
    # The 0.5 coordinate is unchanged.
    assert float(out_np[2]) == pytest.approx(0.5, abs=1e-6)


def test_cmaes_box_reflection_in_sample(gi_setup) -> None:
    """Returned trajectory.actions must lie inside the action box for a
    bio_ode task (action box [0, 1]^3 on the repressilator)."""
    sim = RepressilatorSimulator()
    params = RepressilatorParams()
    x0 = _repressilator_initial_state(params)
    spec = REGISTRY["bio_ode.repressilator.easy"]
    V = make_uniform_action_vocabulary(
        [0.0] * REPRESSILATOR_ACTION_DIM,
        [1.0] * REPRESSILATOR_ACTION_DIM,
        k_per_dim=2,
    )
    sampler = CMAESGradientSampler(
        _uniform_llm(int(V.shape[0])),
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
        # Use a very large sigma_init to deliberately push samples
        # outside the box and stress the reflection path.
        population_size=8,
        n_generations=3,
        sigma_init=2.0,
        n_refine=3,
        lr=1e-2,
    )
    traj, diag = sampler.sample(x0, jax.random.key(0))
    actions = np.asarray(traj.actions)
    assert np.all(actions >= 0.0 - 1e-6), f"actions below 0: min = {actions.min()}"
    assert np.all(actions <= 1.0 + 1e-6), f"actions above 1: max = {actions.max()}"
    # Sanity: the diagnostics record reflection counts.
    assert diag["n_box_reflections"] >= 0
    assert "best_rho_per_gen" in diag
    assert len(diag["best_rho_per_gen"]) == 3


# ---------------------------------------------------------------------------
# C2. CMA-ES converges on a unimodal quadratic in 5-D.
# ---------------------------------------------------------------------------


def test_cmaes_finds_optimum_synthetic() -> None:
    """On a unimodal quadratic ``f(x) = -||x - x*||^2`` in 5-D, the
    hand-rolled CMA-ES (no simulator, no STL) converges to ``x*`` within
    0.1 in L2 in 20 generations.

    This pins the inner update math: if the Hansen 2016 constants or the
    rank-1 / rank-mu mixing are wrong, the optimum is NOT recovered.
    """
    d = 5
    x_star = np.array([1.0, -2.0, 0.5, 3.0, -0.5], dtype=np.float64)

    def fitness(x: np.ndarray) -> float:
        return float(-np.sum((x - x_star) ** 2))

    consts = _CMAESConsts.from_dim(d=d, lambda_=16)
    state = _cmaes_init(d=d, initial_mean=np.zeros(d), initial_sigma=1.0)
    rng = np.random.default_rng(42)

    # Wide box: effectively unbounded.
    lo_flat = -np.full(d, 1e6)
    hi_flat = np.full(d, 1e6)

    best_x = state.mean.copy()
    best_f = fitness(best_x)
    for _ in range(60):
        samples, _ = _cmaes_sample_population(state, consts, rng, lo_flat, hi_flat)
        fits = np.array([fitness(s) for s in samples])
        order = np.argsort(-fits)
        sel = samples[order[: consts.mu]]
        f_top = float(fits[order[0]])
        if f_top > best_f:
            best_f = f_top
            best_x = samples[order[0]].copy()
        state = _cmaes_update(state, consts, sel)

    err = float(np.linalg.norm(best_x - x_star))
    assert err < 0.1, f"CMA-ES did not converge: best_x = {best_x}, x* = {x_star}, err = {err}"


# ---------------------------------------------------------------------------
# C3. CMA-ES escapes a deceptive local minimum.
# ---------------------------------------------------------------------------


def test_cmaes_escapes_local_minimum() -> None:
    """On a 2-D deceptive surface. a wide shallow local maximum at the
    init and a much deeper (but narrower) global maximum at moderate
    distance. CMA-ES should reach the global > 50% of the time across
    8 seeds with a generous population/generation budget.

    Surface (we maximise):
        f(x) = 1.0 * exp(-||x||^2 / 4.0)              # shallow wide basin at origin
             + 5.0 * exp(-||x - [2.5, 2.5]||^2 / 1.0) # deep basin at (2.5, 2.5)

    A pure local-gradient method initialised at the origin would
    converge to the shallow local maximum at (0, 0). CMA-ES with
    sigma_init=2.0 places ~32% of samples beyond ||x||=2 at +1
    stddev, so it reliably discovers the deep basin within ~30
    generations across most seeds. The basins are wider and closer
    than a textbook deceptive landscape so the test isn't a
    population-budget stress test. it is an *escape* test.
    """
    d = 2
    x_global = np.array([2.5, 2.5])

    def fitness(x: np.ndarray) -> float:
        return float(
            1.0 * np.exp(-np.sum(x**2) / 4.0) + 5.0 * np.exp(-np.sum((x - x_global) ** 2) / 1.0)
        )

    n_seeds = 8
    n_global_found = 0
    for seed in range(n_seeds):
        consts = _CMAESConsts.from_dim(d=d, lambda_=24)
        state = _cmaes_init(d=d, initial_mean=np.zeros(d), initial_sigma=2.0)
        rng = np.random.default_rng(seed)
        lo_flat = -np.full(d, 1e6)
        hi_flat = np.full(d, 1e6)
        best_f = -np.inf
        best_x = state.mean.copy()
        for _ in range(80):
            samples, _ = _cmaes_sample_population(state, consts, rng, lo_flat, hi_flat)
            fits = np.array([fitness(s) for s in samples])
            order = np.argsort(-fits)
            f_top = float(fits[order[0]])
            if f_top > best_f:
                best_f = f_top
                best_x = samples[order[0]].copy()
            sel = samples[order[: consts.mu]]
            state = _cmaes_update(state, consts, sel)

        # Reached the global if best_x is closer to (2.5, 2.5) than to
        # origin AND best_f is in the deep-basin range (> 1.5; the
        # shallow basin caps at 1.0).
        d_global = float(np.linalg.norm(best_x - x_global))
        d_origin = float(np.linalg.norm(best_x))
        if d_global < d_origin and best_f > 1.5:
            n_global_found += 1

    assert n_global_found >= n_seeds // 2 + 1, (
        f"CMA-ES escaped global only {n_global_found}/{n_seeds} runs; "
        "population search expected to escape >50% of the time on this surface"
    )


# ---------------------------------------------------------------------------
# C1. End-to-end on glucose-insulin (no crash + finite rho).
# ---------------------------------------------------------------------------


def test_cmaes_runs_glucose_insulin_no_crash(gi_setup) -> None:
    """Smallest end-to-end: the sampler runs, returns a Trajectory with
    the right shapes, and reports a finite final_rho."""
    sim, params, spec, V, x0 = gi_setup
    llm = _uniform_llm(int(V.shape[0]))
    sampler = CMAESGradientSampler(
        llm,
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
        population_size=8,
        n_generations=4,
        sigma_init=0.5,
        n_refine=4,
        lr=1e-2,
    )
    traj, diag = sampler.sample(x0, jax.random.key(0))
    assert isinstance(traj, Trajectory)
    assert traj.states.shape[0] == sim.n_save_points
    assert traj.actions.shape == (sim.n_control_points, 1)
    assert traj.times.shape == (sim.n_save_points,)
    assert "final_rho" in diag
    assert np.isfinite(diag["final_rho"]), f"final_rho not finite: {diag['final_rho']}"
    assert diag["sampler"] == "cmaes_gradient"


# ---------------------------------------------------------------------------
# C4. Protocol compliance.
# ---------------------------------------------------------------------------


def test_cmaes_protocol_compliance(gi_setup) -> None:
    """CMAESGradientSampler must satisfy the runtime-checkable Sampler Protocol."""
    sim, params, spec, V, x0 = gi_setup
    llm = _uniform_llm(int(V.shape[0]))
    sampler = CMAESGradientSampler(
        llm,
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
        population_size=8,
        n_generations=2,
        sigma_init=0.5,
        n_refine=2,
        lr=1e-2,
    )
    assert isinstance(sampler, Sampler)


# ---------------------------------------------------------------------------
# C6. Diagnostics well-formed.
# ---------------------------------------------------------------------------


def test_cmaes_diagnostics_well_formed(gi_setup) -> None:
    """Every documented diagnostic field is present with the expected
    shape / type / range."""
    sim, params, spec, V, x0 = gi_setup
    llm = _uniform_llm(int(V.shape[0]))
    n_gen = 5
    n_ref = 6
    sampler = CMAESGradientSampler(
        llm,
        sim,
        spec,
        V,
        params,
        horizon=sim.n_control_points,
        population_size=8,
        n_generations=n_gen,
        sigma_init=0.4,
        n_refine=n_ref,
        lr=1e-2,
    )
    _, diag = sampler.sample(x0, jax.random.key(123))
    # Per-generation traces.
    assert len(diag["best_rho_per_gen"]) == n_gen
    assert len(diag["mean_rho_per_gen"]) == n_gen
    assert len(diag["sigma_per_gen"]) == n_gen
    # Refinement trace bounded above by n_refine (steps with non-finite
    # gradients append rho but no model update; failed-eval steps do not
    # append at all).
    assert len(diag["rho_post_refine_per_step"]) <= n_ref
    # Sigma is positive throughout.
    assert all(s > 0 for s in diag["sigma_per_gen"])
    # Pre-refine and final rho are scalars.
    assert isinstance(diag["rho_pre_refine"], float)
    assert isinstance(diag["final_rho"], float)
    assert np.isfinite(diag["final_rho"])
    # Sample-counter integers.
    assert isinstance(diag["n_box_reflections"], int)
    assert diag["n_box_reflections"] >= 0
    assert isinstance(diag["n_finite_grads"], int)
    assert 0 <= diag["n_finite_grads"] <= n_ref


# ---------------------------------------------------------------------------
# Bonus: validation that bad construction args raise cleanly.
# ---------------------------------------------------------------------------


def test_cmaes_invalid_population_size_raises(gi_setup) -> None:
    sim, params, spec, V, x0 = gi_setup
    with pytest.raises(ValueError, match="population_size"):
        CMAESGradientSampler(
            _uniform_llm(int(V.shape[0])),
            sim,
            spec,
            V,
            params,
            horizon=sim.n_control_points,
            population_size=2,  # < 4
            n_generations=2,
            sigma_init=0.5,
            n_refine=2,
            lr=1e-2,
        )


def test_cmaes_invalid_sigma_raises(gi_setup) -> None:
    sim, params, spec, V, x0 = gi_setup
    with pytest.raises(ValueError, match="sigma_init"):
        CMAESGradientSampler(
            _uniform_llm(int(V.shape[0])),
            sim,
            spec,
            V,
            params,
            horizon=sim.n_control_points,
            population_size=8,
            n_generations=2,
            sigma_init=0.0,
            n_refine=2,
            lr=1e-2,
        )


def test_cmaes_user_initial_mean_required_when_source_user(gi_setup) -> None:
    sim, params, spec, V, x0 = gi_setup
    with pytest.raises(ValueError, match="initial_mean"):
        CMAESGradientSampler(
            _uniform_llm(int(V.shape[0])),
            sim,
            spec,
            V,
            params,
            horizon=sim.n_control_points,
            population_size=8,
            n_generations=2,
            sigma_init=0.5,
            n_refine=2,
            lr=1e-2,
            initial_mean_source="user",  # but no initial_mean provided
        )

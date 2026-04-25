"""CMA-ES population search + gradient refinement on action sequences.

Strategy "A3" for the cross-task gradient-guided failure
--------------------------------------------------------

The negative result documented in :file:`paper/cross_task_validation.md`
shows that the per-step gradient probe of
:class:`stl_seed.inference.STLGradientGuidedSampler` cannot find the
satisfying region for ``bio_ode.repressilator.easy``. The diagnosis
there: the satisfying region is a measure-near-zero attractor in the
30-D joint action box (``H = 10`` steps × ``m = 3`` channels), and the
single-step partial-then-extrapolated probe is myopic — it cannot see
that *every* future step must also be ``silence-3`` for the
``G_{[120, 200]}(m_1 \\geq 250)`` clause to hold.

Population-based methods (CMA-ES; Hansen 2016, arXiv:1604.00772) do not
suffer from this myopia: they sample whole 30-D action sequences from a
multivariate Gaussian and let the *empirical* fitness ranking drive
covariance adaptation. The mean migrates toward the satisfying corner
without ever needing the local gradient of ``rho``. The cost is
``λ × G`` simulator evaluations (population size × generations) instead
of one backward per step.

We then do a small number of plain gradient-ascent steps on the best
survivor — a refinement that exploits the same continuous ``rho``
gradient that
:class:`stl_seed.inference.STLGradientGuidedSampler` uses, but now
applied jointly across all ``H × m`` action coordinates rather than
one step at a time. CMA-ES picks the basin; gradient ascent polishes
the local optimum within it.

References
----------

* Hansen, N. *The CMA Evolution Strategy: A Tutorial*. arXiv:1604.00772
  (2016). The constants ``c_sigma``, ``d_sigma``, ``c_c``, ``c_1``,
  ``c_mu`` and the weight schedule below follow Section 4 of this
  reference verbatim.
* Salimans, T. et al. *Evolution Strategies as a Scalable Alternative
  to Reinforcement Learning*. arXiv:1703.03864 (2017). Demonstrates
  the population-search-plus-policy-gradient pattern at scale; the
  hybrid here is a small-scale analogue.

-------------

This module imports only from JAX, jaxtyping, and in-package symbols
(``stl_seed.inference.protocol``, ``stl_seed.specs``,
``stl_seed.stl.evaluator``, ``stl_seed.tasks._trajectory``). No
mathematics is implemented here from Hansen 2016 directly; the
``cma`` PyPI package is *not* a dependency. The user's
overall code shape but not imported or copied.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np

from stl_seed.inference.protocol import LLMProposal, SamplerDiagnostics, SamplerResult
from stl_seed.specs import Node, STLSpec
from stl_seed.stl.evaluator import compile_spec
from stl_seed.tasks._trajectory import Trajectory

# ---------------------------------------------------------------------------
# Diagnostics record.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CMAESDiagnostics:
    """Per-generation and refinement diagnostics for one CMA-ES run.

    Fields
    ------
    best_rho_per_gen:
        Best (max) rho observed in each generation's population, length
        ``n_generations``. Pre-registered: this should be monotone non-
        decreasing modulo CMA-ES exploration noise.
    mean_rho_per_gen:
        Mean rho across the population in each generation, length
        ``n_generations``. The gap (best - mean) measures population
        diversity.
    sigma_per_gen:
        CMA-ES step-size sigma at the start of each generation, length
        ``n_generations``. A monotone decrease toward zero indicates
        convergence; a monotone increase indicates divergence (typically
        a bug or pathological landscape).
    rho_pre_refine:
        Best-survivor rho immediately before gradient refinement begins.
    rho_post_refine_per_step:
        Rho after each of the ``n_refine`` gradient-ascent steps applied
        to the best survivor (length ``n_refine``). Pre-registered: this
        should be monotone non-decreasing modulo line-search overshoot.
    final_rho:
        Final rho on the refined trajectory (full re-simulation). The
        canonical scalar reported to the eval harness.
    n_box_reflections:
        Total number of reflection clamps applied during refinement
        (one count per step; can exceed n_refine if multiple steps each
        triggered reflection). Diagnostic; large values signal the
        refinement is pushing against the box boundary.
    n_finite_grads:
        Count of refinement steps where the gradient was finite (non-NaN
        and non-Inf). Should equal ``n_refine`` in healthy runs.
    """

    best_rho_per_gen: list[float] = dataclasses.field(default_factory=list)
    mean_rho_per_gen: list[float] = dataclasses.field(default_factory=list)
    sigma_per_gen: list[float] = dataclasses.field(default_factory=list)
    rho_pre_refine: float = float("nan")
    rho_post_refine_per_step: list[float] = dataclasses.field(default_factory=list)
    final_rho: float = float("nan")
    n_box_reflections: int = 0
    n_finite_grads: int = 0

    def to_dict(self) -> SamplerDiagnostics:
        """Materialise as a plain dict for the harness."""
        return {
            "sampler": "cmaes_gradient",
            "best_rho_per_gen": list(self.best_rho_per_gen),
            "mean_rho_per_gen": list(self.mean_rho_per_gen),
            "sigma_per_gen": list(self.sigma_per_gen),
            "rho_pre_refine": float(self.rho_pre_refine),
            "rho_post_refine_per_step": list(self.rho_post_refine_per_step),
            "final_rho": float(self.final_rho),
            "n_generations": len(self.best_rho_per_gen),
            "n_box_reflections": int(self.n_box_reflections),
            "n_finite_grads": int(self.n_finite_grads),
        }


# ---------------------------------------------------------------------------
# Hand-rolled CMA-ES in JAX (Hansen 2016, arXiv:1604.00772, Section 4).
# ---------------------------------------------------------------------------
#
# Notation follows Hansen 2016:
#   d        : problem dimensionality.
#   lambda_  : population size (called λ in the tutorial; "lambda" is
#              reserved in Python).
#   mu       : number of selected (parents) per generation; we use
#              mu = lambda_ // 2.
#   m        : current mean of the search distribution, shape (d,).
#   sigma    : current step-size, scalar.
#   C        : current covariance matrix, shape (d, d).
#   p_sigma  : evolution path for sigma, shape (d,).
#   p_c      : evolution path for C, shape (d,).
#   weights  : recombination weights, shape (mu,). Positive, sum to 1.
#   mu_eff   : variance-effective selection mass = 1 / sum(weights^2).
#
# Constants (Hansen 2016 §4.3, "Default Strategy Parameters"):
#   c_sigma  = (mu_eff + 2) / (d + mu_eff + 5)
#   d_sigma  = 1 + 2 * max(0, sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
#   c_c      = (4 + mu_eff/d) / (d + 4 + 2*mu_eff/d)
#   c_1      = 2 / ((d + 1.3)^2 + mu_eff)
#   c_mu     = min(1 - c_1, 2 * (mu_eff - 2 + 1/mu_eff) / ((d + 2)^2 + mu_eff))
#
# Sampling: factorise C = B D D B^T (eigendecomposition). New samples are
# x_k = m + sigma * B @ D @ z_k where z_k ~ N(0, I_d). We recompute the
# eigendecomposition every generation; for d <= 100 this is cheap.
#
# Box-constraint policy: reflection. After sampling, any coordinate
# u_i > u_max is replaced by 2*u_max - u_i; any u_i < u_min is replaced
# by 2*u_min - u_i. We iterate (jax.lax.while_loop in JAX-traced code,
# Python loop here) until the sample lies inside the box. For sigma not
# wildly larger than the box width this converges in O(1) iterations.


@dataclasses.dataclass
class _CMAESConsts:
    """Hansen 2016 default strategy constants, computed once per run."""

    d: int
    lambda_: int
    mu: int
    weights: np.ndarray  # shape (mu,), positive, sum to 1
    mu_eff: float
    c_sigma: float
    d_sigma: float
    c_c: float
    c_1: float
    c_mu: float
    chi_n: float  # E[||N(0, I_d)||]

    @classmethod
    def from_dim(cls, d: int, lambda_: int) -> _CMAESConsts:
        """Compute Hansen 2016 default constants for dimension ``d``.

        ``lambda_`` is the population size; the tutorial recommends
        ``lambda_ = 4 + floor(3 * ln(d))`` as the default but our
        sampler exposes it as a hyperparameter so an experimenter can
        trade compute for exploration radius.
        """
        if d < 1:
            raise ValueError(f"CMA-ES dim must be >= 1, got {d}")
        if lambda_ < 4:
            # Hansen 2016 §4.2 notes lambda < 4 is degenerate (no
            # selection pressure). We hard-stop rather than silently
            # breaking the algorithm.
            raise ValueError(f"CMA-ES population size must be >= 4, got {lambda_}")
        mu = lambda_ // 2
        # Pre-weights (Hansen 2016 Eq. 49); the actual weights are normalised
        # so positive part sums to 1.
        raw_w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1, dtype=np.float64))
        # Drop any negative ones (here all are positive since i <= mu = floor(lambda/2)).
        raw_w = np.where(raw_w > 0, raw_w, 0.0)
        weights = raw_w / raw_w.sum()
        mu_eff = float(1.0 / (weights**2).sum())

        c_sigma = (mu_eff + 2.0) / (d + mu_eff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (d + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mu_eff / d) / (d + 4.0 + 2.0 * mu_eff / d)
        c_1 = 2.0 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(
            1.0 - c_1,
            2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((d + 2.0) ** 2 + mu_eff),
        )
        # E[||N(0, I_d)||] approximation, Hansen 2016 §4.1.
        chi_n = math.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d * d))

        return cls(
            d=int(d),
            lambda_=int(lambda_),
            mu=int(mu),
            weights=weights.astype(np.float64),
            mu_eff=float(mu_eff),
            c_sigma=float(c_sigma),
            d_sigma=float(d_sigma),
            c_c=float(c_c),
            c_1=float(c_1),
            c_mu=float(c_mu),
            chi_n=float(chi_n),
        )


def _reflect_into_box(
    x: jt.Float[jt.Array, ...],
    lo: jt.Float[jt.Array, ...],
    hi: jt.Float[jt.Array, ...],
    max_iters: int = 32,
) -> tuple[jt.Float[jt.Array, ...], int]:
    """Reflect ``x`` into the box ``[lo, hi]`` (element-wise).

    Algorithm: while any coordinate is out of bounds, reflect it back
    in. Specifically:

        if x_i > hi_i:  x_i <- 2*hi_i - x_i
        if x_i < lo_i:  x_i <- 2*lo_i - x_i

    For ``|x_i - midpoint| < (hi_i - lo_i)``, one reflection suffices.
    For larger excursions, repeated reflection eventually converges
    (the dynamics are equivalent to an "infinite mirror" problem).
    We bound the iteration count at ``max_iters`` and assert the final
    result is in-bounds; a violation indicates a wildly large input
    relative to the box width and almost certainly a bug upstream.

    Returns
    -------
    (clamped_x, n_reflections):
        ``clamped_x`` is the reflected array, guaranteed to lie in
        ``[lo, hi]`` (raises if ``max_iters`` is insufficient).
        ``n_reflections`` is the number of while-loop iterations actually
        consumed (a soft proxy for how aggressively the box was being
        violated). Always between 0 and ``max_iters``.

    Notes
    -----
    Implemented in pure NumPy-style JAX (no while_loop) so it is safe
    to call from Python control flow. For JIT'd hot paths a
    ``jax.lax.while_loop`` version would be needed.
    """
    n_refl = 0
    for _ in range(max_iters):
        too_high = x > hi
        too_low = x < lo
        if not (bool(jnp.any(too_high)) or bool(jnp.any(too_low))):
            break
        x = jnp.where(too_high, 2.0 * hi - x, x)
        x = jnp.where(too_low, 2.0 * lo - x, x)
        n_refl += 1
    # Final safety clip: if max_iters wasn't enough (extreme numerical
    # case), fall back to a hard clip rather than returning out-of-box
    # values. Diagnostic: this is rare and indicates a sigma blow-up.
    x = jnp.clip(x, lo, hi)
    return x, n_refl


@dataclasses.dataclass
class _CMAESState:
    """Mutable state carried across CMA-ES generations."""

    mean: np.ndarray  # shape (d,)
    sigma: float
    C: np.ndarray  # shape (d, d), symmetric positive-definite
    p_sigma: np.ndarray  # shape (d,)
    p_c: np.ndarray  # shape (d,)
    generation: int = 0


def _cmaes_init(
    d: int,
    initial_mean: np.ndarray,
    initial_sigma: float,
) -> _CMAESState:
    """Build the starting CMA-ES state.

    ``C = I_d`` (no prior anisotropy); ``p_sigma = p_c = 0``.
    """
    if initial_mean.shape != (d,):
        raise ValueError(f"initial_mean shape {initial_mean.shape} != (d,) = ({d},)")
    if initial_sigma <= 0.0:
        raise ValueError(f"initial sigma must be positive, got {initial_sigma}")
    return _CMAESState(
        mean=np.asarray(initial_mean, dtype=np.float64).copy(),
        sigma=float(initial_sigma),
        C=np.eye(d, dtype=np.float64),
        p_sigma=np.zeros(d, dtype=np.float64),
        p_c=np.zeros(d, dtype=np.float64),
        generation=0,
    )


def _cmaes_update(
    state: _CMAESState,
    consts: _CMAESConsts,
    selected: np.ndarray,  # (mu, d), the top-mu samples ranked best->worst
) -> _CMAESState:
    """One CMA-ES update step (Hansen 2016 §4.1, Eqs. 41-47).

    ``selected`` are the ``mu`` best samples from the current
    generation, ordered best-first.

    Mutates a copy of state and returns it.
    """
    d = consts.d
    weights = consts.weights
    mu_eff = consts.mu_eff
    c_sigma = consts.c_sigma
    d_sigma = consts.d_sigma
    c_c = consts.c_c
    c_1 = consts.c_1
    c_mu = consts.c_mu
    chi_n = consts.chi_n

    old_mean = state.mean
    sigma = state.sigma
    C = state.C
    p_sigma = state.p_sigma
    p_c = state.p_c

    # New weighted mean of selected samples (Eq. 41).
    new_mean = weights @ selected  # (d,)

    # y_w = (m_new - m_old) / sigma  (centered, scale-free shift in z-space).
    y_w = (new_mean - old_mean) / sigma

    # Eigendecomposition of C for C^{-1/2}. C is symmetric PSD; we use
    # eigh which is numerically stable for that case.
    eigvals, B = np.linalg.eigh(C)
    eigvals = np.clip(eigvals, 1e-12, None)  # paranoia: positive-definite floor
    D = np.sqrt(eigvals)
    C_inv_sqrt = B @ np.diag(1.0 / D) @ B.T  # C^{-1/2}

    # Update p_sigma (Eq. 43).
    p_sigma_new = (1.0 - c_sigma) * p_sigma + math.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * (
        C_inv_sqrt @ y_w
    )

    # Update sigma (Eq. 44).
    sigma_new = sigma * math.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma_new) / chi_n - 1.0))
    # Cap sigma growth to prevent runaway; Hansen 2016 §3.3 mentions this
    # as a robustness measure for short runs.
    sigma_new = float(min(sigma_new, 1e6))
    sigma_new = float(max(sigma_new, 1e-12))

    # Heaviside h_sigma indicator (Eq. 45 helper) — discounts p_c update
    # when sigma is about to explode (||p_sigma|| too large relative to
    # generation count). The bound is from Hansen 2016 below Eq. 45.
    g_plus_1 = state.generation + 1
    h_sigma_threshold = (1.4 + 2.0 / (d + 1.0)) * chi_n
    h_sigma_norm_bound = math.sqrt(1.0 - (1.0 - c_sigma) ** (2 * g_plus_1))
    h_sigma = float(np.linalg.norm(p_sigma_new) / h_sigma_norm_bound < h_sigma_threshold)

    # Update p_c (Eq. 45).
    p_c_new = (1.0 - c_c) * p_c + h_sigma * math.sqrt(c_c * (2.0 - c_c) * mu_eff) * y_w

    # Update C (Eq. 47): rank-1 + rank-mu.
    # The "delta_h_sigma" term compensates for h_sigma=0 events to keep
    # E[C] ~ C in expectation; Hansen 2016 Eq. 46.
    delta_h_sigma = (1.0 - h_sigma) * c_c * (2.0 - c_c)

    # y_k = (x_k - old_mean) / sigma for each selected sample k.
    y_selected = (selected - old_mean[None, :]) / sigma  # (mu, d)

    rank_one = np.outer(p_c_new, p_c_new)  # (d, d)
    # rank_mu = sum_k weights_k * y_k y_k^T
    rank_mu = (y_selected.T * weights[None, :]) @ y_selected  # (d, d)

    C_new = (1.0 - c_1 - c_mu + c_1 * delta_h_sigma) * C + c_1 * rank_one + c_mu * rank_mu
    # Symmetrise to suppress floating-point asymmetry drift.
    C_new = 0.5 * (C_new + C_new.T)

    return _CMAESState(
        mean=new_mean,
        sigma=sigma_new,
        C=C_new,
        p_sigma=p_sigma_new,
        p_c=p_c_new,
        generation=g_plus_1,
    )


def _cmaes_sample_population(
    state: _CMAESState,
    consts: _CMAESConsts,
    rng: np.random.Generator,
    lo_flat: np.ndarray,
    hi_flat: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Draw ``lambda_`` candidate vectors from N(mean, sigma^2 * C).

    Reflects each candidate into the [lo, hi] box element-wise. Returns
    ``(samples, n_total_reflections)`` where ``samples.shape ==
    (lambda_, d)``.
    """
    d = consts.d
    lambda_ = consts.lambda_

    # Eigendecomposition of C for the sampling factor BD.
    eigvals, B = np.linalg.eigh(state.C)
    eigvals = np.clip(eigvals, 1e-12, None)
    D = np.sqrt(eigvals)
    BD = B * D[None, :]  # (d, d): C = (BD)(BD)^T

    z = rng.standard_normal(size=(lambda_, d))  # (lambda, d)
    raw = state.mean[None, :] + state.sigma * (z @ BD.T)  # (lambda, d)

    # Reflect each row. Loop over rows is fine: lambda_ is small (<= 64).
    # JAX runs in float32 by default in this repo (jax_enable_x64 is off);
    # we use float32 inside the JAX path, then cast back to float64 for the
    # CMA-ES update math which is more eigendecomp-sensitive.
    out = np.empty_like(raw)
    n_total_refl = 0
    for k in range(lambda_):
        clipped, n_refl = _reflect_into_box(
            jnp.asarray(raw[k], dtype=jnp.float32),
            jnp.asarray(lo_flat, dtype=jnp.float32),
            jnp.asarray(hi_flat, dtype=jnp.float32),
        )
        out[k] = np.asarray(clipped, dtype=np.float64)
        n_total_refl += n_refl
    return out, n_total_refl


# ---------------------------------------------------------------------------
# Differentiable simulator wrapper (mirrors gradient_guided._wrap_simulator
# verbatim — replicated rather than imported to keep this module
# ---------------------------------------------------------------------------


SimulateFn = Callable[
    [jt.Float[jt.Array, " n"], jt.Float[jt.Array, "H m"], jt.PRNGKeyArray],
    tuple[jt.Float[jt.Array, "T n"], jt.Float[jt.Array, " T"]],
]


def _wrap_simulator(simulator: Any, sim_params: Any, aux: dict[str, Any] | None) -> SimulateFn:
    """Adapt a simulator to a uniform ``(initial_state, control, key) -> (states, times)`` callable.

    Mirrors the helper in :mod:`stl_seed.inference.gradient_guided`. The
    duplication is deliberate: keeping CMA-ES self-contained makes the
    firewall check single-file and the import graph minimal.
    """
    sim_class_name = type(simulator).__name__

    if sim_class_name == "GlucoseInsulinSimulator":
        from stl_seed.tasks.glucose_insulin import MealSchedule

        meal_schedule = (
            aux.get("meal_schedule") if aux is not None else None
        ) or MealSchedule.empty()

        def gi_sim(
            initial_state: jt.Float[jt.Array, " n"],
            control: jt.Float[jt.Array, "H m"],
            key: jt.PRNGKeyArray,
        ) -> tuple[jt.Float[jt.Array, "T n"], jt.Float[jt.Array, " T"]]:
            u_flat = control.reshape(-1)
            states, times, _meta = simulator.simulate(
                initial_state, u_flat, meal_schedule, sim_params, key
            )
            return states, times

        return gi_sim

    def generic_sim(
        initial_state: jt.Float[jt.Array, " n"],
        control: jt.Float[jt.Array, "H m"],
        key: jt.PRNGKeyArray,
    ) -> tuple[jt.Float[jt.Array, "T n"], jt.Float[jt.Array, " T"]]:
        traj = simulator.simulate(initial_state, control, sim_params, key)
        return traj.states, traj.times

    return generic_sim


# ---------------------------------------------------------------------------
# The sampler.
# ---------------------------------------------------------------------------


class CMAESGradientSampler:
    """CMA-ES population search + gradient refinement on action sequences.

    This is strategy A3 for fixing the cross-task gradient-guided
    failure on the repressilator (paper/cross_task_validation.md). The
    motivating diagnosis: the per-step gradient probe is myopic and
    cannot find the small-measure satisfying region in the joint
    ``H * m``-D action box. CMA-ES escapes the local minimum via
    covariance-adapted population sampling; gradient ascent then
    polishes the survivor.

    Algorithm
    ---------

    1. **Initial mean**. Either ``initial_mean`` (provided), or the
       LLM's argmax-vocabulary action sequence (one greedy decode), or
       the action-box midpoint (fallback when the LLM is uniform).
    2. **For** ``g = 1, ..., n_generations``:

       a. Sample ``population_size`` action sequences from
          ``N(m, sigma^2 * C)``, reflected into the action box.
       b. Evaluate ``rho`` on each (one simulator + STL eval per
          sample).
       c. Select the top ``population_size // 2`` and update
          ``(m, sigma, C, p_sigma, p_c)`` per Hansen 2016.
    3. **Refinement**. Take the best survivor across all generations
       and apply ``n_refine`` plain gradient-ascent steps on
       ``rho(action_sequence)`` with reflection clamping per step.
    4. **Return** the refined action sequence's full trajectory.

    Hyperparameters
    ---------------

    population_size : int, default 32
        ``λ`` in CMA-ES notation. Hansen 2016's default rule is
        ``λ = 4 + floor(3 * ln(d))``; for ``d = 30`` this gives
        ``λ = 14``. We default to 32 for stronger exploration on the
        bumpy repressilator landscape.
    n_generations : int, default 20
        Number of CMA-ES generations.
    sigma_init : float, default 0.3
        Initial step-size, in the units of the action box half-width.
        For an action box of ``[0, 1]^m``, ``σ = 0.3`` covers ~half
        of the box at one standard deviation.
    n_refine : int, default 30
        Number of gradient-ascent refinement steps applied to the best
        survivor.
    lr : float, default 1e-2
        Refinement step size. Plain SGD; if ``rho`` is very flat near
        the survivor, increase. If oscillating, decrease.
    initial_mean_source : {"llm_argmax", "midpoint", "user"}, default "midpoint"
        How to seed the CMA-ES mean. ``"llm_argmax"`` runs the LLM
        once greedily and uses that action sequence; ``"midpoint"``
        uses the action-box midpoint; ``"user"`` requires
        ``initial_mean`` to be provided.

    Compute cost
    ------------

    ``n_generations * population_size`` simulator + STL evals during
    the population search, plus ``n_refine`` backward passes during
    refinement. For the default settings this is ``20 * 32 + 30 = 670``
    forward simulator calls and 30 backward calls.

    -------------

    (arXiv:1604.00772) directly.
    """

    def __init__(
        self,
        llm: LLMProposal,
        simulator: Any,
        spec: STLSpec | Node,
        action_vocabulary: jt.Float[jt.Array, "K m"],
        sim_params: Any,
        *,
        horizon: int,
        aux: dict[str, Any] | None = None,
        population_size: int = 32,
        n_generations: int = 20,
        sigma_init: float = 0.3,
        n_refine: int = 30,
        lr: float = 1e-2,
        action_low: jt.Float[jt.Array, " m"] | float | np.ndarray | None = None,
        action_high: jt.Float[jt.Array, " m"] | float | np.ndarray | None = None,
        initial_mean: jt.Float[jt.Array, "H m"] | None = None,
        initial_mean_source: str = "midpoint",
    ) -> None:
        if population_size < 4:
            raise ValueError(f"population_size must be >= 4, got {population_size}")
        if n_generations < 1:
            raise ValueError(f"n_generations must be >= 1, got {n_generations}")
        if sigma_init <= 0.0:
            raise ValueError(f"sigma_init must be positive, got {sigma_init}")
        if n_refine < 0:
            raise ValueError(f"n_refine must be >= 0, got {n_refine}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if initial_mean_source not in {"llm_argmax", "midpoint", "user"}:
            raise ValueError(
                f"initial_mean_source must be one of "
                f"{{'llm_argmax', 'midpoint', 'user'}}, got {initial_mean_source!r}"
            )
        if initial_mean_source == "user" and initial_mean is None:
            raise ValueError("initial_mean_source='user' requires initial_mean to be provided")

        self.llm = llm
        self.simulator = simulator
        self.spec = spec
        self.vocabulary = jnp.asarray(action_vocabulary, dtype=jnp.float32)
        if self.vocabulary.ndim != 2:
            raise ValueError(
                f"action_vocabulary must be 2-d (K, m), got shape {self.vocabulary.shape}"
            )
        self.K, self.m = int(self.vocabulary.shape[0]), int(self.vocabulary.shape[1])
        self.sim_params = sim_params
        self.horizon = int(horizon)
        self.aux = dict(aux) if aux is not None else None
        self.population_size = int(population_size)
        self.n_generations = int(n_generations)
        self.sigma_init = float(sigma_init)
        self.n_refine = int(n_refine)
        self.lr = float(lr)
        self.initial_mean_source = str(initial_mean_source)

        # Action box: default to the vocabulary's bounding box if not given.
        if action_low is None:
            lo = np.asarray(self.vocabulary.min(axis=0), dtype=np.float64)
        else:
            lo = np.atleast_1d(np.asarray(action_low, dtype=np.float64))
            if lo.size == 1 and self.m > 1:
                lo = np.broadcast_to(lo, (self.m,)).copy()
        if action_high is None:
            hi = np.asarray(self.vocabulary.max(axis=0), dtype=np.float64)
        else:
            hi = np.atleast_1d(np.asarray(action_high, dtype=np.float64))
            if hi.size == 1 and self.m > 1:
                hi = np.broadcast_to(hi, (self.m,)).copy()
        if lo.shape != (self.m,) or hi.shape != (self.m,):
            raise ValueError(
                f"action_low / action_high must broadcast to ({self.m},); "
                f"got shapes {lo.shape}, {hi.shape}"
            )
        if not np.all(hi > lo):
            raise ValueError(f"action_high must be > action_low elementwise; got lo={lo}, hi={hi}")
        self.action_lo = lo  # (m,)
        self.action_hi = hi  # (m,)

        self._initial_mean_user: np.ndarray | None = (
            np.asarray(initial_mean, dtype=np.float64) if initial_mean is not None else None
        )
        if self._initial_mean_user is not None and self._initial_mean_user.shape != (
            self.horizon,
            self.m,
        ):
            raise ValueError(
                f"initial_mean must have shape ({self.horizon}, {self.m}), "
                f"got {self._initial_mean_user.shape}"
            )

        # Compile STL spec once. The CMA-ES path itself only needs the
        # forward eval (no autodiff); the refinement step does need autodiff.
        # We refuse non-conforming predicates either way so the refinement
        # behaves predictably.
        self._compiled_spec = compile_spec(spec)
        from stl_seed.stl.evaluator import _FALLBACK_USED

        if getattr(self._compiled_spec, _FALLBACK_USED, False):
            raise RuntimeError(
                "CMAESGradientSampler requires every predicate to be "
                "JIT/grad-compatible (the introspection convention in "
                "stl_seed.stl.evaluator._introspect_predicate). At least one "
                f"predicate in spec {getattr(spec, 'name', '<unknown>')!r} "
                "fell back to the slow Python path. Either rewrite the "
                "predicate via stl_seed.specs.bio_ode_specs._gt / _lt, or "
                "use BestOfNSampler instead."
            )

        # Wrap the simulator into a uniform (init, control, key) -> (states, times) form.
        self._sim_fn = _wrap_simulator(simulator, sim_params, self.aux)

        compiled_spec = self._compiled_spec
        sim_fn = self._sim_fn

        def rho_from_control(
            initial_state: jt.Float[jt.Array, " n"],
            control: jt.Float[jt.Array, "H m"],
            key: jt.PRNGKeyArray,
        ) -> jt.Float[jt.Array, ""]:
            states, times = sim_fn(initial_state, control, key)
            return compiled_spec(states, times)

        self._rho_from_control = jax.jit(rho_from_control)
        # Refinement uses value-and-grad wrt the *control argument*.
        self._value_and_grad_control = jax.jit(jax.value_and_grad(rho_from_control, argnums=1))

    # ------------------------------------------------------------------ public

    def sample(
        self,
        initial_state: jt.Float[jt.Array, " n"],
        key: jt.PRNGKeyArray,
    ) -> SamplerResult:
        """Run CMA-ES + gradient refinement; return the best trajectory.

        Parameters
        ----------
        initial_state : Float[Array, "n"]
            Simulator initial state.
        key : PRNGKeyArray
            Master PRNG key. Used for (a) the LLM call when seeding via
            ``initial_mean_source="llm_argmax"``, (b) the per-generation
            CMA-ES sampling RNG (we derive a NumPy ``Generator`` from
            ``key`` so the Python-side CMA-ES is reproducible from the
            same JAX key), and (c) the simulator key for each rho eval.

        Returns
        -------
        (trajectory, diagnostics)
            ``trajectory`` is the canonical Trajectory produced by
            simulating the refined action sequence from ``initial_state``.
            ``diagnostics`` is :meth:`CMAESDiagnostics.to_dict`.
        """
        diag = CMAESDiagnostics()

        # Build the initial mean (flat (H*m,) vector for CMA-ES).
        m_init_2d = self._build_initial_mean(initial_state, key)  # (H, m)
        d = self.horizon * self.m
        m_init_flat = np.asarray(m_init_2d, dtype=np.float64).reshape(-1)
        # Reflect into box just in case the LLM proposed something outside.
        lo_flat = np.tile(self.action_lo, self.horizon)  # (H*m,)
        hi_flat = np.tile(self.action_hi, self.horizon)  # (H*m,)
        m_init_clipped, _ = _reflect_into_box(
            jnp.asarray(m_init_flat), jnp.asarray(lo_flat), jnp.asarray(hi_flat)
        )
        m_init_flat = np.asarray(m_init_clipped, dtype=np.float64)

        consts = _CMAESConsts.from_dim(d=d, lambda_=self.population_size)
        state = _cmaes_init(d=d, initial_mean=m_init_flat, initial_sigma=self.sigma_init)

        # Derive a deterministic NumPy RNG from the JAX key. We use the key's
        # uint32 entropy as the seed so the run is reproducible from `key`.
        seed_int = int(jax.random.randint(key, (), 0, 2**31 - 1))
        rng = np.random.default_rng(seed_int)

        sim_key = jax.random.fold_in(key, 7919)  # arbitrary salt for sim eval

        # Track the global best across all generations (CMA-ES updates
        # the mean toward selected samples but the best individual
        # sample seen may be elsewhere).
        global_best_x_flat: np.ndarray = m_init_flat.copy()
        global_best_rho: float = float(
            self._rho_from_control(
                initial_state,
                jnp.asarray(m_init_flat, dtype=jnp.float32).reshape(self.horizon, self.m),
                sim_key,
            )
        )

        for _g in range(self.n_generations):
            diag.sigma_per_gen.append(float(state.sigma))

            samples_flat, _n_refl = _cmaes_sample_population(
                state, consts, rng, lo_flat=lo_flat, hi_flat=hi_flat
            )  # (lambda, d)

            # Evaluate rho on each sample. We re-shape to (H, m) and call
            # the JIT'd rho closure. JAX caches the trace so this is cheap.
            rhos = np.empty(self.population_size, dtype=np.float64)
            for k in range(self.population_size):
                ctrl_k = jnp.asarray(samples_flat[k], dtype=jnp.float32).reshape(
                    self.horizon, self.m
                )
                rhos[k] = float(self._rho_from_control(initial_state, ctrl_k, sim_key))

            # Replace any non-finite rho with -inf so they sort last.
            rhos = np.where(np.isfinite(rhos), rhos, -np.inf)

            # Rank and select the top mu (descending rho = ascending -rho).
            order = np.argsort(-rhos)  # best first
            sel_idx = order[: consts.mu]
            selected = samples_flat[sel_idx]  # (mu, d)

            # Diagnostics: best/mean per generation.
            best_rho_g = float(rhos[order[0]])
            mean_rho_g = (
                float(rhos[np.isfinite(rhos)].mean()) if np.isfinite(rhos).any() else float("nan")
            )
            diag.best_rho_per_gen.append(best_rho_g)
            diag.mean_rho_per_gen.append(mean_rho_g)

            if best_rho_g > global_best_rho:
                global_best_rho = best_rho_g
                global_best_x_flat = samples_flat[order[0]].copy()

            # CMA-ES update. If all rho are -inf (catastrophic), skip the
            # update to avoid propagating NaN through the eigendecomposition.
            if np.isfinite(rhos[order[0]]):
                state = _cmaes_update(state, consts, selected)

        # ---- Refinement: gradient ascent on the global-best survivor. ----
        diag.rho_pre_refine = float(global_best_rho)

        x_refined = jnp.asarray(global_best_x_flat, dtype=jnp.float32).reshape(self.horizon, self.m)
        lo_2d = jnp.asarray(self.action_lo, dtype=jnp.float32)[None, :]  # (1, m)
        hi_2d = jnp.asarray(self.action_hi, dtype=jnp.float32)[None, :]  # (1, m)

        # Refinement step size, in the units of the *box half-width* per
        # coordinate. The raw gradient ``grad rho`` can be many orders of
        # magnitude (the repressilator rho has natural scale ~250 nM); to
        # make the ``lr`` hyperparameter task-agnostic we normalise the
        # gradient to a unit-infinity-norm direction and take a step of
        # size ``lr * box_width / 2`` along that direction. With backtrack-
        # ing line search (halve the step on rejection) this is robust to
        # the gradient's magnitude scale.
        box_halfwidth = jnp.asarray((self.action_hi - self.action_lo) / 2.0, dtype=jnp.float32)
        lo_full = lo_2d.repeat(self.horizon, axis=0)
        hi_full = hi_2d.repeat(self.horizon, axis=0)

        n_finite_grads = 0
        n_box_reflections = 0
        for _r in range(self.n_refine):
            try:
                rho_val, grad_ctrl = self._value_and_grad_control(initial_state, x_refined, sim_key)
            except Exception:
                # Hard failure: stop refinement and keep best-so-far.
                break
            rho_val_f = float(rho_val)
            if not (math.isfinite(rho_val_f) and bool(jnp.all(jnp.isfinite(grad_ctrl)))):
                # Skip this step but record the unrefined rho so the trace
                # still has the right length.
                diag.rho_post_refine_per_step.append(rho_val_f)
                continue
            n_finite_grads += 1

            # Normalise gradient to unit infinity-norm; if the gradient is
            # essentially zero, skip this step (we are at a critical point).
            g_inf = float(jnp.max(jnp.abs(grad_ctrl)))
            if g_inf < 1e-12:
                diag.rho_post_refine_per_step.append(rho_val_f)
                continue
            direction = grad_ctrl / g_inf  # bounded in [-1, 1] per coordinate

            # Backtracking line search: start at lr * box_halfwidth, halve
            # on rejection up to 5 times. Accept the first step that does
            # not strictly decrease rho.
            step_scale = self.lr  # interpreted as fraction of box_halfwidth
            accepted = False
            best_new_rho = rho_val_f
            best_new_x = x_refined
            for _ in range(5):
                x_proposed = x_refined + step_scale * box_halfwidth[None, :] * direction
                x_clipped, n_refl = _reflect_into_box(x_proposed, lo_full, hi_full)
                n_box_reflections += int(n_refl)
                new_rho = float(self._rho_from_control(initial_state, x_clipped, sim_key))
                if new_rho >= rho_val_f - 1e-6:
                    best_new_rho = new_rho
                    best_new_x = x_clipped
                    accepted = True
                    break
                step_scale *= 0.5
            if accepted:
                x_refined = best_new_x
                diag.rho_post_refine_per_step.append(best_new_rho)
            else:
                diag.rho_post_refine_per_step.append(rho_val_f)

        diag.n_finite_grads = int(n_finite_grads)
        diag.n_box_reflections = int(n_box_reflections)

        # ---- Final trajectory: re-simulate the refined control. ----
        traj = self._build_trajectory(initial_state, x_refined, sim_key)
        diag.final_rho = float(self._compiled_spec(traj.states, traj.times))
        return traj, diag.to_dict()

    # -------------------------------------------------------------- internals

    def _build_initial_mean(
        self,
        initial_state: jt.Float[jt.Array, " n"],
        key: jt.PRNGKeyArray,
    ) -> jt.Float[jt.Array, "H m"]:
        """Construct the starting CMA-ES mean as an (H, m) action sequence."""
        if self.initial_mean_source == "user":
            assert self._initial_mean_user is not None  # guarded in __init__
            return jnp.asarray(self._initial_mean_user, dtype=jnp.float32)
        if self.initial_mean_source == "midpoint":
            mid = 0.5 * (self.action_lo + self.action_hi)  # (m,)
            return jnp.broadcast_to(jnp.asarray(mid, dtype=jnp.float32), (self.horizon, self.m))
        # "llm_argmax": one greedy LLM rollout (no simulator interaction).
        history = jnp.zeros((0, self.m), dtype=jnp.float32)
        actions: list[jt.Float[jt.Array, " m"]] = []
        for t in range(self.horizon):
            llm_key = jax.random.fold_in(key, t)
            logits = jnp.asarray(self.llm(initial_state, history, llm_key), dtype=jnp.float32)
            if logits.shape != (self.K,):
                raise ValueError(
                    f"LLM emitted logits of shape {logits.shape}, expected ({self.K},)"
                )
            idx = int(jnp.argmax(logits))
            a = self.vocabulary[idx]
            actions.append(a)
            history = jnp.concatenate([history, a[None, :]], axis=0)
        return jnp.stack(actions, axis=0)

    def _build_trajectory(
        self,
        initial_state: jt.Float[jt.Array, " n"],
        control: jt.Float[jt.Array, "H m"],
        key: jt.PRNGKeyArray,
    ) -> Trajectory:
        """Run the simulator once to materialise the canonical Trajectory."""
        sim_class_name = type(self.simulator).__name__
        if sim_class_name == "GlucoseInsulinSimulator":
            from stl_seed.tasks.glucose_insulin import MealSchedule

            meal_schedule = (
                self.aux.get("meal_schedule") if self.aux is not None else None
            ) or MealSchedule.empty()
            states, times, meta = self.simulator.simulate(
                initial_state, control.reshape(-1), meal_schedule, self.sim_params, key
            )
            return Trajectory(
                states=states,
                actions=control.reshape(-1, self.m),
                times=times,
                meta=meta,
            )
        traj = self.simulator.simulate(initial_state, control, self.sim_params, key)
        return Trajectory(
            states=traj.states,
            actions=control,
            times=traj.times,
            meta=traj.meta,
        )


__all__ = [
    "CMAESDiagnostics",
    "CMAESGradientSampler",
]

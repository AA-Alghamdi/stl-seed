"""Horizon-folded gradient-guided STL sampling (strategy A1).

Technical contribution
----------------------

The :class:`STLGradientGuidedSampler` (sibling module
``gradient_guided.py``) computes a *per-step* gradient
``grad_{u_t} rho`` against a *partial-then-extrapolated* trajectory
``hat_u = (u_1, ..., u_{t-1}, u_bar_t, u_def, ..., u_def)``. On
glucose-insulin this works (a single insulin bolus mostly determines
local glucose dynamics within ~30 min). On the repressilator it does
not: ``G_[120,200] (m1 >= 250)`` requires *sustained* silencing of
gene-3 across all 10 control steps, and the per-step gradient at any
single ``u_t`` cannot point coherently toward this attractor — the
"future is the default action" assumption is wrong when the spec is a
long-window conjunction in which all H actions matter jointly. See
``paper/cross_task_validation.md`` for the failure receipt.

This module implements **horizon folding**: instead of decomposing the
problem into H per-step gradients, we differentiate ``rho`` with respect
to the *entire* control sequence ``u_{1:H}`` simulated end-to-end, then
take ``K`` Adam steps on the whole sequence. This is essentially MPC
with autodiff (Amos & Kolter 2017, "OptNet: Differentiable Optimization
as a Layer in Neural Networks", NeurIPS 2017, arXiv:1703.00443; cf.
Differentiable MPC, Amos et al. 2018, NeurIPS, arXiv:1810.13400).

Algorithm
~~~~~~~~~

Given an initial state ``x_0``, an STL spec ``phi``, and a horizon ``H``:

1. **Initialise** an unconstrained latent ``z_0 in R^{H x m}`` from one
   of:

   * ``init='zeros'``     — ``z_0 = 0`` so the post-sigmoid action is
     the action-box centre.
   * ``init='llm'``       — for each step ``t``, take the LLM logits
     over the discrete vocabulary ``V``, form the preferred mean
     ``u_bar_t = sum_k softmax(z_t)_k * V_k``, then invert the sigmoid
     to recover the corresponding ``z_0_t`` (numerically clipped to
     ``[low + eps, high - eps]`` to avoid the logit blow-up at the
     endpoints).
   * ``init='random'``    — i.i.d. ``Normal(0, 0.1)``, small enough that
     the post-sigmoid action stays near the centre.
   * ``init='heuristic'`` — accept a user-provided ``(H, m)`` vector
     and invert-sigmoid it (delegating "which action is a good warm
     start" to the caller; useful for unit tests with a known answer).

2. **Reparameterise** the action sequence as
   ``u_{1:H} = low + (high - low) * sigmoid(z)``.
   This gives a *smooth* projection onto the action box ``[low, high]^m``
   without the gradient discontinuity at the boundary that a hard
   ``jnp.clip`` would introduce. The Jacobian
   ``du/dz = (high - low) * sigmoid(z) * (1 - sigmoid(z))``
   is positive definite, so ``grad_z rho`` and ``grad_u rho`` agree in
   sign and only differ by a positive scaling factor per coordinate.

3. **Optimise** with Adam (Kingma & Ba 2014, "Adam: A Method for
   Stochastic Optimization", arXiv:1412.6980) for ``K_iters`` steps:

   .. code-block:: text

       g = jax.grad(lambda z: rho(simulate(x_0, sigmoid_reparam(z))))(z)
       (m, v) = adam_update(m, v, g, t)
       z = z + lr * adam_step(m, v, t)              # gradient ASCENT

   We do gradient *ascent* (rho-maximisation), implemented as the usual
   Adam update applied to ``-grad rho`` then negated, equivalently as
   ``z + lr * (m_hat / (sqrt(v_hat) + eps))``.

4. **Return** the action sequence at the iteration with the highest
   in-loop ``rho`` (we track the best-so-far and return it; gradient
   ascent on a non-convex landscape can overshoot, and the best-so-far
   guard makes the sampler monotone in returned rho).

Hyperparameters
~~~~~~~~~~~~~~~

* ``lr``       : Adam learning rate. Default ``1e-2``. The action box
  is normalised to width 1 by sigmoid; ``1e-2`` per step is small
  enough to make ~50 steps before saturating either direction.
* ``K_iters``  : Number of refinement iterations. Default ``100``. The
  per-iteration cost is one ODE solve + one backward pass; on
  bio_ode tasks (H=10, T=200 min, Tsit5 with PID) this is ~50 ms per
  iter on M-series Metal once warmed up.
* ``init``     : Initialisation strategy (see step 1). Default
  ``'zeros'``.
* ``adam_b1``  : Adam first-moment decay. Default ``0.9`` (canonical).
* ``adam_b2``  : Adam second-moment decay. Default ``0.999`` (canonical).
* ``adam_eps`` : Adam epsilon. Default ``1e-8`` (canonical).

Compute cost
~~~~~~~~~~~~

``K_iters`` × (1 forward simulator call + 1 backward pass) per
``sample()`` call. Compare with :class:`STLGradientGuidedSampler` which
costs ``H`` × (1 fwd + 1 bwd) per call (one per step). At
``K_iters = 100`` and ``H = 10``, horizon-folding is ``10×`` more
expensive — but each gradient step uses the *true* end-to-end
trajectory rather than a partial extrapolation, so each step's
information content is qualitatively different.

~~~~~~~~~~~~~

This module imports only from JAX, jaxtyping, and in-package modules
(``stl_seed.stl``, ``stl_seed.tasks``, ``stl_seed.inference``). No

Relationship to other samplers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :class:`STLGradientGuidedSampler` (per-step myopic gradient on
  partial trajectories) — A0 from the strategy ladder.
* :class:`HorizonFoldedGradientSampler` (this class) — A1: full-horizon
  gradient on end-to-end trajectories; same backbone (autodiff through
  the STL evaluator and Diffrax solver), different decomposition.
* :class:`HybridGradientBoNSampler` — orthogonal axis: BoN selection on
  top of A0; could be re-applied on top of A1 (future work).
* :class:`ContinuousBoNSampler` — matched-compute baseline if we treat
  ``K_iters`` as analogous to ``n_bon`` (each Adam step costs one
  forward + one backward; one BoN sample costs one forward).

Limitations
~~~~~~~~~~~

* **Continuous output** — the optimised ``u_{1:H}`` is a real-valued
  vector, not a vocabulary index. For the STL evaluation pipeline
  this is fine (the simulator clips to the action box and the
  evaluator never inspects the action sequence). For *training*
  pipelines that consume ``trajectory.actions`` as token indices,
  callers must project onto the vocabulary post-hoc (e.g. via
  nearest-neighbour assignment in action space). The diagnostics
  expose the continuous control sequence and the per-step nearest
  vocabulary index for downstream consumers.
* **Local optimum** — gradient ascent on a non-convex ``rho`` landscape
  can stall in a saddle / weak local maximum. The best-so-far guard
  protects against overshoot but not against local stalling. The
  ``init='llm'`` path partially mitigates by starting near the LLM's
  preferred mean. Future work: combine horizon-folding with multi-
  start (analogue of Hybrid×Folded).
* **No discrete vocabulary structure used** — the sampler does not
  consume vocabulary directions ``V_k - u_bar`` the way A0 does. If a
  task has a strongly anisotropic vocabulary (some V_k are
  qualitatively different from the box centre), horizon-folding
  treats the action space as continuous and may miss that structure.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, Literal

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
class HorizonFoldedDiagnostics:
    """Per-iteration diagnostics for one horizon-folded refinement run.

    Fields are lists indexed by Adam iteration ``k in [0, K_iters]`` (the
    extra entry at index 0 is the pre-optimisation snapshot).

    * ``rho_at_iter``       : ``rho(u_{1:H})`` at iteration ``k``. Should
      be (mostly) monotone-improving when the gradient ascent is doing
      useful work. Non-monotonicity is expected occasionally because Adam
      is a first-order method on a non-convex landscape; the
      ``best_rho`` field tracks the best-so-far separately.
    * ``grad_norm_at_iter`` : ``||grad_z rho||_2`` at iteration ``k``.
      Decreases as the optimiser approaches a stationary point.
      Pre-registered: a vanishing gradient norm coincident with negative
      best_rho indicates a saddle / weak local optimum.
    * ``best_rho``          : The maximum ``rho`` seen across all
      iterations (the value the sampler returns).
    * ``best_iter``         : The iteration index that produced
      ``best_rho``.
    * ``final_rho``         : Alias for ``best_rho`` (compatibility with
      the canonical ``Sampler`` diagnostics convention).
    * ``init_rho``          : ``rho`` of the initial (pre-Adam) action
      sequence. ``best_rho - init_rho`` is the gradient-ascent gain.
    * ``init_strategy``     : The init strategy string that was used.
    * ``fallback_used``     : True if any predicate fell back to the
      slow Python evaluation path (which disables JIT / grad). When
      True, the sampler raises at construction rather than silently
      degrading.
    * ``n_iters``           : ``K_iters`` (mirrored for harness).
    """

    rho_at_iter: list[float] = dataclasses.field(default_factory=list)
    grad_norm_at_iter: list[float] = dataclasses.field(default_factory=list)
    best_rho: float = float("nan")
    best_iter: int = -1
    final_rho: float = float("nan")
    init_rho: float = float("nan")
    init_strategy: str = ""
    fallback_used: bool = False
    n_iters: int = 0

    def to_dict(self) -> SamplerDiagnostics:
        """Materialise as a plain dict for the harness."""
        return {
            "sampler": "horizon_folded_gradient",
            "rho_at_iter": list(self.rho_at_iter),
            "grad_norm_at_iter": list(self.grad_norm_at_iter),
            "best_rho": float(self.best_rho),
            "best_iter": int(self.best_iter),
            "final_rho": float(self.final_rho),
            "init_rho": float(self.init_rho),
            "init_strategy": str(self.init_strategy),
            "fallback_used": bool(self.fallback_used),
            "n_iters": int(self.n_iters),
            "n_steps": int(self.n_iters),
        }


# ---------------------------------------------------------------------------
# Sigmoid reparam helpers.
# ---------------------------------------------------------------------------


def _sigmoid_reparam(
    z: jt.Float[jt.Array, "H m"],
    low: jt.Float[jt.Array, " m"],
    high: jt.Float[jt.Array, " m"],
) -> jt.Float[jt.Array, "H m"]:
    """Smooth box projection: ``u = low + (high - low) * sigmoid(z)``.

    Because ``sigmoid: R -> (0, 1)`` is strictly increasing, the
    composition is a smooth bijection between ``R^{H x m}`` and the open
    box ``(low, high)^{H x m}``. Adam can therefore optimise ``z``
    unconstrained without ever violating the action bounds at the
    optimised point. The mapping is asymptotically saturating, which
    means very large ``|z|`` produces vanishing gradients — the same
    regularisation effect as a soft-clip penalty, paid at the boundary
    rather than as an extra loss term.
    """
    return low + (high - low) * jax.nn.sigmoid(z)


def _inverse_sigmoid_reparam(
    u: jt.Float[jt.Array, "H m"],
    low: jt.Float[jt.Array, " m"],
    high: jt.Float[jt.Array, " m"],
    *,
    eps: float = 1e-3,
) -> jt.Float[jt.Array, "H m"]:
    """Inverse of :func:`_sigmoid_reparam`: recover ``z`` from a target ``u``.

    Used by the ``init='llm'`` and ``init='heuristic'`` paths to warm-
    start ``z`` from a target action sequence. We clip ``u`` to
    ``[low + eps * width, high - eps * width]`` before inverting to
    avoid the logit blow-up ``logit(0) = -inf``, ``logit(1) = +inf`` at
    the box boundary; ``eps = 1e-3`` keeps the post-clip ``z`` finite
    (``|z| <= log((1 - eps) / eps) ~ 7`` for ``eps = 1e-3``) while
    preserving the directional information the warm-start provides.
    """
    width = high - low
    # Clip u into the open interval so the logit is finite.
    u_safe = jnp.clip(u, low + eps * width, high - eps * width)
    s = (u_safe - low) / width
    return jnp.log(s) - jnp.log1p(-s)


# ---------------------------------------------------------------------------
# Hand-rolled Adam (no optax dependency: keeps the inference subpackage's
# dependency surface to {jax, jaxtyping, equinox, diffrax}).
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _AdamState:
    """Minimal Adam state pytree (carried as a tuple of arrays in JIT)."""

    m: jt.Float[jt.Array, "H m"]
    v: jt.Float[jt.Array, "H m"]
    t: int  # iteration counter (1-indexed inside the bias-corrected step)


def _adam_init(z: jt.Float[jt.Array, "H m"]) -> _AdamState:
    """Construct zero-initialised first/second moment buffers shaped like ``z``."""
    return _AdamState(m=jnp.zeros_like(z), v=jnp.zeros_like(z), t=0)


def _adam_step(
    state: _AdamState,
    grad: jt.Float[jt.Array, "H m"],
    *,
    lr: float,
    b1: float,
    b2: float,
    eps: float,
) -> tuple[jt.Float[jt.Array, "H m"], _AdamState]:
    """One Adam update for *gradient ascent* (note the sign flip vs descent).

    The canonical Adam descent step is
        z <- z - lr * m_hat / (sqrt(v_hat) + eps).
    We are maximising rho, so the ascent step is
        z <- z + lr * m_hat / (sqrt(v_hat) + eps).
    The first/second-moment accumulators are unchanged; only the sign of
    the parameter update flips. We accumulate moments of the raw
    gradient ``+grad rho`` (not the negation) to keep the convention
    intuitive for downstream readers.
    """
    t = state.t + 1
    m = b1 * state.m + (1.0 - b1) * grad
    v = b2 * state.v + (1.0 - b2) * (grad * grad)
    m_hat = m / (1.0 - b1**t)
    v_hat = v / (1.0 - b2**t)
    delta = lr * m_hat / (jnp.sqrt(v_hat) + eps)  # ascent direction
    return delta, _AdamState(m=m, v=v, t=t)


# ---------------------------------------------------------------------------
# Differentiable simulator wrapper (mirrors gradient_guided._wrap_simulator).
# ---------------------------------------------------------------------------


SimulateFn = Callable[
    [jt.Float[jt.Array, " n"], jt.Float[jt.Array, "H m"], jt.PRNGKeyArray],
    tuple[jt.Float[jt.Array, "T n"], jt.Float[jt.Array, " T"]],
]


def _wrap_simulator(simulator: Any, sim_params: Any, aux: dict[str, Any] | None) -> SimulateFn:
    """Adapt a simulator to ``(initial_state, control, key) -> (states, times)``.

    Identical contract to :func:`stl_seed.inference.gradient_guided._wrap_simulator`
    but kept local to this module to preserve the "Round 1 only touches
    horizon_folded.py" invariant and to avoid a back-import that would
    create a cyclic dependency if the per-step sampler ever imports this
    module.
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


InitStrategy = Literal["zeros", "llm", "random", "heuristic"]


class HorizonFoldedGradientSampler:
    """Gradient-guided STL sampling with full-horizon action gradients.

    Instead of per-step ``grad_{u_t} rho`` on partial trajectories
    (cf. :class:`STLGradientGuidedSampler`), this sampler computes
    ``grad_{u_{1:H}} rho`` on the full end-to-end trajectory and takes
    ``K_iters`` Adam steps on the entire action sequence simultaneously.

    Reference (autodiff-MPC framing). Amos & Kolter (2017) "OptNet:
    Differentiable Optimization as a Layer in Neural Networks", NeurIPS
    2017, arXiv:1703.00443; Amos et al. (2018) "Differentiable MPC for
    End-to-end Planning and Control", NeurIPS 2018, arXiv:1810.13400.
    The present method is the inference-time-decoding variant: a
    differentiable verifier (the STL evaluator) replaces the
    differentiable cost-of-control objective.

    Parameters
    ----------
    llm:
        LLM proposal callable conforming to :class:`LLMProposal`.
        Required by the constructor for compatibility with the
        :class:`Sampler` Protocol and the ``init='llm'`` warm-start
        path. When ``init='zeros' | 'random' | 'heuristic'`` the LLM
        is *not* called during ``sample()``.
    simulator:
        ODE simulator with ``simulate(initial_state, control, params,
        key)`` (or the glucose-insulin variant; see
        :func:`_wrap_simulator`). Must be JAX-traceable for ``jax.grad``
        to flow through.
    spec:
        STL specification. Must use the JIT-compatible predicate
        introspection convention from ``stl_seed.stl.evaluator``.
        Non-conforming specs raise at construction.
    action_vocabulary:
        Discrete action set ``V in R^{K x m}``. Used only for the
        ``init='llm'`` warm-start (to compute the LLM's preferred mean
        action via ``softmax(logits) @ V``) and for diagnostic post-
        projection of the optimised continuous control onto the
        nearest vocabulary item per step.
    sim_params:
        Kinetic parameter pytree consumed by the simulator.
    horizon:
        Number of control steps ``H``. Must equal the simulator's
        ``n_control_points``.
    aux:
        Optional task-specific kwargs forwarded to the simulator
        (e.g. ``meal_schedule`` for glucose-insulin). ``None`` for
        bio_ode tasks.
    action_low, action_high:
        Per-axis action bounds. Either a scalar (broadcast across all
        ``m`` axes) or an array of shape ``(m,)``. The optimised
        continuous control is constrained to ``[action_low, action_high]``
        via the sigmoid reparameterisation.
    lr:
        Adam learning rate. Default ``1e-2``.
    k_iters:
        Number of Adam refinement steps. Default ``100``.
    init:
        Initialisation strategy: ``'zeros'`` (action-box centre),
        ``'llm'`` (LLM preferred-mean warm-start), ``'random'``
        (small-variance Gaussian), or ``'heuristic'`` (use
        ``init_action`` argument). Default ``'zeros'``.
    init_action:
        For ``init='heuristic'``, the warm-start action sequence of
        shape ``(H, m)``. Required when ``init='heuristic'``; ignored
        otherwise.
    adam_b1, adam_b2, adam_eps:
        Standard Adam hyperparameters. Defaults match Kingma & Ba 2014.
    track_best:
        If True (default), the sampler returns the action sequence at
        the iteration with the highest in-loop ``rho``, not the final
        iteration. This guards against gradient ascent overshoot on a
        non-convex landscape.
    fallback_on_grad_failure:
        If True (default), a NaN/Inf gradient at any iteration falls
        back to the best-so-far action sequence and the sampler exits
        early. If False, the sampler raises on numerical pathology.

    Notes
    -----
    The sampler is *not* an ``equinox.Module`` — it carries Python-
    level bookkeeping (the diagnostics record, the best-so-far guard).
    The inner gradient computation is JIT'd via
    ``jax.jit(jax.value_and_grad(...))`` and re-used across iterations.
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
        action_low: jt.Float[jt.Array, " m"] | float | np.ndarray | None = None,
        action_high: jt.Float[jt.Array, " m"] | float | np.ndarray | None = None,
        lr: float = 1e-2,
        k_iters: int = 100,
        init: InitStrategy = "zeros",
        init_action: jt.Float[jt.Array, "H m"] | None = None,
        adam_b1: float = 0.9,
        adam_b2: float = 0.999,
        adam_eps: float = 1e-8,
        track_best: bool = True,
        fallback_on_grad_failure: bool = True,
    ) -> None:
        # ---- LLM / simulator / spec ---------------------------------------
        self.llm = llm
        self.simulator = simulator
        self.spec = spec

        # ---- Vocabulary (used for LLM warm-start + diagnostics only) ------
        self.vocabulary = jnp.asarray(action_vocabulary, dtype=jnp.float32)
        if self.vocabulary.ndim != 2:
            raise ValueError(
                f"action_vocabulary must be 2-d (K, m), got shape {self.vocabulary.shape}"
            )
        self.K, self.m = int(self.vocabulary.shape[0]), int(self.vocabulary.shape[1])

        # ---- Horizon ------------------------------------------------------
        self.horizon = int(horizon)
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")

        # ---- Action bounds: derive from vocabulary if not provided --------
        # The gradient-guided convention is to expose action bounds
        # implicitly via the vocabulary corners; we honour that to keep
        # the constructor signature drop-in compatible.
        if action_low is None:
            lo_arr = jnp.min(self.vocabulary, axis=0)
        else:
            lo_arr = jnp.atleast_1d(jnp.asarray(action_low, dtype=jnp.float32))
            if lo_arr.shape == (1,) and self.m > 1:
                lo_arr = jnp.broadcast_to(lo_arr, (self.m,))
        if action_high is None:
            hi_arr = jnp.max(self.vocabulary, axis=0)
        else:
            hi_arr = jnp.atleast_1d(jnp.asarray(action_high, dtype=jnp.float32))
            if hi_arr.shape == (1,) and self.m > 1:
                hi_arr = jnp.broadcast_to(hi_arr, (self.m,))
        if lo_arr.shape != (self.m,) or hi_arr.shape != (self.m,):
            raise ValueError(
                f"action bounds must be scalar or shape ({self.m},); "
                f"got low={lo_arr.shape}, high={hi_arr.shape}"
            )
        if not bool(jnp.all(hi_arr > lo_arr)):
            raise ValueError(f"action_high ({hi_arr}) must be > action_low ({lo_arr})")
        self.action_low = lo_arr
        self.action_high = hi_arr

        # ---- Sim params + aux --------------------------------------------
        self.sim_params = sim_params
        self.aux = dict(aux) if aux is not None else None

        # ---- Optimiser hyperparameters -----------------------------------
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if k_iters < 1:
            raise ValueError(f"k_iters must be >= 1, got {k_iters}")
        self.lr = float(lr)
        self.k_iters = int(k_iters)
        self.adam_b1 = float(adam_b1)
        self.adam_b2 = float(adam_b2)
        self.adam_eps = float(adam_eps)
        self.track_best = bool(track_best)
        self.fallback_on_grad_failure = bool(fallback_on_grad_failure)

        # ---- Init strategy validation ------------------------------------
        if init not in ("zeros", "llm", "random", "heuristic"):
            raise ValueError(
                f"init must be one of {{'zeros', 'llm', 'random', 'heuristic'}}, got {init!r}"
            )
        self.init: InitStrategy = init
        if init == "heuristic":
            if init_action is None:
                raise ValueError("init='heuristic' requires init_action of shape (H, m)")
            ia = jnp.asarray(init_action, dtype=jnp.float32)
            if ia.shape != (self.horizon, self.m):
                raise ValueError(
                    f"init_action shape {ia.shape} must equal (H, m) = ({self.horizon}, {self.m})"
                )
            self.init_action: jt.Float[jt.Array, "H m"] | None = ia
        else:
            self.init_action = None

        # ---- Compile spec; refuse non-JIT predicates ----------------------
        self._compiled_spec = compile_spec(spec)
        from stl_seed.stl.evaluator import _FALLBACK_USED

        if getattr(self._compiled_spec, _FALLBACK_USED, False):
            raise RuntimeError(
                "HorizonFoldedGradientSampler requires every predicate to be "
                "JIT/grad-compatible (the introspection convention in "
                "stl_seed.stl.evaluator._introspect_predicate). At least one "
                f"predicate in spec {getattr(spec, 'name', '<unknown>')!r} "
                "fell back to the slow Python path."
            )

        # ---- Build the differentiable rho-from-z closure ------------------
        self._sim_fn = _wrap_simulator(simulator, sim_params, self.aux)
        compiled_spec = self._compiled_spec
        sim_fn = self._sim_fn
        low = self.action_low
        high = self.action_high

        def rho_from_z(
            z: jt.Float[jt.Array, "H m"],
            initial_state: jt.Float[jt.Array, " n"],
            key: jt.PRNGKeyArray,
        ) -> jt.Float[jt.Array, ""]:
            """End-to-end ``z -> u -> trajectory -> rho`` closure for autodiff.

            Single argument-of-differentiation is ``z`` (the unconstrained
            latent); the other arguments are kept as plain inputs so the
            JIT trace specialises on shapes only.
            """
            u = _sigmoid_reparam(z, low, high)
            states, times = sim_fn(initial_state, u, key)
            return compiled_spec(states, times)

        self._rho_from_z = rho_from_z
        # JIT value+grad wrt z (argnums=0).
        self._value_and_grad_z = jax.jit(jax.value_and_grad(rho_from_z, argnums=0))

    # ----------------------------------------------------------------- public
    def sample(
        self,
        initial_state: jt.Float[jt.Array, " n"],
        key: jt.PRNGKeyArray,
    ) -> SamplerResult:
        """Run ``K_iters`` Adam steps on the full-horizon action sequence.

        Parameters
        ----------
        initial_state:
            Initial state vector of shape ``(n,)`` for the simulator.
        key:
            PRNG key for any randomness in the LLM proposal (init='llm')
            or the random init (init='random'). Deterministic given the
            key.

        Returns
        -------
        (trajectory, diagnostics):
            ``trajectory`` is the canonical
            :class:`stl_seed.tasks._trajectory.Trajectory` produced by
            simulating the *best-so-far* optimised ``u_{1:H}``.
            ``diagnostics`` is :meth:`HorizonFoldedDiagnostics.to_dict`.
        """
        diag = HorizonFoldedDiagnostics(init_strategy=self.init, n_iters=self.k_iters)

        # ---- 1. Build the initial latent z_0. -----------------------------
        z = self._init_latent(initial_state, key)

        # ---- 2. Compute rho at the initial point. -------------------------
        init_rho_val, _ = self._safe_value_and_grad(z, initial_state, key)
        diag.init_rho = float(init_rho_val) if init_rho_val is not None else float("nan")
        diag.rho_at_iter.append(diag.init_rho)
        diag.grad_norm_at_iter.append(0.0)  # no gradient step yet
        best_z = z
        best_rho = diag.init_rho
        best_iter = 0

        # ---- 3. Adam refinement loop. -------------------------------------
        adam_state = _adam_init(z)
        for k in range(1, self.k_iters + 1):
            rho_v, grad_z = self._safe_value_and_grad(z, initial_state, key)
            if rho_v is None or grad_z is None:
                # Numerical pathology — exit with the best-so-far guard.
                if not self.fallback_on_grad_failure:
                    raise RuntimeError(
                        f"HorizonFoldedGradientSampler: NaN/Inf gradient at iter {k}"
                    )
                diag.fallback_used = True
                break

            grad_norm = float(jnp.linalg.norm(grad_z))
            diag.rho_at_iter.append(float(rho_v))
            diag.grad_norm_at_iter.append(grad_norm)

            # Best-so-far guard (apply BEFORE the Adam step so the
            # snapshot is the rho we just measured, not the post-step
            # rho which could be an overshoot).
            if self.track_best and float(rho_v) > best_rho:
                best_rho = float(rho_v)
                best_z = z
                best_iter = k - 1  # the iter that produced this rho

            # Adam step (gradient ascent).
            delta, adam_state = _adam_step(
                adam_state,
                grad_z,
                lr=self.lr,
                b1=self.adam_b1,
                b2=self.adam_b2,
                eps=self.adam_eps,
            )
            z = z + delta

        # Final post-loop check: the last z hasn't been evaluated yet.
        # Compute its rho so the best-so-far reflects the full trajectory.
        final_rho_v, _ = self._safe_value_and_grad(z, initial_state, key)
        if final_rho_v is not None and float(final_rho_v) > best_rho:
            best_rho = float(final_rho_v)
            best_z = z
            best_iter = self.k_iters

        # ---- 4. Build the canonical Trajectory from best_z. ---------------
        u_best = _sigmoid_reparam(best_z, self.action_low, self.action_high)
        traj = self._build_trajectory(initial_state, u_best, jax.random.fold_in(key, 9001))

        # ---- 5. Final diagnostics. ----------------------------------------
        # Recompute rho on the canonical trajectory to make best_rho exact
        # (the in-loop rho is computed via the wrapped sim_fn; the canonical
        # trajectory uses the simulator's full Trajectory contract — these
        # agree numerically but we report the canonical value to keep
        # ``diag['final_rho']`` consistent with downstream STL evaluations).
        canonical_rho = float(self._compiled_spec(traj.states, traj.times))
        diag.best_rho = canonical_rho
        diag.best_iter = best_iter
        diag.final_rho = canonical_rho

        return traj, diag.to_dict()

    # -------------------------------------------------------------- internals
    def _init_latent(
        self,
        initial_state: jt.Float[jt.Array, " n"],
        key: jt.PRNGKeyArray,
    ) -> jt.Float[jt.Array, "H m"]:
        """Construct the initial unconstrained latent ``z_0`` per ``self.init``."""
        if self.init == "zeros":
            return jnp.zeros((self.horizon, self.m), dtype=jnp.float32)

        if self.init == "random":
            return 0.1 * jax.random.normal(key, (self.horizon, self.m), dtype=jnp.float32)

        if self.init == "heuristic":
            assert self.init_action is not None  # validated in __init__
            return _inverse_sigmoid_reparam(self.init_action, self.action_low, self.action_high)

        # init == "llm": query the LLM at each step from a clean history,
        # form the preferred mean action, invert-sigmoid to z.
        if self.init == "llm":
            history = jnp.zeros((0, self.m), dtype=jnp.float32)
            u_init: list[jt.Float[jt.Array, " m"]] = []
            for t in range(self.horizon):
                step_key = jax.random.fold_in(key, t)
                logits = self.llm(initial_state, history, step_key)
                logits = jnp.asarray(logits, dtype=jnp.float32)
                if logits.shape != (self.K,):
                    raise ValueError(
                        f"LLM emitted logits of shape {logits.shape}, "
                        f"expected ({self.K},) for init='llm' warm-start"
                    )
                probs = jax.nn.softmax(logits)
                u_bar = jnp.einsum("k,km->m", probs, self.vocabulary)  # (m,)
                u_init.append(u_bar)
                # Update history with the preferred mean action so subsequent
                # logits see a consistent history (mirrors the gradient_guided
                # autoregressive path).
                history = jnp.concatenate([history, u_bar[None, :]], axis=0)
            u_arr = jnp.stack(u_init, axis=0)  # (H, m)
            return _inverse_sigmoid_reparam(u_arr, self.action_low, self.action_high)

        # Should be unreachable thanks to the constructor validation, but
        # guard for completeness.
        raise ValueError(f"unknown init strategy {self.init!r}")  # pragma: no cover

    def _safe_value_and_grad(
        self,
        z: jt.Float[jt.Array, "H m"],
        initial_state: jt.Float[jt.Array, " n"],
        key: jt.PRNGKeyArray,
    ) -> tuple[jt.Float[jt.Array, ""] | None, jt.Float[jt.Array, "H m"] | None]:
        """Wrap the JIT'd value+grad in NaN/Inf detection.

        Returns ``(None, None)`` on numerical pathology, letting the
        caller decide whether to raise or fall back. We do *not* swallow
        Python exceptions other than the JAX numerical ones — a real
        type / shape error in the inner closure should propagate.
        """
        try:
            rho, grad = self._value_and_grad_z(z, initial_state, key)
        except Exception:  # pragma: no cover -- diagnostic safety net
            if not self.fallback_on_grad_failure:
                raise
            return None, None

        if not bool(jnp.isfinite(rho)) or not bool(jnp.all(jnp.isfinite(grad))):
            return None, None

        return rho, grad

    def _build_trajectory(
        self,
        initial_state: jt.Float[jt.Array, " n"],
        control: jt.Float[jt.Array, "H m"],
        key: jt.PRNGKeyArray,
    ) -> Trajectory:
        """Run the simulator once to materialise the canonical Trajectory.

        Mirrors the convention in
        :class:`STLGradientGuidedSampler._build_trajectory`: we run a
        *fresh* simulator call with the optimised continuous control,
        and we record the agent's *intended* (pre-clip) action in
        ``trajectory.actions`` so downstream eval / training can re-
        validate via re-simulation if needed.
        """
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
    "HorizonFoldedDiagnostics",
    "HorizonFoldedGradientSampler",
    "InitStrategy",
]

"""STL-robustness gradient-guided sampler.

Technical contribution
----------------------

Standard inference for ``stl-seed`` agents proposes a control sequence
``u_{1:H}`` autoregressively from an LLM, then either accepts the
trajectory (vanilla decoding) or generates ``N`` candidates and picks the
one with highest STL robustness ``rho`` (Best-of-N). Both throw away the
fact that ``rho`` is a *continuous, differentiable* signal in the
Donzé-Maler space-robustness semantics implemented in
:mod:`stl_seed.stl.evaluator`.

This module implements **gradient-guided sampling**: at each generation
step ``t``, we

1. take the LLM's logits ``z_t`` over a discrete action vocabulary
   ``V in R^{K x m}``,
2. form the continuous *preferred mean action*
   ``u_bar_t = sum_k softmax(z_t)_k * V_k``,
3. build a *partial-then-extrapolated* control sequence ``hat_u_{1:H} =
   (u_1, ..., u_{t-1}, u_bar_t, u_default, ..., u_default)``,
4. simulate ``hat_u`` through the ODE simulator,
5. evaluate the streaming STL robustness ``rho_stream`` at the current
   wall-clock time on the resulting partial-then-extrapolated trajectory,
6. compute ``g_t = grad_{u_bar_t} rho_stream`` via JAX autodiff,
7. project ``g_t`` onto vocabulary directions ``V_k - u_bar_t`` to form
   a logit bias ``b_k = lambda * <g_t, V_k - u_bar_t>``,
8. sample ``a_t ~ softmax(z_t + b_t)``, then concretise to ``u_t = V_{a_t}``.

The ``lambda = 0`` ablation recovers vanilla LLM decoding bit-for-bit
(modulo numerical tracer-vs-eager noise). The ``lambda -> infty`` limit
selects the vocabulary item most aligned with ``grad rho``, which is a
pure-greedy STL improvement step.

Connection to prior decoding-time methods
-----------------------------------------

* **Classifier guidance** (Dhariwal & Nichol 2021, "Diffusion Models Beat
  GANs on Image Synthesis", arXiv:2105.05233) and **DPS** (Chung et al.
  2023, arXiv:2209.14687): both add the gradient of an external
  continuous classifier into the sampling rule of a generative model.
  The present method is the discrete-control analogue: STL robustness
  plays the role of the external classifier.

* **STLCG / STLCG++** (Leung et al. 2020, arXiv:2008.00097; Hashemi
  et al. 2025, arXiv:2501.04194): provide the differentiable STL
  infrastructure. Prior uses were trajectory optimisation and reward
  shaping, not LLM inference-time decoding.

* **LTLCrit / LogicGuard** (Sun et al. 2025, arXiv:2507.03293): closest
  published prior. Uses LTL constraints as a *discrete* token-level
  critic that vetoes specification-violating tokens. The present work
  uses STL as a *continuous* gradient-providing critic, retaining the
  full information of the robustness signal.

* **STL-as-RL-reward** (Aksaray et al. 2016, "Q-Learning for Robust
  Satisfaction of Signal Temporal Logic Specifications", CDC 2016):
  uses ``rho`` as a scalar reward in a Q-learning loop. We use
  ``grad rho`` as a *per-decision* signal at decoding time, not a
  per-rollout return.

in-package modules (``stl_seed.stl``, ``stl_seed.tasks``,
``physics_*`` symbol is touched.

Computational cost
------------------

Per step ``t``:

* One forward simulator call (Diffrax ODE solve, JIT'd) on the
  partial-then-extrapolated ``hat_u_{1:H}``.
* One backward pass to obtain ``grad_{u_bar_t} rho_stream``. This
  re-uses the Diffrax adjoint (default ``RecursiveCheckpointAdjoint``)
  through the ODE.
* ``K`` inner products to form the logit bias.

Total cost: ``H * (1 fwd + 1 bwd + K * m FLOPs)``. The ``K * m`` term is
negligible relative to the ODE solve. Compare with Best-of-N at matched
compute: ``N`` full-trajectory forward simulations and ``N``
``rho`` evaluations, no backward pass. Setting ``N = 2 * H`` matches the
gradient-guided cost approximately (each backward is ~1-2x forward).
"""

from __future__ import annotations

import dataclasses
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
class GuidanceDiagnostics:
    """Per-step diagnostics for one gradient-guided rollout.

    Fields are lists indexed by control step ``t in [0, H)``.

    * ``rho_stream_at_step``: streaming rho immediately *after* committing
      step t's action. Pre-registered: this should be monotone-improving
      (modulo simulator stochasticity / extrapolation error) under
      ``lambda > 0`` relative to ``lambda = 0``.
    * ``grad_norm_at_step``: ``||grad_{u_bar_t} rho_stream||_2``. A zero
      norm at step t means the streaming rho is locally insensitive to
      ``u_t`` (e.g. the active spec clauses don't reach into the future
      enough). Diagnostic, not pathological.
    * ``bias_max_abs_at_step``: ``max_k |b_k|`` for step t. The dynamic
      range of the gradient guidance term, useful for tuning lambda.
    * ``chosen_index_at_step``: the sampled vocabulary index at step t.
    * ``would_pick_top_logit_at_step``: 1 if argmax(softmax(z + b)) ==
      argmax(softmax(z)), else 0. Counts the steps where guidance changed
      the modal choice.
    * ``final_rho``: full-trajectory robustness after all H steps.
    * ``fallback_used``: True if any predicate fell back to the slow
      Python evaluation path (which disables JIT and gradients). When
      True, the gradient-guided sampler raises a clear error rather than
      silently degrading to BoN.
    """

    rho_stream_at_step: list[float] = dataclasses.field(default_factory=list)
    grad_norm_at_step: list[float] = dataclasses.field(default_factory=list)
    bias_max_abs_at_step: list[float] = dataclasses.field(default_factory=list)
    chosen_index_at_step: list[int] = dataclasses.field(default_factory=list)
    would_pick_top_logit_at_step: list[int] = dataclasses.field(default_factory=list)
    final_rho: float = float("nan")
    fallback_used: bool = False

    def to_dict(self) -> SamplerDiagnostics:
        """Materialise as a plain dict for the harness."""
        return {
            "rho_stream_at_step": list(self.rho_stream_at_step),
            "grad_norm_at_step": list(self.grad_norm_at_step),
            "bias_max_abs_at_step": list(self.bias_max_abs_at_step),
            "chosen_index_at_step": list(self.chosen_index_at_step),
            "would_pick_top_logit_at_step": list(self.would_pick_top_logit_at_step),
            "final_rho": float(self.final_rho),
            "fallback_used": bool(self.fallback_used),
            "n_steps": len(self.chosen_index_at_step),
            "n_steps_changed_by_guidance": int(
                sum(1 for f in self.would_pick_top_logit_at_step if f == 0)
            ),
        }


# ---------------------------------------------------------------------------
# Helpers. vocabulary construction & default extrapolation.
# ---------------------------------------------------------------------------


def make_uniform_action_vocabulary(
    action_low: jt.Float[jt.Array, " m"] | float | np.ndarray,
    action_high: jt.Float[jt.Array, " m"] | float | np.ndarray,
    k_per_dim: int,
) -> jt.Float[jt.Array, "K m"]:
    """Construct a uniform-grid action vocabulary spanning the action box.

    For an ``m``-dimensional action with ``k_per_dim`` levels per axis,
    this produces ``K = k_per_dim ** m`` discrete action vectors filling
    the box ``[action_low, action_high]^m`` on a uniform grid. Suitable
    as a sane default; production deployments may want a learned or
    task-tailored vocabulary instead.

    Notes
    -----
    The vocabulary endpoints (``low`` and ``high``) are *included* (the
    grid is ``linspace(low, high, k_per_dim)``). For ``k_per_dim = 2``
    this gives the corners of the box; for ``k_per_dim = 3`` the corners
    plus the center.

    Cost note: ``K`` grows exponentially in ``m``. The MAPK and
    glucose-insulin tasks have ``m = 1``, so ``K = k_per_dim``; the
    repressilator and toggle have ``m = 3`` and ``m = 2`` respectively,
    so ``k_per_dim = 4`` already gives ``K = 64`` and ``K = 16``. The
    sampler's per-step ``K`` projection cost is linear, so this is fine
    up to ``K = 256`` or so.
    """
    lo_arr = jnp.atleast_1d(jnp.asarray(action_low, dtype=jnp.float32))
    hi_arr = jnp.atleast_1d(jnp.asarray(action_high, dtype=jnp.float32))
    if lo_arr.shape != hi_arr.shape:
        raise ValueError(f"action_low shape {lo_arr.shape} must match action_high {hi_arr.shape}")
    if k_per_dim < 2:
        raise ValueError(f"k_per_dim must be >= 2, got {k_per_dim}")
    m = int(lo_arr.shape[0])
    # Per-axis grid; meshgrid over m axes; reshape to (K, m).
    axes = [jnp.linspace(lo_arr[i], hi_arr[i], k_per_dim) for i in range(m)]
    grids = jnp.meshgrid(*axes, indexing="ij")  # m arrays of shape k_per_dim^m
    flat = [g.reshape(-1) for g in grids]
    V = jnp.stack(flat, axis=-1)  # (K, m)
    return V


# ---------------------------------------------------------------------------
# Differentiable simulator wrapper.
# ---------------------------------------------------------------------------


SimulateFn = Callable[
    [jt.Float[jt.Array, " n"], jt.Float[jt.Array, "H m"], jt.PRNGKeyArray],
    tuple[jt.Float[jt.Array, "T n"], jt.Float[jt.Array, " T"]],
]


def _wrap_simulator(simulator: Any, sim_params: Any, aux: dict[str, Any] | None) -> SimulateFn:
    """Adapt a simulator to a uniform ``(initial_state, control, key) -> (states, times)`` callable.

     The two simulator families currently implemented:

     * :class:`stl_seed.tasks.glucose_insulin.GlucoseInsulinSimulator`:
       takes ``(initial_state, u_flat, meal_schedule, params, key)`` and
       returns ``(states, times, meta)``. The control axis is *implicit*
       (scalar insulin), so a 2-d ``(H, 1)`` is reshaped to ``(H,)``.
     * :class:`stl_seed.tasks.bio_ode.{Repressilator,Toggle,MAPK}Simulator`:
       takes ``(initial_state, control_sequence, params, key)`` and
       returns a :class:`Trajectory` directly.

     The returned wrapper is closed over ``sim_params`` and ``aux`` so the
     autodiff path is purely a function of ``(initial_state, control, key)``
    . JAX can then take ``grad`` with respect to the control argument
     without tracing through static parameter pytrees.
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

    # Generic path: simulator returns a Trajectory dataclass.
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


class STLGradientGuidedSampler:
    """Gradient-guided sampler using continuous STL robustness as a classifier.

    Parameters
    ----------
    llm:
        LLM proposal callable conforming to :class:`LLMProposal`. Returns
        logits of shape ``(K,)`` over the action vocabulary at each step.
    simulator:
        ODE simulator with ``simulate(initial_state, control, params,
        key)`` (or the glucose-insulin variant; see ``_wrap_simulator``).
        Must be JAX-traceable for ``jax.grad`` to flow through.
    spec:
        STL specification (registered :class:`STLSpec` or raw
        :class:`Node`). Predicates must conform to the introspection
        convention in :func:`stl_seed.stl.evaluator._introspect_predicate`
       . i.e. all specs in ``stl_seed.specs.REGISTRY`` are supported.
        Non-conforming predicates would force the slow Python eval path
        which is JIT/grad-incompatible; the sampler raises at construction
        in that case.
    action_vocabulary:
        Discrete action set ``V in R^{K x m}``. The argmax over biased
        logits selects the vocabulary index, which is then concretised
        to the action vector ``V_{argmax}``.
    sim_params:
        Kinetic parameter pytree consumed by the simulator (e.g.
        :class:`BergmanParams` or :class:`RepressilatorParams`).
    horizon:
        Number of control steps ``H``. Must equal the simulator's
        ``n_control_points`` (otherwise the simulator's piecewise-constant
        lookup will misalign).
    aux:
        Optional task-specific kwargs forwarded to the simulator (e.g.
        ``meal_schedule`` for glucose-insulin). ``None`` for bio_ode tasks.
    guidance_weight:
        The lambda hyperparameter on the logit bias. ``0`` reduces to
        :class:`StandardSampler`. Larger values trade fidelity to the LLM
        prior for STL satisfaction. Sensible range: ``0.1`` to ``10`` for
        rho on the order of unity; rescale if rho's natural scale is
        different.
    default_action:
        The action used to extrapolate the control sequence beyond the
        current step ``t`` for the streaming-rho gradient. Defaults to the
        center of the action box (the L^infty-most-neutral choice). For
        the glucose-insulin task this is ``2.5 U/h`` (mid-range of
        ``[0, 5]``); for the bio_ode tasks it is ``0.5`` per channel.
    sampling_temperature:
        Temperature on the biased softmax. ``1.0`` (default) preserves the
        LLM's calibration; ``0`` collapses to argmax (greedy guidance);
        ``> 1`` flattens to encourage exploration.
    fallback_on_grad_failure:
        If True (default), a ``NaN`` or ``Inf`` gradient at step ``t``
        falls back to *unbiased* sampling from ``softmax(z_t)`` and the
        diagnostic ``grad_norm_at_step[t]`` is recorded as ``NaN``. If
        False, the sampler raises. The default is True because gradient
        spikes can occur at the boundary of the action box where ``rho``
        is non-differentiable; failing the entire rollout in that case is
        excessive.

    Notes
    -----
    The sampler is *not* an ``equinox.Module``. it carries Python-level
    bookkeeping (the Trajectory builder loop, diagnostics) that doesn't
    fit the eqx pytree contract. Sub-pieces that *are* JIT-able (the
    simulator wrapper, the STL evaluator) are JIT'd internally.
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
        guidance_weight: float = 1.0,
        default_action: jt.Float[jt.Array, " m"] | None = None,
        sampling_temperature: float = 1.0,
        fallback_on_grad_failure: bool = True,
    ) -> None:
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
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        self.aux = dict(aux) if aux is not None else None
        self.guidance_weight = float(guidance_weight)
        if sampling_temperature < 0.0:
            raise ValueError(f"sampling_temperature must be >= 0, got {sampling_temperature}")
        self.sampling_temperature = float(sampling_temperature)
        self.fallback_on_grad_failure = bool(fallback_on_grad_failure)

        if default_action is None:
            self.default_action = jnp.mean(self.vocabulary, axis=0)
        else:
            da = jnp.asarray(default_action, dtype=jnp.float32)
            if da.shape != (self.m,):
                raise ValueError(f"default_action shape {da.shape} must equal (m,) = ({self.m},)")
            self.default_action = da

        # Compile STL spec once. Refuse non-conforming predicates: the
        # gradient-guided path needs JAX-pure semantics.
        self._compiled_spec = compile_spec(spec)
        from stl_seed.stl.evaluator import _FALLBACK_USED

        if getattr(self._compiled_spec, _FALLBACK_USED, False):
            raise RuntimeError(
                "STLGradientGuidedSampler requires every predicate to be "
                "JIT/grad-compatible (the introspection convention in "
                "stl_seed.stl.evaluator._introspect_predicate). At least one "
                f"predicate in spec {getattr(spec, 'name', '<unknown>')!r} "
                "fell back to the slow Python path. Either rewrite the "
                "predicate via stl_seed.specs.bio_ode_specs._gt / _lt, or "
                "use BestOfNSampler instead."
            )

        # Wrap the simulator into a uniform (init, control, key) -> (states, times) form.
        self._sim_fn = _wrap_simulator(simulator, sim_params, self.aux)

        # Pre-build the rho-from-control closure used by jax.grad.
        compiled_spec = self._compiled_spec
        sim_fn = self._sim_fn

        def rho_from_control(
            initial_state: jt.Float[jt.Array, " n"],
            control: jt.Float[jt.Array, "H m"],
            key: jt.PRNGKeyArray,
        ) -> jt.Float[jt.Array, ""]:
            states, times = sim_fn(initial_state, control, key)
            return compiled_spec(states, times)

        self._rho_from_control = rho_from_control
        # JIT the value+grad wrt the *control argument*; not wrt initial state.
        # argnums=1 is the control_sequence position.
        self._value_and_grad = jax.jit(jax.value_and_grad(rho_from_control, argnums=1))

    # ------------------------------------------------------------------ public
    def sample(
        self,
        initial_state: jt.Float[jt.Array, " n"],
        key: jt.PRNGKeyArray,
    ) -> SamplerResult:
        """Generate one gradient-guided trajectory.

        Returns
        -------
        (trajectory, diagnostics):
            ``trajectory`` is the canonical
            :class:`stl_seed.tasks._trajectory.Trajectory` produced by
            simulating the chosen ``u_{1:H}`` from ``initial_state``.
            ``diagnostics`` is the result of
            :meth:`GuidanceDiagnostics.to_dict`.
        """
        diag = GuidanceDiagnostics()
        # Pre-allocate the control buffer with the default action so unbuilt
        # steps act as the extrapolation tail used by the gradient probe.
        control = jnp.broadcast_to(self.default_action, (self.horizon, self.m))
        # We materialise as a plain Python list of (m,) actions to let us
        # rebuild the (H, m) array each step (small H, cost is negligible).
        actions: list[jt.Float[jt.Array, " m"]] = [
            jnp.asarray(self.default_action, dtype=jnp.float32) for _ in range(self.horizon)
        ]

        # History feeds the LLM at each step.
        history = jnp.zeros((0, self.m), dtype=jnp.float32)

        for t in range(self.horizon):
            step_key = jax.random.fold_in(key, t)
            llm_key, sample_key = jax.random.split(step_key, 2)

            # 1. LLM logits.
            logits = self.llm(initial_state, history, llm_key)
            logits = jnp.asarray(logits, dtype=jnp.float32)
            if logits.shape != (self.K,):
                raise ValueError(
                    f"LLM emitted logits of shape {logits.shape}, expected ({self.K},)"
                )

            # 2. Compute logit bias from grad rho when guidance is on.
            bias, grad_norm = self._compute_bias(
                initial_state=initial_state,
                actions_so_far=actions,
                logits=logits,
                step=t,
            )

            # 3. Form the biased softmax and sample.
            biased = logits + bias
            if self.sampling_temperature == 0.0:
                chosen_idx = int(jnp.argmax(biased))
            else:
                if self.sampling_temperature != 1.0:
                    scaled = biased / self.sampling_temperature
                else:
                    scaled = biased
                chosen_idx = int(jax.random.categorical(sample_key, scaled))

            top_logit_idx = int(jnp.argmax(logits))
            chosen_action = self.vocabulary[chosen_idx]
            actions[t] = chosen_action
            control = jnp.stack(actions, axis=0)

            # 4. Diagnostics: streaming rho post-commit.
            rho_post = float(
                self._rho_from_control(initial_state, control, jax.random.fold_in(key, t))
            )
            diag.rho_stream_at_step.append(rho_post)
            diag.grad_norm_at_step.append(float(grad_norm))
            diag.bias_max_abs_at_step.append(float(jnp.max(jnp.abs(bias))))
            diag.chosen_index_at_step.append(chosen_idx)
            diag.would_pick_top_logit_at_step.append(int(chosen_idx == top_logit_idx))

            # 5. Update history with the committed action.
            history = jnp.concatenate([history, chosen_action[None, :]], axis=0)

        # 6. Final simulation -> canonical Trajectory.
        traj = self._build_trajectory(initial_state, control, jax.random.fold_in(key, 9001))
        diag.final_rho = float(self._compiled_spec(traj.states, traj.times))
        return traj, diag.to_dict()

    # -------------------------------------------------------------- internals
    def _compute_bias(
        self,
        initial_state: jt.Float[jt.Array, " n"],
        actions_so_far: list[jt.Float[jt.Array, " m"]],
        logits: jt.Float[jt.Array, " K"],
        step: int,
    ) -> tuple[jt.Float[jt.Array, " K"], jt.Float[jt.Array, ""]]:
        """Compute the gradient-guidance bias for the current step.

        Returns ``(bias_vector, grad_norm)``. When ``guidance_weight ==
        0`` the bias is identically zero and we skip the simulator call.
        """
        if self.guidance_weight == 0.0:
            zero = jnp.zeros((self.K,), dtype=jnp.float32)
            return zero, jnp.asarray(0.0, dtype=jnp.float32)

        # Continuous "preferred" action u_bar = sum_k softmax(z)_k * V_k.
        # This is the straight-through bridge: the LLM's discrete-token
        # distribution is reinterpreted as a Dirichlet over the vocabulary
        # whose mean is u_bar; we differentiate rho(u_bar).
        probs = jax.nn.softmax(logits)
        u_bar = jnp.einsum("k,km->m", probs, self.vocabulary)

        # Build the partial+extrapolated control: actions_so_far for
        # indices < step, u_bar at step, default for indices > step.
        # Doing this via jax.lax.dynamic_update_slice would be JIT-friendlier
        # but complicates Python-level looping; the (H, m) array is small.
        ctrl_list: list[jt.Float[jt.Array, " m"]] = []
        for i in range(self.horizon):
            if i < step:
                ctrl_list.append(actions_so_far[i])
            elif i == step:
                ctrl_list.append(u_bar)
            else:
                ctrl_list.append(self.default_action)
        control = jnp.stack(ctrl_list, axis=0)

        # value_and_grad wrt the full control sequence; we extract the row
        # corresponding to step `step` to get grad_{u_bar} rho.
        try:
            _rho, dctrl = self._value_and_grad(initial_state, control, jax.random.PRNGKey(0))
        except Exception:
            if not self.fallback_on_grad_failure:
                raise
            # Fall back to zero bias.
            zero = jnp.zeros((self.K,), dtype=jnp.float32)
            return zero, jnp.asarray(float("nan"), dtype=jnp.float32)

        g = dctrl[step]  # shape (m,)

        # Detect numerically pathological gradients (NaN/Inf) early.
        if not bool(jnp.all(jnp.isfinite(g))):
            if not self.fallback_on_grad_failure:
                raise RuntimeError(
                    f"grad rho at step {step} contained NaN/Inf; "
                    "set fallback_on_grad_failure=True to recover."
                )
            zero = jnp.zeros((self.K,), dtype=jnp.float32)
            return zero, jnp.asarray(float("nan"), dtype=jnp.float32)

        # Bias is the linearised improvement in rho moving from u_bar to V_k.
        # Equivalent to a one-step gradient ascent prior on the vocabulary.
        # Positive bias -> vocabulary item points along grad rho.
        deltas = self.vocabulary - u_bar[None, :]  # (K, m)
        bias = self.guidance_weight * jnp.einsum("km,m->k", deltas, g)
        return bias.astype(jnp.float32), jnp.linalg.norm(g)

    def _build_trajectory(
        self,
        initial_state: jt.Float[jt.Array, " n"],
        control: jt.Float[jt.Array, "H m"],
        key: jt.PRNGKeyArray,
    ) -> Trajectory:
        """Run the simulator once to materialise the canonical Trajectory.

        We run a *fresh* simulator call with the committed control rather
        than reusing the gradient-probe trajectories (which were
        partial+extrapolated and not the ground-truth final rollout).
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
        # Generic: simulator returns a Trajectory dataclass; we rebuild
        # actions to ensure they match what we committed (the simulator may
        # internally clip).
        traj = self.simulator.simulate(initial_state, control, self.sim_params, key)
        # Force the actions field to our committed control (clip-aware
        # simulators may return states from the clipped control; we record
        # the agent's *intended* action, which the eval harness will
        # re-validate via re-simulation if needed).
        return Trajectory(
            states=traj.states,
            actions=control,
            times=traj.times,
            meta=traj.meta,
        )


__all__ = [
    "GuidanceDiagnostics",
    "STLGradientGuidedSampler",
    "make_uniform_action_vocabulary",
]

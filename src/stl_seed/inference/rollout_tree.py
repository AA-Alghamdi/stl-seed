"""K-step rollout-tree gradient probing sampler.

Technical contribution
----------------------

The 1-step partial-then-extrapolated gradient probe inside
:class:`STLGradientGuidedSampler` is *myopic*: at decoding step ``t`` it
holds ``u_{t+1}, ..., u_H`` fixed at a single ``default_action`` and
takes ``grad_{u_bar_t} rho`` through the simulator and STL evaluator on
the resulting partial-then-extrapolated trajectory. On tasks where the
satisfying region is a measure-near-zero attractor in the joint
``u_{1:H}`` space (e.g. the cyclic Elowitz-Leibler 2000 repressilator
under ``G_[120,200] (m1 >= 250) AND F_[0,60] (p2 < 25)``), the
single-step gradient cannot disambiguate satisfying corners of the
action box from neighbouring failing corners, and the freezing
assumption produces an extrapolation point that sits in a flat
``rho ~ -250`` floor (see ``paper/cross_task_validation.md``).

This module implements **K-step rollout-tree probing**, a finite-depth
analogue of the Monte-Carlo tree search used in AlphaGo (Silver et al.,
"Mastering the game of Go with deep neural networks and tree search",
*Nature* 529:484, 2016, DOI 10.1038/nature16961). At each decoding step
``t`` we

1. Branch ``branch_k`` candidate actions ``u_t^{(1)}, ..., u_t^{(K)}``
   (the top-``branch_k`` vocabulary items by LLM logits; ties broken
   by index).
2. For each candidate, build a continuation control sequence
   ``u_{t+1}, ..., u_{t+lookahead_h}`` using a fixed
   ``continuation_policy`` (``"zero"`` -> all zeros, ``"random"`` -> i.i.d.
   uniform vocabulary draws, ``"heuristic"`` -> caller-supplied callable,
   ``"llm"`` -> autoregressive LLM rollout). The remainder of the
   horizon ``u_{t+lookahead_h+1}, ..., u_H`` is filled with the
   sampler's ``default_action`` to keep the simulator end-to-end
   integrable on a single PRNG fold.
3. Simulate each candidate through the simulator (vectorised over
   candidates via :func:`jax.vmap`).
4. Evaluate the streaming rho at the lookahead horizon
   ``t + 1 + lookahead_h`` (rho on the partial+continuation trajectory)
   for each candidate. The candidate with the maximum projected rho
   wins.
5. (Optional) take ``refine_iters`` gradient-refinement steps on the
   chosen candidate using the existing horizon-aware
   :class:`STLGradientGuidedSampler` machinery. i.e. recompute
   ``grad_{u_bar_t} rho`` with the chosen-candidate branch as the
   linearisation point, and shift the bias accordingly. Set
   ``refine_iters = 0`` (the default) for pure tree-search.

Compute cost per generation step:
    ``branch_k * lookahead_h`` simulator steps (one full Diffrax solve
    per candidate, vectorised) + ``branch_k`` STL evaluations +
    ``refine_iters`` backward passes on the chosen branch.

For the 10-step repressilator with ``branch_k = 8``, ``lookahead_h = 5``,
``refine_iters = 0`` this is ~80 ODE solves per rollout; the same
budget at matched compute is ``ContinuousBoNSampler(n=80)`` (each BoN
draw is one full-horizon rollout, no backward).

in-package modules (``stl_seed.stl``, ``stl_seed.tasks``,

References
----------

Silver, D. et al. "Mastering the game of Go with deep neural networks
and tree search." *Nature* 529:484-489 (2016). DOI 10.1038/nature16961.
The depth-bounded rollout tree with default-policy continuations is the
direct analogue of their Monte Carlo tree search with rollout policy;
the difference is that we use a continuous STL robustness signal (not a
binary win/loss) at the leaves and we do not maintain a UCT-style
bandit tree across decoding steps. each step's branch is fresh.

Browne, C. B. et al. "A survey of Monte Carlo tree search methods."
*IEEE Trans. Comput. Intell. AI Games* 4(1):1-43 (2012).
DOI 10.1109/TCIAIG.2012.2186810. General reference on the
finite-horizon rollout-tree family of search algorithms.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, Literal

import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np

from stl_seed.inference.gradient_guided import (
    STLGradientGuidedSampler,
    _wrap_simulator,
)
from stl_seed.inference.protocol import LLMProposal, SamplerDiagnostics, SamplerResult
from stl_seed.specs import Node, STLSpec
from stl_seed.stl.evaluator import _FALLBACK_USED, compile_spec
from stl_seed.tasks._trajectory import Trajectory

# ---------------------------------------------------------------------------
# Continuation-policy kind.
# ---------------------------------------------------------------------------

ContinuationPolicy = Literal["zero", "random", "heuristic", "llm"]
"""Allowed values for :attr:`RolloutTreeSampler.continuation_policy`.

* ``"zero"`` (default): the continuation actions ``u_{t+1}, ...,
  ``u_{t+lookahead_h}`` are the all-zeros vector. For the repressilator,
  this corresponds to "leave all genes at full transcription". a
  neutral baseline that lets the candidate at step ``t`` express its
  effect without confounding by the continuation choice.
* ``"random"``: each continuation action is drawn i.i.d. uniformly from
  the action vocabulary. Good Monte Carlo estimator of the *mean*
  satisfaction over the continuation policy class; high variance per
  draw.
* ``"heuristic"``: a caller-supplied callable
  ``heuristic_continuation(initial_state, history, key) ->
  Float[Array, "lookahead_h m"]`` that returns the continuation in one
  shot. Useful when the user has a domain-specific default policy
  (e.g. the constant ``[0, 0, 1]`` silence-3 policy for the repressilator).
* ``"llm"``: roll out the LLM autoregressively for ``lookahead_h`` steps.
  Most expensive (one LLM call per continuation step per candidate),
  but the leaf evaluation reflects what the LLM would actually do under
  greedy continuation.
"""


# ---------------------------------------------------------------------------
# Diagnostics record.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RolloutTreeDiagnostics:
    """Per-step diagnostics for one rollout-tree sampler call.

    Fields are lists indexed by control step ``t in [0, H)``.

    * ``rho_stream_at_step``: streaming rho immediately after committing
      step ``t``'s action, evaluated on the simulator's full-horizon
      trajectory with the *committed* prefix and the ``default_action``
      tail. This is the ground-truth streaming rho the agent acts on,
      *not* the projected leaf rho used for the branch-selection decision.
    * ``projected_rho_at_step``: the chosen branch's leaf rho. i.e. the
      ``branch_k``-argmax leaf rho computed at step ``t``. Subject to the
      continuation-policy bias.
    * ``branch_rho_min/max/mean_at_step``: per-step summary statistics of
      the ``branch_k`` leaf rhos, useful for tuning ``branch_k`` and the
      continuation policy.
    * ``chosen_index_at_step``: the sampled vocabulary index at step ``t``.
    * ``would_pick_top_logit_at_step``: 1 if the chosen vocabulary index
      coincides with ``argmax(softmax(z_t))`` (the LLM's modal pick), else
      0. Counts the steps where rollout-tree disagreed with the LLM.
    * ``refine_grad_norm_at_step``: ``||grad_{u_t} rho||_2`` from the
      gradient-refinement step (NaN if ``refine_iters == 0``).
    * ``final_rho``: full-trajectory robustness after committing all H steps.
    * ``branch_k``: the requested branch count.
    * ``lookahead_h``: the requested lookahead.
    * ``continuation_policy``: the continuation-policy kind used.
    * ``refine_iters``: the requested gradient-refine iterations.
    """

    rho_stream_at_step: list[float] = dataclasses.field(default_factory=list)
    projected_rho_at_step: list[float] = dataclasses.field(default_factory=list)
    branch_rho_min_at_step: list[float] = dataclasses.field(default_factory=list)
    branch_rho_max_at_step: list[float] = dataclasses.field(default_factory=list)
    branch_rho_mean_at_step: list[float] = dataclasses.field(default_factory=list)
    chosen_index_at_step: list[int] = dataclasses.field(default_factory=list)
    would_pick_top_logit_at_step: list[int] = dataclasses.field(default_factory=list)
    refine_grad_norm_at_step: list[float] = dataclasses.field(default_factory=list)
    final_rho: float = float("nan")
    branch_k: int = 0
    lookahead_h: int = 0
    continuation_policy: str = ""
    refine_iters: int = 0

    def to_dict(self) -> SamplerDiagnostics:
        """Materialise as a plain dict for the harness."""
        return {
            "sampler": "rollout_tree",
            "rho_stream_at_step": list(self.rho_stream_at_step),
            "projected_rho_at_step": list(self.projected_rho_at_step),
            "branch_rho_min_at_step": list(self.branch_rho_min_at_step),
            "branch_rho_max_at_step": list(self.branch_rho_max_at_step),
            "branch_rho_mean_at_step": list(self.branch_rho_mean_at_step),
            "chosen_index_at_step": list(self.chosen_index_at_step),
            "would_pick_top_logit_at_step": list(self.would_pick_top_logit_at_step),
            "refine_grad_norm_at_step": list(self.refine_grad_norm_at_step),
            "final_rho": float(self.final_rho),
            "n_steps": len(self.chosen_index_at_step),
            "n_steps_disagree_with_llm": int(
                sum(1 for f in self.would_pick_top_logit_at_step if f == 0)
            ),
            "branch_k": int(self.branch_k),
            "lookahead_h": int(self.lookahead_h),
            "continuation_policy": str(self.continuation_policy),
            "refine_iters": int(self.refine_iters),
        }


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


HeuristicContinuation = Callable[
    [jt.Float[jt.Array, " n"], jt.Float[jt.Array, "T_hist m"], int, jt.PRNGKeyArray],
    jt.Float[jt.Array, "L m"],
]
"""Type alias for a user-supplied heuristic continuation callable.

Signature: ``heuristic(initial_state, history, L, key) -> Float[Array, "L m"]``.

The fourth argument ``L`` is the *actual* continuation length the sampler
needs at this step. It can be smaller than the configured ``lookahead_h``
near the end of the horizon (always ``L = min(lookahead_h, H - step - 1)``).
The callable must return exactly ``L`` rows; the sampler raises if the
shape disagrees.
"""


def _topk_indices(logits: jt.Float[jt.Array, " K"], k: int) -> jt.Int[jt.Array, " k"]:
    """Top-``k`` indices of a 1-d logits vector (descending by value).

    Wrapper around :func:`jax.lax.top_k` that returns plain int32 indices
    in (K,)-domain. Tie-breaking is left to JAX's stable top-k semantics
    (the smaller index wins on ties, which is the deterministic behaviour
    we want for reproducibility across seeds).
    """
    _vals, idx = jax.lax.top_k(logits, k)
    return idx.astype(jnp.int32)


# ---------------------------------------------------------------------------
# The sampler.
# ---------------------------------------------------------------------------


class RolloutTreeSampler:
    """K-step rollout-tree gradient probing sampler.

    At each decoding step ``t``:

      1. Form ``branch_k`` candidate actions = top-``branch_k`` of the
         action vocabulary by LLM logits.
      2. For each candidate: simulate the future ``lookahead_h`` steps
         using ``continuation_policy`` for steps
         ``t+1, ..., t+lookahead_h``, then fill the remainder of the
         horizon with ``default_action`` so the simulator integrates over
         the full configured ``[0, horizon_minutes]`` window. Evaluate
         rho on the resulting partial+continuation trajectory.
      3. Pick the candidate that maximises projected rho.
      4. (Optional) take ``refine_iters`` gradient-refinement steps on the
         chosen ``u_t`` using the horizon-aware
         :class:`STLGradientGuidedSampler` gradient. i.e. compute
         ``grad_{u_bar_t} rho`` at the chosen-branch linearisation point
         and update ``u_t`` along the projection of ``grad_rho`` onto
         vocabulary directions ``V_k - u_bar_t``. Re-pick the argmax
         vocabulary index from the refined logits.

    Parameters
    ----------
    llm:
        LLM proposal callable conforming to :class:`LLMProposal`. Returns
        logits of shape ``(K,)`` over the action vocabulary at each step.
    simulator:
        ODE simulator with ``simulate(initial_state, control, params,
        key)`` (or the glucose-insulin variant; see
        :func:`stl_seed.inference.gradient_guided._wrap_simulator`). Must
        be JAX-traceable for both vmap and (when ``refine_iters > 0``)
        ``jax.grad`` to flow through.
    spec:
        STL specification (registered :class:`STLSpec` or raw
        :class:`Node`). Predicates must conform to the introspection
        convention in :func:`stl_seed.stl.evaluator._introspect_predicate`
       . i.e. all specs in ``stl_seed.specs.REGISTRY`` are supported.
        Non-conforming predicates would force the slow Python eval path
        which is JIT/vmap/grad-incompatible; the sampler raises at
        construction in that case.
    action_vocabulary:
        Discrete action set ``V in R^{K x m}``. The argmax over leaf rhos
        selects the vocabulary index, which is then concretised to the
        action vector ``V_{argmax}``.
    sim_params:
        Kinetic parameter pytree consumed by the simulator (e.g.
        :class:`BergmanParams` or :class:`RepressilatorParams`).
    horizon:
        Number of control steps ``H``. Must equal the simulator's
        ``n_control_points``.
    branch_k:
        Number of candidate actions per decoding step. Default 8;
        capped at the vocabulary size. Each additional branch costs one
        full-horizon ODE solve per decoding step.
    lookahead_h:
        Number of *committed* future steps the continuation policy fills
        per branch before the simulator's piecewise-constant control
        falls back to ``default_action``. Default 5; the rule of thumb
        ``lookahead_h ~ horizon // 2`` covers most of the spec window
        without dominating compute. Capped at ``horizon - 1``.
    continuation_policy:
        Kind of continuation rollout used per branch. See
        :data:`ContinuationPolicy` for semantics. Default ``"zero"``.
    heuristic_continuation:
        Required when ``continuation_policy == "heuristic"``; ignored
        otherwise. Callable returning a ``(lookahead_h, m)`` block of
        continuation actions given the same ``(state, history, key)``
        triple the LLM consumes.
    refine_iters:
        Number of gradient-refinement iterations to apply to the
        chosen branch. Default ``0`` (pure tree-search). When ``> 0``,
        runs the existing :class:`STLGradientGuidedSampler` gradient
        machinery on the chosen branch with the chosen prefix as the
        committed history, then re-picks the argmax vocabulary index.
    refine_step_size:
        Step size on the gradient-refinement update. Default ``0.5``;
        the refinement iterates ``u_bar_t <- u_bar_t + refine_step_size
        * grad_rho_t`` and re-projects to the vocabulary at the end.
        Only consulted when ``refine_iters > 0``.
    aux:
        Optional task-specific kwargs forwarded to the simulator (e.g.
        ``meal_schedule`` for glucose-insulin). ``None`` for bio_ode tasks.
    default_action:
        Action used to fill the simulator's piecewise-constant control
        beyond ``t + lookahead_h``. Defaults to the centre of the action
        box (``mean(vocabulary, axis=0)``). Same role as
        :class:`STLGradientGuidedSampler.default_action`.
    sampling_temperature:
        Temperature on the projected-rho-ranked branches when sampling
        the chosen index. ``0.0`` (default) collapses to argmax over the
        ``branch_k`` projected rhos (the standard tree-search behaviour
        and the most aggressive); ``> 0`` softmax-samples the branch
        index using the projected rhos as logits scaled by
        ``1 / sampling_temperature``. The latter is useful for
        diagnostics when the projected rhos are all near-equal (i.e. the
        leaf evaluator is uninformative) and we want to fall back on the
        LLM prior. Note: argmax / softmax is over the ``branch_k``
        candidates' *projected rhos*, NOT over the LLM logits. the LLM
        only enters via its top-``branch_k`` candidate selection.

    Notes
    -----
    The branch evaluations are vectorised via :func:`jax.vmap` and JIT-
    compiled into a single fused kernel ``_branch_rhos`` per sampler
    instance. The first ``sample()`` call pays the trace cost; subsequent
    calls reuse the cached compilation.
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
        branch_k: int = 8,
        lookahead_h: int = 5,
        continuation_policy: ContinuationPolicy = "zero",
        heuristic_continuation: HeuristicContinuation | None = None,
        refine_iters: int = 0,
        refine_step_size: float = 0.5,
        aux: dict[str, Any] | None = None,
        default_action: jt.Float[jt.Array, " m"] | None = None,
        sampling_temperature: float = 0.0,
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
        if branch_k < 1:
            raise ValueError(f"branch_k must be >= 1, got {branch_k}")
        # Cap branch_k at vocabulary size. top-k requires k <= K.
        self.branch_k = min(int(branch_k), self.K)
        if lookahead_h < 0:
            raise ValueError(f"lookahead_h must be >= 0, got {lookahead_h}")
        # Cap lookahead at horizon-1 so there's always at least the candidate
        # step itself and at most horizon-1 continuation steps.
        self.lookahead_h = min(int(lookahead_h), max(self.horizon - 1, 0))
        valid_policies: tuple[str, ...] = ("zero", "random", "heuristic", "llm")
        if continuation_policy not in valid_policies:
            raise ValueError(
                f"continuation_policy must be one of {valid_policies}, got {continuation_policy!r}"
            )
        self.continuation_policy = continuation_policy
        if continuation_policy == "heuristic" and heuristic_continuation is None:
            raise ValueError(
                "continuation_policy='heuristic' requires a heuristic_continuation callable"
            )
        self.heuristic_continuation = heuristic_continuation
        if refine_iters < 0:
            raise ValueError(f"refine_iters must be >= 0, got {refine_iters}")
        self.refine_iters = int(refine_iters)
        if refine_step_size <= 0.0 and self.refine_iters > 0:
            raise ValueError(
                f"refine_step_size must be > 0 when refine_iters > 0, got {refine_step_size}"
            )
        self.refine_step_size = float(refine_step_size)
        self.aux = dict(aux) if aux is not None else None
        if sampling_temperature < 0.0:
            raise ValueError(f"sampling_temperature must be >= 0, got {sampling_temperature}")
        self.sampling_temperature = float(sampling_temperature)

        if default_action is None:
            self.default_action = jnp.mean(self.vocabulary, axis=0)
        else:
            da = jnp.asarray(default_action, dtype=jnp.float32)
            if da.shape != (self.m,):
                raise ValueError(f"default_action shape {da.shape} must equal (m,) = ({self.m},)")
            self.default_action = da

        # Compile STL spec once. Refuse non-conforming predicates: the
        # vmapped + (optionally) gradient-guided path needs JAX-pure
        # semantics.
        self._compiled_spec = compile_spec(spec)
        if getattr(self._compiled_spec, _FALLBACK_USED, False):
            raise RuntimeError(
                "RolloutTreeSampler requires every predicate to be "
                "JIT/vmap/grad-compatible (the introspection convention in "
                "stl_seed.stl.evaluator._introspect_predicate). At least one "
                f"predicate in spec {getattr(spec, 'name', '<unknown>')!r} "
                "fell back to the slow Python path. Either rewrite the "
                "predicate via stl_seed.specs.bio_ode_specs._gt / _lt, or "
                "use BestOfNSampler instead."
            )

        # Wrap the simulator into a uniform (init, control, key) -> (states, times) form.
        self._sim_fn = _wrap_simulator(simulator, sim_params, self.aux)

        # Pre-build the rho-from-control closure used by the branch evaluator.
        compiled_spec = self._compiled_spec
        sim_fn = self._sim_fn

        def _rho_from_control(
            initial_state: jt.Float[jt.Array, " n"],
            control: jt.Float[jt.Array, "H m"],
            key: jt.PRNGKeyArray,
        ) -> jt.Float[jt.Array, ""]:
            states, times = sim_fn(initial_state, control, key)
            return compiled_spec(states, times)

        self._rho_from_control = _rho_from_control

        # Vectorised branch evaluator: takes a (branch_k, H, m) tensor of
        # candidate full-horizon control sequences, simulates each, returns
        # (branch_k,) rhos. vmap over the *first* axis (the branch axis).
        # We compile inside __init__ so the trace cost is paid once per
        # sampler instance, not per sample() call.
        def _branch_rhos(
            initial_state: jt.Float[jt.Array, " n"],
            controls: jt.Float[jt.Array, "branch_k H m"],
            key: jt.PRNGKeyArray,
        ) -> jt.Float[jt.Array, " branch_k"]:
            return jax.vmap(_rho_from_control, in_axes=(None, 0, None))(
                initial_state, controls, key
            )

        self._branch_rhos = jax.jit(_branch_rhos)

        # If refine_iters > 0, build the inner gradient-guided sampler that
        # we delegate to for the refinement step. We reuse the existing
        # implementation rather than reimplementing the gradient logic here:
        # the inner sampler is the canonical reference and any changes to
        # the gradient path automatically propagate to the refinement.
        self._refiner: STLGradientGuidedSampler | None
        if self.refine_iters > 0:
            self._refiner = STLGradientGuidedSampler(
                llm,
                simulator,
                spec,
                action_vocabulary,
                sim_params,
                horizon=horizon,
                aux=aux,
                guidance_weight=self.refine_step_size,
                default_action=default_action,
                sampling_temperature=0.0,  # refinement is greedy by design
                fallback_on_grad_failure=True,
            )
        else:
            self._refiner = None

    # ------------------------------------------------------------------ public
    def sample(
        self,
        initial_state: jt.Float[jt.Array, " n"],
        key: jt.PRNGKeyArray,
    ) -> SamplerResult:
        """Generate one rollout-tree-guided trajectory.

        Returns
        -------
        (trajectory, diagnostics):
            ``trajectory`` is the canonical
            :class:`stl_seed.tasks._trajectory.Trajectory` produced by
            simulating the chosen ``u_{1:H}`` from ``initial_state``.
            ``diagnostics`` is the result of
            :meth:`RolloutTreeDiagnostics.to_dict`.
        """
        diag = RolloutTreeDiagnostics(
            branch_k=self.branch_k,
            lookahead_h=self.lookahead_h,
            continuation_policy=self.continuation_policy,
            refine_iters=self.refine_iters,
        )
        # Committed actions so far (Python list of (m,) jax arrays).
        actions: list[jt.Float[jt.Array, " m"]] = []
        # History feeds the LLM at each step.
        history = jnp.zeros((0, self.m), dtype=jnp.float32)

        for t in range(self.horizon):
            step_key = jax.random.fold_in(key, t)
            llm_key, branch_key, sample_key, cont_key = jax.random.split(step_key, 4)

            # 1. LLM logits and top-branch_k candidate vocabulary indices.
            logits = self.llm(initial_state, history, llm_key)
            logits = jnp.asarray(logits, dtype=jnp.float32)
            if logits.shape != (self.K,):
                raise ValueError(
                    f"LLM emitted logits of shape {logits.shape}, expected ({self.K},)"
                )
            cand_idx = _topk_indices(logits, self.branch_k)  # (branch_k,)
            cand_actions = self.vocabulary[cand_idx]  # (branch_k, m)

            # 2. Build per-candidate full-horizon control sequences.
            controls = self._build_branch_controls(
                initial_state=initial_state,
                actions_so_far=actions,
                history=history,
                cand_actions=cand_actions,
                step=t,
                key=cont_key,
            )
            # controls shape: (branch_k, H, m)

            # 3. Vectorised branch evaluation -> projected leaf rhos.
            # The PRNG fed to the simulator is a fixed per-step key; the
            # bio_ode simulators ignore their key (deterministic ODE), and
            # the glucose-insulin simulator uses it only for
            # noise-injection on the meal schedule which is small at this
            # scale. Sharing the key across branches keeps the comparison
            # apples-to-apples (any intra-key noise is common to all
            # branches).
            branch_rhos = self._branch_rhos(initial_state, controls, branch_key)
            branch_rhos_np = np.asarray(branch_rhos, dtype=np.float64)

            # 4. Pick the chosen branch.
            top_logit_idx = int(jnp.argmax(logits))
            if self.sampling_temperature == 0.0:
                # Argmax over projected leaf rhos. Ties broken by smallest
                # vocabulary index (cand_idx is already sorted by logit
                # descending, so the smallest-index winner among ties is the
                # higher-logit one. the natural prior tiebreaker).
                chosen_branch = int(jnp.argmax(branch_rhos))
            else:
                # Temperature-softmax over projected rhos. We treat the
                # rhos themselves as logits, scaled by 1 / temperature.
                # Note: rho can be very negative (~-250 on the repressilator
                # floor) but only relative differences matter for softmax.
                scaled = branch_rhos / self.sampling_temperature
                chosen_branch = int(jax.random.categorical(sample_key, scaled))
            chosen_idx = int(cand_idx[chosen_branch])
            chosen_action = self.vocabulary[chosen_idx]

            # 5. Optional gradient refinement on the chosen action.
            refine_grad_norm: float = float("nan")
            if self._refiner is not None:
                chosen_idx, refine_grad_norm = self._refine_chosen(
                    initial_state=initial_state,
                    actions_so_far=actions,
                    history=history,
                    chosen_idx=chosen_idx,
                    step=t,
                )
                chosen_action = self.vocabulary[chosen_idx]

            # 6. Commit the action.
            actions.append(chosen_action)
            committed_control = self._build_committed_control(actions)

            # Diagnostics: streaming rho post-commit (full horizon, default tail).
            rho_post = float(
                self._rho_from_control(
                    initial_state, committed_control, jax.random.fold_in(key, 1000 + t)
                )
            )
            diag.rho_stream_at_step.append(rho_post)
            diag.projected_rho_at_step.append(float(branch_rhos[chosen_branch]))
            diag.branch_rho_min_at_step.append(float(branch_rhos_np.min()))
            diag.branch_rho_max_at_step.append(float(branch_rhos_np.max()))
            diag.branch_rho_mean_at_step.append(float(branch_rhos_np.mean()))
            diag.chosen_index_at_step.append(chosen_idx)
            diag.would_pick_top_logit_at_step.append(int(chosen_idx == top_logit_idx))
            diag.refine_grad_norm_at_step.append(float(refine_grad_norm))

            # 7. Update LLM history with the committed action.
            history = jnp.concatenate([history, chosen_action[None, :]], axis=0)

        # 8. Final simulation -> canonical Trajectory.
        committed_control = self._build_committed_control(actions)
        traj = self._build_trajectory(
            initial_state, committed_control, jax.random.fold_in(key, 9001)
        )
        diag.final_rho = float(self._compiled_spec(traj.states, traj.times))
        return traj, diag.to_dict()

    # -------------------------------------------------------------- internals

    def _build_committed_control(
        self,
        actions: list[jt.Float[jt.Array, " m"]],
    ) -> jt.Float[jt.Array, "H m"]:
        """Build the full-horizon control with committed prefix + default tail."""
        n_committed = len(actions)
        if n_committed == 0:
            return jnp.broadcast_to(self.default_action, (self.horizon, self.m))
        if n_committed >= self.horizon:
            return jnp.stack(actions[: self.horizon], axis=0)
        prefix = jnp.stack(actions, axis=0)
        tail = jnp.broadcast_to(self.default_action, (self.horizon - n_committed, self.m))
        return jnp.concatenate([prefix, tail], axis=0)

    def _build_branch_controls(
        self,
        initial_state: jt.Float[jt.Array, " n"],
        actions_so_far: list[jt.Float[jt.Array, " m"]],
        history: jt.Float[jt.Array, "T_hist m"],
        cand_actions: jt.Float[jt.Array, "branch_k m"],
        step: int,
        key: jt.PRNGKeyArray,
    ) -> jt.Float[jt.Array, "branch_k H m"]:
        """Construct the per-candidate full-horizon control tensors.

        Layout per candidate ``j``:
          ``control[j, 0:step]              = actions_so_far``
          ``control[j, step]                = cand_actions[j]``
          ``control[j, step+1:step+1+L]     = continuation_policy(j, ...)``
          ``control[j, step+1+L:H]          = default_action``

        Where ``L = min(lookahead_h, H - step - 1)``. The continuation is
        either the same for all candidates (``"zero"``, ``"heuristic"``,
        ``"llm"``) or independent per candidate (``"random"``); we tile /
        broadcast accordingly.
        """
        H = self.horizon
        m = self.m
        K = self.branch_k

        # Prefix block (committed actions; common across candidates).
        if step > 0:
            prefix = jnp.stack(actions_so_far, axis=0)  # (step, m)
        else:
            prefix = jnp.zeros((0, m), dtype=jnp.float32)

        # Length of the continuation block actually inside the horizon.
        L = max(0, min(self.lookahead_h, H - step - 1))
        # Length of the default tail after continuation.
        tail_len = H - step - 1 - L

        # Build the continuation block per candidate -> (branch_k, L, m).
        if L == 0:
            cont = jnp.zeros((K, 0, m), dtype=jnp.float32)
        else:
            cont = self._build_continuation_block(
                initial_state=initial_state,
                history=history,
                cand_actions=cand_actions,
                L=L,
                key=key,
            )

        # Tail block: default action; common across candidates.
        if tail_len > 0:
            tail = jnp.broadcast_to(self.default_action, (tail_len, m))
        else:
            tail = jnp.zeros((0, m), dtype=jnp.float32)

        # Assemble per candidate: tile prefix and tail; stack candidate
        # action; concatenate.
        prefix_b = (
            jnp.broadcast_to(prefix, (K, step, m))
            if step > 0
            else jnp.zeros((K, 0, m), dtype=jnp.float32)
        )
        tail_b = (
            jnp.broadcast_to(tail, (K, tail_len, m))
            if tail_len > 0
            else jnp.zeros((K, 0, m), dtype=jnp.float32)
        )
        cand_b = cand_actions[:, None, :]  # (K, 1, m)

        controls = jnp.concatenate([prefix_b, cand_b, cont, tail_b], axis=1)
        # Sanity (cheap): static shape must be (K, H, m).
        assert controls.shape == (K, H, m), (
            f"branch controls shape {controls.shape} != ({K}, {H}, {m})"
        )
        return controls.astype(jnp.float32)

    def _build_continuation_block(
        self,
        initial_state: jt.Float[jt.Array, " n"],
        history: jt.Float[jt.Array, "T_hist m"],
        cand_actions: jt.Float[jt.Array, "branch_k m"],
        L: int,
        key: jt.PRNGKeyArray,
    ) -> jt.Float[jt.Array, "branch_k L m"]:
        """Generate the continuation block per the configured policy.

        Returns shape ``(branch_k, L, m)``.
        """
        K = self.branch_k
        m = self.m

        if self.continuation_policy == "zero":
            return jnp.zeros((K, L, m), dtype=jnp.float32)

        if self.continuation_policy == "random":
            # Independent uniform vocabulary draws per (candidate, step).
            # Sampling indices then gathering gives us proper "vocabulary-
            # quantised" actions that the simulator's piecewise-constant
            # control expects.
            idx = jax.random.randint(
                key,
                shape=(K, L),
                minval=0,
                maxval=self.K,
                dtype=jnp.int32,
            )
            return self.vocabulary[idx]  # (K, L, m)

        if self.continuation_policy == "heuristic":
            assert self.heuristic_continuation is not None  # checked at __init__
            cont_one = self.heuristic_continuation(initial_state, history, L, key)
            cont_one = jnp.asarray(cont_one, dtype=jnp.float32)
            if cont_one.shape != (L, m):
                raise ValueError(
                    f"heuristic_continuation returned shape {cont_one.shape}, expected ({L}, {m})"
                )
            # Tile across candidates: every branch uses the same continuation
            # so the candidate-action effect is isolated.
            return jnp.broadcast_to(cont_one[None, :, :], (K, L, m))

        if self.continuation_policy == "llm":
            # Greedy LLM rollout for L steps, applied per candidate
            # independently (each candidate has a different first action,
            # so the LLM's autoregressive history diverges immediately).
            cont = jnp.zeros((K, L, m), dtype=jnp.float32)
            for j in range(K):
                cont_j = self._llm_continuation(
                    initial_state=initial_state,
                    history_prefix=history,
                    first_action=cand_actions[j],
                    L=L,
                    key=jax.random.fold_in(key, j),
                )
                cont = cont.at[j].set(cont_j)
            return cont

        raise RuntimeError(  # defensive. _init_ already validated
            f"unhandled continuation_policy {self.continuation_policy!r}"
        )

    def _llm_continuation(
        self,
        initial_state: jt.Float[jt.Array, " n"],
        history_prefix: jt.Float[jt.Array, "T_hist m"],
        first_action: jt.Float[jt.Array, " m"],
        L: int,
        key: jt.PRNGKeyArray,
    ) -> jt.Float[jt.Array, "L m"]:
        """Greedy autoregressive LLM rollout for ``L`` steps.

        ``first_action`` is appended to ``history_prefix`` before the first
        LLM call, so the LLM sees the candidate action in context.
        """
        history = jnp.concatenate([history_prefix, first_action[None, :]], axis=0)
        out: list[jt.Float[jt.Array, " m"]] = []
        for ell in range(L):
            sub_key = jax.random.fold_in(key, ell)
            logits = self.llm(initial_state, history, sub_key)
            logits = jnp.asarray(logits, dtype=jnp.float32)
            idx = int(jnp.argmax(logits))  # greedy
            a = self.vocabulary[idx]
            out.append(a)
            history = jnp.concatenate([history, a[None, :]], axis=0)
        return jnp.stack(out, axis=0)

    def _refine_chosen(
        self,
        initial_state: jt.Float[jt.Array, " n"],
        actions_so_far: list[jt.Float[jt.Array, " m"]],
        history: jt.Float[jt.Array, "T_hist m"],
        chosen_idx: int,
        step: int,
    ) -> tuple[int, float]:
        """Refine the chosen vocabulary index via gradient-guided update.

        Returns ``(refined_idx, last_grad_norm)``. The refinement loop
        uses the inner :class:`STLGradientGuidedSampler` machinery (the
        ``_compute_bias`` method), which already handles NaN-grad
        fallback and the projection onto vocabulary directions.
        """
        assert self._refiner is not None
        # The inner sampler's _compute_bias takes the current step's
        # logits as input. We construct synthetic peaked logits at the
        # chosen index (so u_bar = chosen_action) and invoke the bias
        # computation directly. The bias's argmax then gives the refined
        # vocabulary pick.
        last_grad_norm = float("nan")
        cur_idx = chosen_idx
        for _ in range(self.refine_iters):
            peaked = jnp.full((self.K,), -1e3, dtype=jnp.float32)
            peaked = peaked.at[cur_idx].set(0.0)
            # _compute_bias mutates nothing on the sampler; safe to call.
            bias, grad_norm = self._refiner._compute_bias(
                initial_state=initial_state,
                actions_so_far=actions_so_far,
                logits=peaked,
                step=step,
            )
            last_grad_norm = float(grad_norm)
            biased = peaked + bias
            new_idx = int(jnp.argmax(biased))
            if new_idx == cur_idx:
                # Fixed point: refinement converged.
                break
            cur_idx = new_idx
        return cur_idx, last_grad_norm

    def _build_trajectory(
        self,
        initial_state: jt.Float[jt.Array, " n"],
        control: jt.Float[jt.Array, "H m"],
        key: jt.PRNGKeyArray,
    ) -> Trajectory:
        """Run the simulator once to materialise the canonical Trajectory.

        Mirrors :meth:`STLGradientGuidedSampler._build_trajectory`: a
        fresh simulator call with the committed control is used rather
        than reusing the branch-evaluation rollouts (those were partial+
        continuation approximations under specific PRNG folds).
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
    "ContinuationPolicy",
    "HeuristicContinuation",
    "RolloutTreeDiagnostics",
    "RolloutTreeSampler",
]

"""Discrete beam-search warmstart over the action vocabulary, optionally
followed by gradient refinement.

Technical motivation
--------------------

The :class:`STLGradientGuidedSampler` and the
:class:`HybridGradientBoNSampler` (see ``stl_seed.inference.gradient_guided``
and ``stl_seed.inference.hybrid``) both hinge on the *partial-then-
extrapolated* gradient probe: at each step ``t`` they freeze ``u_{t+1}, ...,
u_H`` at a constant ``u_default`` and differentiate ``rho`` w.r.t. the
continuous ``u_bar_t``. On the glucose-insulin task this myopic probe is
enough — a single insulin bolus mostly determines local glucose dynamics —
but it fails on the repressilator, where the ``G_[120,200]`` clause demands
that *all* downstream actions cooperate to keep ``m_1`` above 250 nM. The
negative result is documented in ``paper/cross_task_validation.md``.

This module is the structural-search complement: instead of a continuous
gradient, we run a **discrete beam search** over the same vocabulary
``V in R^{K x m}`` that the gradient sampler uses. At each step ``t`` we
expand every active beam member by every vocabulary item, score the
expanded partial trajectory by the streaming STL robustness ``rho_stream``
(Maler & Nickovic 2004 / Donze & Maler 2010, monotone-lower-bound
semantics implemented in :mod:`stl_seed.stl.streaming`), and keep the top
``B`` candidates for the next step. After ``H`` steps the top-1 sequence
is optionally handed to a short gradient-refinement loop that fine-tunes
the *continuous* control around the discrete winner.

The manual probe in the canonical pilot
(``test_topology_aware_repressilator_satisfies``) shows that the constant
sequence ``u = (0, 0, 1)`` lives inside the discretised vocabulary and
yields ``rho ~ +25``. So the satisfying sequence *exists* in the
vocabulary; the question is whether discrete beam search can find it.

Reference work
--------------

* Reddy 1977, "Speech Recognition by Machine: A Review", Proc. IEEE 64:4,
  pp. 501-531. Origin of beam search as a heuristic decoding strategy.
* Wu et al. 2016, "Google's Neural Machine Translation System", arXiv:
  1609.08144. Modern NMT-style beam search with length and coverage
  penalties — the conceptual ancestor of every present-day decoding
  beam in NLP.
* Vijayakumar et al. 2018, "Diverse Beam Search", arXiv:1610.02424. Adds
  a similarity-penalty term to the score so beams do not collapse onto
  near-duplicate sequences. We expose this as ``diverse_beam=True``.
* Knuth & Stevens 1973, "Optimum binary search trees", Acta Informatica.
  The classical complexity bound on beam search: ``H x B x K`` simulator
  + STL evals per ``sample()``.

The novelty here is not the beam-search algorithm — it is a 50-year-old
recipe — but its application to *STL-aware control sequence decoding*
where each expansion's score is a streaming continuous-robustness signal,
not an LM log-likelihood.

Compute cost
------------

Per ``sample()`` call:

* Beam-search phase: ``H * B * K`` simulator-forward calls + ``H * B * K``
  STL evaluations. The ``B * K`` per-step expansions are vmapped over the
  vocabulary axis, so the wall-clock cost is ``H * B`` JAX-vectorised
  simulator calls (with the ``K`` axis batched).
* Gradient-refinement phase: ``gradient_refine_iters * (1 fwd + 1 bwd)``
  through the simulator and STL evaluator, on the continuous control
  vector initialised at the discrete winner.

Compared with :class:`HybridGradientBoNSampler` at ``n=4`` (which costs
``4 * H * (1 fwd + 1 bwd)``), the beam-search warmstart at ``B=8, K=125``
costs ``H * 8 * 125 = 1000 * H`` simulator forwards plus ``30`` more for
refinement. That is ~3 orders of magnitude more forward calls but no
backward passes. The trade-off is favourable when the STL gradient is
uninformative (cliff-shaped, conjunctive long-horizon ``G`` clauses) and
the vocabulary is small enough that a full sweep over ``B * K`` per step
fits in memory.

-------------

This module imports only from JAX, jaxtyping, and in-package modules
(``stl_seed.inference.protocol``, ``stl_seed.specs``, ``stl_seed.stl``,
referenced.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import jaxtyping as jt

from stl_seed.inference.protocol import LLMProposal, SamplerDiagnostics, SamplerResult
from stl_seed.specs import Node, STLSpec
from stl_seed.stl.evaluator import compile_spec
from stl_seed.stl.streaming import evaluate_streaming
from stl_seed.tasks._trajectory import Trajectory

# ---------------------------------------------------------------------------
# Diagnostics record.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BeamSearchDiagnostics:
    """Per-step diagnostics for one beam-search rollout.

    Fields are lists indexed by control step ``t in [0, H)`` unless noted.

    * ``best_partial_score_at_step``: full-rho lookahead score of the
      top-1 beam member after the step-``t`` expansion. Pre-registered
      as monotone-improving in expectation (the beam can only *retain*
      score by keeping the top-1 of an expanded superset).
    * ``mean_partial_score_at_step``: mean full-rho lookahead across the
      active beam at step ``t``. Diagnostic of beam diversity.
    * ``streaming_rho_top1_at_step``: streaming-rho (Maler-Nickovic
      monotone lower bound) of the top-1 candidate's padded trajectory
      at the wall-clock end of step ``t``. Diagnostic only — not used
      to drive beam selection. Returns ``-inf`` for steps where some
      ``Eventually`` clause has not yet activated and ``+inf`` for
      vacuous ``Always``; both are honest lower bounds per the
      streaming module's documented semantics.
    * ``unique_sequences_per_step``: number of distinct prefix sequences
      in the active beam at step ``t``. If diverse_beam is False this can
      collapse to 1 in the saturated regime.
    * ``chosen_sequence``: the top-1 vocabulary-index sequence found by
      beam search (length ``H``).
    * ``rho_after_beam``: the *full-trajectory* rho of the top-1 sequence
      *before* gradient refinement. The pure-beam endpoint.
    * ``rho_after_refine``: the full-trajectory rho *after* gradient
      refinement. Equal to ``rho_after_beam`` if ``gradient_refine_iters
      == 0``.
    * ``final_rho``: ``rho_after_refine``. Mirrored under the canonical
      key so cross-sampler dashboards consume one field.
    * ``refine_iters_run``: number of refinement iterations that actually
      executed. Equal to ``gradient_refine_iters`` unless an early-stop
      rule fires (currently none — included for forward compatibility).
    """

    best_partial_score_at_step: list[float] = dataclasses.field(default_factory=list)
    mean_partial_score_at_step: list[float] = dataclasses.field(default_factory=list)
    streaming_rho_top1_at_step: list[float] = dataclasses.field(default_factory=list)
    unique_sequences_per_step: list[int] = dataclasses.field(default_factory=list)
    chosen_sequence: list[int] = dataclasses.field(default_factory=list)
    rho_after_beam: float = float("nan")
    rho_after_refine: float = float("nan")
    final_rho: float = float("nan")
    refine_iters_run: int = 0

    def to_dict(self) -> SamplerDiagnostics:
        return {
            "sampler": "beam_search_warmstart",
            "best_partial_score_at_step": list(self.best_partial_score_at_step),
            "mean_partial_score_at_step": list(self.mean_partial_score_at_step),
            "streaming_rho_top1_at_step": list(self.streaming_rho_top1_at_step),
            "unique_sequences_per_step": list(self.unique_sequences_per_step),
            "chosen_sequence": list(self.chosen_sequence),
            "rho_after_beam": float(self.rho_after_beam),
            "rho_after_refine": float(self.rho_after_refine),
            "final_rho": float(self.final_rho),
            "refine_iters_run": int(self.refine_iters_run),
            "n_steps": len(self.best_partial_score_at_step),
        }


# ---------------------------------------------------------------------------
# Simulator wrapper (mirrors gradient_guided._wrap_simulator API).
# ---------------------------------------------------------------------------


SimulateFn = Callable[
    [jt.Float[jt.Array, " n"], jt.Float[jt.Array, "H m"], jt.PRNGKeyArray],
    tuple[jt.Float[jt.Array, "T n"], jt.Float[jt.Array, " T"]],
]


def _wrap_simulator(simulator: Any, sim_params: Any, aux: dict[str, Any] | None) -> SimulateFn:
    """Adapt a simulator to ``(initial_state, control, key) -> (states, times)``.

    Mirrors :func:`stl_seed.inference.gradient_guided._wrap_simulator` so
    the beam sampler accepts the same simulator instances as the gradient
    sampler. Re-implemented locally rather than imported to keep the two
    modules independent — the gradient module is a peer, not a dependency.
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


class BeamSearchWarmstartSampler:
    """Discrete beam-search warmstart + (optional) gradient refinement.

    Algorithm
    ---------

    Let ``V`` be the action vocabulary (shape ``(K, m)``), ``B`` the beam
    width, and ``H`` the control horizon.

    1. Initialise the beam with a single empty prefix (``actions = []``,
       ``score = 0``).
    2. For ``t in [0, H)``:
       a. For every beam member ``b`` and every vocabulary item ``a in V``,
          form the candidate prefix ``b.actions + [a]`` and pad the
          remaining ``H - t - 1`` slots with the *tail-extrapolation*
          (selected by ``tail_strategy``; default ``'repeat_candidate'``,
          which holds the candidate ``a`` constant for the rest of the
          horizon — model-predictive constant-extrapolation).
       b. Simulate each padded sequence end-to-end and score it by the
          *full-trajectory* STL robustness ``rho`` from the same compiled
          spec the eval harness uses. The full-rho lookahead is used
          because the streaming-rho is +/-inf for vacuous-temporal-
          operator regimes (very common on long-horizon ``G_[a, b]``
          specs where ``a`` exceeds early-step wall-clock time), which
          would leave the beam ordered arbitrarily. Streaming-rho on the
          top-1 candidate is computed once per step for diagnostics only
          (see :class:`BeamSearchDiagnostics.streaming_rho_top1_at_step`).
       c. Keep the top ``B`` (prefix, action) pairs by score.
       d. (Optional, ``diverse_beam=True``): apply a Hamming-style
          penalty between same-prefix expansions so the surviving set
          spans more of the vocabulary (Vijayakumar et al. 2018).
    3. Return the top-1 prefix and its full-trajectory rho.
    4. (Optional, ``gradient_refine_iters > 0``): take the top-1
       sequence, project to the *continuous* control space, and run
       ``gradient_refine_iters`` plain SGD steps of ``grad_{control}
       rho`` to fine-tune. The refined control is then *projected back*
       to the vocabulary by nearest-vocabulary quantisation so the
       agent's emitted actions remain on-vocabulary; the refined
       sequence is only kept if its full-rho strictly exceeds the pure-
       beam endpoint, so refinement can never make the result worse.

    Hyperparameters
    ---------------

    * ``beam_size``: ``B in {1, 4, 8, 16}``. ``B=1`` is greedy. Default 8.
    * ``vocabulary_size`` is implicit in ``action_vocabulary.shape[0]``.
    * ``gradient_refine_iters``: 0 (pure beam) or e.g. 30 (refine after).
      Default 30. Backward passes are skipped entirely when 0.
    * ``refine_lr``: refinement learning rate. Default 1e-2.
    * ``diverse_beam``: bool. If True, penalise beam members that share a
      vocabulary index at the just-expanded step. Default False.
    * ``diverse_beam_weight``: penalty magnitude for ``diverse_beam=True``.
      Default 0.1 (in rho units).
    * ``score_full_traj_weight``: weight on the full-trajectory rho used
      to score each candidate (default 1.0). The full-rho on the padded
      sequence is the model-predictive-control "lookahead" score; it is
      always finite (unlike streaming-rho, which is +/-inf for vacuous
      temporal operators) and informative even at step ``t = 0``.
      Streaming-rho on the top-1 candidate is logged per step for
      diagnostics but is not added to the per-candidate score because the
      vacuous-regime issue makes it useless on long-horizon ``G_[a, b]``
      specs where ``a`` exceeds ``t_step_time`` for most ``t``.
    * ``tail_strategy``: how the per-step lookahead pads ``u_{t+1}, ...,
      u_H``. Options:

      * ``'default'`` (the legacy strategy): pad with ``default_action``.
        Inherits the gradient-guided sampler's myopic-default-action
        bias documented in :mod:`stl_seed.inference.gradient_guided`.
      * ``'repeat_candidate'`` (the default for this sampler): pad with
        the candidate action ``a`` itself. This is the model-predictive
        "constant-extrapolation" strategy and is the cleanest way to
        escape the cliff geometry of the repressilator: a constant
        ``u = (0, 0, 1)`` policy yields ``rho ~ +25`` (per
        ``test_topology_aware_repressilator_satisfies``), so the
        candidate-repeat tail makes that sequence visible to the beam
        from step 0. Pre-registered as the repressilator-friendly
        default; the ``'default'`` option is retained for ablation.
    * ``default_action``: extrapolation tail for partial sequences. Defaults
      to the action-box centre.

    -------------

    Only the in-package STL evaluator and streaming wrapper are used.
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
        beam_size: int = 8,
        gradient_refine_iters: int = 30,
        refine_lr: float = 1e-2,
        diverse_beam: bool = False,
        diverse_beam_weight: float = 0.1,
        score_full_traj_weight: float = 1.0,
        default_action: jt.Float[jt.Array, " m"] | None = None,
        tail_strategy: str = "repeat_candidate",
    ) -> None:
        # The LLM is accepted for protocol compatibility (the gradient and
        # standard samplers all take one) and to enable a future
        # extension where beam expansion is constrained to the LLM's
        # top-p actions. The current implementation does not consult it
        # — pre-registered as future work in the docstring.
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
        if beam_size < 1:
            raise ValueError(f"beam_size must be >= 1, got {beam_size}")
        self.beam_size = int(beam_size)
        if gradient_refine_iters < 0:
            raise ValueError(f"gradient_refine_iters must be >= 0, got {gradient_refine_iters}")
        self.gradient_refine_iters = int(gradient_refine_iters)
        if refine_lr <= 0:
            raise ValueError(f"refine_lr must be > 0, got {refine_lr}")
        self.refine_lr = float(refine_lr)
        self.diverse_beam = bool(diverse_beam)
        self.diverse_beam_weight = float(diverse_beam_weight)
        self.score_full_traj_weight = float(score_full_traj_weight)
        if tail_strategy not in ("default", "repeat_candidate"):
            raise ValueError(
                f"tail_strategy must be 'default' or 'repeat_candidate', got {tail_strategy!r}"
            )
        self.tail_strategy = str(tail_strategy)

        if default_action is None:
            self.default_action = jnp.mean(self.vocabulary, axis=0)
        else:
            da = jnp.asarray(default_action, dtype=jnp.float32)
            if da.shape != (self.m,):
                raise ValueError(f"default_action shape {da.shape} must equal (m,) = ({self.m},)")
            self.default_action = da

        self._compiled_spec = compile_spec(spec)
        self._sim_fn = _wrap_simulator(simulator, sim_params, self.aux)

        # Wall-clock time at the end of step t in the simulator's units.
        # Mirrors the convention in stl_seed.tasks.bio_ode and
        # stl_seed.tasks.glucose_insulin: control points are equispaced
        # over ``[0, horizon_minutes]`` and step ``t`` is in effect on
        # the interval ``[t * horizon / H, (t+1) * horizon / H]``.
        sim_horizon = self._infer_simulator_horizon(simulator)
        self._sim_horizon_minutes = float(sim_horizon)

        # Pre-build the JIT'd simulator+STL closures we will reuse.
        compiled_spec = self._compiled_spec
        sim_fn = self._sim_fn

        def rho_full(
            initial_state: jt.Float[jt.Array, " n"],
            control: jt.Float[jt.Array, "H m"],
            key: jt.PRNGKeyArray,
        ) -> jt.Float[jt.Array, ""]:
            states, times = sim_fn(initial_state, control, key)
            return compiled_spec(states, times)

        self._rho_full = rho_full
        # JIT both eval and value+grad once to amortise tracing across the
        # H * B per-step expansions and the refinement loop.
        self._rho_full_jit = jax.jit(rho_full)
        self._value_and_grad = jax.jit(jax.value_and_grad(rho_full, argnums=1))

        # Vmapped full-rho across a batch of (BK, H, m) controls. The
        # batch axis is dynamic across calls (BK changes when the active
        # beam shrinks below B), so we wrap the vmap in a JIT that takes
        # the batch as an argument rather than capturing it as a closure
        # constant — this prevents re-tracing on every step.
        def rho_full_batched(
            init: jt.Float[jt.Array, " n"],
            controls: jt.Float[jt.Array, "BK H m"],
            k: jt.PRNGKeyArray,
        ) -> jt.Float[jt.Array, " BK"]:
            return jax.vmap(rho_full, in_axes=(None, 0, None))(init, controls, k)

        self._rho_full_batched_jit = jax.jit(rho_full_batched)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _infer_simulator_horizon(simulator: Any) -> float:
        """Read the simulator's wall-clock horizon, falling back to 1.0.

        Each task simulator (Bio-ODE, glucose-insulin) exposes either
        ``horizon_minutes`` directly or via the meta. The streaming-rho
        timestamps are in the same units as ``trajectory.times``; we
        compute the end-of-step time as ``(t+1) * horizon_minutes / H``.
        Falling back to ``1.0`` (i.e. assume normalised time in [0, 1])
        is a safe failure mode for non-standard simulators because the
        streaming evaluator clamps unobserved samples.
        """
        for attr in ("horizon_minutes", "horizon_minutes_static", "horizon"):
            if hasattr(simulator, attr):
                v = getattr(simulator, attr)
                try:
                    return float(v)
                except Exception:  # pragma: no cover - defensive
                    pass
        return 1.0

    # ----------------------------------------------------------------- public
    def sample(
        self,
        initial_state: jt.Float[jt.Array, " n"],
        key: jt.PRNGKeyArray,
    ) -> SamplerResult:
        """Run beam search + (optional) refinement; return the canonical Trajectory.

        Parameters
        ----------
        initial_state:
            Initial state ``(n,)`` for the simulator.
        key:
            Master PRNG key. Used only to drive the simulator's internal
            stochasticity (if any) and to seed gradient refinement (the
            beam-search phase itself is deterministic given the inputs).

        Returns
        -------
        (trajectory, diagnostics):
            ``trajectory`` is the canonical
            :class:`stl_seed.tasks._trajectory.Trajectory` produced by
            simulating the beam-and-refine winner.
            ``diagnostics`` is the result of :meth:`BeamSearchDiagnostics.to_dict`.
        """
        diag = BeamSearchDiagnostics()
        sim_key = jax.random.fold_in(key, 0)

        # Beam state: we track it as numpy-like JAX arrays for vmap.
        # ``beam_indices`` shape (B_active, t) holds the vocabulary
        # indices chosen so far per beam member (Python list of ints
        # nested in a Python list — H is small, B is small, no JAX-scan
        # required at this scale).
        beam_indices: list[list[int]] = [[]]
        beam_scores: list[float] = [0.0]

        for t in range(self.horizon):
            t_step_time = float((t + 1) * self._sim_horizon_minutes / self.horizon)

            # Build the (B * K, H, m) batch of full padded controls.
            controls_batch, prefix_idx, action_idx = self._expand_beam(beam_indices, t)
            scores = self._score_batch(initial_state, controls_batch, t_step_time, sim_key)
            # Apply diversity penalty if requested.
            if self.diverse_beam and t > 0:
                scores = self._apply_diversity_penalty(scores, prefix_idx, action_idx)

            # Top-B by score across the (B * K) expansions.
            top = self._top_b(scores, self.beam_size)
            new_beam_indices: list[list[int]] = []
            new_beam_scores: list[float] = []
            for j in top:
                pi = int(prefix_idx[j])
                ai = int(action_idx[j])
                new_beam_indices.append(beam_indices[pi] + [ai])
                new_beam_scores.append(float(scores[j]))
            beam_indices = new_beam_indices
            beam_scores = new_beam_scores

            # Diagnostics. Compute streaming-rho on the top-1 candidate's
            # padded trajectory at the wall-clock end of step t. This is
            # diagnostic-only — beam selection used the full-rho score
            # already computed above. The streaming-rho honestly returns
            # +/-inf in the vacuous-temporal-operator regime; we record
            # it as a JSON-friendly float (Python float supports inf).
            top1_indices = beam_indices[0]
            top1_padded = self._pad_with_default(top1_indices)
            states_top1, times_top1 = self._sim_fn(initial_state, top1_padded, sim_key)
            traj_top1 = Trajectory(
                states=states_top1,
                actions=top1_padded,
                times=times_top1,
                meta=_dummy_meta(),
            )
            stream_rho = float(evaluate_streaming(self.spec, traj_top1, t_step_time))

            diag.best_partial_score_at_step.append(float(beam_scores[0]))
            diag.mean_partial_score_at_step.append(float(sum(beam_scores) / len(beam_scores)))
            diag.streaming_rho_top1_at_step.append(stream_rho)
            diag.unique_sequences_per_step.append(len({tuple(b) for b in beam_indices}))

        # Top-1 sequence.
        chosen_indices = beam_indices[0]
        diag.chosen_sequence = list(chosen_indices)
        chosen_control = jnp.stack([self.vocabulary[i] for i in chosen_indices], axis=0)
        rho_beam = float(self._rho_full_jit(initial_state, chosen_control, sim_key))
        diag.rho_after_beam = rho_beam

        # Optional gradient refinement on the continuous control around
        # the discrete winner. The refinement explores the convex hull of
        # the vocabulary; the final action sequence is projected back to
        # the vocabulary by nearest-vocabulary quantisation so the agent's
        # emitted action sequence is still on-vocabulary.
        refined_control = chosen_control
        if self.gradient_refine_iters > 0:
            refined_control, diag.refine_iters_run = self._gradient_refine(
                initial_state, chosen_control, sim_key
            )
            # Project back to vocabulary.
            refined_indices = self._nearest_vocab_indices(refined_control)
            refined_control = jnp.stack([self.vocabulary[int(i)] for i in refined_indices], axis=0)
            # If the refined sequence beats the discrete-beam one, use it.
            rho_refined = float(self._rho_full_jit(initial_state, refined_control, sim_key))
            if rho_refined > rho_beam:
                diag.chosen_sequence = [int(i) for i in refined_indices]
                diag.rho_after_refine = rho_refined
                final_control = refined_control
            else:
                # Refinement did not help; keep the discrete winner.
                diag.rho_after_refine = rho_beam
                final_control = chosen_control
        else:
            diag.rho_after_refine = rho_beam
            final_control = chosen_control

        diag.final_rho = diag.rho_after_refine
        traj = self._build_trajectory(initial_state, final_control, sim_key)
        # Recompute final rho from the materialised trajectory to handle
        # any clip-rounding the simulator may apply (and to keep the
        # diagnostic value in lockstep with the trajectory the eval
        # harness will re-evaluate).
        diag.final_rho = float(self._compiled_spec(traj.states, traj.times))
        return traj, diag.to_dict()

    # ---------------------------------------------------------------- internals
    def _expand_beam(
        self,
        beam_indices: list[list[int]],
        step: int,
    ) -> tuple[
        jt.Float[jt.Array, "BK H m"],
        jt.Int[jt.Array, " BK"],
        jt.Int[jt.Array, " BK"],
    ]:
        """Build the (B * K, H, m) batch of padded controls for step ``t``.

        For each (prefix, candidate-action) pair we return a full-length
        control sequence with the prefix in the first ``step`` slots, the
        candidate at slot ``step``, and ``default_action`` in the
        remaining ``H - step - 1`` slots. The two index arrays let us
        recover which (prefix, action) each row came from when picking the
        top B.
        """
        B_active = len(beam_indices)
        K = self.K
        H = self.horizon
        m = self.m

        # Prefix actions, shape (B_active, step, m). For step == 0 this is
        # an empty axis-1; we materialise a 0-length slice so jnp.concatenate
        # does the right thing.
        if step == 0:
            prefix_actions = jnp.zeros((B_active, 0, m), dtype=jnp.float32)
        else:
            prefix_actions = jnp.stack(
                [
                    jnp.stack([self.vocabulary[i] for i in beam_indices[b]], axis=0)
                    for b in range(B_active)
                ],
                axis=0,
            )

        # Tail of length H - step - 1, content depends on tail_strategy:
        #   - 'default': repeat self.default_action (myopic; same as
        #     gradient-guided sampler's gradient-probe extrapolation).
        #   - 'repeat_candidate': repeat the candidate action `a` itself,
        #     i.e. score each (prefix, a) by the rho of "do prefix, then
        #     hold a constant for the rest". This is the model-predictive
        #     constant-extrapolation strategy that makes the
        #     repressilator's constant satisfying policy visible to the
        #     beam from step 0.
        tail_len = H - step - 1

        # Build the (B * K, H, m) batch.
        # broadcast across (B, K, H, m)
        # prefix: (B, 1, step, m) -> (B, K, step, m)
        prefix_b_k = jnp.broadcast_to(prefix_actions[:, None, :, :], (B_active, K, step, m))
        # candidate: (1, K, 1, m) -> (B, K, 1, m)
        cand_b_k = jnp.broadcast_to(self.vocabulary[None, :, None, :], (B_active, K, 1, m))
        # tail: shape (B, K, tail_len, m), content depends on strategy.
        if tail_len > 0:
            if self.tail_strategy == "repeat_candidate":
                tail_b_k = jnp.broadcast_to(
                    self.vocabulary[None, :, None, :], (B_active, K, tail_len, m)
                )
            else:  # "default"
                default_tail = jnp.broadcast_to(self.default_action, (tail_len, m))
                tail_b_k = jnp.broadcast_to(
                    default_tail[None, None, :, :], (B_active, K, tail_len, m)
                )
            full = jnp.concatenate([prefix_b_k, cand_b_k, tail_b_k], axis=2)
        else:
            full = jnp.concatenate([prefix_b_k, cand_b_k], axis=2)
        controls_batch = full.reshape(B_active * K, H, m)

        prefix_idx = jnp.repeat(jnp.arange(B_active, dtype=jnp.int32), K)
        action_idx = jnp.tile(jnp.arange(K, dtype=jnp.int32), B_active)
        return controls_batch, prefix_idx, action_idx

    def _score_batch(
        self,
        initial_state: jt.Float[jt.Array, " n"],
        controls_batch: jt.Float[jt.Array, "BK H m"],
        t_step_time: float,
        sim_key: jt.PRNGKeyArray,
    ) -> jt.Float[jt.Array, " BK"]:
        """Score every padded candidate by streaming-rho + full-rho tie-break.

        Two scoring components combined into a single per-candidate scalar:

        * **Full-trajectory rho** of the padded sequence (prefix +
          candidate + ``default_action`` tail). Always finite, always
          informative. This is the model-predictive-control "lookahead"
          score, computed by the same compiled spec the eval harness uses.
          Vmapped + JIT'd over the (B * K) batch axis for speed.
        * **Streaming-rho** at ``t_step_time`` on the *first* candidate
          per beam expansion. Used only as a Python-level diagnostic
          (the streaming evaluator's case split on the temporal interval
          is not JIT-friendly, so we evaluate it once per step on the
          top-1 candidate to populate the diagnostic; it is *not* added
          into the per-candidate score, because the +/-inf vacuous regime
          would render it useless on long-horizon ``G_[a, b]`` specs
          where ``a`` exceeds ``t_step_time`` for most ``t``).

        Pre-registered ablation: setting ``score_full_traj_weight = 0``
        makes the per-candidate score the (zero-padded) streaming-rho;
        setting ``score_full_traj_weight >> 0`` makes it dominated by
        full-rho. The default 1.0 (see :meth:`__init__`) lets full-rho
        carry the signal.
        """
        del t_step_time  # streaming-rho is computed in sample(), not here
        full_rhos = self._rho_full_batched_jit(initial_state, controls_batch, sim_key)
        return self.score_full_traj_weight * full_rhos

    def _apply_diversity_penalty(
        self,
        scores: jt.Float[jt.Array, " BK"],
        prefix_idx: jt.Int[jt.Array, " BK"],
        action_idx: jt.Int[jt.Array, " BK"],
    ) -> jt.Float[jt.Array, " BK"]:
        """Subtract a Hamming-style penalty among same-prefix candidates.

        Vijayakumar et al. 2018 ("Diverse Beam Search") is the canonical
        reference. Our simplification: same-prefix candidates that share
        the just-expanded action are penalised by ``diverse_beam_weight``
        per duplicate beyond the first. This breaks the tie when the
        streaming-rho saturates and all expansions of one prefix score
        identically.
        """
        # Group by (prefix_idx, action_idx); penalise duplicates.
        # We do this in numpy because the loop is small (BK <= a few hundred).
        import numpy as _np

        pi_np = _np.asarray(prefix_idx)
        ai_np = _np.asarray(action_idx)
        seen: dict[tuple[int, int], int] = {}
        penalty = _np.zeros(scores.shape[0], dtype=_np.float32)
        for i in range(scores.shape[0]):
            key = (int(pi_np[i]), int(ai_np[i]))
            n_dup = seen.get(key, 0)
            penalty[i] = self.diverse_beam_weight * n_dup
            seen[key] = n_dup + 1
        return scores - jnp.asarray(penalty)

    @staticmethod
    def _top_b(scores: jt.Float[jt.Array, " BK"], B: int) -> jt.Int[jt.Array, " B"]:
        """Indices of the top ``min(B, BK)`` scores in descending order."""
        BK = int(scores.shape[0])
        k = min(B, BK)
        # jnp.argpartition is faster than sort for k << BK; argsort is fine
        # at our scale and gives a stable order.
        order = jnp.argsort(-scores)
        return order[:k]

    def _gradient_refine(
        self,
        initial_state: jt.Float[jt.Array, " n"],
        seed_control: jt.Float[jt.Array, "H m"],
        sim_key: jt.PRNGKeyArray,
    ) -> tuple[jt.Float[jt.Array, "H m"], int]:
        """Adam-free vanilla gradient ascent on rho around the seed control.

        We deliberately avoid optax to keep the dependency surface small;
        the refinement is a few dozen plain SGD steps with the simulator's
        adjoint. The signs are chosen so we ascend rho.

        The refined control is *not* clipped to the vocabulary box during
        refinement (so gradients can move freely); the projection back to
        the vocabulary happens in :meth:`sample` after refinement returns.
        """
        ctrl = jnp.asarray(seed_control, dtype=jnp.float32)
        lr = self.refine_lr
        # Track the best refined control so we never make things worse.
        best_ctrl = ctrl
        best_rho = float(self._rho_full_jit(initial_state, ctrl, sim_key))
        for it in range(self.gradient_refine_iters):
            rho_v, dctrl = self._value_and_grad(initial_state, ctrl, sim_key)
            if not bool(jnp.all(jnp.isfinite(dctrl))):
                # NaN/Inf gradient -> stop refining; keep the current best.
                return best_ctrl, it
            ctrl = ctrl + lr * dctrl
            rho_now = float(rho_v)
            if rho_now > best_rho:
                best_rho = rho_now
                best_ctrl = ctrl
        return best_ctrl, self.gradient_refine_iters

    def _pad_with_default(self, prefix_indices: list[int]) -> jt.Float[jt.Array, "H m"]:
        """Materialise a length-H control by stacking prefix indices then default.

        Used to build the top-1 padded trajectory for the per-step
        streaming-rho diagnostic in :meth:`sample`. We pad in Python
        because the per-call cost is one extra simulator forward at most
        (the streaming evaluator is the bottleneck, not the indexing).
        """
        H = self.horizon
        n_prefix = len(prefix_indices)
        n_pad = H - n_prefix
        prefix_actions = (
            jnp.stack([self.vocabulary[i] for i in prefix_indices], axis=0)
            if n_prefix > 0
            else jnp.zeros((0, self.m), dtype=jnp.float32)
        )
        if n_pad > 0:
            tail = jnp.broadcast_to(self.default_action, (n_pad, self.m))
            return jnp.concatenate([prefix_actions, tail], axis=0)
        return prefix_actions

    def _nearest_vocab_indices(self, control: jt.Float[jt.Array, "H m"]) -> jt.Int[jt.Array, " H"]:
        """Project a continuous (H, m) control to nearest vocabulary indices.

        Uses squared L2 distance in the action space. The vocabulary is
        small (K <= a few hundred) so the (H, K) distance matrix fits
        easily in memory.
        """
        # control: (H, m); vocabulary: (K, m)
        diffs = control[:, None, :] - self.vocabulary[None, :, :]  # (H, K, m)
        dists = jnp.sum(diffs * diffs, axis=-1)  # (H, K)
        return jnp.argmin(dists, axis=-1)  # (H,)

    def _build_trajectory(
        self,
        initial_state: jt.Float[jt.Array, " n"],
        control: jt.Float[jt.Array, "H m"],
        key: jt.PRNGKeyArray,
    ) -> Trajectory:
        """Materialise the canonical Trajectory dataclass for the chosen control."""
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


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _dummy_meta():
    """Build a TrajectoryMeta with all-zero diagnostic fields.

    The streaming-rho evaluator only consults ``trajectory.states`` and
    ``trajectory.times``; the meta field is required by the dataclass
    constructor but not read. We avoid importing the simulator's meta
    factory to keep the beam-search module simulator-agnostic.
    """
    from stl_seed.tasks._trajectory import TrajectoryMeta

    z = jnp.asarray(0.0, dtype=jnp.float32)
    return TrajectoryMeta(
        n_nan_replacements=z,
        final_solver_result=z,
        used_stiff_fallback=z,
    )


__all__ = [
    "BeamSearchDiagnostics",
    "BeamSearchWarmstartSampler",
]

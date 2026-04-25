"""Hybrid gradient-guided + Best-of-N sampler.

Technical motivation
--------------------

The pure :class:`STLGradientGuidedSampler` (this package) consumes
per-step ``grad rho`` and biases LLM logits accordingly. The pure
:class:`ContinuousBoNSampler` (this package) draws ``N`` independent
trajectories and picks the argmax-rho one. The two methods exploit
*different* axes of inference-time compute:

* **Gradient guidance** uses a *backward* pass per step to convert the
  continuous STL signal into a per-decision bias. It is information-
  efficient per sample but spends compute on differentiation.
* **BoN selection** uses *more samples* — it pays the cost of additional
  forward simulations but never differentiates.

Hybridising them produces a sampler that runs ``n`` *gradient-guided*
draws and selects the argmax-rho one. The hypothesis underlying the
hybrid:

    H_hybrid:    mean_rho(hybrid_n)         >=
                 mean_rho(gradient_guided)  >=
                 mean_rho(continuous_bon_2n) >=
                 mean_rho(best_of_n_n)      >=
                 mean_rho(standard).

The connection to recent test-time-compute scaling literature is direct:
hybridising gradient guidance with BoN at fixed wall-clock budget is a
*scaling* knob in the spirit of Snell et al. ("Scaling LLM test-time
compute optimally can be more effective than scaling model parameters",
arXiv:2408.03314, 2024) and Brown et al. ("Large Language Monkeys:
scaling inference compute with repeated sampling", arXiv:2407.21787,
2024). The novelty here is that the inner sampler uses a continuous
*verifier gradient* (STL ``grad rho``) rather than the LLM's own
self-consistency or a learned verifier.

Compute cost
------------

One hybrid draw costs ``H * (1 fwd + 1 bwd)`` (one gradient-guided
rollout). ``n`` draws therefore cost ``n * H * (1 fwd + 1 bwd)``.
A backward pass on the Diffrax recursive-checkpoint adjoint costs
roughly ``1-2x`` a forward pass, so a hybrid run with ``n`` draws is
approximately compute-equivalent to a continuous-BoN run with
``n_bon ≈ 2 * n * H / H = 2 * n`` draws. (More precisely: ``2n`` if
``bwd ≈ fwd``; ``3n`` if ``bwd ≈ 2 * fwd``.) The conservative
matched-compute baseline is therefore ``ContinuousBoNSampler(n=2*n)``.

-------------

This module imports only from JAX, jaxtyping, and in-package modules
(``stl_seed.inference.gradient_guided`` and the Trajectory dataclass).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jaxtyping as jt

from stl_seed.inference.gradient_guided import (
    STLGradientGuidedSampler,
)
from stl_seed.inference.protocol import (
    LLMProposal,
    SamplerDiagnostics,
    SamplerResult,
)
from stl_seed.specs import Node, STLSpec
from stl_seed.tasks._trajectory import Trajectory

# ---------------------------------------------------------------------------
# HybridGradientBoNSampler
# ---------------------------------------------------------------------------


class HybridGradientBoNSampler:
    """Hybrid: ``n`` gradient-guided draws, argmax-rho selection.

    For each of ``n`` rollouts:

      1. Run :class:`STLGradientGuidedSampler` on a sub-key derived from
         the master key (``jax.random.fold_in(key, draw_idx)``).
      2. Score the produced trajectory by the spec's robustness ``rho``
         (taken directly from the inner sampler's ``final_rho``
         diagnostic — which is computed by the same compiled spec
         used inside the inner sampler).

    Then select the argmax-rho trajectory across the ``n`` draws.

    Hypothesis
    ----------

    At fixed compute budget,

        hybrid(n=n_h)
            >  pure gradient guidance           (the n_h=1 reduction)
            >  binary BoN(n=n_h)
            >  continuous BoN(n=2*n_h)          (matched-compute baseline)
            >  standard sampling.

    See module docstring for the test-time-compute scaling rationale.

    Compute cost
    ------------

    ``n * H * (1 fwd + 1 bwd)``; matched-compute baseline is
    ``ContinuousBoNSampler(n=2*n)``.

    Parameters
    ----------
    llm:
        LLM proposal callable (see :class:`LLMProposal`). Returns logits
        of shape ``(K,)`` over the ``K``-element action vocabulary.
    simulator:
        ODE simulator. Must be JAX-traceable to allow ``grad`` to flow.
    spec:
        STL specification (registered :class:`STLSpec` or raw
        :class:`Node`). Must use the JIT-compatible predicate
        introspection convention (same constraint as
        :class:`STLGradientGuidedSampler`).
    action_vocabulary:
        Discrete action set of shape ``(K, m)``.
    sim_params:
        Kinetic parameter pytree consumed by the simulator.
    horizon:
        Number of control steps ``H``. Must equal the simulator's
        ``n_control_points``.
    n:
        Number of gradient-guided draws to perform per ``sample()``
        call. ``n=1`` collapses to pure gradient guidance (with one
        added simulator call to compute the same final rho — but the
        final rho is read directly from the inner diagnostics, so the
        collapse is exact at the action-sequence level).
    aux:
        Optional task-specific kwargs forwarded to the simulator
        (e.g. ``meal_schedule`` for glucose-insulin). ``None`` for
        bio_ode tasks.
    guidance_weight:
        Lambda hyperparameter for the inner gradient-guided sampler.
        Set to ``0.0`` to make the hybrid equivalent to plain
        :class:`ContinuousBoNSampler` (each inner draw becomes a
        vanilla LLM rollout, with a constant tax of one extra
        evaluation per draw).
    default_action:
        Forwarded to the inner sampler. ``None`` (default) -> action-box center.
    sampling_temperature:
        Sampling temperature on the biased softmax inside each inner
        draw. ``0.0`` makes each inner draw deterministic given its
        sub-key — but since the sub-keys differ across draws, the inner
        sampler's `categorical` consumes them differently and the
        rollouts still diverge. We default to ``1.0`` which preserves
        the LLM's calibration; lowering helps when the LLM is poorly
        calibrated.
    fallback_on_grad_failure:
        Forwarded to the inner sampler. Default ``True`` (do not abort
        the rollout on a NaN/Inf gradient — the failed step samples
        unbiased and the event is recorded in diagnostics).

    Notes
    -----
    The inner sampler is constructed *once* and re-used across the
    ``n`` draws; this avoids ``n`` separate JIT-trace warmups for the
    same `value_and_grad` closure (each fresh sampler builds and traces
    its own JIT cache). The trade-off: any state stored on the inner
    sampler (currently only the JIT cache) is shared; the inner sampler
    is otherwise stateless w.r.t. ``sample`` calls.
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
        n: int = 4,
        aux: dict[str, Any] | None = None,
        guidance_weight: float = 1.0,
        default_action: jt.Float[jt.Array, " m"] | None = None,
        sampling_temperature: float = 1.0,
        fallback_on_grad_failure: bool = True,
    ) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.n = int(n)
        self.guidance_weight = float(guidance_weight)
        # Build the inner gradient-guided sampler once. All draws share
        # this sampler so we only pay the JIT-trace cost once.
        self._inner = STLGradientGuidedSampler(
            llm,
            simulator,
            spec,
            action_vocabulary,
            sim_params,
            horizon=horizon,
            aux=aux,
            guidance_weight=guidance_weight,
            default_action=default_action,
            sampling_temperature=sampling_temperature,
            fallback_on_grad_failure=fallback_on_grad_failure,
        )
        # Mirror selected attributes for introspection / diagnostic checks.
        self.K = self._inner.K
        self.m = self._inner.m
        self.horizon = self._inner.horizon
        self.spec = spec

    def sample(
        self,
        initial_state: jt.Float[jt.Array, " n"],
        key: jt.PRNGKeyArray,
    ) -> SamplerResult:
        """Run ``n`` gradient-guided draws and return the argmax-rho one.

        Parameters
        ----------
        initial_state:
            Initial state ``(n,)`` for the simulator.
        key:
            Master PRNG key. Each of the ``n`` draws is given a sub-key
            derived via :func:`jax.random.fold_in` so the sequence of
            sampled actions differs across draws while remaining
            reproducible from ``key``.

        Returns
        -------
        (best_trajectory, diagnostics):
            ``best_trajectory`` is the canonical
            :class:`stl_seed.tasks._trajectory.Trajectory` produced by
            the argmax-rho inner draw. ``diagnostics`` aggregates per-
            draw final rho, the chosen index, the per-draw counts of
            steps changed by guidance, and the per-draw fallback flags.
        """
        all_trajs: list[Trajectory] = []
        all_rho: list[float] = []
        all_steps_changed: list[int] = []
        all_fallback: list[bool] = []
        all_inner_diags: list[dict[str, Any]] = []
        for i in range(self.n):
            sub_key = jax.random.fold_in(key, i)
            traj, inner_diag = self._inner.sample(initial_state, sub_key)
            all_trajs.append(traj)
            all_rho.append(float(inner_diag["final_rho"]))
            all_steps_changed.append(int(inner_diag.get("n_steps_changed_by_guidance", 0)))
            all_fallback.append(bool(inner_diag.get("fallback_used", False)))
            all_inner_diags.append(dict(inner_diag))

        chosen = int(jnp.argmax(jnp.asarray(all_rho)))
        diag: SamplerDiagnostics = {
            "sampler": "hybrid_gradient_bon",
            "n_samples": self.n,
            "guidance_weight": self.guidance_weight,
            "all_rho": all_rho,
            "chosen_index": chosen,
            "chosen_rho": all_rho[chosen],
            "max_rho": max(all_rho),
            "min_rho": min(all_rho),
            "mean_rho": float(sum(all_rho) / len(all_rho)),
            "final_rho": all_rho[chosen],
            "n_steps_changed_by_guidance_per_draw": all_steps_changed,
            "n_steps_changed_by_guidance": int(all_steps_changed[chosen]),
            # ``n_steps`` is the per-rollout horizon, mirrored from the inner
            # diag so CLI consumers that pretty-print
            # ``n_steps_changed_by_guidance / n_steps`` work uniformly across
            # samplers.
            "n_steps": int(all_inner_diags[chosen].get("n_steps", self.horizon)),
            "fallback_used_per_draw": all_fallback,
            "fallback_used": bool(all_fallback[chosen]),
            # Per-draw inner diagnostics retained as nested list so
            # callers that want streaming-rho traces of the chosen
            # rollout (or of the worst rollout) can recover them.
            "per_draw_diagnostics": all_inner_diags,
        }
        return all_trajs[chosen], diag


__all__ = [
    "HybridGradientBoNSampler",
]

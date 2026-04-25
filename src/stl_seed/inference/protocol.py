"""Sampler protocol shared by all inference-time decoding strategies.

The four samplers in this subpackage (:class:`StandardSampler`,
:class:`BestOfNSampler`, :class:`ContinuousBoNSampler`,
:class:`STLGradientGuidedSampler`) all conform to the same
:class:`Sampler` interface. The eval harness can therefore swap them at
zero call-site cost.

LLM abstraction
---------------

The :class:`LLMProposal` callable is the minimal LLM interface every
sampler consumes:

    LLMProposal(state, history, key) -> Float[Array, " K"]
        # logits over the K-element discrete action vocabulary

The vocabulary itself (``Float[Array, "K m"]``) is owned by the sampler,
not by the LLM. This keeps the LLM stateless: in tests we use a
synthetic deterministic Markov policy as the LLM proxy; in production
the same protocol wraps an MLX/transformers token-distribution head
projected onto the action vocabulary via a small MLP or via
nearest-vocabulary embedding lookup. The vocabulary discretization is a
straight-through estimator (Bengio et al. 2013) endpoint: the gradient
``grad_u rho`` flows through the *continuous* preferred mean action
``u_bar = sum_k p_k * V_k``, then is projected onto vocabulary directions
``V_k - u_bar`` to form the logit bias. Section 2 of
``paper/inference_method.md`` derives this estimator formally.

REDACTED firewall. This module imports only from typing / jaxtyping / JAX. No
``REDACTED`` artifact is referenced.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import jaxtyping as jt

from stl_seed.tasks._trajectory import Trajectory

# A sampler's diagnostics are an arbitrary str-keyed dict so each sampler
# can record what it cares about (per-step rho, gradient norms, accepted
# count, etc.). The eval harness treats them as opaque metadata.
SamplerDiagnostics = dict[str, Any]


# A SamplerResult is the canonical return shape of `Sampler.sample(...)`.
# Kept as a tuple alias rather than a NamedTuple so that callers can do
# `traj, diag = sampler.sample(...)` without an extra import.
SamplerResult = tuple[Trajectory, SamplerDiagnostics]


@runtime_checkable
class LLMProposal(Protocol):
    """Minimal LLM interface for inference-time decoding.

    Conceptually: the LLM consumes ``(state, history)`` and returns a
    distribution over the next-action vocabulary. We expose the *logits*
    directly (not the softmax) because gradient-guided sampling adds a
    bias to logits, then renormalises. Working in log-space keeps the
    addition numerically stable for both very confident and very uncertain
    distributions.

    The ``key`` is provided so that *stochastic* LLMs (e.g. MC-dropout
    decoders) can split their own randomness; deterministic LLMs ignore it.

    Returned shape: ``(K,)`` where ``K`` is the vocabulary size set at
    sampler construction. Implementers must return a JAX array (not a
    numpy array) so the autodiff machinery has a tracer to bind to.
    """

    def __call__(
        self,
        state: jt.Float[jt.Array, " n"],
        history: jt.Float[jt.Array, "T_hist m"],
        key: jt.PRNGKeyArray,
    ) -> jt.Float[jt.Array, " K"]: ...


@runtime_checkable
class Sampler(Protocol):
    """The unified inference-time sampler interface.

    Every sampler in :mod:`stl_seed.inference` implements ``sample``.
    The harness in ``scripts/run_canonical_eval.py`` (and the new
    ``stl-seed sample`` CLI subcommand) calls into this protocol.

    A canonical ``Trajectory`` is produced; per-step diagnostics live in
    the second element of the returned tuple.
    """

    def sample(
        self,
        initial_state: jt.Float[jt.Array, " n"],
        key: jt.PRNGKeyArray,
    ) -> SamplerResult: ...


__all__ = [
    "LLMProposal",
    "Sampler",
    "SamplerDiagnostics",
    "SamplerResult",
]

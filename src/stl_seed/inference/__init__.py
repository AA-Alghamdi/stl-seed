"""Inference-time decoding strategies for STL-aware LLM control agents.

This subpackage implements the technical contribution of `stl-seed`:
**STL-robustness gradient-guided sampling** for inference-time decoding.

Standard LLM-control inference produces a control sequence ``u_{1:H}`` by
autoregressive sampling from the model's logits. Best-of-N (BoN) generates
``N`` such sequences and picks the one with the highest STL robustness ``rho``
post-hoc. This throws away two structural facts:

1. STL robustness is a *continuous*, *differentiable* signal (Donzé-Maler
   2010), not a binary satisfaction predicate.
2. Selection happens after generation, so the LLM cannot use the verifier's
   information *while* deciding which token to emit.

Gradient-guided sampling fixes both: at each generation step ``t``, we
compute ``grad_{u_t} rho(tau, phi)`` via JAX autodiff through the simulator
and the streaming STL evaluator, then bias the LLM's logits toward
candidates that locally increase ``rho``.

Public classes
--------------

* :class:`STLGradientGuidedSampler` — the technical contribution. Uses
  ``grad rho`` as a continuous classifier in the spirit of classifier
  guidance (Dhariwal & Nichol 2021), DPS (Chung et al. 2023), and
  signal-temporal-logic-as-RL-reward (Aksaray et al. 2016). Closest prior
  decoding-time method is LTLCrit / LogicGuard (Sun et al. 2025,
  arXiv:2507.03293), which uses LTL as a *discrete* token-level critic;
  our method uses STL as a *continuous, differentiable* one.

* :class:`StandardSampler` — vanilla LLM sampling (the ``lambda = 0``
  ablation); no verifier feedback at all.

* :class:`BestOfNSampler` — standard BoN with binary STL filtering
  (``rho > 0`` selects).

* :class:`ContinuousBoNSampler` — BoN with continuous ``rho`` scoring
  (argmax over the ``N`` samples).

All four samplers implement the :class:`Sampler` Protocol and produce a
canonical :class:`stl_seed.tasks._trajectory.Trajectory` plus a
diagnostics dict.

REDACTED firewall. None of this code imports from ``REDACTED``,
``REDACTED``, ``REDACTED``, ``REDACTED``, or any
``REDACTED*`` artifact. The autodiff path goes through the from-
scratch evaluator in ``stl_seed.stl.evaluator``.
"""

from __future__ import annotations

from stl_seed.inference.baselines import (
    BestOfNSampler,
    ContinuousBoNSampler,
    StandardSampler,
)
from stl_seed.inference.gradient_guided import (
    GuidanceDiagnostics,
    STLGradientGuidedSampler,
)
from stl_seed.inference.hybrid import HybridGradientBoNSampler
from stl_seed.inference.protocol import LLMProposal, Sampler, SamplerResult

__all__ = [
    "BestOfNSampler",
    "ContinuousBoNSampler",
    "GuidanceDiagnostics",
    "HybridGradientBoNSampler",
    "LLMProposal",
    "STLGradientGuidedSampler",
    "Sampler",
    "SamplerResult",
    "StandardSampler",
]

"""Empirical analysis of the Goodhart spec-completeness gap (theory.md §6).

This subpackage operationalizes the decomposition

    R_gold(tau) - R_proxy(tau)
        = [R_gold(tau) - R_spec(tau)]      (spec-completeness gap)
        + [R_spec(tau) - R_verifier(tau)]  (verifier-fidelity gap)

For STL with Donze-Maler robustness as the verifier, the second term is
identically zero (modulo float64 round-off, see ``stl_seed.stl.evaluator``
docstring), so the entire R_gold - R_proxy gap is the spec-completeness gap.
This package provides:

* :class:`TrajectoryAdversary` --- a JAX-autodiff search for trajectories
  that satisfy the proxy STL spec (high rho) yet score badly under a
  user-supplied "gold" objective. The adversary's worst-case finding is a
  direct empirical lower bound on the spec-completeness gap.

* :mod:`gold_scorers` --- a library of literature-derived gold-score
  callables for the ``glucose_insulin`` and ``bio_ode`` task families.
  Each function decomposes spec-aligned quality from a *separate*
  unstated-intent term whose neglect is what the adversary exploits.

* :func:`measure_goodhart_gap` --- empirical measurement of the gap as a
  function of policy: correlation, regression slope, top-decile divergence
  between proxy rho and gold score.

REDACTED firewall posture
---------------------
This subpackage operates only on the in-package STL evaluator
(``stl_seed.stl.evaluator``) and the in-package ``Simulator`` protocol
(``stl_seed.tasks.bio_ode.Simulator``). It does not import ``REDACTED``,
``REDACTED``, ``REDACTED``, ``REDACTED``, or any
``REDACTED`` artifact, and it does not reuse any REDACTED-tuned theta or
spec literal. The "gold" augmentations are sourced exclusively from
clinical / molecular-biology literature cited inline in
``gold_scorers.py``.
"""

from __future__ import annotations

from stl_seed.analysis.adversary import (
    AdversaryResult,
    TrajectoryAdversary,
)
from stl_seed.analysis.decomposition import (
    GoodhartGapResult,
    PerPolicyGap,
    measure_goodhart_gap,
)
from stl_seed.analysis.gold_scorers import (
    GoldScorer,
    bio_ode_gold_score,
    bio_ode_repressilator_gold,
    bio_ode_toggle_gold,
    glucose_insulin_gold_score,
)

__all__ = [
    "AdversaryResult",
    "GoldScorer",
    "GoodhartGapResult",
    "PerPolicyGap",
    "TrajectoryAdversary",
    "bio_ode_gold_score",
    "bio_ode_repressilator_gold",
    "bio_ode_toggle_gold",
    "glucose_insulin_gold_score",
    "measure_goodhart_gap",
]

"""Learned-verifier baselines for empirical comparison against STL-rho.

Currently provides:

* :class:`PAVProcessRewardModel` — Process Advantage Verifier
  (Setlur et al. 2024, arXiv:2410.08146): a small MLP trained to predict
  per-step "advantage" — the change in expected success probability before
  vs. after a step — from per-step Monte-Carlo continuation labels.

* :func:`compare_pav_vs_stl` — empirical apples-to-apples comparison
  on a held-out trajectory split: predictive AUC vs. terminal success,
  Spearman correlation, sample-efficiency curve, and training cost.

Theoretical motivation. ``paper/theory.md`` §6 defines the Goodhart
decomposition  R_gold − R_proxy = [R_gold − R_spec] + [R_spec − R_verifier].
For STL-rho the *verifier-fidelity* term is identically zero modulo float
round-off; for a learned PRM it is a regression residual that grows
with trajectory diversity. This module provides the learned baseline
needed to *measure* that residual on a real corpus.

JAX + Equinox + the in-repo STL evaluator and trajectory types.
"""

from __future__ import annotations

from stl_seed.baselines.comparison import (
    ComparisonResult,
    SampleEfficiencyPoint,
    compare_pav_vs_stl,
)
from stl_seed.baselines.pav import (
    PAVProcessRewardModel,
    compute_per_step_mc_labels,
)

__all__ = [
    "PAVProcessRewardModel",
    "compute_per_step_mc_labels",
    "ComparisonResult",
    "SampleEfficiencyPoint",
    "compare_pav_vs_stl",
]

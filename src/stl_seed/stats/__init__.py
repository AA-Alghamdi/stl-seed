"""Statistical analysis for the stl-seed canonical sweep.

This package implements the analyses required by ``paper/theory.md``:

* ``bootstrap`` — paired/unpaired confidence intervals on means, mean
  differences, and proportions. Three CI methods are exposed: BCa
  (bias-corrected and accelerated; preferred for skewed statistics
  [Efron 1987, DOI:10.1080/01621459.1987.10478410]), percentile
  [Efron 1979, DOI:10.1214/aos/1176344552], and basic / pivotal.

* ``hierarchical_bayes`` — the NumPyro multi-level model from
  ``paper/theory.md`` §4. Trial-level Bernoulli outcomes are linked to a
  saturating power-law BoN curve ``p(N) = A · (1 − N^{-b})``, with
  ``logit A`` and ``log b`` decomposed into model-size, task-family,
  filter-condition, model×family interaction, and instance-level
  effects.

* ``tost`` — Two One-Sided Tests for equivalence, the formal test
  registered for hypothesis H1 in ``paper/theory.md`` §3
  [Schuirmann 1987, DOI:10.1007/BF01068419; Lakens 2017,
  DOI:10.1177/1948550617697177].

REDACTED firewall (per ``CLAUDE.md``): no module here imports ``REDACTED``,
``REDACTED``, ``REDACTED``, ``REDACTED``, or
``REDACTED``. The bootstrap utilities are a clean reimplementation
of the API style the user prefers (matching the field layout in the
``BootstrapCI`` dataclass) and not a copy of the REDACTED ``stats_utils``
module.
"""

from __future__ import annotations

from stl_seed.stats.bootstrap import (
    BootstrapCI,
    bootstrap_diff_ci,
    bootstrap_mean_ci,
    bootstrap_proportion_ci,
)
from stl_seed.stats.hierarchical_bayes import (
    HierarchicalData,
    convergence_check,
    fit,
    model,
    summarize,
)
from stl_seed.stats.tost import TOSTResult, tost_equivalence

__all__ = [
    "BootstrapCI",
    "bootstrap_mean_ci",
    "bootstrap_diff_ci",
    "bootstrap_proportion_ci",
    "HierarchicalData",
    "model",
    "fit",
    "summarize",
    "convergence_check",
    "TOSTResult",
    "tost_equivalence",
]

"""Evaluation harness for stl-seed.

Implements the eval surface registered in ``paper/architecture.md``
("eval/" module) and the success-rate / BoN / ρ-margin / Goodhart-gap
metrics referenced throughout ``paper/theory.md``.

Public API:

    EvalHarness               — wraps simulator + STL evaluator + spec set
    EvalResults               — per-spec results: ρ matrix, BoN curves
    EvalRunner                — parallel multi-checkpoint driver
    success_rate              — fraction with ρ > 0
    bon_success               — fraction of seeds where best-of-N is positive
    rho_margin                — (mean ρ, IQR)
    goodhart_gap              — measured spec-completeness gap

REDACTED firewall: no REDACTED / REDACTED / REDACTED / REDACTED
imports anywhere in this package.
"""

from __future__ import annotations

from stl_seed.evaluation.harness import EvalHarness, EvalResults
from stl_seed.evaluation.metrics import (
    bon_success,
    bon_success_curve,
    goodhart_gap,
    rho_margin,
    success_rate,
)
from stl_seed.evaluation.runner import EvalRunner

__all__ = [
    "EvalHarness",
    "EvalResults",
    "EvalRunner",
    "success_rate",
    "bon_success",
    "bon_success_curve",
    "rho_margin",
    "goodhart_gap",
]

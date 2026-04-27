"""STL specs for the toy coding-agent task family.

Spike for ``paper/coding_task_design.md``. Only the easy spec is wired here;
medium and hard are deferred to the full implementation per the design doc.

Conjunction-only structure (firewall §C.1): every formula is built from
``Always``, ``Eventually``, n-ary ``And``, and predicate-level ``Negation``.

Channels (matches ``stl_seed.tasks.coding_toy``):
* ``m[0] = test_pass_rate ∈ [0, 1]``  fraction of unit tests passing.

Times for the toy simulator are integer step indices ``0, 1, ..., H``.
"""

from __future__ import annotations

from stl_seed.specs import (
    Eventually,
    Interval,
    Predicate,
    STLSpec,
    register,
)


def _gt(name: str, channel: int, threshold: float) -> Predicate:
    """``signal[channel] - threshold`` predicate-level robustness."""

    return Predicate(
        f"{name}>{threshold}",
        fn=lambda traj, t, c=channel, th=threshold: float(traj[t, c]) - th,
    )


# ---------------------------------------------------------------------------
# coding.fix.easy
# ---------------------------------------------------------------------------
#
# Formula: F_[0, H] (test_pass_rate >= 0.5).
#
# Meaning: at some step on the trajectory, more than half of the unit tests
# pass. This is the smoothest reachability spec a coding agent can be asked
# to satisfy. If a sampler fails this on the toy simulator, the failure is
# structural (vocabulary or simulator), not an STL artifact.
#
# Threshold derivation: the toy simulator's ``score`` callable lives in
# [0, 1] with resolution 1/3 (3 unit tests per task). The 0.5 threshold puts
# the spec frontier between "1 of 3 pass" (rho < 0) and "2 of 3 pass"
# (rho > 0), giving a positive STL gradient on each independent fix.
# Horizon = 6 edit steps (chosen so the worst-case task. the two-bug task
# in TINY_TASKS. is reachable but not trivially so).

CODING_HORIZON: int = 6

_easy_predicate = _gt("test_pass_rate", channel=0, threshold=0.5)
_easy_formula = Eventually(
    inner=_easy_predicate,
    interval=Interval(t_lo=0.0, t_hi=float(CODING_HORIZON)),
)

CODING_FIX_EASY = STLSpec(
    name="coding.fix.easy",
    formula=_easy_formula,
    signal_dim=1,
    horizon_minutes=float(CODING_HORIZON),  # "minutes" reused as step-index unit
    description=(
        "Eventually within H edit steps, the candidate code passes more than "
        "half of its unit tests. Smoothest reachability spec on the toy "
        "coding-agent simulator."
    ),
    citations=(
        "Donze, A. & Maler, O. 'Robust Satisfaction of Temporal Logic over "
        "Real-Valued Signals.' FORMATS 2010, LNCS 6246: 92-106.",
        "paper/coding_task_design.md (this artifact, Section 5).",
    ),
    formula_text="F_[0, 6] (test_pass_rate > 0.5)",
)

register(CODING_FIX_EASY)

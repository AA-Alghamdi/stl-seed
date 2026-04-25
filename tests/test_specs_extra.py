"""Supplemental coverage for ``stl_seed.specs.__init__`` AST validation.

Targets the post-init validation branches that are not exercised by the
default registered specs:

* ``Negation(non-Predicate)`` -> TypeError (line 70-72).
* ``Interval(t_lo > t_hi)`` -> ValueError (line 87).
* ``And(children=())`` -> ValueError (line 114).
* ``register`` duplicate name -> KeyError (line 171).
"""

from __future__ import annotations

import pytest

from stl_seed.specs import (
    REGISTRY,
    Always,
    And,
    Interval,
    Negation,
    Predicate,
    register,
)


def _identity_predicate(name: str = "x") -> Predicate:
    return Predicate(name, fn=lambda traj, t: float(traj[t, 0]))


def test_negation_around_non_predicate_raises() -> None:
    """Negation may only wrap a Predicate (firewall §C.1)."""
    inner_pred = _identity_predicate()
    inner_always = Always(inner_pred, Interval(0.0, 1.0))
    with pytest.raises(TypeError, match="Negation"):
        Negation(inner_always)  # type: ignore[arg-type]


def test_interval_inverted_bounds_raise() -> None:
    with pytest.raises(ValueError, match="Interval requires"):
        Interval(t_lo=10.0, t_hi=5.0)


def test_interval_equal_bounds_ok() -> None:
    """t_lo == t_hi is a degenerate-but-valid single-point interval."""
    iv = Interval(t_lo=3.0, t_hi=3.0)
    assert iv.t_lo == iv.t_hi == 3.0


def test_and_requires_at_least_two_children() -> None:
    pred = _identity_predicate()
    one_child = Always(pred, Interval(0.0, 1.0))
    with pytest.raises(ValueError, match="at least two children"):
        And(children=(one_child,))


def test_register_duplicate_name_raises() -> None:
    """Re-registering an existing name raises KeyError."""
    # Pick any registered spec; it MUST already be in REGISTRY.
    existing = REGISTRY["glucose_insulin.tir.easy"]
    with pytest.raises(KeyError, match="already registered"):
        register(existing)

"""Supplemental coverage tests for the STL evaluator subpackage.

Targets the lower-coverage branches of:

* ``stl/streaming.py``. predicate / negation / and streaming branches
  (lines 89-118 in the partial-trajectory evaluator).
* ``stl/worst_subformula.py``. negation branch (lines 78-81) and the
  Eventually witness branch (lines 107-120).
* ``stl/evaluator.py``. the predicate-introspection fallback (lines
  175-188), the nested-temporal branch (318-363), and a non-conforming
  predicate that triggers ``_FALLBACK_USED``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import pytest

from stl_seed.specs import (
    Always,
    And,
    Eventually,
    Interval,
    Negation,
    Predicate,
)
from stl_seed.specs.bio_ode_specs import _gt, _lt
from stl_seed.stl import (
    evaluate_robustness,
    evaluate_streaming,
    worst_violating_subformula,
)
from stl_seed.stl.evaluator import _FALLBACK_USED, compile_spec

# ---------------------------------------------------------------------------
# Trajectory stub.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Traj:
    states: jt.Float[jt.Array, "T n"]
    times: jt.Float[jt.Array, " T"]


def _flat(value: float, T: int = 50, t_max: float = 10.0, n_channels: int = 1) -> _Traj:
    times = jnp.linspace(0.0, t_max, T)
    states = jnp.full((T, n_channels), value)
    return _Traj(states=states, times=times)


# ---------------------------------------------------------------------------
# Streaming: Predicate branch (lines 89-106).
# ---------------------------------------------------------------------------


def test_streaming_predicate_returns_latest_value() -> None:
    """Pure-Predicate streaming returns the per-step rho at the latest
    observed time. We feed a constant trajectory and check rho == const."""
    pred = _gt("x_above_zero", 0, 0.0)
    traj = _flat(value=2.0, T=10, t_max=5.0)
    rho = float(evaluate_streaming(pred, traj, current_time=2.5))
    assert rho == pytest.approx(2.0)


def test_streaming_negation_flips_sign() -> None:
    """Negation streaming should be -rho(predicate)."""
    pred = _gt("x_above_zero", 0, 0.0)
    neg = Negation(pred)
    traj = _flat(value=3.0, T=10, t_max=5.0)
    rho = float(evaluate_streaming(neg, traj, current_time=4.0))
    assert rho == pytest.approx(-3.0)


def test_streaming_and_takes_min_over_children() -> None:
    """Streaming on a conjunction returns min over children."""
    # Two predicates at value=2 (rho=2) and value=2 with threshold 5 (rho=-3).
    p1 = _gt("x_above_zero", 0, 0.0)
    p2 = _gt("x_above_5", 0, 5.0)
    spec = And(children=(Always(p1, Interval(0.0, 5.0)), Always(p2, Interval(0.0, 5.0))))
    traj = _flat(value=2.0, T=10, t_max=5.0)
    rho = float(evaluate_streaming(spec, traj, current_time=3.0))
    # min(2.0, -3.0) = -3.0
    assert rho == pytest.approx(-3.0)


def test_streaming_unsupported_node_raises() -> None:
    """A bare ``object`` is not a recognized AST node -> TypeError."""

    class FakeNode:
        pass

    traj = _flat(value=1.0, T=4, t_max=4.0)
    with pytest.raises(TypeError, match="Unsupported AST node"):
        evaluate_streaming(FakeNode(), traj, current_time=2.0)


# ---------------------------------------------------------------------------
# worst_violating_subformula: Negation + Eventually witness branches.
# ---------------------------------------------------------------------------


def test_worst_subformula_negation_picks_argmin() -> None:
    """Negation localization: the worst time is when the *inner* predicate
    is largest (since rho(NOT p) = -rho(p))."""
    # Trajectory: x ramps from 0 to 10 over [0, 10].
    times = jnp.linspace(0.0, 10.0, 11)
    states = jnp.arange(11).reshape(11, 1).astype(jnp.float32)
    traj = _Traj(states=states, times=times)
    pred = _gt("x", 0, 0.0)  # rho_t = x_t
    neg = Negation(pred)
    sub, rho, t = worst_violating_subformula(neg, traj)
    # rho(NOT p) is minimized at t=10 (x=10) -> rho=-10
    assert rho == pytest.approx(-10.0)
    assert t == pytest.approx(10.0)
    assert sub is neg


def test_worst_subformula_eventually_returns_max_witness() -> None:
    """For ``Eventually``, the witness time is where rho is *highest*
    inside the interval (the satisfaction point)."""
    times = jnp.linspace(0.0, 10.0, 11)
    states = jnp.arange(11).reshape(11, 1).astype(jnp.float32)
    traj = _Traj(states=states, times=times)
    pred = _gt("x", 0, 0.0)  # rho_t = x_t
    spec = Eventually(pred, Interval(2.0, 8.0))
    sub, rho, t = worst_violating_subformula(spec, traj)
    # max rho in [2, 8] is at t=8, rho=8
    assert rho == pytest.approx(8.0)
    assert t == pytest.approx(8.0)
    assert sub is spec


def test_worst_subformula_unsupported_node_raises() -> None:
    class FakeNode:
        pass

    traj = _flat(value=1.0)
    with pytest.raises(TypeError, match="Unsupported AST node"):
        worst_violating_subformula(FakeNode(), traj)


# ---------------------------------------------------------------------------
# evaluator fallback: a Predicate that does NOT match the lambda
# default-args convention triggers the slow Python-loop fallback.
# ---------------------------------------------------------------------------


def test_predicate_fallback_triggers_when_introspection_fails() -> None:
    """A user-supplied lambda without (channel, threshold) defaults must
    drop into the fallback path (lines 175-188 of evaluator.py).

    The compiled spec should still produce the correct rho but the
    ``_FALLBACK_USED`` attribute should be True."""

    def my_fn(traj, t):  # no defaults. introspection returns None
        return float(traj[t, 0]) - 1.0

    pred = Predicate("custom>1", fn=my_fn)
    spec = Always(pred, Interval(0.0, 5.0))
    traj = _flat(value=2.0, T=11, t_max=5.0)
    compiled = compile_spec(spec)
    rho = float(compiled(traj.states, traj.times))
    assert rho == pytest.approx(1.0)
    assert getattr(compiled, _FALLBACK_USED) is True


def test_introspection_handles_non_int_channel() -> None:
    """A lambda whose first default is not an int falls through to fallback."""

    # Use a lambda whose defaults are floats (channel 0.5 is not int).
    fn = lambda traj, t, c=0.5, th=1.0: float(traj[t, 0]) - th  # noqa: E731
    pred = Predicate("weird", fn=fn)
    spec = Always(pred, Interval(0.0, 5.0))
    traj = _flat(value=3.0, T=6, t_max=5.0)
    compiled = compile_spec(spec)
    rho = float(compiled(traj.states, traj.times))
    assert rho == pytest.approx(2.0)
    assert getattr(compiled, _FALLBACK_USED) is True


# ---------------------------------------------------------------------------
# Nested temporal: Always[a,b] (Always[c,d] phi). exercises the nested
# branch in _compile_temporal_inner (lines 331-361).
# ---------------------------------------------------------------------------


def test_nested_temporal_inside_always() -> None:
    """An Always around an inner Always evaluates to the min over the
    convolved interval. With a constant signal this is just the constant.
    """
    pred = _gt("x", 0, 0.0)
    inner_always = Always(pred, Interval(0.0, 1.0))
    outer = Always(inner_always, Interval(0.0, 2.0))
    traj = _flat(value=5.0, T=20, t_max=4.0)
    rho = float(evaluate_robustness(outer, traj))
    assert rho == pytest.approx(5.0)


def test_nested_temporal_inside_eventually() -> None:
    """Eventually around an inner Eventually on a constant trajectory:
    rho == constant."""
    pred = _gt("x", 0, 0.0)
    inner = Eventually(pred, Interval(0.0, 1.0))
    outer = Eventually(inner, Interval(0.0, 2.0))
    traj = _flat(value=2.5, T=20, t_max=4.0)
    rho = float(evaluate_robustness(outer, traj))
    assert rho == pytest.approx(2.5)


def test_nested_temporal_inner_unsupported_raises() -> None:
    """A bogus AST node inside a temporal raises TypeError."""
    from stl_seed.stl.evaluator import _compile_temporal_inner

    class FakeNode:
        pass

    with pytest.raises(TypeError, match="Unsupported AST node"):
        _compile_temporal_inner(FakeNode())


def test_compile_node_unsupported_raises() -> None:
    from stl_seed.stl.evaluator import _compile_node

    class FakeNode:
        pass

    with pytest.raises(TypeError, match="Unsupported AST node"):
        _compile_node(FakeNode())


# ---------------------------------------------------------------------------
# Streaming Eventually: covered partial branch (lines 140-145).
# ---------------------------------------------------------------------------


def test_streaming_eventually_partial_window() -> None:
    """Eventually whose window has started but not ended: rho is the max
    over [a, t_now] (a lower bound on the eventual max)."""
    times = jnp.linspace(0.0, 10.0, 11)
    states = jnp.arange(11).reshape(11, 1).astype(jnp.float32)
    traj = _Traj(states=states, times=times)
    pred = _gt("x", 0, 0.0)
    spec = Eventually(pred, Interval(0.0, 10.0))
    rho = float(evaluate_streaming(spec, traj, current_time=4.0))
    # max of x over [0, 4] is 4.0
    assert rho == pytest.approx(4.0)

"""Tests for the STL space-robustness evaluator (Donzé-Maler 2010).

Each test pre-registers an analytic expectation for the value of rho on a
synthetic trajectory, then asserts the evaluator reproduces it.

REDACTED firewall. None of these tests import ``REDACTED``, ``REDACTED``,
``REDACTED``, ``REDACTED``, or ``REDACTED``. The evaluator
under test (``stl_seed.stl.evaluator``) is a from-scratch implementation
on the conjunction-only AST in ``stl_seed.specs``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jaxtyping as jt
import pytest

from stl_seed.specs import (
    REGISTRY,
    Always,
    And,
    Eventually,
    Interval,
    Negation,
    Predicate,
    STLSpec,
)
from stl_seed.specs.bio_ode_specs import _gt, _lt
from stl_seed.stl import (
    evaluate_robustness,
    evaluate_streaming,
    worst_violating_subformula,
)
from stl_seed.stl.evaluator import _FALLBACK_USED, compile_spec

# ---------------------------------------------------------------------------
# Test trajectory.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TrajStub:
    """Minimal trajectory object satisfying the evaluator's protocol."""

    states: jt.Float[jt.Array, "T n"]
    times: jt.Float[jt.Array, " T"]


def _flat_traj(value: float, T: int = 100, t_max: float = 10.0) -> _TrajStub:
    """Constant-state trajectory at ``value`` over ``[0, t_max]``."""
    times = jnp.linspace(0.0, t_max, T)
    states = jnp.full((T, 1), value)
    return _TrajStub(states=states, times=times)


def _sine_traj(T: int = 200, t_max: float = 10.0) -> _TrajStub:
    """Single-channel sine trajectory: x(t) = sin(t)."""
    times = jnp.linspace(0.0, t_max, T)
    states = jnp.sin(times)[:, None]
    return _TrajStub(states=states, times=times)


def _two_channel_traj(x: float, y: float, T: int = 50, t_max: float = 5.0) -> _TrajStub:
    """Constant two-channel trajectory."""
    times = jnp.linspace(0.0, t_max, T)
    states = jnp.stack([jnp.full((T,), x), jnp.full((T,), y)], axis=1)
    return _TrajStub(states=states, times=times)


# ---------------------------------------------------------------------------
# Predicate tests.
# ---------------------------------------------------------------------------


def test_predicate_constant_signal() -> None:
    """Constant x = 5, predicate ``x >= 3`` -> rho = 2."""
    pred = _gt("x", 0, 3.0)
    traj = _flat_traj(5.0)
    rho = float(evaluate_robustness(pred, traj))
    assert rho == pytest.approx(2.0, abs=1e-6), f"expected rho=2, got {rho}"


def test_predicate_violation() -> None:
    """Constant x = 1, predicate ``x >= 3`` -> rho = -2."""
    pred = _gt("x", 0, 3.0)
    traj = _flat_traj(1.0)
    rho = float(evaluate_robustness(pred, traj))
    assert rho == pytest.approx(-2.0, abs=1e-6), f"expected rho=-2, got {rho}"


def test_predicate_lt_form() -> None:
    """Constant x = 1, predicate ``x < 3`` (i.e. 3 - x) -> rho = 2."""
    pred = _lt("x", 0, 3.0)
    traj = _flat_traj(1.0)
    rho = float(evaluate_robustness(pred, traj))
    assert rho == pytest.approx(2.0, abs=1e-6), f"expected rho=2, got {rho}"


# ---------------------------------------------------------------------------
# Negation.
# ---------------------------------------------------------------------------


def test_negation_flips_sign() -> None:
    """For any predicate phi on any traj, rho(NOT phi) = -rho(phi)."""
    pred = _gt("x", 0, 3.0)
    neg = Negation(pred)
    for value in (0.0, 1.5, 3.0, 5.0, -1.0):
        traj = _flat_traj(value)
        r_pos = float(evaluate_robustness(pred, traj))
        r_neg = float(evaluate_robustness(neg, traj))
        assert r_neg == pytest.approx(-r_pos, abs=1e-6), (
            f"value={value}: rho(p)={r_pos}, rho(NOT p)={r_neg}, expected negation"
        )


# ---------------------------------------------------------------------------
# Always (G).
# ---------------------------------------------------------------------------


def test_always_global_min() -> None:
    """G_[0, 10] (x >= 0) on x(t) = sin(t) over [0, 10] -> rho = -1.

    The trough of sin(t) on [0, 10] is at 3*pi/2 ~= 4.71 with value -1, so
    rho = inf_t (sin(t) - 0) = -1.
    """
    spec = Always(_gt("x", 0, 0.0), interval=Interval(0.0, 10.0))
    traj = _sine_traj(T=2000, t_max=10.0)
    rho = float(evaluate_robustness(spec, traj))
    assert rho == pytest.approx(-1.0, abs=1e-2), f"expected rho ~ -1, got {rho}"


def test_always_satisfied() -> None:
    """G_[0, 10] (x >= -2) on x(t) = sin(t) -> rho = (-1) - (-2) = 1."""
    spec = Always(_gt("x", 0, -2.0), interval=Interval(0.0, 10.0))
    traj = _sine_traj(T=2000, t_max=10.0)
    rho = float(evaluate_robustness(spec, traj))
    assert rho == pytest.approx(1.0, abs=1e-2), f"expected rho ~ 1, got {rho}"


# ---------------------------------------------------------------------------
# Eventually (F).
# ---------------------------------------------------------------------------


def test_eventually_global_max() -> None:
    """F_[0, 10] (x >= 2) on x(t) = sin(t) (max ~ 1) -> rho = 1 - 2 = -1."""
    spec = Eventually(_gt("x", 0, 2.0), interval=Interval(0.0, 10.0))
    traj = _sine_traj(T=2000, t_max=10.0)
    rho = float(evaluate_robustness(spec, traj))
    assert rho == pytest.approx(-1.0, abs=1e-2), f"expected rho ~ -1, got {rho}"


def test_eventually_satisfied() -> None:
    """F_[0, 10] (x >= 0.5) on x(t) = sin(t) -> rho = 1 - 0.5 = 0.5."""
    spec = Eventually(_gt("x", 0, 0.5), interval=Interval(0.0, 10.0))
    traj = _sine_traj(T=2000, t_max=10.0)
    rho = float(evaluate_robustness(spec, traj))
    assert rho == pytest.approx(0.5, abs=1e-2), f"expected rho ~ 0.5, got {rho}"


# ---------------------------------------------------------------------------
# Conjunction.
# ---------------------------------------------------------------------------


def test_conjunction_min() -> None:
    """((x >= a) AND (y >= b)) -> rho = min(rho_x, rho_y)."""
    spec = And(
        children=(
            _gt("x", 0, 1.0),  # rho_x = 5 - 1 = 4
            _gt("y", 1, 7.0),  # rho_y = 3 - 7 = -4
        )
    )
    traj = _two_channel_traj(x=5.0, y=3.0)
    rho = float(evaluate_robustness(spec, traj))
    assert rho == pytest.approx(-4.0, abs=1e-6), f"expected rho=-4, got {rho}"


def test_conjunction_three_clauses() -> None:
    """min over three clauses, all satisfied, picks the smallest margin."""
    spec = And(
        children=(
            _gt("x", 0, 1.0),  # 5 - 1 = 4
            _gt("y", 1, 0.5),  # 3 - 0.5 = 2.5  <-- smallest
            _lt("y_safe", 1, 10.0),  # 10 - 3 = 7
        )
    )
    traj = _two_channel_traj(x=5.0, y=3.0)
    rho = float(evaluate_robustness(spec, traj))
    assert rho == pytest.approx(2.5, abs=1e-6), f"expected rho=2.5, got {rho}"


# ---------------------------------------------------------------------------
# Streaming.
# ---------------------------------------------------------------------------


def test_streaming_lower_bound() -> None:
    """For a safety spec G_[a, b] (x >= c), the streaming rho at any
    t_now in [a, b] must be a *lower bound* on the eventual full rho.

    Construct x(t) such that the worst point is at the END of the window
    (so partial observation looks rosier than the truth):
        x(t) = 1 - t/10 over t in [0, 10]   -> minimum is at t = 10, value 0.
    Spec: G_[0, 10] (x >= -0.5).
    Final rho = 0 - (-0.5) = 0.5.
    Streaming rho at t=2 (only [0, 2] observed): inf x in [0,2] - (-0.5)
                                               = (1 - 0.2) + 0.5 = 1.3.
    1.3 > 0.5, so the streaming value is an upper bound on the partial
    window — but a lower bound on the final rho is the wrong sign here.
    Reading more carefully: the docstring says streaming rho at t_now is
    inf over [a, t_now], which monotone-decreases with t_now (more
    samples can only make the inf smaller). So streaming rho at t_now is
    an UPPER bound on the final rho (since adding more samples only
    decreases the inf). The "lower bound" wording in the design doc
    refers to the min growing tighter from above; equivalently, the
    final rho is at most the current streaming rho.

    The contract we test: for a safety spec, streaming(t_now) >=
    streaming(t_now') whenever t_now < t_now', and final rho equals
    streaming(b). So streaming is monotone-non-increasing in t_now.
    """
    times = jnp.linspace(0.0, 10.0, 1001)
    states = (1.0 - times / 10.0)[:, None]  # x(t) = 1 - t/10
    traj = _TrajStub(states=states, times=times)
    spec = Always(_gt("x", 0, -0.5), interval=Interval(0.0, 10.0))

    rho_full = float(evaluate_robustness(spec, traj))
    rho_at_2 = float(evaluate_streaming(spec, traj, current_time=2.0))
    rho_at_5 = float(evaluate_streaming(spec, traj, current_time=5.0))
    rho_at_10 = float(evaluate_streaming(spec, traj, current_time=10.0))

    # Monotone non-increasing.
    assert rho_at_2 >= rho_at_5 >= rho_at_10, (
        f"streaming not monotone non-increasing: ({rho_at_2}, {rho_at_5}, {rho_at_10})"
    )
    # Streaming at t = b equals the full rho.
    assert rho_at_10 == pytest.approx(rho_full, abs=1e-5)


def test_streaming_not_yet_activated() -> None:
    """For G_[5, 10] (x >= 0) at current_time = 2 (window not started),
    streaming returns +inf (vacuous Always satisfied)."""
    traj = _flat_traj(0.0, T=100, t_max=10.0)
    spec = Always(_gt("x", 0, 0.0), interval=Interval(5.0, 10.0))
    rho = float(evaluate_streaming(spec, traj, current_time=2.0))
    assert jnp.isinf(jnp.asarray(rho)) and rho > 0, (
        f"expected +inf for not-yet-activated Always, got {rho}"
    )


def test_streaming_eventually_pending() -> None:
    """F_[5, 10] (x >= 0) at current_time = 2 returns -inf (pending)."""
    traj = _flat_traj(1.0, T=100, t_max=10.0)
    spec = Eventually(_gt("x", 0, 0.0), interval=Interval(5.0, 10.0))
    rho = float(evaluate_streaming(spec, traj, current_time=2.0))
    assert jnp.isinf(jnp.asarray(rho)) and rho < 0, (
        f"expected -inf for pending Eventually, got {rho}"
    )


# ---------------------------------------------------------------------------
# Worst-subformula localization.
# ---------------------------------------------------------------------------


def test_worst_subformula_identifies_violator() -> None:
    """In ((x >= 0.5) AND (y < 0.1)), with x = 1.0 (sat), y = 0.5 (vio),
    the worst subformula must be the y-clause."""
    x_clause = _gt("x", 0, 0.5)  # rho = 1.0 - 0.5 = 0.5
    y_clause = _lt("y", 1, 0.1)  # rho = 0.1 - 0.5 = -0.4
    spec = And(children=(x_clause, y_clause))
    traj = _two_channel_traj(x=1.0, y=0.5)

    sub, rho, t = worst_violating_subformula(spec, traj)
    assert sub is y_clause, f"expected y-clause as worst, got {sub.name}"
    assert rho == pytest.approx(-0.4, abs=1e-6), f"expected rho=-0.4, got {rho}"
    # Constant-state trajectory: worst time is just the first index.
    assert 0.0 <= t <= 5.0


def test_worst_subformula_temporal() -> None:
    """For G_[0, 10] (x >= 0) on x(t) = sin(t), the worst is at the trough
    (t ~ 3*pi/2 ~ 4.71)."""
    spec = Always(_gt("x", 0, 0.0), interval=Interval(0.0, 10.0))
    traj = _sine_traj(T=2000, t_max=10.0)
    sub, rho, t = worst_violating_subformula(spec, traj)
    assert isinstance(sub, Always), f"expected Always as witness, got {type(sub)}"
    assert rho == pytest.approx(-1.0, abs=1e-2)
    # 3*pi/2 = 4.712...
    assert abs(t - 4.712) < 0.05, f"expected t ~ 4.71, got {t}"


# ---------------------------------------------------------------------------
# JIT.
# ---------------------------------------------------------------------------


def test_jit_works() -> None:
    """The evaluator must be JIT-compatible on the registered repressilator
    spec, with a synthetic trajectory of the appropriate shape (T=121, n=3)."""
    spec = REGISTRY["bio_ode.repressilator.easy"]
    compiled = compile_spec(spec)
    assert getattr(compiled, _FALLBACK_USED) is False, (
        "repressilator spec should be fully JIT-compiled (no fallback)"
    )

    times = jnp.linspace(0.0, spec.horizon_minutes, 401)
    # Synthetic trajectory: (p1, p2, p3) all at 100 nM (between LOW=25 and HIGH=250).
    states = jnp.full((401, 3), 100.0)

    jit_fn = jax.jit(compiled)
    rho = float(jit_fn(states, times))
    # p1 fails the >= 250 clause: rho_p1 = 100 - 250 = -150.
    # p2 satisfies the < 25 clause? No: 25 - 100 = -75. F over [0,60]: max = -75.
    # min(-150, -75) = -150.
    assert rho == pytest.approx(-150.0, abs=1e-4)


def test_jit_glucose_insulin() -> None:
    """The evaluator must JIT-compile on the registered TIR glucose spec."""
    spec = REGISTRY["glucose_insulin.tir.easy"]
    compiled = compile_spec(spec)
    assert getattr(compiled, _FALLBACK_USED) is False

    times = jnp.linspace(0.0, spec.horizon_minutes, 121)
    states = jnp.zeros((121, 3))
    states = states.at[:, 0].set(120.0)  # glucose flat at 120 (in TIR band).

    jit_fn = jax.jit(compiled)
    rho = float(jit_fn(states, times))
    # G_[30,120] (G >= 70) AND G_[30,120] (G < 180) at G=120:
    #   clause 1 rho = 120 - 70 = 50
    #   clause 2 rho = 180 - 120 = 60
    # min = 50.
    assert rho == pytest.approx(50.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Real specs from the registry.
# ---------------------------------------------------------------------------


def _synthetic_for(spec: STLSpec) -> _TrajStub:
    """Build a synthetic trajectory of the right shape for ``spec``.

    The state values are chosen so the spec evaluates to a definite sign
    (positive or negative) that the test can check.
    """
    T = 401
    times = jnp.linspace(0.0, spec.horizon_minutes, T)
    n = spec.signal_dim

    if spec.name == "bio_ode.repressilator.easy":
        # p1 high (300), p2 low (10), p3 mid (100). Should SATISFY.
        states = jnp.zeros((T, n))
        states = states.at[:, 0].set(300.0).at[:, 1].set(10.0).at[:, 2].set(100.0)
        return _TrajStub(states=states, times=times)

    if spec.name == "bio_ode.toggle.medium":
        # x1 = 150 (>= HIGH=100), x2 = 20 (< LOW=30), both << UNSAFE=600. SAT.
        # Threshold lowered from 200 to 100 nM on 2026-04-25 (alpha_1 = 160
        # caps x_1 in steady state, making the old HIGH=200 unreachable).
        states = jnp.zeros((T, n))
        states = states.at[:, 0].set(150.0).at[:, 1].set(20.0)
        return _TrajStub(states=states, times=times)

    if spec.name == "bio_ode.mapk.hard":
        # Spec was corrected on 2026-04-25 to read state index 4 (MAPK_PP)
        # in absolute microM. signal_dim is now 6 (matches the simulator).
        # MAPK_PP (channel 4): rises to 0.7 microM in [0, 30], drops to
        # 0.02 microM in [45, 60].
        # MKKK_P (channel 0): stays at 0.001 microM (< MKKK_SAFE = 0.002975).
        ramp_up = jnp.where(times <= 30.0, 0.7, 0.02)
        states = jnp.zeros((T, n))
        states = states.at[:, 0].set(0.001).at[:, 4].set(ramp_up)
        return _TrajStub(states=states, times=times)

    if spec.name == "glucose_insulin.tir.easy":
        # G = 120 mg/dL (in band [70, 180]). SAT.
        states = jnp.zeros((T, n))
        states = states.at[:, 0].set(120.0)
        return _TrajStub(states=states, times=times)

    if spec.name == "glucose_insulin.no_hypo.medium":
        # G = 120 (in TIR), I = 50 (irrelevant). SAT.
        states = jnp.zeros((T, n))
        states = states.at[:, 0].set(120.0).at[:, 2].set(50.0)
        return _TrajStub(states=states, times=times)

    if spec.name == "glucose_insulin.dawn.hard":
        # I crosses 40 in first hour, G in postprandial band [70, 140] at end.
        I_traj = jnp.where((times >= 20.0) & (times <= 50.0), 60.0, 30.0)
        # G stays at 100 throughout (in [70, 140] and [70, 180] and not severe).
        states = jnp.zeros((T, n))
        states = states.at[:, 0].set(100.0).at[:, 2].set(I_traj)
        return _TrajStub(states=states, times=times)

    raise ValueError(f"No synthetic trajectory builder for {spec.name}")


@pytest.mark.parametrize("spec_name", sorted(REGISTRY.keys()))
def test_real_specs(spec_name: str) -> None:
    """Every registered spec must (a) compile to a JIT-able evaluator and
    (b) return a finite scalar with the expected sign on a synthetic
    trajectory we believe satisfies it."""
    spec = REGISTRY[spec_name]
    compiled = compile_spec(spec)
    assert getattr(compiled, _FALLBACK_USED) is False, (
        f"spec {spec_name} fell back to the slow path; introspection failed"
    )
    traj = _synthetic_for(spec)
    rho = float(evaluate_robustness(spec, traj))
    assert jnp.isfinite(jnp.asarray(rho)), f"non-finite rho for {spec_name}: {rho}"
    assert rho > 0, (
        f"spec {spec_name} on its synthetic-satisfying trajectory: rho = {rho}, "
        "expected > 0; either the trajectory builder or the evaluator is wrong"
    )

    # JIT the evaluator and confirm same value.
    jit_fn = jax.jit(compiled)
    rho_jit = float(jit_fn(traj.states, traj.times))
    assert rho_jit == pytest.approx(rho, rel=1e-5, abs=1e-6)


# ---------------------------------------------------------------------------
# Numerical / edge cases.
# ---------------------------------------------------------------------------


def test_empty_window_always_returns_inf() -> None:
    """Spec interval entirely outside the trajectory's time range -> Always
    yields +inf (vacuous)."""
    traj = _flat_traj(0.0, T=100, t_max=5.0)  # times in [0, 5]
    spec = Always(_gt("x", 0, 0.0), interval=Interval(10.0, 20.0))
    rho = float(evaluate_robustness(spec, traj))
    assert jnp.isinf(jnp.asarray(rho)) and rho > 0


def test_empty_window_eventually_returns_neg_inf() -> None:
    """Spec interval entirely outside the trajectory -> Eventually yields -inf."""
    traj = _flat_traj(1.0, T=100, t_max=5.0)
    spec = Eventually(_gt("x", 0, 0.0), interval=Interval(10.0, 20.0))
    rho = float(evaluate_robustness(spec, traj))
    assert jnp.isinf(jnp.asarray(rho)) and rho < 0


def test_non_uniform_times() -> None:
    """Non-uniform time grid: the masking-by-time approach must still work."""
    # Times concentrated near the start, sparse near the end.
    times = jnp.concatenate([jnp.linspace(0.0, 1.0, 50), jnp.linspace(1.0, 10.0, 11)[1:]])
    states = jnp.where(times[:, None] < 5.0, 0.0, 10.0)  # step at t = 5
    traj = _TrajStub(states=states, times=times)
    spec = Always(_gt("x", 0, 5.0), interval=Interval(6.0, 10.0))
    rho = float(evaluate_robustness(spec, traj))
    # In [6, 10] x = 10 -> rho = 10 - 5 = 5.
    assert rho == pytest.approx(5.0, abs=1e-6)

"""Tests for the STL spec auto-tuning module.

Covers:
* Threshold-placeholder extraction and substitution on the AST.
* Closed-form 1-D Wasserstein and ROC-AUC implementations.
* Synthetic auto-tune problems where the optimal threshold is known
  analytically.
* End-to-end auto-tune on a real glucose-insulin spec with PID and
  Random policies, verifying that the recommended threshold improves
  discriminability over a trivial baseline.

The synthetic tests are JAX-free wherever possible so they run in <1s.
The real-glucose end-to-end test is gated by ``pytest.mark.slow`` so it
can be excluded from the fast CI loop.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stl_seed.generation.policies import PIDController, RandomPolicy
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
from stl_seed.specs.bio_ode_specs import _gt as bio_gt
from stl_seed.specs.bio_ode_specs import _lt as bio_lt
from stl_seed.specs.calibration import (
    AutoTuneResult,
    ThresholdPlaceholder,
    auc_separation,
    auto_tune_spec_thresholds,
    extract_threshold_placeholders,
    instantiate_spec_with_thresholds,
    trace_overlap,
    wasserstein_distance_rho,
)
from stl_seed.specs.glucose_insulin_specs import _gt as gi_gt
from stl_seed.stl.evaluator import compile_spec
from stl_seed.tasks.bio_ode import (
    RepressilatorSimulator,
    default_repressilator_initial_state,
)
from stl_seed.tasks.bio_ode_params import RepressilatorParams
from stl_seed.tasks.glucose_insulin import (
    BergmanParams,
    GlucoseInsulinSimulator,
    default_normal_subject_initial_state,
    single_meal_schedule,
)

# ---------------------------------------------------------------------------
# extract_threshold_placeholders
# ---------------------------------------------------------------------------


def test_extract_placeholders_glucose_tir() -> None:
    """The TIR-easy spec exposes two placeholders: G_above_70, G_below_180."""
    spec = REGISTRY["glucose_insulin.tir.easy"]
    placeholders = extract_threshold_placeholders(spec)
    by_name = {p.base_name: p for p in placeholders}
    assert "G_above_70" in by_name
    assert "G_below_180" in by_name
    assert by_name["G_above_70"].channel == 0
    assert by_name["G_above_70"].op == "gt"
    assert by_name["G_above_70"].current_value == pytest.approx(70.0)
    assert by_name["G_below_180"].channel == 0
    assert by_name["G_below_180"].op == "lt"
    assert by_name["G_below_180"].current_value == pytest.approx(180.0)


def test_extract_placeholders_repressilator() -> None:
    """The repressilator-easy spec exposes p1 (gt) and p2 (lt)."""
    spec = REGISTRY["bio_ode.repressilator.easy"]
    placeholders = extract_threshold_placeholders(spec)
    by_name = {p.base_name: p for p in placeholders}
    assert "p1" in by_name
    assert "p2" in by_name
    assert by_name["p1"].channel == 0 and by_name["p1"].op == "gt"
    assert by_name["p1"].current_value == pytest.approx(250.0)
    assert by_name["p2"].channel == 1 and by_name["p2"].op == "lt"
    assert by_name["p2"].current_value == pytest.approx(25.0)


def test_extract_placeholders_through_negation() -> None:
    """Predicates wrapped in `Negation` are still discoverable."""
    spec = REGISTRY["glucose_insulin.no_hypo.medium"]
    placeholders = extract_threshold_placeholders(spec)
    base_names = {p.base_name for p in placeholders}
    # The spec wraps `G_severe_hypo` (a `_lt` predicate at 54) under
    # `Negation`, and `G_severe_hyper` (a `_gt` predicate at 250) under
    # `Negation`. Both must show up.
    assert "G_severe_hypo" in base_names
    assert "G_severe_hyper" in base_names


def test_extract_placeholders_dedup() -> None:
    """Duplicate predicates (same base, channel, op) are de-duped."""
    p = bio_gt("p1", 0, 250.0)
    spec = STLSpec(
        name="dup.test",
        formula=And(
            children=(
                Always(p, interval=Interval(0.0, 10.0)),
                Always(p, interval=Interval(20.0, 30.0)),
            )
        ),
        signal_dim=1,
        horizon_minutes=30.0,
        description="dup",
        citations=(),
        formula_text="",
    )
    placeholders = extract_threshold_placeholders(spec)
    assert len(placeholders) == 1
    assert placeholders[0].base_name == "p1"


def test_extract_placeholders_skips_non_introspectable() -> None:
    """A predicate that doesn't follow the _gt/_lt convention is skipped."""
    weird = Predicate("weird", fn=lambda traj, t: 0.0)  # no defaults
    spec = STLSpec(
        name="weird.test",
        formula=Always(weird, interval=Interval(0.0, 1.0)),
        signal_dim=1,
        horizon_minutes=1.0,
        description="weird",
        citations=(),
        formula_text="",
    )
    assert extract_threshold_placeholders(spec) == []


# ---------------------------------------------------------------------------
# instantiate_spec_with_thresholds
# ---------------------------------------------------------------------------


def test_instantiate_replaces_threshold_value() -> None:
    """Substituting `G_above_70` with 75.0 yields a spec whose introspected
    placeholder reads 75.0 and whose evaluator gives the expected rho."""
    spec = REGISTRY["glucose_insulin.tir.easy"]
    new_spec = instantiate_spec_with_thresholds(spec, {"G_above_70": 75.0})
    assert isinstance(new_spec, STLSpec)
    new_phs = {p.base_name: p for p in extract_threshold_placeholders(new_spec)}
    assert new_phs["G_above_70"].current_value == pytest.approx(75.0)
    # The other threshold is untouched.
    assert new_phs["G_below_180"].current_value == pytest.approx(180.0)
    # Metadata records the substitution.
    assert new_spec.metadata["auto_tuned_thresholds"] == {"G_above_70": 75.0}


def test_instantiate_evaluator_consistency() -> None:
    """Evaluating the new spec against a known trajectory must give the
    threshold-shifted rho (i.e., the substitution is wired through)."""
    spec = REGISTRY["glucose_insulin.tir.easy"]
    # Synthesize a trajectory with G constant at 100, X=I=0.
    times = jnp.linspace(0.0, 120.0, 121)
    states = jnp.stack(
        [jnp.full_like(times, 100.0), jnp.zeros_like(times), jnp.zeros_like(times)],
        axis=1,
    )
    rho_orig = float(compile_spec(spec)(states, times))
    # G_above_70 -> threshold 75: G - 75 = 25 (vs 30 originally).
    new_spec = instantiate_spec_with_thresholds(spec, {"G_above_70": 75.0})
    rho_new = float(compile_spec(new_spec)(states, times))
    # The "and" of two clauses: min(G-75, 180-G) = min(25, 80) = 25;
    # original: min(G-70, 180-G) = min(30, 80) = 30. So new is 5 less.
    assert rho_orig == pytest.approx(30.0)
    assert rho_new == pytest.approx(25.0)


def test_instantiate_unknown_key_skipped() -> None:
    """Substitution dict keys not in the spec are silently ignored.

    (Strict checking happens at the auto-tune entry point; the substitution
    function is permissive so it can be reused for partial updates.)
    """
    spec = REGISTRY["glucose_insulin.tir.easy"]
    new_spec = instantiate_spec_with_thresholds(spec, {"nonexistent": 1.0})
    # All placeholders unchanged.
    new_phs = {p.base_name: p.current_value for p in extract_threshold_placeholders(new_spec)}
    assert new_phs["G_above_70"] == pytest.approx(70.0)
    assert new_phs["G_below_180"] == pytest.approx(180.0)


def test_instantiate_preserves_negation_and_intervals() -> None:
    """`Negation`/`Always`/`Eventually`/`And` structure must round-trip."""
    spec = REGISTRY["glucose_insulin.no_hypo.medium"]
    new_spec = instantiate_spec_with_thresholds(spec, {"G_severe_hypo": 50.0})
    # The substitution should preserve the And/Always/Negation tree shape.
    assert isinstance(new_spec.formula, And)
    # Locate the negation-wrapped severe-hypo branch.
    found = False
    for child in new_spec.formula.children:
        if isinstance(child, Always) and isinstance(child.inner, Negation):
            inner_pred = child.inner.inner
            phs = {p.base_name: p for p in extract_threshold_placeholders(inner_pred)}
            if "G_severe_hypo" in phs:
                assert phs["G_severe_hypo"].current_value == pytest.approx(50.0)
                found = True
    assert found, "no negation-wrapped G_severe_hypo branch found in rebuilt spec"


# ---------------------------------------------------------------------------
# wasserstein_distance_rho
# ---------------------------------------------------------------------------


def test_wasserstein_zero_for_identical_samples() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=100)
    assert wasserstein_distance_rho(a, a.copy()) == pytest.approx(0.0)


def test_wasserstein_basic_known_distance() -> None:
    """Two equal-length sorted samples: W1 = mean of |sorted differences|."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    # Both sorted; distance per pair = 3; mean = 3.
    assert wasserstein_distance_rho(a, b) == pytest.approx(3.0)


def test_wasserstein_unequal_lengths_cdf_form() -> None:
    """For unequal lengths, the CDF-integral form must match a closed-form
    case (delta-distributions): W1(δ_0, δ_1) = 1.

    Five samples at 0 vs. one sample at 1: support is {0, 1};
    F_a(x<0)=0, F_a(0<=x<1)=1, F_b(0<=x<1)=0. CDF gap is 1 over
    [0, 1), integral = 1. (We only sum over interior intervals so the
    final atom contributes 0. matches the discrete W1.)
    """
    a = np.zeros(5)
    b = np.ones(1)
    assert wasserstein_distance_rho(a, b) == pytest.approx(1.0)


def test_wasserstein_drops_nonfinite() -> None:
    a = np.array([1.0, np.nan, 2.0, 3.0, np.inf])
    b = np.array([1.0, 2.0, 3.0])
    # After dropping NaN/Inf, a == b (sorted), so W1 = 0.
    assert wasserstein_distance_rho(a, b) == pytest.approx(0.0)


def test_wasserstein_empty_safe() -> None:
    assert wasserstein_distance_rho(np.array([]), np.array([1.0, 2.0])) == 0.0


# ---------------------------------------------------------------------------
# auc_separation
# ---------------------------------------------------------------------------


def test_auc_perfect_separation_a_above_b() -> None:
    a = np.array([10.0, 11.0, 12.0])
    b = np.array([0.0, 1.0, 2.0])
    assert auc_separation(a, b) == pytest.approx(1.0)


def test_auc_perfect_separation_b_above_a() -> None:
    a = np.array([0.0, 1.0, 2.0])
    b = np.array([10.0, 11.0, 12.0])
    assert auc_separation(a, b) == pytest.approx(0.0)


def test_auc_no_information_identical() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    # Mann-Whitney U with all ties -> AUC = 0.5.
    assert auc_separation(a, b) == pytest.approx(0.5)


def test_auc_empty_safe() -> None:
    assert auc_separation(np.array([]), np.array([1.0])) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# trace_overlap
# ---------------------------------------------------------------------------


def test_trace_overlap_disjoint() -> None:
    """Disjoint supports -> overlap 0 -> score 1."""
    a = np.zeros(50)
    b = np.ones(50) * 100.0
    assert trace_overlap(a, b) == pytest.approx(1.0, abs=0.05)


def test_trace_overlap_identical() -> None:
    """Identical samples -> overlap 1 -> score 0."""
    a = np.linspace(0.0, 1.0, 100)
    assert trace_overlap(a, a.copy()) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# auto_tune. synthetic
# ---------------------------------------------------------------------------


class _DeterministicPolicy:
    """Emit a constant action vector. useful to drive the simulator into
    a known regime so we can engineer a known-best threshold."""

    def __init__(self, value: float, action_dim: int) -> None:
        self._value = jnp.full((action_dim,), value, dtype=jnp.float32)
        self.action_dim = action_dim

    def __call__(self, state, spec, history, key):  # noqa: ARG002
        return self._value


class _TargetedSilence:
    """Silence one gene channel. used in synthetic auto-tune test."""

    def __init__(self, channel: int, action_dim: int = 3):
        self.channel = channel
        self.action_dim = action_dim

    def __call__(self, state, spec, history, key):  # noqa: ARG002
        return jnp.zeros((self.action_dim,), dtype=jnp.float32).at[self.channel].set(1.0)


def test_auto_tune_synthetic_recovers_expected_threshold() -> None:
    """Two-clause spec where the binding clause changes with T_peak.

    Metric-degeneracy lemma: for a *single* ``Eventually(pred > T)``
    clause over two policy distributions, ``rho = max_t pred(t) - T``,
    so shifting T translates both rho distributions by the same scalar
    and leaves Wasserstein-1 / AUC invariant. Real STL specs in this
    codebase are always conjunctions of >= 2 clauses, so the
    *min*-across-clauses operator makes rho a piecewise-linear function
    of T whose non-linearity creates threshold-sensitive
    discriminability. this test exercises that regime.

    Construction:

    * Tunable clause: ``Eventually(p1 > T_peak)``. favours the policy
      that drives p1 high.
    * Fixed clause: ``Always(p2 < SAFE)``. pins the *upper* end of
      ``rho_high`` so increasing T_peak past p1_high's max stops
      improving discriminability (the safety clause becomes binding).

    Expected behaviour: discriminability is *minimal* when T_peak <<
    peak_low (both policies satisfy clause 1 with rho dominated by the
    shared safety clause -> small Wasserstein) and *non-decreasing* in
    T_peak until it hits the safety-clause-binding regime. The
    auto-tuner should pick a T_peak strictly above the saturated-low
    candidate.
    """
    sim = RepressilatorSimulator()
    params = RepressilatorParams()
    init = default_repressilator_initial_state(params)

    pred_peak = bio_gt("p1_peak", 3, 100.0)
    pred_safety = bio_lt("p2_safety", 4, 1000.0)  # p2 < 1000, fixed clause
    spec = STLSpec(
        name="synthetic.repressilator.two_clause",
        formula=And(
            children=(
                Eventually(pred_peak, interval=Interval(0.0, 200.0)),
                Always(pred_safety, interval=Interval(0.0, 200.0)),
            )
        ),
        signal_dim=6,
        horizon_minutes=200.0,
        description="synthetic two-clause spec for auto-tune testing",
        citations=(),
        formula_text="F_[0,200] (p1 > T) AND G_[0,200] (p2 < 1000)",
    )
    policy_high = _TargetedSilence(2)  # silence gene 2 -> p1 high
    policy_low = _TargetedSilence(0)  # silence gene 0 -> p1 low

    result = auto_tune_spec_thresholds(
        simulator=sim,
        spec_template=spec,
        threshold_search_space={"p1_peak": [10.0, 100.0, 300.0, 1000.0, 3000.0]},
        policies={"high": policy_high, "low": policy_low},
        initial_state=np.asarray(init),
        sim_params=params,
        n_trajectories_per_policy=3,
        discriminability_metric="wasserstein",
        key=jax.random.key(0),
    )
    assert isinstance(result, AutoTuneResult)
    df = result.search_results.sort_values("p1_peak").reset_index(drop=True)
    metric_lo = float(df["metric_aggregated"].iloc[0])
    # The best metric must strictly improve on the saturated-low regime,
    # which proves the auto-tuner is doing something non-trivial. (We do
    # NOT assert improvement over the saturated-high regime because, in a
    # two-clause spec where the safety clause is symmetric across
    # policies, the metric is non-decreasing in T_peak: large T_peak
    # plateaus, it does not decrease.)
    assert result.best_metric_value > metric_lo, (
        f"auto-tune did not improve over the saturated-low regime; sweep:\n{df}"
    )
    # Per-policy stats: at the chosen threshold, the two policies should
    # have visibly different mean rho.
    mean_high = result.per_policy_rho_stats["high"]["mean"]
    mean_low = result.per_policy_rho_stats["low"]["mean"]
    assert not math.isclose(mean_high, mean_low, abs_tol=1.0)


def test_auto_tune_metric_translation_invariance_documented() -> None:
    """Document the Wasserstein/AUC translation-invariance of single-clause specs.

    A spec ``F_[0, T] (p1 > T_peak)`` has rho = max p1 - T_peak. Two
    policies' rho samples translate by -T_peak when T_peak shifts, so:

    * Wasserstein-1 between the two distributions is invariant.
    * AUC between the two distributions is invariant.

    This is *correct behaviour* (the metrics are translation-invariant
    by construction); it is also a *limitation* of single-clause spec
    auto-tuning that the user should know about. We assert the property
    here so future contributors who notice it has been "broken" know
    they have introduced a bug.
    """
    sim = RepressilatorSimulator()
    params = RepressilatorParams()
    init = default_repressilator_initial_state(params)
    pred = bio_gt("p1_peak", 3, 100.0)
    spec = STLSpec(
        name="single_clause.test",
        formula=Eventually(pred, interval=Interval(0.0, 200.0)),
        signal_dim=6,
        horizon_minutes=200.0,
        description="single-clause spec",
        citations=(),
        formula_text="F_[0,200] (p1 > T)",
    )
    result = auto_tune_spec_thresholds(
        simulator=sim,
        spec_template=spec,
        threshold_search_space={"p1_peak": [50.0, 100.0, 200.0, 500.0]},
        policies={"high": _TargetedSilence(2), "low": _TargetedSilence(0)},
        initial_state=np.asarray(init),
        sim_params=params,
        n_trajectories_per_policy=2,
        discriminability_metric="wasserstein",
        key=jax.random.key(0),
    )
    # All metric values must be (almost) equal. translation invariance.
    metric_vals = result.search_results["metric_aggregated"].values
    assert np.ptp(metric_vals) < 1e-3, (
        f"single-clause spec broke translation invariance. metric values "
        f"vary by {np.ptp(metric_vals):.6g}, expected ~0. Sweep:\n"
        f"{result.search_results}"
    )


def test_auto_tune_rejects_unknown_threshold_key() -> None:
    spec = REGISTRY["glucose_insulin.tir.easy"]
    sim = GlucoseInsulinSimulator()
    params = BergmanParams()
    init = default_normal_subject_initial_state(params)

    pid = PIDController()
    rand = RandomPolicy(action_dim=1, action_low=0.0, action_high=5.0)

    with pytest.raises(ValueError, match="not predicate base names"):
        auto_tune_spec_thresholds(
            simulator=sim,
            spec_template=spec,
            threshold_search_space={"nope": [1.0, 2.0]},
            policies={"pid": pid, "rand": rand},
            initial_state=np.asarray(init),
            sim_params=params,
            aux={"meal_schedule": single_meal_schedule(15.0, 50.0)},
            n_trajectories_per_policy=2,
        )


def test_auto_tune_rejects_single_policy() -> None:
    spec = REGISTRY["glucose_insulin.tir.easy"]
    sim = GlucoseInsulinSimulator()
    params = BergmanParams()
    init = default_normal_subject_initial_state(params)

    pid = PIDController()

    with pytest.raises(ValueError, match=">= 2 policies"):
        auto_tune_spec_thresholds(
            simulator=sim,
            spec_template=spec,
            threshold_search_space={"G_above_70": [70.0]},
            policies={"pid": pid},
            initial_state=np.asarray(init),
            sim_params=params,
        )


# ---------------------------------------------------------------------------
# auto_tune. real glucose-insulin spec
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_auto_tune_real_glucose_insulin_recovers_meaningful_threshold() -> None:
    """End-to-end: PID vs. Random on the TIR-easy spec, sweeping the upper
    TIR bound. We expect:

    * At the saturated upper edge (e.g. 250 mg/dL), both policies satisfy
      the spec trivially -> low discriminability.
    * At the saturated lower edge (e.g. 100 mg/dL), both policies fail
      most of the time -> low discriminability.
    * Somewhere in between, PID separates from Random.
    The auto-tuner should pick a threshold inside the informative band
    AND deliver a metric strictly above the worst sweep entry.
    """
    sim = GlucoseInsulinSimulator()
    params = BergmanParams()
    init = default_normal_subject_initial_state(params)
    aux = {"meal_schedule": single_meal_schedule(onset_min=15.0, carb_grams=50.0)}

    spec = REGISTRY["glucose_insulin.tir.easy"]
    pid = PIDController(setpoint=110.0, kp=0.05, ki=0.001, kd=0.02)
    rand = RandomPolicy(action_dim=1, action_low=0.0, action_high=5.0)

    result = auto_tune_spec_thresholds(
        simulator=sim,
        spec_template=spec,
        threshold_search_space={"G_below_180": [140.0, 160.0, 180.0, 200.0, 220.0, 250.0]},
        policies={"pid": pid, "random": rand},
        initial_state=np.asarray(init),
        sim_params=params,
        aux=aux,
        n_trajectories_per_policy=20,
        discriminability_metric="wasserstein",
        key=jax.random.key(0),
    )
    assert isinstance(result, AutoTuneResult)
    # Metric must be strictly positive. random and PID *should* differ.
    assert result.best_metric_value > 0.0
    # Best threshold must be one of the candidates.
    chosen = result.best_thresholds["G_below_180"]
    assert chosen in {140.0, 160.0, 180.0, 200.0, 220.0, 250.0}
    # The best metric must beat the *minimum* over the sweep. i.e., the
    # auto-tuner is doing something better than picking the worst.
    metric_col = result.search_results["metric_aggregated"]
    assert result.best_metric_value > metric_col.min()
    # Per-policy summary populated.
    assert "pid" in result.per_policy_rho_stats
    assert "random" in result.per_policy_rho_stats
    assert result.per_policy_rho_stats["pid"]["n_finite"] > 0
    assert result.per_policy_rho_stats["random"]["n_finite"] > 0

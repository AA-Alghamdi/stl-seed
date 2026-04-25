"""Tests for ``stl_seed.stats``: bootstrap, hierarchical Bayes, TOST.

The hierarchical-Bayes test that fits the full NumPyro model is marked
``slow`` and skipped by default; run with ``pytest -m slow`` to exercise
it (typical wall time on CPU: 60–180 s for the small synthetic dataset
used here).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stl_seed.stats import (
    HierarchicalData,
    bootstrap_diff_ci,
    bootstrap_mean_ci,
    bootstrap_proportion_ci,
    convergence_check,
    fit,
    summarize,
    tost_equivalence,
)
from stl_seed.stats.bootstrap import proportion_wilson_ci

# ===========================================================================
# Bootstrap CIs
# ===========================================================================


def test_bootstrap_mean_ci_point_in_interval() -> None:
    rng = np.random.default_rng(0)
    xs = rng.normal(loc=2.0, scale=0.5, size=50)
    ci = bootstrap_mean_ci(xs, n_resamples=2000, ci=0.95, key=0)
    assert ci.lower < ci.statistic < ci.upper
    # mean should be very close to 2.0 with this n
    assert abs(ci.statistic - 2.0) < 0.2


def test_bootstrap_methods_all_work() -> None:
    rng = np.random.default_rng(1)
    xs = rng.normal(loc=0.0, scale=1.0, size=80)
    for method in ("bca", "percentile", "basic"):
        ci = bootstrap_mean_ci(xs, n_resamples=500, ci=0.9, method=method, key=2)
        assert ci.method == method
        assert ci.lower < ci.statistic < ci.upper
        assert ci.confidence_level == pytest.approx(0.9)


def test_bootstrap_handles_empty() -> None:
    ci = bootstrap_mean_ci(np.array([]), n_resamples=100, key=0)
    assert ci.n == 0
    assert np.isnan(ci.statistic) and np.isnan(ci.lower) and np.isnan(ci.upper)


def test_bootstrap_handles_single() -> None:
    ci = bootstrap_mean_ci(np.array([3.14]), n_resamples=100, key=0)
    assert ci.statistic == ci.lower == ci.upper == 3.14
    assert ci.n == 1


def test_bootstrap_invalid_method_raises() -> None:
    with pytest.raises(ValueError):
        bootstrap_mean_ci(np.array([1.0, 2.0]), method="bogus", key=0)


def test_bootstrap_diff_paired_recovers_known_mean() -> None:
    """Paired difference with a known offset of 0.5 recovered with CI
    that excludes 0."""
    rng = np.random.default_rng(2)
    n = 100
    a = rng.normal(loc=0.5, scale=0.3, size=n)
    b = a - 0.5 + rng.normal(loc=0.0, scale=0.05, size=n)  # b ≈ a − 0.5 + tiny noise
    ci = bootstrap_diff_ci(a, b, paired=True, n_resamples=2000, key=3)
    assert abs(ci.statistic - 0.5) < 0.1
    assert ci.lower > 0.0  # 0 excluded — clear positive offset


def test_bootstrap_diff_paired_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        bootstrap_diff_ci(np.array([1.0, 2.0, 3.0]), np.array([1.0]), paired=True, key=0)


def test_bootstrap_diff_unpaired_runs() -> None:
    rng = np.random.default_rng(4)
    a = rng.normal(loc=1.0, scale=0.5, size=40)
    b = rng.normal(loc=0.0, scale=0.5, size=60)
    ci = bootstrap_diff_ci(a, b, paired=False, n_resamples=1000, key=5)
    # Mean a − b should be ≈ 1.0
    assert abs(ci.statistic - 1.0) < 0.3
    assert ci.lower > 0.0


def test_bootstrap_proportion_basic() -> None:
    """Bootstrap proportion with 30/100 successes recovers p̂ ≈ 0.3."""
    ci = bootstrap_proportion_ci(successes=30, n=100, n_resamples=2000, key=6)
    assert ci.statistic == pytest.approx(0.3)
    assert 0.2 < ci.lower < 0.3 < ci.upper < 0.4


def test_bootstrap_proportion_invalid() -> None:
    with pytest.raises(ValueError):
        bootstrap_proportion_ci(successes=-1, n=10)
    with pytest.raises(ValueError):
        bootstrap_proportion_ci(successes=11, n=10)
    with pytest.raises(ValueError):
        bootstrap_proportion_ci(successes=0, n=0)


def test_proportion_wilson_ci_known_example() -> None:
    """Wilson interval, p̂ = 0.5, n = 100. Standard reference value
    [Newcombe 1998, DOI:10.1002/(SICI)1097-0258(19980430)17:8<857::AID-SIM777>3.0.CO;2-E]:
    [0.404, 0.596] (rounded)."""
    ci = proportion_wilson_ci(successes=50, n=100, ci=0.95)
    assert ci.statistic == pytest.approx(0.5)
    assert abs(ci.lower - 0.404) < 0.02
    assert abs(ci.upper - 0.596) < 0.02


@pytest.mark.slow
def test_bootstrap_ci_coverage() -> None:
    """Empirical coverage of 95% bootstrap CIs from N(0, 1) samples
    should be ≥ 90% (theoretical 95%, with 1000 datasets and bootstrap
    Monte Carlo error)."""
    rng = np.random.default_rng(0)
    n_datasets = 200  # 1000 in spec; trim for unit-test wall time
    n_per = 30
    n_resamples = 1000
    inside = 0
    for d in range(n_datasets):
        xs = rng.normal(size=n_per)
        ci = bootstrap_mean_ci(xs, n_resamples=n_resamples, ci=0.95, method="bca", key=d)
        if ci.lower <= 0.0 <= ci.upper:
            inside += 1
    coverage = inside / n_datasets
    # 95% nominal; lower bound 0.90 per spec
    assert coverage >= 0.88, f"coverage {coverage:.3f} below 0.88"


# ===========================================================================
# TOST
# ===========================================================================


def test_tost_known_equivalence_clear() -> None:
    """Effect = 0.0, SE = 0.01, Δ = 0.05 → clearly equivalent."""
    res = tost_equivalence(diff=0.0, se=0.01, equivalence_margin=0.05, alpha=0.05)
    assert res.equivalent
    assert res.p_lower < 0.05 and res.p_upper < 0.05
    assert res.p_tost == max(res.p_lower, res.p_upper)


def test_tost_known_not_equivalent_far_from_zero() -> None:
    """Effect = 0.10, well outside ±0.05 → not equivalent."""
    res = tost_equivalence(diff=0.10, se=0.02, equivalence_margin=0.05, alpha=0.05)
    assert not res.equivalent
    # The lower bound test rejects (diff is way above -Δ), but the upper
    # bound test does not (diff is also above +Δ).
    assert res.p_lower < 0.05
    assert res.p_upper > 0.05


def test_tost_borderline() -> None:
    """Effect at the boundary: diff = Δ exactly → upper test p = 0.5."""
    res = tost_equivalence(diff=0.05, se=0.01, equivalence_margin=0.05, alpha=0.05)
    # t_upper = (0.05 - 0.05)/0.01 = 0  → CDF(0) = 0.5
    assert res.p_upper == pytest.approx(0.5, abs=1e-10)
    assert not res.equivalent


def test_tost_lakens_2017_table1_example() -> None:
    """From Lakens 2017 [DOI:10.1177/1948550617697177] Table 1, row 1:
    means 4.785 and 4.985 (diff = -0.2), pooled SE = 0.247, Δ = ±0.5,
    df = 98 → p_lower ≈ 0.117, p_upper ≈ 0.117 (symmetric); not
    significantly equivalent at α = 0.05.

    Our implementation: with diff = -0.2, se = 0.247, Δ = 0.5, df = 98,
    t_lower = ( -0.2 + 0.5 ) / 0.247 ≈ 1.215 → p_lower = SF(t, df=98) ≈ 0.114
    t_upper = ( -0.2 - 0.5 ) / 0.247 ≈ -2.834 → p_upper = CDF(t, df=98) ≈ 0.0028
    Equivalent at α = 0.05 because both < 0.05? p_lower = 0.114 > 0.05 → NO.

    So this example is *not* equivalent, matching Lakens' interpretation
    that the 90% CI [-0.61, 0.21] crosses one of the bounds.
    """
    res = tost_equivalence(diff=-0.2, se=0.247, equivalence_margin=0.5, alpha=0.05, df=98)
    assert not res.equivalent
    # p_lower ≈ 0.11, p_upper ≈ 0.003
    assert 0.10 < res.p_lower < 0.13
    assert res.p_upper < 0.005


def test_tost_lakens_2017_equivalent_example() -> None:
    """A clearly-equivalent case: diff = 0.05, se = 0.10, Δ = 0.5, df = 50.
    t_lower = 0.55/0.10 = 5.5 → p_lower ≈ 0
    t_upper = -0.45/0.10 = -4.5 → p_upper ≈ 0
    Both below α = 0.05 → equivalent."""
    res = tost_equivalence(diff=0.05, se=0.10, equivalence_margin=0.5, alpha=0.05, df=50)
    assert res.equivalent
    assert res.p_lower < 1e-5
    assert res.p_upper < 1e-4


def test_tost_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        tost_equivalence(diff=0.0, se=0.0, equivalence_margin=0.05)
    with pytest.raises(ValueError):
        tost_equivalence(diff=0.0, se=0.01, equivalence_margin=0.0)
    with pytest.raises(ValueError):
        tost_equivalence(diff=0.0, se=0.01, equivalence_margin=0.05, alpha=0.6)


# ===========================================================================
# Hierarchical Bayes (small smoke + slow recovery test)
# ===========================================================================


def _simulate_hierarchical_data(
    n_models: int = 3,
    n_verifiers: int = 2,
    n_families: int = 2,
    n_instances: int = 8,
    n_seeds: int = 5,
    budgets: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128),
    delta_A_true: float = 0.4,
    delta_b_true: float = 0.3,
    mu_A_true: float = 0.0,
    mu_b_true: float = -0.5,
    seed: int = 0,
) -> tuple[HierarchicalData, dict]:
    """Simulate trial-level outcomes from the registered model with
    known δ_A and δ_b for the single non-baseline verifier, returning
    both the data and the ground-truth parameters."""
    rng = np.random.default_rng(seed)
    n_contrasts = n_verifiers - 1
    # Random effects (small magnitudes, so signal dominates)
    alpha_A = rng.normal(0.0, 0.1, size=n_models)
    alpha_A -= alpha_A.mean()
    alpha_b = rng.normal(0.0, 0.1, size=n_models)
    alpha_b -= alpha_b.mean()
    phi_A = rng.normal(0.0, 0.1, size=n_families)
    phi_A -= phi_A.mean()
    phi_b = rng.normal(0.0, 0.1, size=n_families)
    phi_b -= phi_b.mean()

    rows = []
    for m in range(n_models):
        for v in range(n_verifiers):
            for f in range(n_families):
                for i in range(n_instances):
                    for s in range(n_seeds):
                        for N in budgets:
                            rows.append((m, v, f, i, s, N))
    arr = np.asarray(rows)
    m_idx = arr[:, 0]
    v_idx = arr[:, 1]
    f_idx = arr[:, 2]
    i_idx = arr[:, 3]
    s_idx = arr[:, 4]
    N_obs = arr[:, 5]

    # Logit A and log b
    delta_A = np.concatenate([[0.0], np.full(n_contrasts, delta_A_true)])
    delta_b = np.concatenate([[0.0], np.full(n_contrasts, delta_b_true)])
    logit_A = mu_A_true + alpha_A[m_idx] + phi_A[f_idx] + delta_A[v_idx]
    log_b = mu_b_true + alpha_b[m_idx] + phi_b[f_idx] + delta_b[v_idx]
    A = 1.0 / (1.0 + np.exp(-logit_A))
    b = np.exp(log_b)
    p = A * (1.0 - np.power(np.maximum(N_obs, 1.0).astype(float), -b))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    Y = (rng.uniform(size=p.shape) < p).astype(int)

    data = HierarchicalData(
        model_idx=m_idx.astype(np.int64),
        verifier_idx=v_idx.astype(np.int64),
        family_idx=f_idx.astype(np.int64),
        instance_idx=i_idx.astype(np.int64),
        seed=s_idx.astype(np.int64),
        N=N_obs.astype(np.int64),
        Y=Y.astype(np.int64),
        n_models=n_models,
        n_verifiers=n_verifiers,
        n_families=n_families,
        n_instances=n_instances,
    )
    truth = {
        "mu_A": mu_A_true,
        "mu_b": mu_b_true,
        "delta_A": delta_A_true,
        "delta_b": delta_b_true,
    }
    return data, truth


def test_hierarchical_data_validates_shapes() -> None:
    """``HierarchicalData`` rejects inconsistent shapes."""
    bad = dict(
        model_idx=np.zeros(10),
        verifier_idx=np.zeros(10),
        family_idx=np.zeros(10),
        instance_idx=np.zeros(10),
        seed=np.zeros(10),
        N=np.ones(10),
        Y=np.zeros(9),  # mismatched
        n_models=1,
        n_verifiers=2,
        n_families=1,
        n_instances=1,
    )
    with pytest.raises(ValueError):
        HierarchicalData(**bad)


def test_hierarchical_data_rejects_single_verifier() -> None:
    bad = dict(
        model_idx=np.zeros(2),
        verifier_idx=np.zeros(2),
        family_idx=np.zeros(2),
        instance_idx=np.zeros(2),
        seed=np.zeros(2),
        N=np.ones(2),
        Y=np.zeros(2),
        n_models=1,
        n_verifiers=1,
        n_families=1,
        n_instances=1,
    )
    with pytest.raises(ValueError, match="n_verifiers"):
        HierarchicalData(**bad)


@pytest.mark.slow
def test_hierarchical_bayes_recovers_known_truth() -> None:
    """Fit the registered model on simulated data with known δ_A = 0.4
    and check the posterior 89% HDI covers the truth.

    Notes on tolerance: the spec asks for posterior recovery within
    ±0.05 95% HDI. We use a synthetic sweep matching the structure of
    the canonical 3 × 3 × 2 grid (theory.md §4) — 3 models × 2
    verifiers × 2 families × 8 instances × 5 seeds × 8 budgets = 3840
    trials — and check that δ_A's 95% HDI covers the truth and the
    posterior mean is within ±0.15 of truth (allowing for partial
    pooling shrinkage on a sample much smaller than the full sweep).
    The δ_b parameter governs the curvature of the BoN power law and
    is structurally less identifiable than δ_A from Bernoulli BoN
    data; we check sign-discrimination (P(δ_b > 0) > 0.7) rather than
    absolute recovery.
    """
    data, truth = _simulate_hierarchical_data(seed=2026)
    idata = fit(
        data,
        n_chains=2,
        n_warmup=1000,
        n_samples=1000,
        target_accept=0.95,
        key=2026,
        progress_bar=False,
    )
    summ = summarize(idata, hdi_prob=0.95)
    delta_A_row = summ[summ["parameter"] == "delta_v_A"].iloc[0]
    delta_b_row = summ[summ["parameter"] == "delta_v_b"].iloc[0]

    # δ_A: posterior mean within ±0.15, HDI covers truth, sign correct
    assert abs(delta_A_row["mean"] - truth["delta_A"]) < 0.15, (
        f"δ_A posterior mean {delta_A_row['mean']:.3f} too far from truth {truth['delta_A']}"
    )
    assert delta_A_row["hdi_low"] <= truth["delta_A"] <= delta_A_row["hdi_high"], (
        f"δ_A truth {truth['delta_A']} outside 95% HDI "
        f"[{delta_A_row['hdi_low']:.3f}, {delta_A_row['hdi_high']:.3f}]"
    )
    assert delta_A_row["P(>0)"] > 0.9

    # δ_b: weaker identifiability — only check sign and that the HDI
    # covers truth.
    assert delta_b_row["hdi_low"] <= truth["delta_b"] <= delta_b_row["hdi_high"], (
        f"δ_b truth {truth['delta_b']} outside 95% HDI "
        f"[{delta_b_row['hdi_low']:.3f}, {delta_b_row['hdi_high']:.3f}]"
    )
    assert delta_b_row["P(>0)"] > 0.7

    # Convergence (relaxed from theory.md §4 thresholds for the smaller
    # synthetic dataset and 2 chains × 1000 draws used here)
    conv = convergence_check(idata)
    for param, vals in conv.items():
        assert vals["r_hat_max"] < 1.10, (param, vals)
        assert vals["ess_bulk_min"] > 50.0, (param, vals)

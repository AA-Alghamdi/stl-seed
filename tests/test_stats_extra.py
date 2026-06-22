"""Supplemental stats tests covering edge cases of bootstrap CI and
hierarchical-Bayes data validation.

Targets:

* ``stats/bootstrap.py``. degenerate jackknife (BCa fallback to percentile)
  and ``proportion_wilson_ci`` invalid-n branch (lines 78, 82, 85, 105,
  111, 113, 177, 183, 189, 210, 339-340, 420).
* ``stats/hierarchical_bayes.py``. _hdi degenerate path (322-330) and
  the convergence_check empty-rows skip (403).
* ``stats/tost.py``. df<=0 raise path (172).
"""

from __future__ import annotations

import numpy as np
import pytest

from stl_seed.stats.bootstrap import (
    BootstrapCI,
    bootstrap_diff_ci,
    bootstrap_mean_ci,
    bootstrap_proportion_ci,
    proportion_wilson_ci,
)
from stl_seed.stats.hierarchical_bayes import _hdi
from stl_seed.stats.tost import tost_equivalence

# ---------------------------------------------------------------------------
# bootstrap_mean_ci edge cases
# ---------------------------------------------------------------------------


def test_bootstrap_mean_all_equal_falls_back_gracefully() -> None:
    """All-equal sample -> jackknife std is zero -> BCa falls back to
    percentile interval (which is degenerate but finite)."""
    xs = np.full(20, 3.14)
    ci = bootstrap_mean_ci(xs, n_resamples=200, method="bca", key=0)
    # statistic and endpoints all equal the constant.
    assert ci.statistic == pytest.approx(3.14)
    assert ci.lower == pytest.approx(3.14)
    assert ci.upper == pytest.approx(3.14)


def test_bootstrap_mean_invalid_ci_raises() -> None:
    with pytest.raises(ValueError, match="confidence level"):
        bootstrap_mean_ci(np.array([1.0, 2.0]), ci=0.0, key=0)
    with pytest.raises(ValueError, match="confidence level"):
        bootstrap_mean_ci(np.array([1.0, 2.0]), ci=1.5, key=0)


def test_bootstrap_methods_basic_runs_without_jackknife() -> None:
    """basic and percentile methods should not need jackknife."""
    xs = np.linspace(-1, 1, 20)
    for method in ("basic", "percentile"):
        ci = bootstrap_mean_ci(xs, n_resamples=200, method=method, key=0)
        assert ci.method == method
        assert isinstance(ci, BootstrapCI)


def test_bootstrap_diff_unpaired_one_empty_returns_nan() -> None:
    """Empty B array -> CI is NaN."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([np.nan, np.nan])
    ci = bootstrap_diff_ci(a, b, paired=False, n_resamples=100, key=0)
    assert np.isnan(ci.statistic)
    assert np.isnan(ci.lower)
    assert np.isnan(ci.upper)


def test_bootstrap_diff_unpaired_basic_method() -> None:
    """Unpaired diff with the basic method runs."""
    rng = np.random.default_rng(0)
    a = rng.normal(loc=1.0, size=30)
    b = rng.normal(loc=0.0, size=30)
    ci = bootstrap_diff_ci(a, b, paired=False, method="basic", n_resamples=200, key=0)
    assert ci.method == "basic"


def test_bootstrap_proportion_n_zero_raises() -> None:
    with pytest.raises(ValueError, match="n must be positive"):
        bootstrap_proportion_ci(successes=0, n=0)


def test_proportion_wilson_n_zero_raises() -> None:
    with pytest.raises(ValueError, match="n must be positive"):
        proportion_wilson_ci(successes=0, n=0)


def test_bootstrap_ci_dataclass_helpers() -> None:
    """as_dict / width / contains helpers."""
    ci = BootstrapCI(
        statistic=0.5,
        lower=0.3,
        upper=0.7,
        n=10,
        n_resamples=100,
        confidence_level=0.95,
        method="bca",
    )
    d = ci.as_dict()
    assert d["statistic"] == 0.5
    assert ci.width == pytest.approx(0.4)
    assert ci.contains(0.5) is True
    assert ci.contains(1.0) is False


# ---------------------------------------------------------------------------
# tost edge cases
# ---------------------------------------------------------------------------


def test_tost_negative_df_raises() -> None:
    """df <= 0 with the t-test reference distribution -> ValueError."""
    with pytest.raises(ValueError, match="df"):
        tost_equivalence(diff=0.0, se=0.1, equivalence_margin=0.5, df=-1)


def test_tost_str_renders_dist_label() -> None:
    """__str__ includes the reference distribution label (z or t(df=...))."""
    z_res = tost_equivalence(diff=0.0, se=0.01, equivalence_margin=0.05)
    assert "z:" in str(z_res)
    t_res = tost_equivalence(diff=0.0, se=0.01, equivalence_margin=0.05, df=20)
    assert "t(df=20)" in str(t_res)


def test_tost_as_dict_round_trip() -> None:
    res = tost_equivalence(diff=0.0, se=0.01, equivalence_margin=0.05, df=10)
    d = res.as_dict()
    assert d["diff"] == 0.0
    assert d["df"] == 10


# ---------------------------------------------------------------------------
# _hdi degenerate paths
# ---------------------------------------------------------------------------


def test_hdi_empty_returns_nan_pair() -> None:
    lo, hi = _hdi(np.array([]))
    assert np.isnan(lo) and np.isnan(hi)


def test_hdi_tiny_sample_falls_back_to_quantile() -> None:
    """For a 2-sample input the integer interval-width index is degenerate;
    _hdi falls back to the symmetric-quantile interval."""
    lo, hi = _hdi(np.array([1.0, 5.0]), prob=0.5)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo <= hi


def test_hdi_returns_tighter_when_well_defined() -> None:
    samples = np.linspace(0.0, 1.0, 100)
    lo, hi = _hdi(samples, prob=0.5)
    assert (hi - lo) <= 0.6  # roughly the central 50%

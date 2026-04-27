"""Bootstrap confidence intervals for paper-grade reporting.

Three CI methods are exposed via the ``method`` keyword:

* ``"bca"``. bias-corrected and accelerated [Efron 1987,
  DOI:10.1080/01621459.1987.10478410]. Preferred default. Adjusts the
  percentile endpoints with a bias correction ``z_0`` (computed from the
  fraction of bootstrap statistics below the point estimate) and an
  acceleration ``a`` (computed via jackknife). Approximately
  second-order accurate; corrects the percentile method's failure on
  skewed statistics.

* ``"percentile"``. naive empirical-quantile interval [Efron 1979,
  DOI:10.1214/aos/1176344552]. First-order accurate; assumes the
  bootstrap distribution is roughly symmetric around the true value.

* ``"basic"``. pivotal interval ``(2θ̂ − q_{1-α/2}, 2θ̂ − q_{α/2})``.
  Useful when the bootstrap distribution is not centred at the point
  estimate.

The implementations use ``numpy.random.Generator`` so RNG state is
explicit and seedable. We do not use ``scipy.stats.bootstrap`` because
(a) we want the exact same code path for paired and unpaired
differences, and (b) we want the BCa intermediate quantities exposed
for tests.

``statistic``, ``lower``, ``upper``, ``n``, ``n_resamples``,
``confidence_level``, ``method``) intentionally matches the user's
preferred style, but the implementation here is independent.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapCI:
    """A point estimate paired with a bootstrap confidence interval.

    Fields
    ------
    statistic:
        Point estimate computed on the original (un-resampled) data.
    lower, upper:
        Two-sided CI endpoints at confidence level ``confidence_level``.
    n:
        Sample size (paired sample size for paired diffs).
    n_resamples:
        Number of bootstrap replicates used.
    confidence_level:
        Two-sided coverage target, e.g. ``0.95``.
    method:
        ``"bca"``, ``"percentile"``, or ``"basic"``.
    """

    statistic: float
    lower: float
    upper: float
    n: int
    n_resamples: int
    confidence_level: float
    method: str

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)

    @property
    def width(self) -> float:
        return float(self.upper - self.lower)

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_VALID_METHODS = ("bca", "percentile", "basic")


def _validate_method(method: str) -> str:
    m = method.lower()
    if m not in _VALID_METHODS:
        raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}")
    return m


def _validate_ci(ci: float) -> float:
    if not (0.0 < ci < 1.0):
        raise ValueError(f"confidence level must be in (0, 1), got {ci}")
    return float(ci)


def _seeded_rng(key: int | np.random.Generator | None) -> np.random.Generator:
    if key is None:
        return np.random.default_rng()
    if isinstance(key, np.random.Generator):
        return key
    return np.random.default_rng(int(key))


def _resample_indices(n: int, n_resamples: int, rng: np.random.Generator) -> np.ndarray:
    """Vectorized iid bootstrap resample indices, shape ``(n_resamples, n)``."""
    return rng.integers(0, n, size=(n_resamples, n))


def _percentile_endpoints(boot_stats: np.ndarray, ci: float) -> tuple[float, float]:
    """Empirical-quantile percentile interval."""
    alpha = 1.0 - ci
    lo = float(np.quantile(boot_stats, alpha / 2.0, method="linear"))
    hi = float(np.quantile(boot_stats, 1.0 - alpha / 2.0, method="linear"))
    return lo, hi


def _basic_endpoints(boot_stats: np.ndarray, theta_hat: float, ci: float) -> tuple[float, float]:
    """Basic / pivotal interval: ``2 θ̂ − q_{1−α/2},  2 θ̂ − q_{α/2}``."""
    alpha = 1.0 - ci
    q_hi = float(np.quantile(boot_stats, 1.0 - alpha / 2.0, method="linear"))
    q_lo = float(np.quantile(boot_stats, alpha / 2.0, method="linear"))
    return 2.0 * theta_hat - q_hi, 2.0 * theta_hat - q_lo


def _bca_endpoints(
    boot_stats: np.ndarray,
    theta_hat: float,
    jackknife_stats: np.ndarray,
    ci: float,
) -> tuple[float, float]:
    """BCa interval [Efron 1987]. Falls back to percentile if degenerate.

    Bias correction:
        z0 = Φ^{-1}( (#{boot < θ̂} + 0.5 · #{boot == θ̂}) / B )

    Acceleration:
        a = Σ (mean(jack) − jack_i)^3 / [ 6 · (Σ (mean(jack) − jack_i)^2)^{3/2} ]

    Adjusted endpoints:
        α₁ = Φ( z0 + (z0 + z_{α/2}) / (1 − a (z0 + z_{α/2})) )
        α₂ = Φ( z0 + (z0 + z_{1−α/2}) / (1 − a (z0 + z_{1−α/2})) )
    """
    alpha = 1.0 - ci
    z_lo = sp_stats.norm.ppf(alpha / 2.0)
    z_hi = sp_stats.norm.ppf(1.0 - alpha / 2.0)

    below = float(np.mean(boot_stats < theta_hat))
    equal = float(np.mean(boot_stats == theta_hat))
    p_below = below + 0.5 * equal
    # Clamp to avoid ±inf at the boundaries
    p_below = min(max(p_below, 1e-12), 1.0 - 1e-12)
    z0 = float(sp_stats.norm.ppf(p_below))

    jack_mean = float(np.mean(jackknife_stats))
    diffs = jack_mean - jackknife_stats
    num = float(np.sum(diffs**3))
    den = 6.0 * (float(np.sum(diffs**2)) ** 1.5)
    if den == 0.0 or not math.isfinite(num / den):
        # Degenerate jackknife (e.g., all-equal sample). back off to percentile
        return _percentile_endpoints(boot_stats, ci)
    a_hat = num / den

    def _adjust(z_q: float) -> float:
        denom = 1.0 - a_hat * (z0 + z_q)
        if denom <= 0:
            return float("nan")
        return float(sp_stats.norm.cdf(z0 + (z0 + z_q) / denom))

    alpha_lo = _adjust(z_lo)
    alpha_hi = _adjust(z_hi)
    if not (math.isfinite(alpha_lo) and math.isfinite(alpha_hi)):
        return _percentile_endpoints(boot_stats, ci)
    lo = float(np.quantile(boot_stats, alpha_lo, method="linear"))
    hi = float(np.quantile(boot_stats, alpha_hi, method="linear"))
    return lo, hi


def _finalize(
    statistic: float,
    boot_stats: np.ndarray,
    jackknife_stats: np.ndarray | None,
    ci: float,
    method: str,
    n: int,
    n_resamples: int,
) -> BootstrapCI:
    if method == "percentile":
        lo, hi = _percentile_endpoints(boot_stats, ci)
    elif method == "basic":
        lo, hi = _basic_endpoints(boot_stats, statistic, ci)
    else:  # bca
        if jackknife_stats is None:
            raise RuntimeError("BCa requires jackknife stats")
        lo, hi = _bca_endpoints(boot_stats, statistic, jackknife_stats, ci)
    return BootstrapCI(
        statistic=float(statistic),
        lower=float(lo),
        upper=float(hi),
        n=int(n),
        n_resamples=int(n_resamples),
        confidence_level=float(ci),
        method=method,
    )


# ---------------------------------------------------------------------------
# Public bootstrap APIs
# ---------------------------------------------------------------------------


def bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    method: str = "bca",
    key: int | np.random.Generator | None = 0,
) -> BootstrapCI:
    """Bootstrap CI for the sample mean.

    Parameters
    ----------
    values:
        1-D array of observations. NaN/Inf values are dropped.
    n_resamples:
        Number of bootstrap replicates. The standard minimum for
        publication-grade CIs is 10,000 [Efron 1979,
        DOI:10.1214/aos/1176344552]; use a larger value for tighter
        BCa endpoints if compute permits.
    ci:
        Two-sided confidence level, e.g. ``0.95``.
    method:
        ``"bca"`` (default), ``"percentile"``, or ``"basic"``.
    key:
        Seed for ``np.random.default_rng`` (or a ``Generator`` directly).
        Pass ``None`` for non-deterministic resampling.
    """

    method = _validate_method(method)
    ci = _validate_ci(ci)
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        nan = float("nan")
        return BootstrapCI(nan, nan, nan, 0, n_resamples, ci, method)
    if n == 1:
        v = float(arr[0])
        return BootstrapCI(v, v, v, 1, n_resamples, ci, method)

    rng = _seeded_rng(key)
    idx = _resample_indices(n, n_resamples, rng)
    boot_stats = arr[idx].mean(axis=1)

    jack = None
    if method == "bca":
        jack = _jackknife_mean(arr)

    return _finalize(
        statistic=float(arr.mean()),
        boot_stats=boot_stats,
        jackknife_stats=jack,
        ci=ci,
        method=method,
        n=n,
        n_resamples=n_resamples,
    )


def _jackknife_mean(arr: np.ndarray) -> np.ndarray:
    """Leave-one-out jackknife estimates of the mean."""
    n = arr.size
    total = float(arr.sum())
    return (total - arr) / (n - 1)


def bootstrap_diff_ci(
    values_a: Sequence[float] | np.ndarray,
    values_b: Sequence[float] | np.ndarray,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    paired: bool = True,
    method: str = "bca",
    key: int | np.random.Generator | None = 0,
) -> BootstrapCI:
    """Bootstrap CI for ``mean(values_a) − mean(values_b)``.

    Parameters
    ----------
    values_a, values_b:
        Observation arrays. If ``paired=True`` both must have the same
        length and entries are paired by position; the bootstrap
        resamples pair indices jointly. If ``paired=False`` the two
        samples are resampled independently.
    paired:
        See above.
    method:
        For ``paired=True``: ``"bca"``, ``"percentile"``, or
        ``"basic"`` are all supported. For ``paired=False``: BCa is
        supported via paired-sample jackknife on the longer sample with
        a fallback to percentile if the jackknife is degenerate.
    """

    method = _validate_method(method)
    ci = _validate_ci(ci)
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)

    if paired:
        if a.shape != b.shape:
            raise ValueError(f"paired bootstrap requires equal shapes; got {a.shape} vs {b.shape}")
        diffs = a - b
        return bootstrap_mean_ci(diffs, n_resamples=n_resamples, ci=ci, method=method, key=key)

    finite_a = a[np.isfinite(a)]
    finite_b = b[np.isfinite(b)]
    na, nb = finite_a.size, finite_b.size
    if na == 0 or nb == 0:
        nan = float("nan")
        return BootstrapCI(nan, nan, nan, min(na, nb), n_resamples, ci, method)

    rng = _seeded_rng(key)
    idx_a = _resample_indices(na, n_resamples, rng)
    idx_b = _resample_indices(nb, n_resamples, rng)
    boot_stats = finite_a[idx_a].mean(axis=1) - finite_b[idx_b].mean(axis=1)

    statistic = float(finite_a.mean() - finite_b.mean())
    jack = None
    if method == "bca":
        # Combined-sample jackknife on the difference of means: drop one
        # observation from whichever sample shares the index modulo the
        # combined size. Standard reference: Efron & Tibshirani 1993,
        # "An Introduction to the Bootstrap", §14.3.
        jack_a = _jackknife_mean(finite_a)
        jack_b = _jackknife_mean(finite_b)
        # Jackknife of the difference under the leave-one-out replication
        # scheme: a stack of (n_a + n_b) leave-one-outs.
        mean_b_full = float(finite_b.mean())
        mean_a_full = float(finite_a.mean())
        jack = np.concatenate([jack_a - mean_b_full, mean_a_full - jack_b])
    return _finalize(
        statistic=statistic,
        boot_stats=boot_stats,
        jackknife_stats=jack,
        ci=ci,
        method=method,
        n=min(na, nb),
        n_resamples=n_resamples,
    )


def bootstrap_proportion_ci(
    successes: int,
    n: int,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    method: str = "bca",
    key: int | np.random.Generator | None = 0,
) -> BootstrapCI:
    """Bootstrap CI for a sample proportion ``p̂ = successes / n``.

    Resamples a synthetic 0/1 array of length ``n`` with the observed
    success count. For paper-grade proportion CIs the closed-form
    Wilson interval (``scipy.stats.binomtest(...).proportion_ci(method=
    "wilson")``) is generally preferred. see ``BootstrapCI.method`` for
    a ``"wilson"`` route which is provided as a one-liner via
    ``proportion_wilson_ci``. The bootstrap implementation here is
    primarily for consistency with the rest of the API and for cases
    where the user is composing the proportion estimator with a
    nonlinear transform downstream.
    """

    method = _validate_method(method)
    ci = _validate_ci(ci)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not (0 <= successes <= n):
        raise ValueError(f"successes must be in [0, n]; got {successes}/{n}")
    arr = np.zeros(n, dtype=float)
    arr[:successes] = 1.0
    return bootstrap_mean_ci(arr, n_resamples=n_resamples, ci=ci, method=method, key=key)


def proportion_wilson_ci(successes: int, n: int, ci: float = 0.95) -> BootstrapCI:
    """Closed-form Wilson score interval [Wilson 1927,
    DOI:10.1080/01621459.1927.10502953].

    Returned as a ``BootstrapCI`` with ``method="wilson"`` and
    ``n_resamples=0`` so it slots into the same reporting helpers.
    """

    ci = _validate_ci(ci)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    res = sp_stats.binomtest(successes, n).proportion_ci(confidence_level=ci, method="wilson")
    p_hat = successes / n
    return BootstrapCI(
        statistic=float(p_hat),
        lower=float(res.low),
        upper=float(res.high),
        n=int(n),
        n_resamples=0,
        confidence_level=ci,
        method="wilson",
    )


__all__ = [
    "BootstrapCI",
    "bootstrap_mean_ci",
    "bootstrap_diff_ci",
    "bootstrap_proportion_ci",
    "proportion_wilson_ci",
]

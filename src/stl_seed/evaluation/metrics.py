"""Per-spec evaluation metrics.

These are the building blocks the harness reduces over to produce its
final ``EvalResults``. All functions accept either ``jax.Array`` or
``numpy.ndarray`` and return Python floats / tuples (the JIT boundary
ends inside ``Simulator.simulate`` — these summary stats are
post-processing on small arrays).

Definitions
-----------

* **Success rate**: ``Pr[ρ(τ, φ) > 0]``. Implemented as the fraction
  of finite ρ values that exceed ``0`` (NaN/Inf are excluded; per
  ``paper/architecture.md`` NaN policy, the upstream simulator should
  have filtered or zeroed these already, but we exclude defensively).

* **Best-of-N success**: ``Pr_{S ⊂ samples, |S| = N}[max_{i ∈ S} ρ_i > 0]``,
  evaluated *exactly* by sample reuse. Given a per-seed array of K
  samples, ``bon_success(rhos, N)`` returns the fraction of seeds for
  which the maximum of the first ``N`` samples is positive. The
  ``bon_success_curve`` helper returns the BoN-success vector at all
  budgets in a passed list.

* **ρ margin**: ``(mean(ρ), IQR(ρ))``. The IQR (Q3 − Q1) is reported
  alongside the mean as a robust dispersion measure.

* **Goodhart gap**: ``mean(R_proxy) − mean(R_gold)``, where ``R_proxy``
  is the σ-squashed ρ on the training spec and ``R_gold`` the
  σ-squashed ρ on the held-out tightened ``φ_gold`` (paper §6).
  Positive values quantify how much the trained policy over-optimizes
  the proxy reward relative to the gold reward — the operational
  definition of the spec-completeness term in the Goodhart
  decomposition theorem of §6.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_numpy(x: jnp.ndarray | np.ndarray | Sequence[float]) -> np.ndarray:
    """Convert to numpy and drop non-finite entries."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _finite(arr: np.ndarray) -> np.ndarray:
    return arr[np.isfinite(arr)]


# ---------------------------------------------------------------------------
# Success rate
# ---------------------------------------------------------------------------


def success_rate(rhos: jnp.ndarray | np.ndarray | Sequence[float]) -> float:
    """Fraction of trajectories with ρ > 0.

    Parameters
    ----------
    rhos:
        1-D array of robustness values. Non-finite entries are excluded
        from both numerator and denominator. Returns ``nan`` if the
        finite subarray is empty.
    """
    arr = _finite(_to_numpy(rhos))
    if arr.size == 0:
        return float("nan")
    return float((arr > 0).mean())


# ---------------------------------------------------------------------------
# Best-of-N success
# ---------------------------------------------------------------------------


def bon_success(
    rhos_per_seed: jnp.ndarray | np.ndarray,
    n: int,
) -> float:
    """Best-of-N success probability with sample reuse.

    Parameters
    ----------
    rhos_per_seed:
        2-D array of shape ``(n_seeds, K)`` where ``K >= n``. Entry
        ``[s, k]`` is the ρ of the ``k``-th draw from the policy under
        seed ``s``. We use the *first* ``n`` columns (sample reuse:
        BoN-K success at K is read from the same draws used at K' > K).
    n:
        BoN budget; must satisfy ``1 <= n <= K``.

    Returns
    -------
    Probability ``Pr[max(ρ_{:, :n}) > 0]`` averaged over seeds, with
    NaN entries treated as ``-inf`` (i.e., they cannot be the
    successful sample).
    """
    arr = np.asarray(rhos_per_seed, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"rhos_per_seed must be 2-D (n_seeds, K), got shape {arr.shape}")
    n_seeds, k_max = arr.shape
    if not (1 <= n <= k_max):
        raise ValueError(f"BoN budget n={n} must be in [1, {k_max}]")
    if n_seeds == 0:
        return float("nan")
    sub = arr[:, :n]
    # NaNs cannot be the success — treat as -inf.
    sub = np.where(np.isfinite(sub), sub, -np.inf)
    best = sub.max(axis=1)
    return float((best > 0).mean())


def bon_success_curve(
    rhos_per_seed: jnp.ndarray | np.ndarray,
    budgets: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128),
) -> dict[int, float]:
    """BoN success at each budget in ``budgets`` via sample reuse."""
    arr = np.asarray(rhos_per_seed, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"rhos_per_seed must be 2-D (n_seeds, K), got shape {arr.shape}")
    out: dict[int, float] = {}
    k_max = arr.shape[1]
    for n in budgets:
        if n < 1 or n > k_max:
            out[int(n)] = float("nan")
            continue
        out[int(n)] = bon_success(arr, n)
    return out


# ---------------------------------------------------------------------------
# ρ margin
# ---------------------------------------------------------------------------


def rho_margin(
    rhos: jnp.ndarray | np.ndarray | Sequence[float],
) -> tuple[float, float]:
    """Return ``(mean(ρ), IQR(ρ))`` over finite entries.

    IQR is ``Q3 − Q1`` per the standard definition (linear-interpolation
    quantiles, matching numpy default).
    """
    arr = _finite(_to_numpy(rhos))
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    q1, q3 = np.quantile(arr, [0.25, 0.75])
    return mean, float(q3 - q1)


# ---------------------------------------------------------------------------
# Goodhart gap (paper §6)
# ---------------------------------------------------------------------------


def goodhart_gap(
    rho_proxy: jnp.ndarray | np.ndarray | Sequence[float],
    rho_gold: jnp.ndarray | np.ndarray | Sequence[float],
    kappa: float = 1.0,
) -> float:
    """Measured spec-completeness gap of paper §6.

    Defined as ``mean(σ(ρ_proxy / κ)) − mean(σ(ρ_gold / κ))`` where σ is
    the logistic. Returns ``0.0`` exactly when ``ρ_proxy ≡ ρ_gold``
    (identical specs evaluated on identical trajectories).

    Both arrays must have the same shape — the comparison is paired by
    trajectory index. NaN entries are dropped pairwise.
    """
    a = _to_numpy(rho_proxy)
    b = _to_numpy(rho_gold)
    if a.shape != b.shape:
        raise ValueError(f"goodhart_gap requires equal shapes; got {a.shape} vs {b.shape}")
    finite = np.isfinite(a) & np.isfinite(b)
    a = a[finite]
    b = b[finite]
    if a.size == 0:
        return float("nan")
    sa = 1.0 / (1.0 + np.exp(-a / float(kappa)))
    sb = 1.0 / (1.0 + np.exp(-b / float(kappa)))
    return float(sa.mean() - sb.mean())


__all__ = [
    "success_rate",
    "bon_success",
    "bon_success_curve",
    "rho_margin",
    "goodhart_gap",
]

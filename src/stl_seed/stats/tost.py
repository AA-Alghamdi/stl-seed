"""Two One-Sided Tests (TOST) for equivalence.

This is the formal test registered for hypothesis H1 in
``paper/theory.md`` §3 ("soft is *as good as* hard within Δ = 0.05").

References
----------
[Schuirmann 1987, DOI:10.1007/BF01068419]
    Donald J. Schuirmann, "A comparison of the two one-sided tests
    procedure and the power approach for assessing the equivalence of
    average bioavailability." J. Pharmacokinet. Biopharm. 15:657-680.
    Original derivation in the bioequivalence setting.

[Lakens 2017, DOI:10.1177/1948550617697177]
    Daniel Lakens, "Equivalence tests: A practical primer for t-tests,
    correlations, and meta-analyses." Soc. Psychol. Personal. Sci.
    8(4):355-362. Accessible treatment with worked examples.

Procedure
---------
For an observed effect ``diff = θ̂_a − θ̂_b`` with standard error ``se``,
TOST tests two one-sided null hypotheses simultaneously at level α:

    H0_lower : diff <= −Δ      (the effect is at least Δ below zero)
    H0_upper : diff >=  +Δ     (the effect is at least Δ above zero)

Rejecting *both* one-sided nulls is the equivalence conclusion: the
effect is bounded inside ``(−Δ, +Δ)`` with confidence ``1 − 2α`` (note:
``1 − 2α``, not ``1 − α``, because the joint test reuses both tails of
a single two-sided ``1 − 2α`` interval).

The default is the t-distribution (Schuirmann's original derivation);
when ``df`` is omitted the normal approximation is used. Both t-test
variants and the z-test reduce to the same arithmetic on the test
statistics, only the reference distribution differs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from scipy import stats as sp_stats


@dataclass(frozen=True)
class TOSTResult:
    """Two one-sided test result for an equivalence comparison.

    Fields
    ------
    diff:
        Observed effect ``θ̂_a − θ̂_b``.
    se:
        Standard error of ``diff``.
    equivalence_margin:
        Absolute margin ``Δ`` such that ``|diff| < Δ`` is the
        equivalence claim.
    alpha:
        Per-one-sided-test level. The joint coverage of "equivalent" is
        ``1 − 2 alpha``.
    p_lower:
        p-value for ``H0_lower : diff ≤ −Δ`` (rejecting → ``diff > −Δ``).
    p_upper:
        p-value for ``H0_upper : diff ≥ +Δ`` (rejecting → ``diff < +Δ``).
    p_tost:
        ``max(p_lower, p_upper)``. the equivalence p-value under the
        intersection-union test principle [Berger & Hsu 1996,
        DOI:10.1214/ss/1032280304].
    equivalent:
        ``True`` iff both p-values are below ``alpha``.
    df:
        Degrees of freedom for the t-test, or ``None`` if z-test was
        used.
    """

    diff: float
    se: float
    equivalence_margin: float
    alpha: float
    p_lower: float
    p_upper: float
    p_tost: float
    equivalent: bool
    df: int | float | None

    def as_dict(self) -> dict[str, float | bool | int | None]:
        return asdict(self)

    def __str__(self) -> str:
        verdict = "equivalent" if self.equivalent else "not equivalent"
        dist = "z" if self.df is None else f"t(df={self.df})"
        return (
            f"TOST {dist}: diff={self.diff:+.4f} se={self.se:.4f} "
            f"Δ=±{self.equivalence_margin:.4f}  "
            f"p_low={self.p_lower:.4g} p_up={self.p_upper:.4g} "
            f"p_tost={self.p_tost:.4g}  α={self.alpha:.3f} → {verdict}"
        )


def tost_equivalence(
    diff: float,
    se: float,
    equivalence_margin: float,
    alpha: float = 0.05,
    df: int | float | None = None,
) -> TOSTResult:
    """Two one-sided tests for equivalence.

    Parameters
    ----------
    diff:
        Observed effect ``θ̂_a − θ̂_b``.
    se:
        Standard error of the observed effect (must be > 0).
    equivalence_margin:
        Symmetric margin ``Δ > 0``. The equivalence claim is
        ``|θ_a − θ_b| < Δ``.
    alpha:
        Per-one-sided level. Default ``0.05`` (the standard registered
        choice in our preregistration).
    df:
        Degrees of freedom for the t reference distribution. If
        ``None``, uses the normal approximation (z-test). For our
        paired-sample analyses ``df = n − 1`` where ``n`` is the number
        of paired (m, v, f, i, s) cells.

    Returns
    -------
    TOSTResult

    Raises
    ------
    ValueError
        If ``se <= 0`` or ``equivalence_margin <= 0`` or ``alpha`` is
        not in ``(0, 0.5)``.

    Notes
    -----
    Test statistics:
        t_lower = (diff − (−Δ)) / se = (diff + Δ) / se
        t_upper = (diff − (+Δ)) / se = (diff − Δ) / se

    p-values (one-sided, in the rejection direction):
        p_lower = P(T <= t_lower under H0_lower)?. actually we test
            "diff > −Δ" so reject H0_lower when t_lower is large; thus
            p_lower = 1 − F(t_lower) = SF(t_lower).
        p_upper = P(T >= t_upper under H0_upper)?. reject H0_upper
            when t_upper is small; thus p_upper = F(t_upper) = CDF(t_upper).

    The intersection-union test [Berger & Hsu 1996,
    DOI:10.1214/ss/1032280304] gives the equivalence p-value as the
    maximum of the two one-sided p-values.
    """

    if se <= 0.0:
        raise ValueError(f"standard error must be positive, got {se}")
    if equivalence_margin <= 0.0:
        raise ValueError(f"equivalence_margin must be positive, got {equivalence_margin}")
    if not (0.0 < alpha < 0.5):
        raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")

    delta = float(equivalence_margin)
    t_lower = (float(diff) + delta) / float(se)
    t_upper = (float(diff) - delta) / float(se)

    if df is None:
        # Standard-normal reference (z-test)
        p_lower = float(sp_stats.norm.sf(t_lower))
        p_upper = float(sp_stats.norm.cdf(t_upper))
    else:
        if df <= 0:
            raise ValueError(f"df must be positive, got {df}")
        p_lower = float(sp_stats.t.sf(t_lower, df=df))
        p_upper = float(sp_stats.t.cdf(t_upper, df=df))

    p_tost = max(p_lower, p_upper)
    equivalent = (p_lower < alpha) and (p_upper < alpha)

    return TOSTResult(
        diff=float(diff),
        se=float(se),
        equivalence_margin=delta,
        alpha=float(alpha),
        p_lower=p_lower,
        p_upper=p_upper,
        p_tost=float(p_tost),
        equivalent=bool(equivalent),
        df=df,
    )


__all__ = [
    "TOSTResult",
    "tost_equivalence",
]

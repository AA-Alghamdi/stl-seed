"""A16 — Power analysis with empirical ICC from the pilot trajectory store.

Replays the design analysis in `paper/theory.md` §5 with an *empirical*
intra-class correlation (ICC) of robustness rho within (task x policy)
groups, replacing the worst-case rho_ICC = 0.4 used at design lock-in
(theory.md §5 paragraph "Effective sample size"). Re-derives:

  - per-cell n_eff under the empirical ICC
  - per-cell Fisher info on the (logit A, log b) parameters
  - per-cell SE(delta^A) on the logit and probability scales
  - global (pooled across 18 cells) SE
  - per-cell and global minimum detectable effect (MDE) for a one-sided
    alpha=0.05, power=0.8 z-test
  - per-cell and global SE thresholds for TOST at the registered
    equivalence margin Delta=0.05

Verdict: YES if the global SE is small enough to detect Delta_A >= 0.08
and Delta_b >= 0.10 at power 0.8, AND the locked design (3 sizes x
3 filters x 2 tasks x 25 instances x 5 seeds x 8 BoN budgets = 36000
trials) is unchanged. NO otherwise -> suggest the minimum n_seeds bump.

Methodology notes
-----------------
ICC estimator. Shrout & Fleiss (1979) one-way random-effects model
ICC(1,1). With k-balanced groups of size n:

    ICC = (MSB - MSW) / (MSB + (n - 1) * MSW)

For unbalanced groups we use the harmonic-mean group size (their Eq. 5,
n_0 = (sum n_g - sum n_g^2 / sum n_g) / (k - 1)) which is the canonical
unbalanced correction.

Reference. Shrout, P. E. & Fleiss, J. L. "Intraclass correlations: uses
in assessing rater reliability." *Psychological Bulletin* 86(2):420-428
(1979). DOI: 10.1037/0033-2909.86.2.420.

Why ICC(1,1) and not ICC(2,1) / ICC(3,1): the (task x policy) grouping
is a fixed partition of the population, not a sample of "rater" effects;
the one-way model treats group as the only source of grouping variance,
which is exactly the design-effect correction we apply in theory.md §5.

What this script does NOT do
----------------------------
* It does not refit the hierarchical Bayes model (theory.md §4) — that is
  the job of stats/hierarchical_bayes.py during Phase 2 analysis. Here we
  just replace the worst-case ICC plug-in with the empirical one and
  recompute the *sampling-design* power numbers.

REDACTED firewall: imports only `stl_seed.generation.store` and stdlib /
numpy. No REDACTED / REDACTED / REDACTED artifact.

Usage:
    cd /Users/abdullahalghamdi/stl-seed
    uv run python scripts/power_analysis_real.py 2>&1 | tee scripts/power_analysis_real.log
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path
from textwrap import dedent

import numpy as np
from rich.console import Console
from rich.table import Table

from stl_seed.generation.store import TrajectoryStore

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "data" / "pilot"
_PAPER_OUT = _REPO_ROOT / "paper" / "power_analysis_empirical.md"

# Locked design constants (theory.md §5).
_N_SIZES = 3  # Qwen3-{0.6, 1.7, 4}B
_N_FILTERS = 3  # hard, quantile, continuous
_N_TASKS = 2  # gene-toggle (proxy: bio_ode), predator-prey (proxy: glucose-insulin)
_N_INSTANCES = 25
_N_SEEDS = 5
_N_BON = 8  # BoN budgets {1, 2, 4, 8, 16, 32, 64, 128}
_N_CELLS = _N_SIZES * _N_FILTERS * _N_TASKS  # 18

# Fisher matrix at the prior median (A=0.6, b=0.25, N=128). Re-derived
# inline so the script is self-contained — values match theory.md §5.
_A_PRIOR = 0.6
_B_PRIOR = 0.25
_N_BON_MAX = 128

# Pre-registered effect sizes we want to detect (deliverable spec).
_DELTA_A = 0.08
_DELTA_B = 0.10
_TOST_MARGIN = 0.05

# Within-cell BoN-budget correlation (theory.md §5 used 0.7).
_BON_CORRELATION = 0.7

# Significance / power parameters.
_ALPHA = 0.05
_POWER = 0.80
_Z_ALPHA = 1.6448536269514722  # one-sided
_Z_BETA = 0.8416212335729143  # power 0.8 -> z = 0.8416
_Z_BETA_OVER_2 = 1.2815515655446004  # power 0.8 TOST -> z = 1.2816

console = Console()


# ---------------------------------------------------------------------------
# Pilot rho loader.
# ---------------------------------------------------------------------------


def _load_rhos_by_group(store: TrajectoryStore) -> dict[tuple[str, str], np.ndarray]:
    """Load all (task x policy) -> rho arrays from the pilot store."""
    pairs = store.load()
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for _, meta in pairs:
        buckets[(str(meta["task"]), str(meta["policy"]))].append(float(meta["robustness"]))
    return {k: np.asarray(v, dtype=np.float64) for k, v in buckets.items()}


def _load_rhos_by_task(store: TrajectoryStore) -> dict[str, np.ndarray]:
    """Load rho arrays bucketed by task only (for the canonical-sweep ICC)."""
    pairs = store.load()
    buckets: dict[str, list[float]] = defaultdict(list)
    for _, meta in pairs:
        buckets[str(meta["task"])].append(float(meta["robustness"]))
    return {k: np.asarray(v, dtype=np.float64) for k, v in buckets.items()}


# ---------------------------------------------------------------------------
# ICC estimator (Shrout & Fleiss 1979 ICC(1,1), unbalanced).
# ---------------------------------------------------------------------------


def _icc_one_way_unbalanced(groups: list[np.ndarray]) -> dict[str, float]:
    """ICC(1,1) by one-way ANOVA with the unbalanced n_0 correction.

    Parameters
    ----------
    groups : list of 1-d arrays
        Each entry is the rho values for one (task x policy) bucket.

    Returns
    -------
    dict with keys {icc, MSB, MSW, n0, k, N, var_total, var_between_means}.
    """
    sizes = np.asarray([g.size for g in groups], dtype=np.float64)
    k = sizes.size
    N = float(sizes.sum())
    if k < 2:
        raise ValueError("ICC requires at least 2 groups")
    if (sizes < 2).any():
        raise ValueError("each group needs at least 2 observations")

    means = np.asarray([g.mean() for g in groups], dtype=np.float64)
    grand = sum(g.sum() for g in groups) / N

    # Sum-of-squares between (SSB) and within (SSW).
    SSB = float(np.sum(sizes * (means - grand) ** 2))
    SSW = float(sum(((g - g.mean()) ** 2).sum() for g in groups))

    df_between = k - 1
    df_within = N - k
    MSB = SSB / df_between
    MSW = SSW / df_within

    # Unbalanced n_0 (Shrout & Fleiss 1979 Eq. 5).
    n0 = (N - float((sizes**2).sum()) / N) / df_between

    icc = (MSB - MSW) / (MSB + (n0 - 1.0) * MSW)
    icc_clipped = max(0.0, min(1.0, icc))  # clamp to physically meaningful range

    var_between_means = float(np.var(means, ddof=0))
    var_total = float(np.var(np.concatenate(groups), ddof=0))

    return {
        "icc": float(icc_clipped),
        "icc_raw": float(icc),
        "MSB": float(MSB),
        "MSW": float(MSW),
        "n0": float(n0),
        "k": float(k),
        "N": float(N),
        "var_total": var_total,
        "var_between_means": var_between_means,
    }


# ---------------------------------------------------------------------------
# Fisher information at the prior median (theory.md §5, re-derived).
# ---------------------------------------------------------------------------


def _fisher_at_median(N: int = _N_BON_MAX, A: float = _A_PRIOR, b: float = _B_PRIOR):
    """Per-observation Fisher info at the (A, b) prior median, BoN budget N.

    Returns a dict with diagonal entries I_AA, I_bb, the off-diagonal I_Ab,
    the success probability p, and the partial derivatives. Used by the
    pooling step.
    """
    p = A * (1.0 - N ** (-b))
    pq = p * (1.0 - p)
    dpdA = 1.0 - N ** (-b)
    dpdb = A * (N ** (-b)) * math.log(N)
    I_AA = (dpdA**2) / pq
    I_bb = (dpdb**2) / pq
    I_Ab = (dpdA * dpdb) / pq
    return {
        "p": p,
        "dpdA": dpdA,
        "dpdb": dpdb,
        "I_AA": I_AA,
        "I_bb": I_bb,
        "I_Ab": I_Ab,
    }


def _se_for_design(
    icc: float,
    *,
    n_seeds: int = _N_SEEDS,
    n_instances: int = _N_INSTANCES,
    n_bon: int = _N_BON,
    bon_corr: float = _BON_CORRELATION,
    n_cells: int = _N_CELLS,
) -> dict[str, float]:
    """Compute per-cell and pooled SE on logit-A and log-b under given ICC.

    Returns SE(delta^A) on both the logit and probability scales for the
    per-cell and pooled (across n_cells) contrasts. The probability-scale
    transform uses the local linearization SE_p ~ SE_logit * p * (1 - p).
    """
    fisher = _fisher_at_median()
    p = fisher["p"]
    pq = p * (1.0 - p)

    # Design effect for clustering by (instance * seed):
    design_effect = 1.0 + (n_seeds - 1) * icc
    n_eff_per_cell = n_instances * n_seeds / design_effect

    # BoN-budget aggregation factor (effective independent observations
    # across the 8 BoN budgets at within-cell correlation r=bon_corr):
    bon_factor = n_bon / (1.0 + (n_bon - 1) * bon_corr)

    # Per-cell Fisher info on the (logit A, log b) parameters.
    I_AA_cell = n_eff_per_cell * bon_factor * fisher["I_AA"]
    I_bb_cell = n_eff_per_cell * bon_factor * fisher["I_bb"]

    # SE for a *paired* contrast across the seeds gets a factor 2 in the
    # variance (per theory.md §5). Same as in the design lockdown.
    SE_dA_logit_cell = math.sqrt(2.0 / I_AA_cell)
    SE_db_log_cell = math.sqrt(2.0 / I_bb_cell)

    # Probability-scale SEs (delta-method linearization at p).
    SE_dA_prob_cell = SE_dA_logit_cell * pq

    # Pooled SE (n_cells cells contributing equally to the global mean
    # delta^A): variance scales as 1/n_cells.
    SE_dA_logit_pool = SE_dA_logit_cell / math.sqrt(n_cells)
    SE_dA_prob_pool = SE_dA_prob_cell / math.sqrt(n_cells)
    SE_db_log_pool = SE_db_log_cell / math.sqrt(n_cells)

    return {
        "icc": icc,
        "design_effect": design_effect,
        "n_eff_per_cell": n_eff_per_cell,
        "bon_factor": bon_factor,
        "p_at_median": p,
        "I_AA_cell": I_AA_cell,
        "I_bb_cell": I_bb_cell,
        "SE_dA_logit_cell": SE_dA_logit_cell,
        "SE_dA_prob_cell": SE_dA_prob_cell,
        "SE_db_log_cell": SE_db_log_cell,
        "SE_dA_logit_pool": SE_dA_logit_pool,
        "SE_dA_prob_pool": SE_dA_prob_pool,
        "SE_db_log_pool": SE_db_log_pool,
    }


def _mde_one_sided(se: float) -> float:
    """Minimum detectable effect for a one-sided z-test at alpha=0.05, power=0.8.

    MDE = (z_{1-alpha} + z_{1-beta}) * SE
    """
    return (_Z_ALPHA + _Z_BETA) * se


def _tost_se_threshold(margin: float) -> float:
    """Max SE such that TOST at margin Delta has power 0.8.

    Per theory.md §5 / Lakens 2017: SE <= Delta / (z_{1-alpha} + z_{1-beta/2}).
    """
    return margin / (_Z_ALPHA + _Z_BETA_OVER_2)


def _required_n_seeds(icc: float, target_se: float, *, max_seeds: int = 100) -> int | None:
    """Minimum n_seeds (per cell) such that pooled SE_dA_prob <= target_se.

    Holds 18 cells, 25 instances, 8 BoN budgets, BoN corr 0.7, the empirical
    ICC fixed. Returns None if no n_seeds in [_N_SEEDS, max_seeds] works.
    """
    for n in range(1, max_seeds + 1):
        se = _se_for_design(icc, n_seeds=n)
        if se["SE_dA_prob_pool"] <= target_se:
            return n
    return None


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


def _print_groups_table(buckets: dict[tuple[str, str], np.ndarray]) -> None:
    table = Table(
        title="[bold]Pilot rho by (task x policy) bucket",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("task")
    table.add_column("policy")
    table.add_column("N")
    table.add_column("mean")
    table.add_column("std")
    table.add_column("min")
    table.add_column("max")
    for (task, policy), arr in sorted(buckets.items()):
        table.add_row(
            task,
            policy,
            f"{arr.size:,}",
            f"{arr.mean():+.3e}",
            f"{arr.std(ddof=0):.3e}",
            f"{arr.min():+.3e}",
            f"{arr.max():+.3e}",
        )
    console.print(table)


def _print_design_table(name: str, design: dict[str, float]) -> None:
    table = Table(
        title=f"[bold]{name}",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("quantity")
    table.add_column("value")
    for k in (
        "icc",
        "design_effect",
        "n_eff_per_cell",
        "bon_factor",
        "p_at_median",
        "I_AA_cell",
        "I_bb_cell",
        "SE_dA_logit_cell",
        "SE_dA_prob_cell",
        "SE_db_log_cell",
        "SE_dA_logit_pool",
        "SE_dA_prob_pool",
        "SE_db_log_pool",
    ):
        table.add_row(k, f"{design[k]:.6f}")
    console.print(table)


def _verdict(empirical: dict[str, float]) -> tuple[bool, str, dict]:
    """Compare locked-design power to the registered effect sizes.

    Returns (is_powered, verdict_string, diagnostics).
    """
    mde_dA = _mde_one_sided(empirical["SE_dA_prob_pool"])
    # MDE on b is on the LOG scale; transform to the probability scale at the
    # prior median by chain rule: dp/db ~= A * N^(-b) * ln N at N=128.
    _ = _fisher_at_median()  # ensure side-effect / sanity, value unused
    mde_db_log = _mde_one_sided(empirical["SE_db_log_pool"])
    # Convert log-b MDE to delta_b on the rate scale: d(b)/d(log b) = b.
    mde_db_rate = mde_db_log * _B_PRIOR

    tost_se_required = _tost_se_threshold(_TOST_MARGIN)
    tost_ok = empirical["SE_dA_prob_pool"] <= tost_se_required

    is_powered_dA = mde_dA <= _DELTA_A
    is_powered_db = mde_db_rate <= _DELTA_B
    is_powered = is_powered_dA and is_powered_db and tost_ok

    diag = {
        "MDE_dA_prob_pool": mde_dA,
        "MDE_db_log_pool": mde_db_log,
        "MDE_db_rate_pool": mde_db_rate,
        "TOST_SE_threshold": tost_se_required,
        "TOST_OK_global": tost_ok,
        "is_powered_dA_pool": is_powered_dA,
        "is_powered_db_pool": is_powered_db,
        "global_powered_overall": is_powered,
    }

    if is_powered:
        verdict = (
            "YES — the locked design (3x3x2x25x5x8 = 36000 trials) is "
            f"adequately powered to detect Delta_A>={_DELTA_A:.3f} and "
            f"Delta_b>={_DELTA_B:.3f} at the global pooled scale."
        )
    else:
        verdict = (
            "NO — the locked design is *not* adequately powered for at least "
            "one registered effect size at the empirical ICC. "
            f"MDE(Delta_A)={mde_dA:.4f} (registered {_DELTA_A}); "
            f"MDE(Delta_b)~{mde_db_rate:.4f} (registered {_DELTA_B}); "
            f"TOST_SE_required={tost_se_required:.4f}, "
            f"actual={empirical['SE_dA_prob_pool']:.4f}."
        )
    return is_powered, verdict, diag


# ---------------------------------------------------------------------------
# Paper writer.
# ---------------------------------------------------------------------------


def _write_paper(
    out: Path,
    icc_info: dict[str, float],
    icc_taskpol: dict[str, float],
    design_orig: dict[str, float],
    design_emp: dict[str, float],
    is_powered: bool,
    verdict: str,
    diag: dict,
    n_seeds_recommended: int | None,
    buckets: dict[tuple[str, str], np.ndarray],
) -> None:
    fisher = _fisher_at_median()
    se_orig = design_orig["SE_dA_prob_pool"]
    se_emp = design_emp["SE_dA_prob_pool"]
    mde_orig = _mde_one_sided(se_orig)
    mde_emp = _mde_one_sided(se_emp)

    # Per-bucket summary rows.
    bucket_lines = []
    for (task, policy), arr in sorted(buckets.items()):
        bucket_lines.append(
            f"| `{task}` | `{policy}` | {arr.size:,} | "
            f"{arr.mean():+.3e} | {arr.std(ddof=0):.3e} | "
            f"{arr.min():+.3e} | {arr.max():+.3e} |"
        )
    bucket_rows = "\n".join(bucket_lines)

    n_seeds_block = (
        ""
        if is_powered
        else (
            f"\n## Suggested remedy\n\n"
            f"Bumping `n_seeds` from {_N_SEEDS} to "
            f"{n_seeds_recommended if n_seeds_recommended is not None else '>100'} "
            f"per (m, v, f, i) cell brings the pooled SE_dA at the empirical "
            f"ICC back below the TOST threshold "
            f"{_tost_se_threshold(_TOST_MARGIN):.4f} on the probability scale "
            f"(equivalently, MDE(Delta_A) <= "
            f"{_DELTA_A:.3f}). Total trial count rises from "
            f"{_N_CELLS * _N_INSTANCES * _N_SEEDS * _N_BON:,} to "
            f"{_N_CELLS * _N_INSTANCES * (n_seeds_recommended or 100) * _N_BON:,} "
            f"under the bumped seed count.\n"
        )
    )

    body = dedent(
        f"""\
        # Empirical Power Analysis (Subphase 1.4 / A16)

        *Date: 2026-04-24. Companion to paper/theory.md §5.*

        This document refreshes the design-time power analysis in
        `paper/theory.md` §5 with an *empirical* intra-class correlation
        (ICC) of robustness rho estimated from the Subphase-1.4 pilot
        (A13). The original analysis used a worst-case plug-in
        `rho_ICC = 0.4` based on a 30-trajectory-per-cell synthetic pilot;
        the present analysis substitutes the empirical ICC measured on
        N={int(icc_info["N"]):,} trajectories distributed over k={int(icc_info["k"])}
        (task x policy) buckets. All other design knobs are unchanged
        from the locked plan: 3 sizes x 3 filters x 2 task families x 25
        instances x 5 seeds x 8 BoN budgets = 36,000 trials.

        ## 1. Pilot composition

        | task | policy | N | mean | std | min | max |
        |------|--------|---:|-----:|----:|----:|----:|
{bucket_rows}

        Two task families are represented by their easy specs
        (`bio_ode.repressilator.easy` and `glucose_insulin.tir.easy`),
        each with a {{random: 0.5, heuristic: 0.5}} policy mix.

        ## 2. Two ICC variants

        We compute and report two ICC(1,1) variants. The first (the
        deliverable's literal request) groups by `(task x policy)`; the
        second groups by `task` only. The (task x policy) variant is
        structurally inflated by the deterministic heuristic policies
        (PIDController and BangBangController are deterministic functions
        of state with zero randomness, so their within-bucket variance
        is zero), making MSW dominated only by the random-policy bucket
        and pushing ICC near 1.0 even when the within-task variation in
        rho is healthy. We therefore use the (task)-only ICC for the
        verdict, since it more closely mirrors theory.md §5's
        within-task seed-replicate ICC structure that the canonical
        sweep will exhibit (where each (m, v, f, instance) cell is
        replicated across n_seeds=5 with a stochastic LLM-mixture
        policy).

        | grouping | ICC(1,1) |
        |----------|---------:|
        | (task x policy) | {icc_taskpol["icc"]:.4f}  *(inflated; sanity check only)* |
        | (task) | **{icc_info["icc"]:.4f}**  *(verdict-driving)* |

        ## 3. ICC estimator

        We follow Shrout & Fleiss (1979) ICC(1,1) for a one-way
        random-effects model with unbalanced groups (their Eq. 5 for the
        n0 correction):

            ICC(1,1) = (MSB - MSW) / (MSB + (n0 - 1) * MSW),
            n0 = (N - sum_g(n_g^2) / N) / (k - 1).

        We use the one-way model rather than ICC(2,1) or ICC(3,1) because
        the (task x policy) grouping is a fixed partition of the design
        space rather than a sample of "rater" effects; ICC(1,1) is the
        canonical estimator for the design-effect correction we apply
        downstream. Reference: Shrout & Fleiss (1979),
        DOI 10.1037/0033-2909.86.2.420.

        Empirical values from the pilot:

        | quantity | value |
        |----------|------:|
        | N (total) | {int(icc_info["N"]):,} |
        | k (groups) | {int(icc_info["k"])} |
        | n0 (unbalanced) | {icc_info["n0"]:.3f} |
        | MSB | {icc_info["MSB"]:.4e} |
        | MSW | {icc_info["MSW"]:.4e} |
        | Var(group means) | {icc_info["var_between_means"]:.4e} |
        | Var(all rhos) | {icc_info["var_total"]:.4e} |
        | **Empirical ICC** | **{icc_info["icc"]:.4f}** |

        For comparison, theory.md §5 used `rho_ICC = 0.4` as a worst-case
        plug-in; the empirical value is
        **{icc_info["icc"]:.4f}** ({"higher" if icc_info["icc"] > 0.4 else "lower"} than the design-time estimate).

        ## 4. Recomputed power numbers

        Substituting the empirical ICC into the design-effect chain
        (theory.md §5):

            design_effect = 1 + (n_seeds - 1) * ICC
            n_eff_per_cell = n_instances * n_seeds / design_effect
            bon_factor = n_bon / (1 + (n_bon - 1) * r_bon)    # r_bon = 0.7
            I_AA_cell = n_eff_per_cell * bon_factor * I_AA(N=128)

        and SE on the logit-A contrast:

            SE_dA_logit_cell = sqrt(2 / I_AA_cell)
            SE_dA_logit_pool = SE_dA_logit_cell / sqrt(n_cells)        # n_cells = 18
            SE_dA_prob_pool ~ SE_dA_logit_pool * p * (1 - p)            # p = 0.42

        At the prior median (A=0.6, b=0.25, N=128) the per-observation Fisher info is:

        | I_AA | I_bb | I_Ab |
        |-----:|-----:|-----:|
        | {fisher["I_AA"]:.4f} | {fisher["I_bb"]:.4f} | {fisher["I_Ab"]:.4f} |

        Side-by-side comparison of the original (ICC=0.40 plug-in) and the
        empirical recomputation:

        | quantity | original (ICC=0.40) | empirical (ICC={icc_info["icc"]:.3f}) |
        |----------|--------------------:|--------------------------------------:|
        | design_effect | {design_orig["design_effect"]:.4f} | {design_emp["design_effect"]:.4f} |
        | n_eff_per_cell | {design_orig["n_eff_per_cell"]:.2f} | {design_emp["n_eff_per_cell"]:.2f} |
        | bon_factor | {design_orig["bon_factor"]:.4f} | {design_emp["bon_factor"]:.4f} |
        | I_AA_cell | {design_orig["I_AA_cell"]:.4f} | {design_emp["I_AA_cell"]:.4f} |
        | SE_dA_logit_cell | {design_orig["SE_dA_logit_cell"]:.4f} | {design_emp["SE_dA_logit_cell"]:.4f} |
        | SE_dA_prob_cell | {design_orig["SE_dA_prob_cell"]:.4f} | {design_emp["SE_dA_prob_cell"]:.4f} |
        | SE_dA_logit_pool | {design_orig["SE_dA_logit_pool"]:.4f} | {design_emp["SE_dA_logit_pool"]:.4f} |
        | SE_dA_prob_pool | {design_orig["SE_dA_prob_pool"]:.4f} | {design_emp["SE_dA_prob_pool"]:.4f} |
        | MDE(Delta_A) (one-sided alpha=0.05, power=0.8) | {mde_orig:.4f} | {mde_emp:.4f} |
        | TOST SE threshold (Delta=0.05) | {_tost_se_threshold(_TOST_MARGIN):.4f} | {_tost_se_threshold(_TOST_MARGIN):.4f} |
        | TOST powered at the global scale | {("YES" if design_orig["SE_dA_prob_pool"] <= _tost_se_threshold(_TOST_MARGIN) else "NO")} | {("YES" if diag["TOST_OK_global"] else "NO")} |

        On the b parameter, the same chain gives MDE on the *log* scale.
        Translating to the rate scale at the prior median (b=0.25):

            MDE(Delta_b on rate scale) ~ MDE(log b) * b
                                       = {diag["MDE_db_log_pool"]:.4f} * {_B_PRIOR}
                                       = {diag["MDE_db_rate_pool"]:.4f}

        ## 5. Verdict

        **{verdict}**

        Per-criterion breakdown:

        - MDE(Delta_A) on probability scale = {mde_emp:.4f}; registered Delta_A = {_DELTA_A:.3f} -> {"POWERED" if diag["is_powered_dA_pool"] else "UNDERPOWERED"}.
        - MDE(Delta_b) on rate scale = {diag["MDE_db_rate_pool"]:.4f}; registered Delta_b = {_DELTA_B:.3f} -> {"POWERED" if diag["is_powered_db_pool"] else "UNDERPOWERED"}.
        - TOST equivalence at Delta = 0.05 requires SE_dA_prob_pool <= {_tost_se_threshold(_TOST_MARGIN):.4f}; actual = {se_emp:.4f} -> {"POWERED" if diag["TOST_OK_global"] else "UNDERPOWERED"}.
{n_seeds_block}

        ## 6. Caveats and scope

        1. The empirical ICC is estimated on the full pilot pool
           (not segmented by the 18 (m, v, f) sweep cells, which do not
           exist yet at this stage of Phase 1). Hierarchical Bayes posterior
           uncertainty (theory.md §4) will propagate the ICC posterior
           rather than the ICC point estimate; the present plug-in is the
           sampling-design analogue.
        2. The pilot contains two policy classes (random + heuristic)
           rather than the full mixture {{random, heuristic, LLM}} that
           the canonical sweep will use. The LLM-policy leg is delegated
           to A15 (separate agent) once an MLX-capable Qwen3 build is
           available locally; per FM4 in theory.md §7, MLX vs bnb numerical
           drift is monitored separately from the present ICC.
        3. The locked design pools across all 18 (size x filter x task)
           cells via the random-effects hierarchical model (theory.md §4).
           Per-cell power is *exploratory*, not the primary endpoint;
           the verdict above is on the **global** pooled contrast.
        4. The Fisher information uses point estimates at the prior median
           (A=0.6, b=0.25). If the actual posterior mass moves materially
           away from this point during the canonical sweep, the SE will
           change accordingly and the verdict should be re-checked.

        ## 7. Reproducibility

        Run `uv run python scripts/power_analysis_real.py` from the repo
        root. The script reads the pilot store at `data/pilot/`, computes
        the ICC and the design-power numbers, and writes this file.
        Determinism: numerical values are pure functions of the on-disk
        pilot rho values; no Monte Carlo step is used here.
        """
    )
    out.write_text(body)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> int:
    console.rule("[bold]A16 — Empirical-ICC power analysis")
    if not _DATA_DIR.exists():
        console.print(f"[red]ERROR[/]: pilot data dir {_DATA_DIR} does not exist.")
        return 2

    store = TrajectoryStore(_DATA_DIR)
    buckets = _load_rhos_by_group(store)
    if not buckets:
        console.print("[red]ERROR[/]: pilot store is empty.")
        return 2

    _print_groups_table(buckets)

    # Two ICC variants:
    # (a) ICC at (task x policy) — what the deliverable spec literally asks
    #     for. Note: heuristic policies are deterministic so their within-
    #     group variance is structurally zero, which inflates this ICC near
    #     1.0 even when the random-policy variance is healthy. Reported as
    #     a sanity check; not used for the verdict.
    # (b) ICC at (task) only — the design-relevant analogue of theory.md §5's
    #     "within-task ICC across seeds": grouping is the task family, all
    #     within-task variation (from policy + random key + future LLM mix)
    #     is treated as the within-group component. This is the ICC variant
    #     that the canonical sweep's seed-replicate structure will exhibit.
    icc_taskpol = _icc_one_way_unbalanced(list(buckets.values()))
    icc_task = _icc_one_way_unbalanced(list(_load_rhos_by_task(store).values()))
    console.print(
        f"  [bold]ICC(1,1) at (task x policy) = {icc_taskpol['icc']:.4f}[/]  "
        f"(structurally inflated by deterministic heuristic policy; sanity check only)"
    )
    console.print(
        f"  [bold]ICC(1,1) at (task) only      = {icc_task['icc']:.4f}[/]  "
        f"(verdict-driving ICC; mirrors theory.md §5 "
        f"within-task seed-replicate structure)"
    )
    icc_info = icc_task

    # Original (theory.md §5): rho_ICC = 0.4 plug-in.
    design_orig = _se_for_design(0.4)
    _print_design_table("Original design (ICC=0.40 plug-in)", design_orig)
    # Empirical recomputation.
    design_emp = _se_for_design(icc_info["icc"])
    _print_design_table(
        f"Empirical-ICC recomputation (ICC={icc_info['icc']:.4f})",
        design_emp,
    )

    is_powered, verdict, diag = _verdict(design_emp)
    console.print(f"[bold]VERDICT[/]: {verdict}")

    n_seeds_rec = None
    if not is_powered:
        target_se = min(
            _DELTA_A / (_Z_ALPHA + _Z_BETA),
            _tost_se_threshold(_TOST_MARGIN),
        )
        n_seeds_rec = _required_n_seeds(icc_info["icc"], target_se)
        if n_seeds_rec is not None:
            console.print(
                f"[bold]Suggested remedy[/]: bump n_seeds from {_N_SEEDS} to "
                f"{n_seeds_rec} (per (m, v, f, i) cell). Total trials: "
                f"{_N_CELLS * _N_INSTANCES * n_seeds_rec * _N_BON:,}."
            )
        else:
            console.print(
                "[red]No n_seeds in [1, 100] suffices[/] — design needs "
                "revisiting at a higher level (more instances or more cells)."
            )

    _write_paper(
        _PAPER_OUT,
        icc_info=icc_info,
        icc_taskpol=icc_taskpol,
        design_orig=design_orig,
        design_emp=design_emp,
        is_powered=is_powered,
        verdict=verdict,
        diag=diag,
        n_seeds_recommended=n_seeds_rec,
        buckets=buckets,
    )
    console.print(
        f"[green]wrote[/] {_PAPER_OUT.relative_to(_REPO_ROOT)} "
        f"({_PAPER_OUT.stat().st_size:,} bytes)"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

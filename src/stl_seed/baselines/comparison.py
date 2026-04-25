"""Empirical PAV vs STL-rho comparison on a held-out trajectory split.

Three quantities of interest, all computed on the same train/test split
of an existing trajectory corpus:

1. *Predictive AUC vs terminal success*. Each verifier emits a per-trajectory
   score; ROC-AUC against the binary terminal-success label measures how
   well the score discriminates winners from losers. Baseline-free,
   threshold-free; the standard "is the verifier informative" check.

2. *Spearman rank correlation* between the verifier score and (a) terminal
   rho, (b) terminal success. Captures monotonic rather than threshold
   discrimination.

3. *Sample efficiency*. Refit PAV at varying train-set sizes
   ``n_train in {100, 250, 500, 1000, 2000, ...}`` and chart its
   held-out AUC against STL-rho's (which is *constant* in n_train, since
   STL-rho needs no training). The "crossover n" --- the smallest train
   size at which PAV's AUC matches STL-rho's --- is the headline figure.

-------------
Imports only stdlib + numpy + scipy + the in-repo PAV / Trajectory /
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import jax
import numpy as np

from stl_seed.baselines.pav import PAVProcessRewardModel
from stl_seed.specs import REGISTRY, STLSpec
from stl_seed.stl.evaluator import evaluate_robustness
from stl_seed.tasks._trajectory import Trajectory

# ---------------------------------------------------------------------------
# Result containers.
# ---------------------------------------------------------------------------


@dataclass
class SampleEfficiencyPoint:
    """One point on the PAV sample-efficiency curve."""

    n_train: int
    pav_auc: float
    pav_train_seconds: float
    pav_train_loss_final: float
    pav_val_loss_final: float
    n_test: int


@dataclass
class ComparisonResult:
    """Outputs of :func:`compare_pav_vs_stl`.

    Fields mirror the headline quantities the comparison script reports.
    """

    task: str
    spec_key: str
    n_train: int
    n_test: int
    seed: int

    pav_scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    stl_scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    test_terminal_success: np.ndarray = field(default_factory=lambda: np.zeros(0))

    pav_auc: float = float("nan")
    stl_auc: float = float("nan")
    pav_spearman_success: float = float("nan")
    stl_spearman_success: float = float("nan")
    pav_spearman_rho: float = float("nan")
    stl_spearman_rho: float = float("nan")

    pav_train_seconds: float = 0.0
    stl_score_seconds: float = 0.0
    pav_train_loss_final: float = float("nan")
    pav_val_loss_final: float = float("nan")

    sample_efficiency: list[SampleEfficiencyPoint] = field(default_factory=list)
    crossover_n_train: int | None = None  # smallest n_train s.t. pav_auc >= stl_auc


# ---------------------------------------------------------------------------
# Metric helpers.
# ---------------------------------------------------------------------------


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC without scikit-learn (Mann-Whitney U formulation).

    AUC = P(score(positive) > score(negative)). Uses the rank-sum identity
    AUC = (U_+) / (n_+ * n_-) where U_+ = sum_{i in pos} rank_i - n_+(n_+ + 1)/2.
    Ties contribute 0.5 (handled by ``scipy.stats.rankdata`` average ranks).

    Returns NaN if either class has zero samples.
    """
    y = np.asarray(y_true).astype(np.float64)
    s = np.asarray(y_score).astype(np.float64)
    if y.shape != s.shape:
        raise ValueError(f"Shape mismatch: y={y.shape} s={s.shape}")
    n_pos = int((y > 0.5).sum())
    n_neg = int((y <= 0.5).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Robust rank with NaN handling.
    finite = np.isfinite(s)
    if not finite.all():
        s = s.copy()
        s[~finite] = np.nanmin(s[finite]) if finite.any() else 0.0
    from scipy.stats import rankdata

    ranks = rankdata(s, method="average")
    u_pos = float(ranks[y > 0.5].sum() - n_pos * (n_pos + 1) / 2.0)
    return u_pos / (n_pos * n_neg)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, NaN-safe. Returns NaN if undefined."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return float("nan")
    from scipy.stats import spearmanr

    r, _ = spearmanr(x[mask], y[mask])
    return float(r)


# ---------------------------------------------------------------------------
# STL scoring helper.
# ---------------------------------------------------------------------------


def _stl_score(spec: STLSpec, trajectory: Trajectory) -> float:
    """Wrap STL rho into a plain Python float, NaN-safe."""
    rho = float(evaluate_robustness(spec, trajectory))
    if not np.isfinite(rho):
        # Replace +/- inf with large finite sentinel; never NaN-poison the AUC.
        return 1e9 if rho > 0 else -1e9
    return rho


# ---------------------------------------------------------------------------
# Main comparison routine.
# ---------------------------------------------------------------------------


def compare_pav_vs_stl(
    trajectories: list[Trajectory],
    terminal_success: np.ndarray,
    spec: STLSpec | str,
    n_train: int = 1000,
    n_test: int = 500,
    seed: int = 42,
    sample_efficiency_grid: list[int] | None = None,
    pav_n_epochs: int = 50,
    pav_lr: float = 1e-3,
    pav_hidden: int = 256,
    pav_dropout: float = 0.1,
    k_neighbors: int = 16,
    task_name: str = "<unspecified>",
    verbose: bool = False,
    terminal_rho: np.ndarray | None = None,
) -> ComparisonResult:
    """Empirically compare PAV vs STL-rho as process verifiers.

    Splits ``trajectories`` into train and test, fits PAV on train, scores
    test trajectories with both PAV and STL-rho, and reports AUC + Spearman
    + sample-efficiency curve.

    Parameters
    ----------
    trajectories:
        Full corpus to draw the train/test split from. Must be at least
        ``n_train + n_test`` items.
    terminal_success:
        Shape ``(N,)`` 0/1 indicators aligned with ``trajectories``.
    spec:
        Either an :class:`STLSpec` instance or a registry key string.
    n_train, n_test:
        Sizes of the train and test splits. The remainder of the corpus is
        unused (kept aside as the candidate pool for sample-efficiency
        sweeps drawing larger train sizes).
    seed:
        Controls the train/test permutation and PAV initialization.
    sample_efficiency_grid:
        Optional list of train-set sizes to refit PAV at and measure AUC.
        ``None`` -> ``[100, 250, 500, 1000]`` clipped to the available
        train-pool size.
    pav_n_epochs, pav_lr, pav_hidden, pav_dropout, k_neighbors:
        PAV hyperparameters (see :class:`PAVProcessRewardModel.fit`).
    task_name:
        Human-readable task label for the result record.
    verbose:
        Print progress.
    terminal_rho:
        Optional precomputed terminal STL rho per trajectory (so we don't
        recompute it for every Spearman call). If ``None``, computed on
        the test split.

    Returns
    -------
    ComparisonResult
    """
    if isinstance(spec, str):
        if spec not in REGISTRY:
            raise KeyError(f"Spec key {spec!r} not in REGISTRY")
        spec_obj = REGISTRY[spec]
        spec_key = spec
    else:
        spec_obj = spec
        spec_key = spec.name

    ts = np.asarray(terminal_success, dtype=np.float64)
    if ts.shape[0] != len(trajectories):
        raise ValueError(
            f"terminal_success length {ts.shape[0]} != n_trajectories {len(trajectories)}"
        )

    n_total = len(trajectories)
    if n_train + n_test > n_total:
        raise ValueError(f"n_train + n_test ({n_train} + {n_test}) exceeds n_total ({n_total})")

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n_total)
    train_idx = perm[:n_train]
    test_idx = perm[n_train : n_train + n_test]
    pool_idx = perm[n_train + n_test :]  # candidate pool for sample-eff sweep

    train_trajs = [trajectories[int(i)] for i in train_idx]
    train_ts = ts[train_idx]
    test_trajs = [trajectories[int(i)] for i in test_idx]
    test_ts = ts[test_idx]

    # ---- STL: zero training cost, just score the test set. -----------------
    t0 = time.time()
    stl_test_scores = np.array([_stl_score(spec_obj, t) for t in test_trajs])
    stl_score_seconds = float(time.time() - t0)

    # ---- PAV: fit on train, score test. ------------------------------------
    pav = PAVProcessRewardModel(
        state_dim=int(np.asarray(train_trajs[0].states).shape[1]),
        hidden=pav_hidden,
        dropout=pav_dropout,
    )
    key = jax.random.PRNGKey(int(seed))
    history = pav.fit(
        train_trajs,
        train_ts,
        n_epochs=pav_n_epochs,
        lr=pav_lr,
        key=key,
        k_neighbors=k_neighbors,
        verbose=verbose,
    )
    pav_test_scores = pav.score_batch(test_trajs)

    # ---- Metrics. ----------------------------------------------------------
    pav_auc = _roc_auc(test_ts, pav_test_scores)
    stl_auc = _roc_auc(test_ts, stl_test_scores)

    if terminal_rho is None:
        # Compute terminal rho on the test split for Spearman correlation.
        # (For pure STL this equals stl_test_scores; we recompute generally
        # so callers can pass an externally-validated terminal_rho.)
        terminal_rho_test = stl_test_scores.copy()
    else:
        terminal_rho_test = np.asarray(terminal_rho)[test_idx]

    pav_sp_succ = _spearman(pav_test_scores, test_ts)
    stl_sp_succ = _spearman(stl_test_scores, test_ts)
    pav_sp_rho = _spearman(pav_test_scores, terminal_rho_test)
    stl_sp_rho = _spearman(stl_test_scores, terminal_rho_test)

    # ---- Sample-efficiency sweep (optional). -------------------------------
    sample_eff: list[SampleEfficiencyPoint] = []
    if sample_efficiency_grid is None:
        # Default: a couple of points below n_train and one at n_train.
        candidate = [100, 250, 500, 1000]
        sample_efficiency_grid = [n for n in candidate if n <= n_train and n <= n_total - n_test]
        if n_train not in sample_efficiency_grid and n_train <= n_total - n_test:
            sample_efficiency_grid.append(n_train)

    for n_t in sample_efficiency_grid:
        if n_t < 2:
            continue
        # Build a train pool of size n_t by combining train_idx and pool_idx
        # (use train_idx first so smaller n_t stays nested in larger n_t).
        combined_pool = np.concatenate([train_idx, pool_idx])
        n_t_eff = min(int(n_t), len(combined_pool))
        if n_t_eff < 2:
            continue
        sub_idx = combined_pool[:n_t_eff]
        sub_trajs = [trajectories[int(i)] for i in sub_idx]
        sub_ts = ts[sub_idx]
        pav_sub = PAVProcessRewardModel(
            state_dim=int(np.asarray(sub_trajs[0].states).shape[1]),
            hidden=pav_hidden,
            dropout=pav_dropout,
        )
        key, sub_key = jax.random.split(key)
        sub_hist = pav_sub.fit(
            sub_trajs,
            sub_ts,
            n_epochs=pav_n_epochs,
            lr=pav_lr,
            key=sub_key,
            k_neighbors=min(k_neighbors, max(1, n_t_eff - 1)),
            verbose=False,
        )
        sub_test_scores = pav_sub.score_batch(test_trajs)
        sub_auc = _roc_auc(test_ts, sub_test_scores)
        sample_eff.append(
            SampleEfficiencyPoint(
                n_train=n_t_eff,
                pav_auc=float(sub_auc),
                pav_train_seconds=float(sub_hist["wall_time_s"]),
                pav_train_loss_final=float(sub_hist["train_loss"][-1])
                if sub_hist["train_loss"]
                else float("nan"),
                pav_val_loss_final=float(sub_hist["val_loss"][-1])
                if sub_hist["val_loss"]
                else float("nan"),
                n_test=int(len(test_trajs)),
            )
        )

    # ---- Crossover n_train. ------------------------------------------------
    crossover: int | None = None
    if not np.isnan(stl_auc):
        for pt in sorted(sample_eff, key=lambda p: p.n_train):
            if not np.isnan(pt.pav_auc) and pt.pav_auc >= stl_auc:
                crossover = pt.n_train
                break

    return ComparisonResult(
        task=task_name,
        spec_key=spec_key,
        n_train=int(n_train),
        n_test=int(n_test),
        seed=int(seed),
        pav_scores=pav_test_scores,
        stl_scores=stl_test_scores,
        test_terminal_success=test_ts,
        pav_auc=float(pav_auc),
        stl_auc=float(stl_auc),
        pav_spearman_success=float(pav_sp_succ),
        stl_spearman_success=float(stl_sp_succ),
        pav_spearman_rho=float(pav_sp_rho),
        stl_spearman_rho=float(stl_sp_rho),
        pav_train_seconds=float(history["wall_time_s"]),
        stl_score_seconds=float(stl_score_seconds),
        pav_train_loss_final=float(history["train_loss"][-1])
        if history["train_loss"]
        else float("nan"),
        pav_val_loss_final=float(history["val_loss"][-1]) if history["val_loss"] else float("nan"),
        sample_efficiency=sample_eff,
        crossover_n_train=crossover,
    )


def result_to_summary_dict(result: ComparisonResult) -> dict[str, Any]:
    """Flatten a :class:`ComparisonResult` into a JSON-serializable dict."""
    return {
        "task": result.task,
        "spec_key": result.spec_key,
        "n_train": result.n_train,
        "n_test": result.n_test,
        "seed": result.seed,
        "pav_auc": result.pav_auc,
        "stl_auc": result.stl_auc,
        "pav_spearman_success": result.pav_spearman_success,
        "stl_spearman_success": result.stl_spearman_success,
        "pav_spearman_rho": result.pav_spearman_rho,
        "stl_spearman_rho": result.stl_spearman_rho,
        "pav_train_seconds": result.pav_train_seconds,
        "stl_score_seconds": result.stl_score_seconds,
        "pav_train_loss_final": result.pav_train_loss_final,
        "pav_val_loss_final": result.pav_val_loss_final,
        "crossover_n_train": result.crossover_n_train,
        "sample_efficiency": [
            {
                "n_train": pt.n_train,
                "pav_auc": pt.pav_auc,
                "pav_train_seconds": pt.pav_train_seconds,
                "pav_train_loss_final": pt.pav_train_loss_final,
                "pav_val_loss_final": pt.pav_val_loss_final,
                "n_test": pt.n_test,
            }
            for pt in result.sample_efficiency
        ],
    }


__all__ = [
    "ComparisonResult",
    "SampleEfficiencyPoint",
    "compare_pav_vs_stl",
    "result_to_summary_dict",
]

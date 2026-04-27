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

V2 (2026-04-26): the audit at ``paper/pav_v2.md`` added
:func:`compare_pav_v2_vs_stl`, which differs from
:func:`compare_pav_vs_stl` in three places:

* PAV is fit via ``PAVProcessRewardModel.fit_with_selection`` --- a
  hidden-width / weight-decay grid search with early stopping on val MSE.
* MC labels can come from either the legacy kNN estimator
  (``label_source="knn"``) or the on-policy rollout estimator
  (``label_source="onpolicy"``; see
  :mod:`stl_seed.baselines.pav_rollout`).
* The model-selection report is propagated into the result so the v2
  markdown can show which (hidden, wd) won and at what val MSE.

The legacy :func:`compare_pav_vs_stl` is kept unchanged so the original
``paper/pav_comparison.md`` remains regenerable bit-for-bit.
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


# ---------------------------------------------------------------------------
# V2: model-selection + on-policy-rollout comparison.
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResultV2:
    """Outputs of :func:`compare_pav_v2_vs_stl`.

    Mirrors :class:`ComparisonResult` but adds the model-selection report
    and the (optional) on-policy rollout statistics. Kept as a separate
    dataclass so the v1 schema is not broken for existing serialized
    results.
    """

    task: str
    spec_key: str
    n_train: int
    n_test: int
    seed: int
    label_source: str  # "knn" or "onpolicy"

    pav_scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    stl_scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    test_terminal_success: np.ndarray = field(default_factory=lambda: np.zeros(0))

    pav_auc: float = float("nan")
    stl_auc: float = float("nan")
    pav_spearman_success: float = float("nan")
    stl_spearman_success: float = float("nan")
    pav_spearman_rho: float = float("nan")
    stl_spearman_rho: float = float("nan")

    pav_train_seconds: float = 0.0  # incl. selection wall time
    stl_score_seconds: float = 0.0
    label_compute_seconds: float = 0.0  # MC-label cost (kNN or on-policy)
    pav_train_loss_final: float = float("nan")
    pav_val_loss_final: float = float("nan")
    pav_best_val_mse: float = float("nan")
    pav_best_hidden: int = -1
    pav_best_weight_decay: float = float("nan")
    pav_best_epoch: int = -1

    n_onpolicy_simulations: int = 0  # 0 if label_source != "onpolicy"
    selection_grid: list[dict[str, Any]] = field(default_factory=list)


def _onpolicy_labels_with_split(
    train_trajs: list[Trajectory],
    train_ts: np.ndarray,
    val_trajs: list[Trajectory],
    val_ts: np.ndarray,
    *,
    task: str,
    spec_key: str,
    K: int,
    seed: int,
) -> tuple[Any, Any, float, int]:
    """Run on-policy rollouts on train and val trajectory subsets.

    Returns ``(train_ds, val_ds, wall_time_s, n_simulations)``. The val
    side is rolled out with a *disjoint* RNG seed so the K random tails
    are independent across the train/val boundary. ``train_ts`` and
    ``val_ts`` are unused (terminal success is recomputed inside the
    rollout from rho > 0); they are accepted only to surface mismatched-
    length errors loudly at the call site.
    """
    from stl_seed.baselines.pav_rollout import compute_per_step_mc_labels_onpolicy

    if len(train_ts) != len(train_trajs):
        raise ValueError(
            f"train terminal_success ({len(train_ts)}) != train_trajs ({len(train_trajs)})"
        )
    if len(val_ts) != len(val_trajs):
        raise ValueError(f"val terminal_success ({len(val_ts)}) != val_trajs ({len(val_trajs)})")
    t0 = time.time()
    train_ds, train_stats = compute_per_step_mc_labels_onpolicy(
        train_trajs, spec_key=spec_key, task=task, K=K, seed=int(seed)
    )
    val_ds = None
    val_n_sims = 0
    if val_trajs:
        val_ds, val_stats = compute_per_step_mc_labels_onpolicy(
            val_trajs, spec_key=spec_key, task=task, K=K, seed=int(seed) + 1
        )
        val_n_sims = int(val_stats.n_simulations)
    wall = float(time.time() - t0)
    n_sims = int(train_stats.n_simulations) + val_n_sims
    return train_ds, val_ds, wall, n_sims


def compare_pav_v2_vs_stl(
    trajectories: list[Trajectory],
    terminal_success: np.ndarray,
    spec: STLSpec | str,
    *,
    task: str,
    n_train: int = 1000,
    n_test: int = 500,
    seed: int = 42,
    label_source: str = "onpolicy",
    K_rollout: int = 5,
    hidden_grid: tuple[int, ...] = (64, 128, 256, 512),
    weight_decay_grid: tuple[float, ...] = (0.0, 1e-4, 1e-3, 1e-2),
    pav_n_epochs: int = 100,
    pav_lr: float = 1e-3,
    pav_dropout: float = 0.1,
    early_stopping_patience: int = 5,
    val_frac: float = 0.2,
    k_neighbors: int = 16,
    task_name: str = "<unspecified>",
    verbose: bool = False,
    terminal_rho: np.ndarray | None = None,
) -> ComparisonResultV2:
    """V2 PAV-vs-STL comparison: model selection + (optional) on-policy MC.

    Differences vs :func:`compare_pav_vs_stl`:

    * PAV is fit via :meth:`PAVProcessRewardModel.fit_with_selection`,
      which sweeps ``hidden_grid x weight_decay_grid`` and picks the cell
      with the lowest *trajectory-disjoint* val MSE. Each cell uses
      early-stopping-on-val-MSE-plateau (``early_stopping_patience``).
    * MC labels for both train and val sides come from
      ``label_source in {"knn", "onpolicy"}``. With ``"onpolicy"`` we
      reconstruct the canonical simulator (see
      :mod:`stl_seed.baselines.pav_rollout`) and roll out
      ``K_rollout`` random tails per (trajectory, prefix-length) pair.
    * No sample-efficiency sweep: the v2 design treats the headline
      ``n_train`` as the published number; the v1 sweep is preserved
      for the original ``pav_comparison.md`` artifact.

    Parameters
    ----------
    trajectories, terminal_success, spec, n_train, n_test, seed,
    task_name, verbose, terminal_rho:
        Same meaning as :func:`compare_pav_vs_stl`.
    task:
        Canonical-store task identifier (e.g. ``"glucose_insulin"``,
        ``"bio_ode.repressilator"``). Required when ``label_source ==
        "onpolicy"`` because we reconstruct the simulator from it.
    label_source:
        ``"knn"`` (legacy kNN MC) or ``"onpolicy"`` (Setlur §3.2 fresh
        rollouts). Defaults to ``"onpolicy"`` because that's the
        Setlur-faithful baseline.
    K_rollout:
        Number of i.i.d. random tails per prefix length. Setlur uses
        ``M ~ 8-32``; we default to 5 to keep wall-clock tractable on
        the M-series. Bumping this number reduces MC variance at the
        cost of linear time.
    hidden_grid, weight_decay_grid, pav_n_epochs, pav_lr, pav_dropout,
    early_stopping_patience, val_frac:
        Forwarded to ``fit_with_selection``.
    k_neighbors:
        kNN pool size, used only when ``label_source == "knn"``.

    Returns
    -------
    ComparisonResultV2
    """
    # Local import to avoid a hard dependency on pav at module import
    # time (pav imports JAX, equinox; tests that don't touch v2 should
    # not pay that startup cost).
    from stl_seed.baselines.pav import (
        PAVProcessRewardModel,
        compute_per_step_mc_labels,
    )

    if isinstance(spec, str):
        if spec not in REGISTRY:
            raise KeyError(f"Spec key {spec!r} not in REGISTRY")
        spec_obj = REGISTRY[spec]
        spec_key = spec
    else:
        spec_obj = spec
        spec_key = spec.name

    if label_source not in {"knn", "onpolicy"}:
        raise ValueError(f"label_source must be 'knn' or 'onpolicy', got {label_source!r}")

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

    train_trajs = [trajectories[int(i)] for i in train_idx]
    train_ts = ts[train_idx]
    test_trajs = [trajectories[int(i)] for i in test_idx]
    test_ts = ts[test_idx]

    # ---- STL: zero training cost, just score the test set. -----------------
    t0 = time.time()
    stl_test_scores = np.array([_stl_score(spec_obj, t) for t in test_trajs])
    stl_score_seconds = float(time.time() - t0)

    # ---- Train/val split *within* the train pool. --------------------------
    n_val = max(2, int(round(val_frac * n_train))) if val_frac > 0.0 else 0
    val_perm = np.random.default_rng(int(seed) + 7).permutation(n_train)
    val_local = val_perm[:n_val]
    tr_local = val_perm[n_val:]
    if len(tr_local) < 2:
        tr_local = val_perm
        val_local = np.array([], dtype=np.int64)

    train_inner = [train_trajs[int(i)] for i in tr_local]
    train_ts_inner = train_ts[tr_local]
    val_inner = [train_trajs[int(i)] for i in val_local]
    val_ts_inner = train_ts[val_local]

    # ---- Compute MC labels for both halves. --------------------------------
    if label_source == "onpolicy":
        train_ds, val_ds, label_wall, n_sims = _onpolicy_labels_with_split(
            train_inner,
            train_ts_inner,
            val_inner,
            val_ts_inner,
            task=task,
            spec_key=spec_key,
            K=int(K_rollout),
            seed=int(seed),
        )
    else:
        t_lab = time.time()
        train_ds = compute_per_step_mc_labels(
            train_inner, train_ts_inner, k_neighbors=int(k_neighbors)
        )
        val_ds = None
        if len(val_inner) >= 2:
            val_ds = compute_per_step_mc_labels(
                val_inner,
                val_ts_inner,
                k_neighbors=min(int(k_neighbors), len(val_inner) - 1),
            )
        label_wall = float(time.time() - t_lab)
        n_sims = 0

    if val_ds is None:
        raise RuntimeError(
            "compare_pav_v2_vs_stl requires a non-empty val split; increase n_train or val_frac."
        )

    state_dim = int(np.asarray(train_inner[0].states).shape[1])

    # ---- Hidden-width / weight-decay sweep. --------------------------------
    t_fit = time.time()
    pav, sel_report = PAVProcessRewardModel.fit_with_selection(
        trajectories=train_trajs,  # whole train pool, for shape/checks
        terminal_success=train_ts,
        state_dim=state_dim,
        hidden_grid=hidden_grid,
        weight_decay_grid=weight_decay_grid,
        dropout=pav_dropout,
        n_epochs=int(pav_n_epochs),
        lr=float(pav_lr),
        key=jax.random.PRNGKey(int(seed)),
        val_frac=val_frac,
        k_neighbors=int(k_neighbors),
        early_stopping_patience=int(early_stopping_patience),
        precomputed_train=train_ds,
        precomputed_val=val_ds,
        verbose=bool(verbose),
    )
    fit_wall = float(time.time() - t_fit)
    pav_test_scores = pav.score_batch(test_trajs)

    # ---- Metrics. ----------------------------------------------------------
    pav_auc = _roc_auc(test_ts, pav_test_scores)
    stl_auc = _roc_auc(test_ts, stl_test_scores)
    if terminal_rho is None:
        terminal_rho_test = stl_test_scores.copy()
    else:
        terminal_rho_test = np.asarray(terminal_rho)[test_idx]

    pav_sp_succ = _spearman(pav_test_scores, test_ts)
    stl_sp_succ = _spearman(stl_test_scores, test_ts)
    pav_sp_rho = _spearman(pav_test_scores, terminal_rho_test)
    stl_sp_rho = _spearman(stl_test_scores, terminal_rho_test)

    final_hist = sel_report["final_history"]
    train_loss_list = list(final_hist["train_loss"])
    val_loss_list = list(final_hist["val_loss"])

    return ComparisonResultV2(
        task=task_name,
        spec_key=spec_key,
        n_train=int(n_train),
        n_test=int(n_test),
        seed=int(seed),
        label_source=str(label_source),
        pav_scores=pav_test_scores,
        stl_scores=stl_test_scores,
        test_terminal_success=test_ts,
        pav_auc=float(pav_auc),
        stl_auc=float(stl_auc),
        pav_spearman_success=float(pav_sp_succ),
        stl_spearman_success=float(stl_sp_succ),
        pav_spearman_rho=float(pav_sp_rho),
        stl_spearman_rho=float(stl_sp_rho),
        pav_train_seconds=float(fit_wall),
        stl_score_seconds=float(stl_score_seconds),
        label_compute_seconds=float(label_wall),
        pav_train_loss_final=float(train_loss_list[-1]) if train_loss_list else float("nan"),
        pav_val_loss_final=float(val_loss_list[-1]) if val_loss_list else float("nan"),
        pav_best_val_mse=float(sel_report["best_val_mse"]),
        pav_best_hidden=int(sel_report["best_hidden"]),
        pav_best_weight_decay=float(sel_report["best_weight_decay"]),
        pav_best_epoch=int(sel_report["best_epoch"]),
        n_onpolicy_simulations=int(n_sims),
        selection_grid=list(sel_report["grid"]),
    )


def result_v2_to_summary_dict(result: ComparisonResultV2) -> dict[str, Any]:
    """Flatten a :class:`ComparisonResultV2` into a JSON-serializable dict."""
    return {
        "task": result.task,
        "spec_key": result.spec_key,
        "n_train": result.n_train,
        "n_test": result.n_test,
        "seed": result.seed,
        "label_source": result.label_source,
        "pav_auc": result.pav_auc,
        "stl_auc": result.stl_auc,
        "pav_spearman_success": result.pav_spearman_success,
        "stl_spearman_success": result.stl_spearman_success,
        "pav_spearman_rho": result.pav_spearman_rho,
        "stl_spearman_rho": result.stl_spearman_rho,
        "pav_train_seconds": result.pav_train_seconds,
        "stl_score_seconds": result.stl_score_seconds,
        "label_compute_seconds": result.label_compute_seconds,
        "pav_train_loss_final": result.pav_train_loss_final,
        "pav_val_loss_final": result.pav_val_loss_final,
        "pav_best_val_mse": result.pav_best_val_mse,
        "pav_best_hidden": result.pav_best_hidden,
        "pav_best_weight_decay": result.pav_best_weight_decay,
        "pav_best_epoch": result.pav_best_epoch,
        "n_onpolicy_simulations": result.n_onpolicy_simulations,
        "selection_grid": result.selection_grid,
    }


__all__ = [
    "ComparisonResult",
    "ComparisonResultV2",
    "SampleEfficiencyPoint",
    "compare_pav_v2_vs_stl",
    "compare_pav_vs_stl",
    "result_to_summary_dict",
    "result_v2_to_summary_dict",
]

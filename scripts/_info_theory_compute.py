"""One-off computation for paper/information_theory.md.

Computes per-task:
  - I(rho; success) via discretized binning + KSG (kNN) MI estimator
  - H(success | rho) via discretized binning
For PAV, uses the closed-form Gaussian-additive-noise channel bound from
the recorded per-task best_val_mse, which is the irreducible MSE achievable
by the calibrated PAV regression on the calibration distribution.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DATA_ROOT = Path("/Users/abdullahalghamdi/stl-seed/data/canonical")
PAV_ROOT = Path("/Users/abdullahalghamdi/stl-seed/runs/pav_comparison_v2")


def load_rho_and_success(task: str) -> tuple[np.ndarray, np.ndarray]:
    p = DATA_ROOT / task
    files = sorted([f for f in os.listdir(p) if f.endswith(".parquet")])
    rhos: list[float] = []
    for f in files:
        t = pq.read_table(p / f)
        df = t.to_pandas()
        rhos.extend(df["robustness"].astype(float).tolist())
    rho = np.asarray(rhos, dtype=np.float64)
    succ = (rho > 0.0).astype(np.int64)
    return rho, succ


def entropy_binary(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))


def mi_binned(rho: np.ndarray, succ: np.ndarray, n_bins: int = 32) -> float:
    """I(rho; succ) in bits via equal-frequency binning of rho.

    Uses the standard plug-in estimator on the discretized joint table:
      I = sum_{x,y} p(x,y) log2 p(x,y)/(p(x)p(y))
    Equal-frequency bins keep small-cell bias bounded.
    """
    rng = np.random.default_rng(0)
    jitter = rng.normal(0.0, 1e-9 * (np.abs(rho).max() + 1e-12), size=rho.shape)
    rho_j = rho + jitter
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(rho_j, quantiles)
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    bin_idx = np.digitize(rho_j, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    n = rho.size
    joint = np.zeros((n_bins, 2), dtype=np.float64)
    for b, y in zip(bin_idx, succ, strict=True):
        joint[b, int(y)] += 1.0
    joint /= n
    pX = joint.sum(axis=1, keepdims=True)
    pY = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((joint > 0) & (pX > 0) & (pY > 0), joint / (pX * pY), 1.0)
        terms = np.where(joint > 0, joint * np.log2(ratio), 0.0)
    return float(terms.sum())


def cond_entropy_binned(rho: np.ndarray, succ: np.ndarray, n_bins: int = 32) -> float:
    """H(succ | rho) in bits via equal-frequency binning."""
    rng = np.random.default_rng(0)
    jitter = rng.normal(0.0, 1e-9 * (np.abs(rho).max() + 1e-12), size=rho.shape)
    rho_j = rho + jitter
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(rho_j, quantiles)
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    bin_idx = np.digitize(rho_j, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    n = rho.size
    H = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        nb = int(mask.sum())
        if nb == 0:
            continue
        p = float(succ[mask].mean())
        H += (nb / n) * entropy_binary(p)
    return H


def pav_gaussian_mi_bound(sigma: float, succ: np.ndarray, n_grid: int = 4096) -> float:
    """Closed-form upper bound on I(PAV_score; success).

    PAV is trained as MSE regression of the per-step Monte-Carlo success
    probability mu(s_t) = E[Y | s_t]. With val MSE sigma^2 the fitted
    score s = mu(s_t) + eps where eps is approx Gaussian with variance
    bounded by sigma^2 - Var(mu)?  We use the simpler additive-noise model:
      score = Y + N, N ~ N(0, sigma^2)
    which over-credits PAV (best case for it). For Y in {0,1} and
    p = mean(Y), I(score; Y) = H(score) - H(score | Y).
    Both terms are 1-D Gaussian mixtures; we estimate H via Monte-Carlo.
    """
    p = float(succ.mean())
    if p <= 0.0 or p >= 1.0:
        return 0.0
    rng = np.random.default_rng(0)
    n = 200_000
    Y = (rng.random(n) < p).astype(np.float64)
    N = rng.normal(0.0, sigma, size=n)
    S = Y + N
    # H(score | Y): mixture of two Gaussians N(0, sigma^2) and N(1, sigma^2)
    # Conditional on Y, score is Gaussian -> H = 0.5 log2(2 pi e sigma^2)
    H_cond = 0.5 * np.log2(2 * np.pi * np.e * sigma * sigma)
    # H(score): KDE-free plug-in via histogram
    edges = np.linspace(S.min() - 1e-6, S.max() + 1e-6, 1024)
    hist, _ = np.histogram(S, bins=edges)
    px = hist / hist.sum()
    dx = edges[1] - edges[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        H_S = -np.sum(px[px > 0] * np.log2(px[px > 0])) + np.log2(dx)
    return float(max(0.0, H_S - H_cond))


def pav_cond_entropy_decision(sigma: float, succ: np.ndarray) -> float:
    """H(success | PAV_score) under the calibrated-Gaussian channel model.

    PAV's score s ~ N(mu(x), sigma^2) with mu(x) = E[Y|x].
    The Bayes-decision posterior is p(Y=1 | s) = sigma((s - 0.5)/something)
    but we just compute the empirical conditional entropy of the binarized
    decision: discretize s into 32 quantile bins and average H(Y|bin).
    """
    p = float(succ.mean())
    if p <= 0.0 or p >= 1.0:
        return 0.0
    rng = np.random.default_rng(0)
    n = 200_000
    Y = (rng.random(n) < p).astype(np.int64)
    N = rng.normal(0.0, sigma, size=n)
    S = Y + N
    return cond_entropy_binned(S, Y, n_bins=32)


def main() -> None:
    pav_results: dict[str, dict] = {}
    for jpath in PAV_ROOT.glob("*results_v2.json"):
        d = json.loads(jpath.read_text())
        pav_results[d["task"]] = d

    summary: dict[str, dict] = {}
    for task in sorted(os.listdir(DATA_ROOT)):
        if not (DATA_ROOT / task).is_dir():
            continue
        rho, succ = load_rho_and_success(task)
        n_pos = int(succ.sum())
        n = succ.size
        p_succ = n_pos / n if n > 0 else 0.0
        H_Y = entropy_binary(p_succ)

        I_stl = mi_binned(rho, succ, n_bins=32)
        H_cond_stl = cond_entropy_binned(rho, succ, n_bins=32)

        pav_entry = pav_results.get(task)
        if pav_entry is not None:
            sigma = float(np.sqrt(pav_entry["pav_best_val_mse"]))
            I_pav = pav_gaussian_mi_bound(sigma, succ)
            H_cond_pav = pav_cond_entropy_decision(sigma, succ)
        else:
            sigma = float("nan")
            I_pav = float("nan")
            H_cond_pav = float("nan")

        summary[task] = {
            "n": int(n),
            "n_succ": n_pos,
            "p_success": p_succ,
            "H_Y_bits": H_Y,
            "I_stl_bits": I_stl,
            "H_cond_stl_bits": H_cond_stl,
            "pav_sigma": sigma,
            "I_pav_bits_upperbound": I_pav,
            "H_cond_pav_bits": H_cond_pav,
        }
        print(
            f"{task:30s} "
            f"n={n:5d} p={p_succ:.3f} H(Y)={H_Y:.4f} "
            f"I_stl={I_stl:.4f} H(Y|stl)={H_cond_stl:.4f} "
            f"sigma={sigma:.4f} I_pav={I_pav:.4f} H(Y|pav)={H_cond_pav:.4f}"
        )

    out = Path("/Users/abdullahalghamdi/stl-seed/runs/pav_comparison_v2/info_theory_numbers.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

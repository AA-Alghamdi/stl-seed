"""Hierarchical Bayesian model for the canonical 3 × 3 × 2 sweep.

This implements *exactly* the model from ``paper/theory.md`` §4. The
trial-level outcome ``Y ∈ {0, 1}`` is a Bernoulli success indicator at
best-of-N budget ``N`` for cell ``(m, v, f, i)`` (model size, filter
condition, task family, instance), with the per-cell success-vs-budget
curve following a saturating power law

    p(N) = A · (1 − N^{-b}),    A ∈ (0, 1),   b > 0.

Hierarchical structure on ``logit A`` and ``log b``:

    logit A = μ_A + α^A_m + φ^A_f + δ^A_v · 1{v ≠ hard}
              + γ^A_{mf} + ε^A_{mvfi}
    log b   = μ_b + α^b_m + φ^b_f + δ^b_v · 1{v ≠ hard}
              + γ^b_{mf} + ε^b_{mvfi}

The ``δ^A_v, δ^b_v`` are the parameters of interest: contrasts of
``v ∈ {quant, cont}`` against ``v = hard`` (the baseline). Sum-to-zero
parametrizations on ``α, φ, γ`` avoid the standard intercept-confounding
identifiability trap.

Priors and MCMC config are the registered values from §4 of the paper:

    μ_A ~ N(0, 1²);                μ_b ~ N(0, 1²)
    δ^A_v, δ^b_v ~ N(0, 1²)         (for v ∈ {quant, cont})
    α, φ, γ random effects with
       τ ~ HalfNormal(0, 0.5²)      (random-effects scale)
       ε ~ N(0, σ²),
       σ ~ HalfNormal(0, 0.3²)      (cell-level idiosyncratic)

NUTS with 4 chains, 2000 warmup + 2000 draws each, target_accept = 0.9.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS

# Suppress arviz refactor banner — it's noisy for unit testing
warnings.filterwarnings("ignore", category=FutureWarning, module="arviz")


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HierarchicalData:
    """Trial-level data for the hierarchical model.

    All array fields are 1-D and aligned: row ``r`` describes the
    success indicator ``Y[r]`` for trial ``(model_idx[r], verifier_idx[r],
    family_idx[r], instance_idx[r], seed[r], N[r])``.

    Index conventions
    -----------------
    model_idx ∈ {0, ..., n_models − 1}        — model size (Q06, Q17, Q40)
    verifier_idx ∈ {0, ..., n_verifiers − 1}  — verifier-density v
        Index 0 is the *baseline* (e.g., hard); the δ_v contrasts cover
        indices 1 .. n_verifiers − 1.
    family_idx ∈ {0, ..., n_families − 1}     — task family
    instance_idx ∈ {0, ..., n_instances − 1}  — global instance id
        (the model treats each (family, instance) as its own
        observation; instances are nested inside families but indexed
        flat for vectorization.)
    seed ∈ {0, ..., n_seeds − 1}              — RNG seed of the trial
    N ∈ ℤ_>0                                  — best-of-N budget

    Y ∈ {0, 1}                                — success indicator

    Notes
    -----
    ``seed`` is carried through but not used as a structural effect in
    the registered model — its role is to allow held-out posterior
    predictive checks on a withheld seed (theory.md §4 "PPC via held-
    out seed 6").
    """

    model_idx: np.ndarray  # shape (R,)
    verifier_idx: np.ndarray  # shape (R,)
    family_idx: np.ndarray  # shape (R,)
    instance_idx: np.ndarray  # shape (R,)
    seed: np.ndarray  # shape (R,)
    N: np.ndarray  # shape (R,) integer
    Y: np.ndarray  # shape (R,) {0, 1}
    n_models: int
    n_verifiers: int
    n_families: int
    n_instances: int
    coords: dict[str, list[Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Light shape consistency check
        n = self.Y.shape[0]
        for name in ("model_idx", "verifier_idx", "family_idx", "instance_idx", "seed", "N"):
            arr = getattr(self, name)
            if arr.shape[0] != n:
                raise ValueError(
                    f"HierarchicalData: {name} has shape {arr.shape}, expected ({n},) to match Y"
                )
        if self.n_verifiers < 2:
            raise ValueError("n_verifiers must be >= 2 (one baseline + at least one contrast)")

    @property
    def n_rows(self) -> int:
        return int(self.Y.shape[0])


# ---------------------------------------------------------------------------
# NumPyro model
# ---------------------------------------------------------------------------


def model(data: HierarchicalData) -> None:
    """NumPyro model implementing theory.md §4.

    All effects are non-centered to improve sampler geometry on the
    typically small (M, F) cells.
    """

    n_models = data.n_models
    n_verifiers = data.n_verifiers
    n_families = data.n_families
    n_instances = data.n_instances
    n_contrasts = n_verifiers - 1  # baseline excluded

    m_idx = jnp.asarray(data.model_idx, dtype=jnp.int32)
    v_idx = jnp.asarray(data.verifier_idx, dtype=jnp.int32)
    f_idx = jnp.asarray(data.family_idx, dtype=jnp.int32)
    i_idx = jnp.asarray(data.instance_idx, dtype=jnp.int32)
    N_obs = jnp.asarray(data.N, dtype=jnp.float32)
    Y_obs = jnp.asarray(data.Y, dtype=jnp.int32)

    # ----- Hyperpriors on the global intercepts -----
    mu_A = numpyro.sample("mu_A", dist.Normal(0.0, 1.0))
    mu_b = numpyro.sample("mu_b", dist.Normal(0.0, 1.0))

    # ----- Filter-condition contrasts (the parameters of interest for H1) -----
    # δ_v for v ∈ {1, ..., n_verifiers - 1}; index 0 is the baseline (hard).
    delta_v_A = numpyro.sample(
        "delta_v_A",
        dist.Normal(0.0, 1.0).expand([n_contrasts]).to_event(1),
    )
    delta_v_b = numpyro.sample(
        "delta_v_b",
        dist.Normal(0.0, 1.0).expand([n_contrasts]).to_event(1),
    )
    # Pad zero for the baseline so we can index ``delta_full[v_idx]``.
    delta_full_A = jnp.concatenate([jnp.zeros(1), delta_v_A])
    delta_full_b = jnp.concatenate([jnp.zeros(1), delta_v_b])

    # ----- Random effects: model size α, family φ, model×family γ -----
    # All non-centered with HalfNormal(0, 0.5²) on the scale.
    tau_alpha_A = numpyro.sample("tau_alpha_A", dist.HalfNormal(0.5))
    tau_alpha_b = numpyro.sample("tau_alpha_b", dist.HalfNormal(0.5))
    tau_phi_A = numpyro.sample("tau_phi_A", dist.HalfNormal(0.5))
    tau_phi_b = numpyro.sample("tau_phi_b", dist.HalfNormal(0.5))
    tau_gamma_A = numpyro.sample("tau_gamma_A", dist.HalfNormal(0.5))
    tau_gamma_b = numpyro.sample("tau_gamma_b", dist.HalfNormal(0.5))

    # Non-centered model-size effect (sum-to-zero via deviation coding)
    alpha_A_raw = numpyro.sample(
        "alpha_A_raw", dist.Normal(0.0, 1.0).expand([n_models]).to_event(1)
    )
    alpha_b_raw = numpyro.sample(
        "alpha_b_raw", dist.Normal(0.0, 1.0).expand([n_models]).to_event(1)
    )
    # Soft sum-to-zero: subtract mean. This is the standard trick to
    # avoid the joint identifiability of (μ, α) — strict sum-to-zero
    # would require a constrained simplex which complicates HMC; the
    # subtract-mean projection has the same fixed point.
    alpha_A = tau_alpha_A * (alpha_A_raw - alpha_A_raw.mean())
    alpha_b = tau_alpha_b * (alpha_b_raw - alpha_b_raw.mean())

    phi_A_raw = numpyro.sample("phi_A_raw", dist.Normal(0.0, 1.0).expand([n_families]).to_event(1))
    phi_b_raw = numpyro.sample("phi_b_raw", dist.Normal(0.0, 1.0).expand([n_families]).to_event(1))
    phi_A = tau_phi_A * (phi_A_raw - phi_A_raw.mean())
    phi_b = tau_phi_b * (phi_b_raw - phi_b_raw.mean())

    # Model × family interaction: shape (n_models, n_families)
    gamma_A_raw = numpyro.sample(
        "gamma_A_raw",
        dist.Normal(0.0, 1.0).expand([n_models, n_families]).to_event(2),
    )
    gamma_b_raw = numpyro.sample(
        "gamma_b_raw",
        dist.Normal(0.0, 1.0).expand([n_models, n_families]).to_event(2),
    )
    # Center within each row and column to enforce sum-to-zero on both
    # margins (so γ is a "pure interaction").
    gamma_A = tau_gamma_A * (
        gamma_A_raw
        - gamma_A_raw.mean(axis=0, keepdims=True)
        - gamma_A_raw.mean(axis=1, keepdims=True)
        + gamma_A_raw.mean()
    )
    gamma_b = tau_gamma_b * (
        gamma_b_raw
        - gamma_b_raw.mean(axis=0, keepdims=True)
        - gamma_b_raw.mean(axis=1, keepdims=True)
        + gamma_b_raw.mean()
    )

    # ----- Cell-level idiosyncratic noise -----
    sigma_eps_A = numpyro.sample("sigma_eps_A", dist.HalfNormal(0.3))
    sigma_eps_b = numpyro.sample("sigma_eps_b", dist.HalfNormal(0.3))

    # Per-cell ε indexed by a unique (m, v, f, i) cell id constructed
    # from the input arrays. We use ``instance_idx`` as the leaf grain.
    cell_id = ((m_idx * n_verifiers + v_idx) * n_families + f_idx) * n_instances + i_idx
    n_cells = int(n_models * n_verifiers * n_families * n_instances)
    eps_A_raw = numpyro.sample("eps_A_raw", dist.Normal(0.0, 1.0).expand([n_cells]).to_event(1))
    eps_b_raw = numpyro.sample("eps_b_raw", dist.Normal(0.0, 1.0).expand([n_cells]).to_event(1))
    eps_A = sigma_eps_A * eps_A_raw
    eps_b = sigma_eps_b * eps_b_raw

    # ----- Compose per-trial logit A and log b -----
    logit_A = (
        mu_A
        + alpha_A[m_idx]
        + phi_A[f_idx]
        + delta_full_A[v_idx]
        + gamma_A[m_idx, f_idx]
        + eps_A[cell_id]
    )
    log_b = (
        mu_b
        + alpha_b[m_idx]
        + phi_b[f_idx]
        + delta_full_b[v_idx]
        + gamma_b[m_idx, f_idx]
        + eps_b[cell_id]
    )

    A = jax.nn.sigmoid(logit_A)
    b = jnp.exp(log_b)

    # Saturating power law: p(N) = A · (1 - N^{-b})
    # For numerical stability we evaluate ``1 - exp(-b · log N)``.
    # Note: at N = 1, log N = 0 → p = 0, which is the BoN-1 baseline.
    p = A * (1.0 - jnp.exp(-b * jnp.log(jnp.maximum(N_obs, 1.0))))
    # Clip to (eps, 1 - eps) to avoid log(0) inside Bernoulli.
    p = jnp.clip(p, 1e-6, 1.0 - 1e-6)

    # Deterministics for downstream summarization
    numpyro.deterministic("delta_A_full", delta_full_A)
    numpyro.deterministic("delta_b_full", delta_full_b)

    numpyro.sample("Y", dist.Bernoulli(probs=p), obs=Y_obs)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def fit(
    data: HierarchicalData,
    n_chains: int = 4,
    n_warmup: int = 2000,
    n_samples: int = 2000,
    target_accept: float = 0.9,
    max_tree_depth: int = 10,
    key: int | jax.Array = 0,
    progress_bar: bool = False,
) -> az.InferenceData:
    """Fit the hierarchical model with NUTS and return ArviZ InferenceData.

    Defaults match ``paper/theory.md`` §4: 4 chains, 2000 warmup + 2000
    draws, target_accept 0.9, max_tree_depth 10.
    """

    rng_key = jax.random.PRNGKey(key) if isinstance(key, int) else key

    nuts = NUTS(
        model,
        target_accept_prob=target_accept,
        max_tree_depth=max_tree_depth,
    )
    mcmc = MCMC(
        nuts,
        num_warmup=n_warmup,
        num_samples=n_samples,
        num_chains=n_chains,
        chain_method="sequential",  # robust on macOS without distributed
        progress_bar=progress_bar,
    )
    mcmc.run(rng_key, data=data)
    idata = az.from_numpyro(mcmc)
    return idata


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _hdi(samples: np.ndarray, prob: float = 0.89) -> tuple[float, float]:
    """Highest Density Interval (Kruschke). Falls back to quantile if degenerate."""
    arr = np.sort(np.asarray(samples).ravel())
    n = arr.size
    if n == 0:
        return float("nan"), float("nan")
    k = int(np.floor(prob * n))
    if k <= 0 or k >= n:
        lo = float(np.quantile(arr, (1 - prob) / 2))
        hi = float(np.quantile(arr, 1 - (1 - prob) / 2))
        return lo, hi
    widths = arr[k:] - arr[: n - k]
    j = int(np.argmin(widths))
    return float(arr[j]), float(arr[j + k])


def summarize(
    idata: az.InferenceData,
    hdi_prob: float = 0.89,
) -> pd.DataFrame:
    """Posterior summary for the contrast parameters of interest.

    Returns one row per (parameter, contrast index) with:
        mean, sd, hdi_low, hdi_high, P(δ > 0), P(δ < 0)

    Parameters covered: ``delta_v_A``, ``delta_v_b``.
    """
    rows = []
    posterior = idata.posterior  # type: ignore[attr-defined]
    for name in ("delta_v_A", "delta_v_b"):
        if name not in posterior:
            continue
        # Shape (chain, draw, n_contrasts)
        arr = np.asarray(posterior[name].values)
        n_contrasts = arr.shape[-1]
        flat = arr.reshape(-1, n_contrasts)
        for v in range(n_contrasts):
            samples = flat[:, v]
            lo, hi = _hdi(samples, prob=hdi_prob)
            rows.append(
                {
                    "parameter": name,
                    "contrast_idx": v + 1,  # 1-indexed (0 is baseline)
                    "mean": float(samples.mean()),
                    "sd": float(samples.std(ddof=1)),
                    "hdi_low": lo,
                    "hdi_high": hi,
                    "hdi_prob": hdi_prob,
                    "P(>0)": float((samples > 0).mean()),
                    "P(<0)": float((samples < 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def convergence_check(
    idata: az.InferenceData,
    parameters: tuple[str, ...] = (
        "mu_A",
        "mu_b",
        "delta_v_A",
        "delta_v_b",
        "tau_alpha_A",
        "tau_alpha_b",
        "tau_phi_A",
        "tau_phi_b",
        "tau_gamma_A",
        "tau_gamma_b",
        "sigma_eps_A",
        "sigma_eps_b",
    ),
) -> dict[str, dict[str, float]]:
    """Compute R̂ and ESS for the supplied parameters.

    Returns a nested dict ``{param: {"r_hat_max": .., "ess_bulk_min": ..,
    "ess_tail_min": ..}}``.

    Pass / fail thresholds (from theory.md §4):
        R̂  < 1.01  for primary parameters
        ESS > 400  for primary parameters
    """
    out: dict[str, dict[str, float]] = {}
    summ = az.summary(
        idata,
        var_names=list(parameters),
        kind="diagnostics",
    )
    for p in parameters:
        # az.summary indexes vector parameters as e.g. "delta_v_A[0]"
        rows = summ.loc[summ.index.str.startswith(f"{p}[") | (summ.index == p)]
        if rows.empty:
            continue
        out[p] = {
            "r_hat_max": float(rows["r_hat"].max()),
            "ess_bulk_min": float(rows["ess_bulk"].min()),
            "ess_tail_min": float(rows["ess_tail"].min()),
        }
    return out


__all__ = [
    "HierarchicalData",
    "model",
    "fit",
    "summarize",
    "convergence_check",
]

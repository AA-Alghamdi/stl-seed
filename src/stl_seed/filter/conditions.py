"""STL filter conditions implementing the three filter densities of
`paper/theory.md` §2.

Filter contract (locked, `paper/architecture.md` §"Filter condition interface"):

    class FilterCondition(Protocol):
        name: str

        def filter(
            self,
            trajectories: list[Trajectory],
            robustness: jt.Float[jt.Array, " N"],
        ) -> tuple[list[Trajectory], jt.Float[jt.Array, " N_kept"]]: ...

The returned weights satisfy:
* HardFilter / QuantileFilter: w_i ≡ 1.0. uniform-weight SFT on a kept
  subset.
* ContinuousWeightedFilter: w_i = N_total · softmax(ρ_i / β)_i; the N
  rescaling matches the H1-aligned "unbiased gradient" property. the
  expected per-trajectory weight is 1.0 (so the optimizer's effective
  learning rate matches hard/quantile when the spread of ρ is small).
  See `paper/theory.md` §2 stage 4 derivation.

Defensive: every filter raises `FilterError` if it would return fewer
than `min_kept` trajectories (default 10), per the deliverable spec
"if filter would return < 10 trajectories (too few for SFT), raise
informative error".
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from stl_seed.tasks._trajectory import Trajectory

_MIN_KEPT_DEFAULT = 10


class FilterError(ValueError):
    """Raised when a filter cannot produce a usable training subset."""


def _to_numpy(rho: Float[Array, " N"] | np.ndarray | Sequence[float]) -> np.ndarray:
    """Best-effort conversion to a 1-d float64 numpy array."""
    arr = np.asarray(rho, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"robustness must be 1-d, got shape {arr.shape}")
    return arr


# -----------------------------------------------------------------------------
# HardFilter
# -----------------------------------------------------------------------------


class HardFilter:
    """Keep trajectories with ρ > `rho_threshold`. Weights uniform 1.0.

    This is the SERA / RFT baseline (`paper/theory.md` §2 D_hard). The
    SERA-empty-bucket fallback ("retain top-1 with a flag") is NOT
    automatically activated here; if all ρ_i ≤ threshold and the
    resulting subset is below `min_kept`, the caller gets a `FilterError`
    so they can decide explicitly to fall back. This is intentional: a
    silent fallback would mask a calibration failure (FM2 in
    `paper/theory.md` §7).
    """

    name = "hard"

    def __init__(
        self,
        rho_threshold: float = 0.0,
        *,
        min_kept: int = _MIN_KEPT_DEFAULT,
    ) -> None:
        self.rho_threshold = float(rho_threshold)
        self.min_kept = int(min_kept)

    def filter(
        self,
        trajectories: Sequence[Trajectory],
        robustness: Float[Array, " N"] | np.ndarray | Sequence[float],
    ) -> tuple[list[Trajectory], Float[Array, " N_kept"]]:
        rho_np = _to_numpy(robustness)
        if len(trajectories) != rho_np.size:
            raise ValueError(
                f"trajectories ({len(trajectories)}) and robustness ({rho_np.size}) length mismatch"
            )
        keep_mask = rho_np > self.rho_threshold
        kept_idx = np.flatnonzero(keep_mask)
        if kept_idx.size < self.min_kept:
            raise FilterError(
                f"HardFilter (threshold={self.rho_threshold}) kept "
                f"{kept_idx.size} of {rho_np.size} trajectories. below "
                f"min_kept={self.min_kept}. ρ summary: "
                f"min={rho_np.min():.4f}, median={np.median(rho_np):.4f}, "
                f"max={rho_np.max():.4f}."
            )
        kept_traj = [trajectories[int(i)] for i in kept_idx]
        weights = jnp.ones((kept_idx.size,), dtype=jnp.float32)
        return kept_traj, weights


# -----------------------------------------------------------------------------
# QuantileFilter
# -----------------------------------------------------------------------------


class QuantileFilter:
    """Keep the top `top_k_pct`% of trajectories by ρ. Weights uniform 1.0.

    The top-K cut is taken on the *full* corpus regardless of sign. i.e.
    even if ρ < 0 throughout, the top quartile is still selected (this
    matches `paper/theory.md` §2 D_quant which uses the empirical top
    quartile, not the positive subset).

    Parameters
    ----------
    top_k_pct:
        Percentage in (0, 100]. The number kept is
        `ceil(top_k_pct/100 * N)`.
    min_kept:
        Defensive lower bound (raises `FilterError` if the corpus is too
        small to satisfy this).
    """

    name = "quantile"

    def __init__(
        self,
        top_k_pct: float = 25.0,
        *,
        min_kept: int = _MIN_KEPT_DEFAULT,
    ) -> None:
        if not (0.0 < top_k_pct <= 100.0):
            raise ValueError(f"top_k_pct must be in (0, 100], got {top_k_pct}")
        self.top_k_pct = float(top_k_pct)
        self.min_kept = int(min_kept)

    def filter(
        self,
        trajectories: Sequence[Trajectory],
        robustness: Float[Array, " N"] | np.ndarray | Sequence[float],
    ) -> tuple[list[Trajectory], Float[Array, " N_kept"]]:
        rho_np = _to_numpy(robustness)
        N = rho_np.size
        if len(trajectories) != N:
            raise ValueError(
                f"trajectories ({len(trajectories)}) and robustness ({N}) length mismatch"
            )
        n_keep = int(np.ceil(self.top_k_pct / 100.0 * N))
        if n_keep < self.min_kept:
            raise FilterError(
                f"QuantileFilter (top_k_pct={self.top_k_pct}) on N={N} "
                f"would keep only {n_keep} < min_kept={self.min_kept}."
            )
        # argsort ascending, take last n_keep.
        sorted_idx = np.argsort(rho_np, kind="stable")
        kept_idx = sorted_idx[-n_keep:]
        # Preserve original order for downstream determinism.
        kept_idx = np.sort(kept_idx)
        kept_traj = [trajectories[int(i)] for i in kept_idx]
        weights = jnp.ones((kept_idx.size,), dtype=jnp.float32)
        return kept_traj, weights


# -----------------------------------------------------------------------------
# ContinuousWeightedFilter
# -----------------------------------------------------------------------------


class ContinuousWeightedFilter:
    """Keep all trajectories; weights = N · softmax(ρ / β).

    Per `paper/theory.md` §2 stage 4 / §2 continuous derivation:
    * `β = std({ρ_i})` if `temperature is None`, else the supplied value;
    * weights are softmax-normalized so they sum to 1, then multiplied
      by `N` so the *expected* per-trajectory weight is 1.0. This makes
      the L_continuous loss directly comparable in scale to L_hard /
      L_quant (each of which weights its kept subset uniformly with
      w_i = 1.0).

    Defensive: if `len(rho) < min_kept` the filter raises before computing
    softmax (a tiny corpus is not usable for SFT regardless of weighting).
    Also raises if `std(ρ) == 0` and no explicit temperature was given
    (degenerate softmax).
    """

    name = "continuous"

    def __init__(
        self,
        temperature: float | None = None,
        *,
        min_kept: int = _MIN_KEPT_DEFAULT,
    ) -> None:
        if temperature is not None and temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature
        self.min_kept = int(min_kept)

    def filter(
        self,
        trajectories: Sequence[Trajectory],
        robustness: Float[Array, " N"] | np.ndarray | Sequence[float],
    ) -> tuple[list[Trajectory], Float[Array, " N_kept"]]:
        rho_np = _to_numpy(robustness)
        N = rho_np.size
        if len(trajectories) != N:
            raise ValueError(
                f"trajectories ({len(trajectories)}) and robustness ({N}) length mismatch"
            )
        if self.min_kept > N:
            raise FilterError(f"ContinuousWeightedFilter on N={N} below min_kept={self.min_kept}.")
        if self.temperature is not None:
            beta = float(self.temperature)
        else:
            beta = float(np.std(rho_np, ddof=0))
            if beta == 0.0:
                raise FilterError(
                    "ContinuousWeightedFilter: empirical std(ρ)=0 (all "
                    "trajectories have identical robustness); supply an "
                    "explicit temperature= to break the degeneracy."
                )
        # Numerically stable softmax.
        z = rho_np / beta
        z = z - np.max(z)
        e = np.exp(z)
        w = e / np.sum(e)
        # Rescale to sum to N so the expected per-traj weight is 1.0.
        w_scaled = w * N
        kept_traj = list(trajectories)
        return kept_traj, jnp.asarray(w_scaled, dtype=jnp.float32)


__all__ = [
    "ContinuousWeightedFilter",
    "FilterError",
    "HardFilter",
    "QuantileFilter",
]

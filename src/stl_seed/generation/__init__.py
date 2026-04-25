"""Trajectory generation for stl-seed (Subphase 1.3, A8).

This subpackage produces the `Trajectory` corpus that the STL filter and the
SFT loop consume. The three first-class objects are:

* `Policy` implementations (`policies.py`): random, constant, PID,
  bang-bang, MLX-LLM, and a heuristic-router. All conform to the
  `(state, spec, history, key) -> action` contract from
  `paper/architecture.md` §"Policy interface".
* `TrajectoryRunner` (`runner.py`): orchestrates batched per-task rollouts
  under a configurable policy mix and computes per-trajectory STL
  robustness. Resumable, NaN-aware, deterministic in `key`.
* `TrajectoryStore` (`store.py`): append-only Parquet-backed corpus with
  filter queries, summary statistics, and concurrent-read safety.

"""

from __future__ import annotations

from stl_seed.generation.policies import (
    BangBangController,
    ConstantPolicy,
    HeuristicPolicy,
    MLXModelPolicy,
    PerturbedHeuristicPolicy,
    PIDController,
    RandomPolicy,
    TopologyAwareController,
)
from stl_seed.generation.runner import TrajectoryRunner
from stl_seed.generation.store import TrajectoryStore

__all__ = [
    "BangBangController",
    "ConstantPolicy",
    "HeuristicPolicy",
    "MLXModelPolicy",
    "PerturbedHeuristicPolicy",
    "PIDController",
    "RandomPolicy",
    "TopologyAwareController",
    "TrajectoryRunner",
    "TrajectoryStore",
]

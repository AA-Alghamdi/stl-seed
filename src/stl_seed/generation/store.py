"""TrajectoryStore: append-only Parquet-backed corpus of trajectories.

Design notes
------------
* Append-only. `save(...)` writes a *new* shard file under `root/`. Existing
  shards are never rewritten — this matches the project rule that "results
  files (`sweep.jsonl`, `*.jsonl` in results directories) are append-only
  artifacts" (`~/CLAUDE.md`).
* Each shard is an Apache Parquet file holding one row per trajectory.
  Rows include both metadata columns (id, task, spec_key, policy, seed,
  robustness, nan_count, generated_at) and binary-encoded array columns
  (states, actions, times). The binary columns store `numpy` array bytes
  via `numpy.lib.format.write_array(...)` so the array dtype + shape +
  byte order are recoverable without external schema.
* Concurrent reads are safe: writers always create a new file, never edit
  existing shards. Readers list-and-merge shards in a single pass.
* Per-trajectory ID lookup uses a small in-memory index maintained on
  `load(...)`. The index is a dict `id -> (shard_path, row_index)`, so a
  caller can pull a single trajectory in O(shard) time without rescanning
  the whole corpus.

REDACTED firewall: this module imports only `pyarrow`, `numpy`, and the
locked `Trajectory` / `TrajectoryMeta` types. No REDACTED artifact appears.
"""

from __future__ import annotations

import io
import os
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from stl_seed.tasks._trajectory import Trajectory, TrajectoryMeta

_SHARD_PREFIX = "trajectories"
_SHARD_SUFFIX = ".parquet"
_SCHEMA_FIELDS = (
    "id",
    "task",
    "spec_key",
    "policy",
    "states",
    "actions",
    "times",
    "robustness",
    "nan_count",
    "seed",
    "generated_at",
    "n_save_points",
    "horizon",
    "state_dim",
    "action_dim",
    "final_solver_result",
    "used_stiff_fallback",
)


def _array_to_bytes(arr: np.ndarray) -> bytes:
    """Serialize a numpy array to bytes via the .npy format."""
    buf = io.BytesIO()
    np.lib.format.write_array(buf, np.asarray(arr), allow_pickle=False)
    return buf.getvalue()


def _array_from_bytes(buf: bytes | bytearray | memoryview) -> np.ndarray:
    """Deserialize a numpy array previously written by `_array_to_bytes`."""
    bio = io.BytesIO(bytes(buf))
    return np.lib.format.read_array(bio, allow_pickle=False)


class TrajectoryStore:
    """Append-only Parquet-backed corpus of trajectories.

    Parameters
    ----------
    root:
        Directory under which shard files are written. Created on demand.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # In-memory id -> (shard_path, row_index) index.
        self._index: dict[str, tuple[Path, int]] = {}

    # ------------------------------------------------------------------- write
    def save(
        self,
        trajectories: Iterable[Trajectory],
        metadata: Iterable[dict[str, Any]],
    ) -> Path:
        """Append `trajectories` and aligned `metadata` as a new shard.

        The shard filename is `trajectories-<unix_ms>-<uuid8>.parquet` so
        concurrent writers never collide.
        """
        traj_list = list(trajectories)
        meta_list = list(metadata)
        if len(traj_list) != len(meta_list):
            raise ValueError(
                f"trajectories ({len(traj_list)}) and metadata ({len(meta_list)}) length mismatch"
            )
        if not traj_list:
            raise ValueError("save() requires at least one trajectory")

        rows: dict[str, list[Any]] = {f: [] for f in _SCHEMA_FIELDS}
        for traj, meta in zip(traj_list, meta_list, strict=True):
            states_np = np.asarray(traj.states)
            actions_np = np.asarray(traj.actions)
            times_np = np.asarray(traj.times)
            T, n_state = states_np.shape
            H, m = actions_np.shape

            rows["id"].append(str(meta["id"]))
            rows["task"].append(str(meta["task"]))
            rows["spec_key"].append(str(meta["spec_key"]))
            rows["policy"].append(str(meta["policy"]))
            rows["states"].append(_array_to_bytes(states_np))
            rows["actions"].append(_array_to_bytes(actions_np))
            rows["times"].append(_array_to_bytes(times_np))
            rows["robustness"].append(float(meta["robustness"]))
            rows["nan_count"].append(int(meta["nan_count"]))
            rows["seed"].append(int(meta["seed"]))
            rows["generated_at"].append(str(meta["generated_at"]))
            rows["n_save_points"].append(int(T))
            rows["horizon"].append(int(H))
            rows["state_dim"].append(int(n_state))
            rows["action_dim"].append(int(m))
            rows["final_solver_result"].append(int(np.asarray(traj.meta.final_solver_result)))
            rows["used_stiff_fallback"].append(int(np.asarray(traj.meta.used_stiff_fallback)))

        table = pa.table(rows)
        shard_path = (
            self.root / f"{_SHARD_PREFIX}-{int(time.time() * 1000)}"
            f"-{uuid.uuid4().hex[:8]}{_SHARD_SUFFIX}"
        )
        pq.write_table(table, shard_path)

        # Maintain in-memory index incrementally.
        for row_idx, m in enumerate(meta_list):
            self._index[str(m["id"])] = (shard_path, row_idx)

        return shard_path

    # ------------------------------------------------------------------- read
    def _shard_paths(self) -> list[Path]:
        return sorted(
            p
            for p in self.root.iterdir()
            if p.name.startswith(_SHARD_PREFIX) and p.suffix == _SHARD_SUFFIX
        )

    def load(
        self,
        filter_query: dict[str, Any] | None = None,
    ) -> list[tuple[Trajectory, dict[str, Any]]]:
        """Load all trajectories matching `filter_query`.

        Filter semantics: every (key, value) pair in `filter_query` must
        match exactly. If `value` is a list/tuple/set, membership in the
        collection is required.

        Returns a list of `(trajectory, metadata)` pairs preserving the
        on-disk order. The in-memory index is rebuilt as a side effect so
        subsequent `get_by_id(...)` calls are fast.
        """
        if filter_query is None:
            filter_query = {}
        results: list[tuple[Trajectory, dict[str, Any]]] = []
        for shard in self._shard_paths():
            table = pq.read_table(shard)
            df = table.to_pandas()
            for row_idx, row in df.iterrows():
                # Update index.
                self._index[str(row["id"])] = (shard, int(row_idx))
                if not _row_matches(row, filter_query):
                    continue
                traj = _row_to_trajectory(row)
                meta = _row_to_meta(row)
                results.append((traj, meta))
        return results

    def get_by_id(self, traj_id: str) -> tuple[Trajectory, dict[str, Any]] | None:
        """Fetch a single trajectory by ID, using the in-memory index."""
        if traj_id not in self._index:
            # Rebuild the index by listing all shards.
            for shard in self._shard_paths():
                table = pq.read_table(shard, columns=["id"])
                ids = table.column("id").to_pylist()
                for row_idx, this_id in enumerate(ids):
                    self._index[str(this_id)] = (shard, row_idx)
            if traj_id not in self._index:
                return None
        shard, row_idx = self._index[traj_id]
        table = pq.read_table(shard)
        df = table.to_pandas()
        row = df.iloc[row_idx]
        return _row_to_trajectory(row), _row_to_meta(row)

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict[str, Any]:
        """Aggregate counts per task/policy and ρ histogram + NaN rate."""
        per_task: dict[str, int] = {}
        per_policy: dict[str, int] = {}
        per_task_policy: dict[tuple[str, str], int] = {}
        rhos: list[float] = []
        nan_counts: list[int] = []
        n_total = 0
        for shard in self._shard_paths():
            table = pq.read_table(
                shard,
                columns=[
                    "task",
                    "policy",
                    "robustness",
                    "nan_count",
                    "n_save_points",
                ],
            )
            for task, policy, rho, n_nan, T in zip(
                table.column("task").to_pylist(),
                table.column("policy").to_pylist(),
                table.column("robustness").to_pylist(),
                table.column("nan_count").to_pylist(),
                table.column("n_save_points").to_pylist(),
                strict=True,
            ):
                per_task[task] = per_task.get(task, 0) + 1
                per_policy[policy] = per_policy.get(policy, 0) + 1
                per_task_policy[(task, policy)] = per_task_policy.get((task, policy), 0) + 1
                rhos.append(float(rho))
                nan_counts.append(int(n_nan))
                _ = T
                n_total += 1

        rhos_arr = np.asarray(rhos, dtype=np.float64) if rhos else np.zeros((0,))
        nan_arr = np.asarray(nan_counts, dtype=np.int64) if nan_counts else np.zeros((0,))
        # Tukey bins: 7 fixed boundaries make the histogram comparable across runs.
        bins = np.array([-np.inf, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0, np.inf])
        counts: list[int] = []
        if rhos_arr.size:
            hist, _ = np.histogram(rhos_arr, bins=bins)
            counts = hist.tolist()
        return {
            "n_total": n_total,
            "per_task": per_task,
            "per_policy": per_policy,
            "per_task_policy": {f"{t}/{p}": c for (t, p), c in per_task_policy.items()},
            "rho_histogram_bins": bins.tolist(),
            "rho_histogram_counts": counts,
            "rho_mean": float(rhos_arr.mean()) if rhos_arr.size else 0.0,
            "rho_std": float(rhos_arr.std()) if rhos_arr.size else 0.0,
            "nan_rate_total": float(nan_arr.sum()) / max(1, int(nan_arr.size))
            if nan_arr.size
            else 0.0,
        }


# -----------------------------------------------------------------------------
# Row <-> Trajectory helpers
# -----------------------------------------------------------------------------


def _row_to_trajectory(row: Any) -> Trajectory:
    states = jnp.asarray(_array_from_bytes(row["states"]))
    actions = jnp.asarray(_array_from_bytes(row["actions"]))
    times = jnp.asarray(_array_from_bytes(row["times"]))
    meta = TrajectoryMeta(
        n_nan_replacements=jnp.asarray(int(row["nan_count"]), dtype=jnp.int32),
        final_solver_result=jnp.asarray(int(row["final_solver_result"]), dtype=jnp.int32),
        used_stiff_fallback=jnp.asarray(int(row["used_stiff_fallback"]), dtype=jnp.int32),
    )
    return Trajectory(states=states, actions=actions, times=times, meta=meta)


def _row_to_meta(row: Any) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "task": str(row["task"]),
        "spec_key": str(row["spec_key"]),
        "policy": str(row["policy"]),
        "robustness": float(row["robustness"]),
        "nan_count": int(row["nan_count"]),
        "seed": int(row["seed"]),
        "generated_at": str(row["generated_at"]),
    }


def _row_matches(row: Any, query: dict[str, Any]) -> bool:
    for k, v in query.items():
        if k not in row:
            return False
        cell = row[k]
        if isinstance(v, (list, tuple, set)):
            if cell not in v:
                return False
        else:
            if cell != v:
                return False
    return True


__all__ = ["TrajectoryStore"]

"""A14 — Apply the three STL filter conditions to the pilot trajectory store.

For each task family in `data/pilot/`, runs:

* HardFilter (rho > 0)
* QuantileFilter (top 25% by rho)
* ContinuousWeightedFilter (all kept; weights = N * softmax(rho / std(rho)))

For each (task, filter) we report:
  - N_input -> N_kept
  - per-trajectory weight statistics (min, max, mean, sum)
  - worst-kept rho  (smallest rho still in the kept subset)
  - best-dropped rho (largest rho excluded; N/A for ContinuousWeightedFilter
    which keeps everything)

Filtered datasets are persisted as Parquet under
``data/pilot/filtered_<task>_<filter>.parquet`` with columns:
  - traj_id (str)        — backreference to the source trajectory store
  - policy (str)
  - robustness (float)
  - weight (float)

This file format is **separate** from the source TrajectoryStore (which is
append-only and keeps full trajectory arrays); the filtered files are
metadata + weights only, sufficient to drive the training-data joiner. The
trajectory arrays remain in `data/pilot/trajectories-*.parquet` and are
joined by id at training time.

REDACTED firewall: imports only `stl_seed.filter`, `stl_seed.generation.store`,
and the locked Trajectory pytree. No REDACTED / REDACTED artifact is
touched.

Usage:
    cd /Users/abdullahalghamdi/stl-seed
    uv run python scripts/filter_pilot.py 2>&1 | tee scripts/filter_pilot.log
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console
from rich.table import Table

from stl_seed.filter.conditions import (
    ContinuousWeightedFilter,
    HardFilter,
    QuantileFilter,
)
from stl_seed.generation.store import TrajectoryStore

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "data" / "pilot"

console = Console()


def _load_per_task(store: TrajectoryStore) -> dict[str, list]:
    """Load all (trajectory, metadata) pairs and bucket by task."""
    pairs = store.load()
    buckets: dict[str, list] = {}
    for traj, meta in pairs:
        buckets.setdefault(meta["task"], []).append((traj, meta))
    return buckets


def _apply_filter(
    name: str,
    flt,
    trajectories,
    rhos: np.ndarray,
    metas: list[dict],
):
    """Apply `flt` and return (kept_traj, kept_meta, weights, kept_idx)."""
    kept_traj, weights = flt.filter(trajectories, rhos)
    weights_np = np.asarray(weights, dtype=np.float64)
    # Recover indices: HardFilter / QuantileFilter return a strict subset in
    # original order; ContinuousWeightedFilter returns ALL trajectories. We
    # need the indices into the source list to look up policies / IDs.
    if name == "continuous":
        kept_idx = np.arange(len(trajectories), dtype=np.int64)
    else:
        # Match by membership: the filters preserve original order, and the
        # `Trajectory` objects are the same Python instances passed in, so an
        # `id(...)` comparison recovers the indices unambiguously.
        idmap = {id(t): i for i, t in enumerate(trajectories)}
        kept_idx = np.asarray([idmap[id(t)] for t in kept_traj], dtype=np.int64)
    kept_meta = [metas[int(i)] for i in kept_idx]
    return kept_traj, kept_meta, weights_np, kept_idx


def _report_filter(
    task: str,
    filter_name: str,
    n_input: int,
    rhos_input: np.ndarray,
    kept_idx: np.ndarray,
    weights: np.ndarray,
) -> None:
    n_kept = int(kept_idx.size)
    rhos_kept = rhos_input[kept_idx]
    dropped_idx = np.setdiff1d(np.arange(n_input), kept_idx, assume_unique=False)
    if dropped_idx.size > 0 and filter_name != "continuous":
        best_dropped = float(rhos_input[dropped_idx].max())
    else:
        best_dropped = float("nan")
    worst_kept = float(rhos_kept.min())

    table = Table(
        title=f"[bold]{task} / {filter_name}",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("metric")
    table.add_column("value")
    table.add_row("N_input", f"{n_input:,}")
    table.add_row("N_kept", f"{n_kept:,}  ({n_kept / n_input:.2%})")
    table.add_row("weight min", f"{weights.min():.4e}")
    table.add_row("weight max", f"{weights.max():.4e}")
    table.add_row("weight mean", f"{weights.mean():.4e}")
    table.add_row("weight sum", f"{weights.sum():.4e}")
    table.add_row("worst-kept rho", f"{worst_kept:+.4e}")
    table.add_row(
        "best-dropped rho",
        "N/A (continuous keeps all)"
        if filter_name == "continuous"
        else f"{best_dropped:+.4e}",
    )
    console.print(table)


def _persist_filtered(
    out_path: Path,
    task: str,
    filter_name: str,
    kept_meta: list[dict],
    weights: np.ndarray,
) -> None:
    """Write the filtered manifest as a Parquet file (id + weight + meta)."""
    rows = {
        "traj_id": [str(m["id"]) for m in kept_meta],
        "task": [task] * len(kept_meta),
        "filter_name": [filter_name] * len(kept_meta),
        "policy": [str(m["policy"]) for m in kept_meta],
        "robustness": [float(m["robustness"]) for m in kept_meta],
        "weight": [float(w) for w in weights],
    }
    pq.write_table(pa.table(rows), out_path)


def main() -> int:
    console.rule("[bold]A14 — Apply STL filter conditions to pilot store")
    console.print(f"input store: {_DATA_DIR}")

    if not _DATA_DIR.exists():
        console.print(f"[red]ERROR[/]: pilot data dir {_DATA_DIR} does not exist.")
        return 2

    store = TrajectoryStore(_DATA_DIR)
    buckets = _load_per_task(store)
    if not buckets:
        console.print("[red]ERROR[/]: no trajectories in pilot store.")
        return 2

    filter_factories = {
        "hard": lambda: HardFilter(rho_threshold=0.0),
        "quantile": lambda: QuantileFilter(top_k_pct=25.0),
        "continuous": lambda: ContinuousWeightedFilter(),
    }

    # Filter failures are EXPECTED for HardFilter on tasks where the
    # uncalibrated pilot policy mix produces zero satisfying trajectories
    # (this is FM2 in paper/theory.md §7). They must surface in the report
    # but they do not mark the script as "failed" — A14's contract is to
    # *try* every (task, filter) combination and *report* the outcome.
    overall_ok = True
    for task in sorted(buckets):
        pairs = buckets[task]
        trajectories = [t for t, _ in pairs]
        metas = [m for _, m in pairs]
        rhos = np.asarray([m["robustness"] for m in metas], dtype=np.float64)
        n_input = len(trajectories)
        console.rule(f"[bold]{task} (N={n_input:,})")
        console.print(
            f"  rho summary: min={rhos.min():+.3e}  median={np.median(rhos):+.3e}  "
            f"max={rhos.max():+.3e}  positive_fraction={(rhos > 0).mean():.2%}"
        )
        for filter_name, factory in filter_factories.items():
            flt = factory()
            try:
                _, kept_meta, weights, kept_idx = _apply_filter(
                    filter_name, flt, trajectories, rhos, metas
                )
            except Exception as e:  # noqa: BLE001
                # FilterError on a calibration pilot is informative, not a
                # script-level failure (see comment in main() above).
                from stl_seed.filter.conditions import FilterError

                if isinstance(e, FilterError):
                    console.print(
                        f"  [yellow]{filter_name} INAPPLICABLE[/]: "
                        f"{type(e).__name__}: {e}"
                    )
                else:
                    console.print(
                        f"  [red]{filter_name} FAILED[/]: "
                        f"{type(e).__name__}: {e}"
                    )
                    overall_ok = False
                continue
            _report_filter(
                task=task,
                filter_name=filter_name,
                n_input=n_input,
                rhos_input=rhos,
                kept_idx=kept_idx,
                weights=weights,
            )
            out_path = _DATA_DIR / f"filtered_{task}_{filter_name}.parquet"
            _persist_filtered(out_path, task, filter_name, kept_meta, weights)
            console.print(f"  wrote {out_path.name} ({out_path.stat().st_size:,} B)")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())

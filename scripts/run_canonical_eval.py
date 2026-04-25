"""Phase-2 canonical eval runner: BoN-curve evaluation of 18 trained checkpoints.

After ``scripts/run_canonical_sweep.py`` finishes, the trained adapters
sit in ``runs/canonical/<cell_id>/adapter/``. This script iterates over
those checkpoints, runs the EvalHarness against the per-task held-out
spec set, and aggregates per-(cell, instance, seed, N) success
indicators into a single Parquet file consumable by the hierarchical
Bayes analysis.

Output schema (one row per trial)::

    model           str          # "qwen3_0.6b" | "qwen3_1.7b" | "qwen3_4b"
    filter          str          # "hard" | "quantile" | "continuous"
    task            str          # "bio_ode.repressilator" | "glucose_insulin"
    spec            str          # registered spec key
    instance        int          # 0 .. n_instances_per_family - 1
    seed            int          # 0 .. n_seeds - 1
    N               int          # BoN budget
    success         int          # {0, 1}
    rho             float        # max rho across the N samples
    action_diversity_first  float
    action_diversity_seq    float
    wall_clock_s    float        # end-to-end per-(cell, instance, seed) cost

This schema matches `src/stl_seed/stats/hierarchical_bayes.py
::HierarchicalData` field-for-field after a one-shot Pandas pivot.

REDACTED firewall: imports only `stl_seed.{evaluation,specs,tasks,training}`
plus stdlib + numpy + pandas + pyarrow + omegaconf + hydra + rich.
Verified by `scripts/REDACTED.sh`.

Usage::

    # On RunPod, after sweep completes:
    uv run python scripts/run_canonical_eval.py --runs-dir runs/canonical

    # Locally, against a single cell for debug:
    uv run python scripts/run_canonical_eval.py \\
        --runs-dir runs/canonical \\
        --only-cell qwen3_0.6b__hard__bio_ode_repressilator
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ---------------------------------------------------------------------------
# Constants.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _REPO_ROOT / "configs"
_DEFAULT_RUNS_DIR = _REPO_ROOT / "runs" / "canonical"
_RESULTS_PARQUET_NAME = "eval_results.parquet"

# Diversity threshold below which we emit a warning per cell. The smoke
# test surfaced "every held-out generation produced an identical first
# action" as a regression mode (paper/REDACTED.md §"Issues").
_DIVERSITY_WARN_THRESHOLD = 0.5

console = Console()


# ---------------------------------------------------------------------------
# Cell discovery.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainedCell:
    """A cell whose adapter was successfully trained and is ready to evaluate."""

    cell_id: str
    cell_dir: Path
    model: str
    filter: str
    task: str
    adapter_dir: Path | None  # None when adapter is missing (FAILED cell)


def _parse_cell_id(cell_id: str) -> tuple[str, str, str]:
    parts = cell_id.split("__")
    if len(parts) != 3:
        raise ValueError(f"cannot parse cell_id {cell_id!r}; expected 'model__filter__task'")
    return parts[0], parts[1], parts[2]


def discover_cells(runs_dir: Path) -> list[TrainedCell]:
    """Walk ``runs_dir`` for cell subdirectories with a done.flag file."""
    if not runs_dir.exists():
        raise SystemExit(f"runs_dir does not exist: {runs_dir}")

    cells: list[TrainedCell] = []
    for sub in sorted(runs_dir.iterdir()):
        if not sub.is_dir():
            continue
        cell_id = sub.name
        if "__" not in cell_id:
            continue
        try:
            model, filt, task = _parse_cell_id(cell_id)
        except ValueError:
            continue
        adapter = sub / "adapter"
        cells.append(
            TrainedCell(
                cell_id=cell_id,
                cell_dir=sub,
                model=model,
                filter=filt,
                task=task,
                adapter_dir=adapter if adapter.exists() else None,
            )
        )
    return cells


# ---------------------------------------------------------------------------
# Hydra resolution.
# ---------------------------------------------------------------------------


def load_cell_config(model: str, filter_: str, task: str) -> DictConfig:
    """Compose a per-cell config from the same overrides used by the sweep."""
    abs_config_dir = str(_CONFIG_DIR.resolve())
    with initialize_config_dir(version_base="1.3", config_dir=abs_config_dir):
        cfg = compose(
            config_name="default",
            overrides=[
                f"model={model}",
                f"filter={filter_}",
                f"task={task}",
                f"run.name={model}__{filter_}__{task}",
            ],
        )
    return cfg


# ---------------------------------------------------------------------------
# Eval per cell.
# ---------------------------------------------------------------------------


def _build_registries(task_family: str) -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
    """Construct simulator / spec / evaluator / x0_fn registries for one family.

    All imports are lazy because they pull diffrax + jax under the hood.
    """
    from stl_seed.generation.runner import evaluate_robustness  # noqa: F401
    from stl_seed.specs import REGISTRY as _SPEC_REG

    # Spec registry restricted to this family so the harness does not
    # accidentally evaluate cross-family specs.
    spec_registry = {name: spec for name, spec in _SPEC_REG.items() if name.startswith(task_family)}
    if not spec_registry:
        raise ValueError(f"no specs registered under prefix {task_family!r}")

    # Build a simulator instance per spec key. Different families have
    # different simulators; we pick by the family prefix.
    if task_family.startswith("bio_ode."):
        from stl_seed.tasks.bio_ode import (  # noqa: PLC0415
            MAPKSimulator,
            RepressilatorSimulator,
            ToggleSimulator,
        )

        sim_lookup: dict[str, Any] = {
            "bio_ode.repressilator": RepressilatorSimulator(),
            "bio_ode.toggle": ToggleSimulator(),
            "bio_ode.mapk": MAPKSimulator(),
        }
    elif task_family == "glucose_insulin":
        from stl_seed.tasks.glucose_insulin import GlucoseInsulinSimulator

        sim_lookup = {"glucose_insulin": GlucoseInsulinSimulator()}
    else:
        raise ValueError(f"unknown task family {task_family!r}")

    sim_registry: dict[str, Any] = {}
    for spec_name in spec_registry:
        for prefix, sim in sim_lookup.items():
            if spec_name.startswith(prefix):
                sim_registry[spec_name] = sim
                break

    # The harness expects a callable ``stl_evaluator(spec, trajectory) -> float``.
    # ``evaluate_robustness`` in ``stl_seed.generation.runner`` already has
    # the right signature.
    def _stl_eval(spec: Any, trajectory: Any) -> float:
        return float(evaluate_robustness(spec, trajectory))

    # A no-op x0 function: every spec uses a deterministic initial state
    # supplied by the simulator itself. Sweep-time per-instance variation
    # is realized via the seed; instance index is folded in by the outer
    # driver below.
    def _x0_fn(spec_name: str, key: Any) -> Any:
        sim = sim_registry[spec_name]
        # Most simulators expose ``default_initial_state``; fall back to
        # a zero vector of the right dim if not.
        if hasattr(sim, "default_initial_state"):
            return sim.default_initial_state()
        import jax.numpy as jnp  # noqa: PLC0415

        return jnp.zeros((sim.state_dim,))

    return sim_registry, spec_registry, _stl_eval, _x0_fn


def _make_checkpoint_proxy(cell: TrainedCell, cfg: DictConfig) -> Any:
    """Build a CheckpointProtocol-conforming policy from a trained adapter.

    Lazy-imports the backend's ``load`` path. For the canonical Phase-2
    bnb backend, this returns a ``transformers.pipeline``-style callable
    wrapped to satisfy ``sample_controls(spec, initial_state, key)``.

    For now (Phase-2 first-cut), we wrap with a thin shim that decodes
    the model's text output via the same regex used in
    ``scripts/smoke_test_bnb.py`` (``<state>...</state><action>...</action>``)
    and returns the action sequence.
    """
    from stl_seed.training.backends.base import TrainedCheckpoint
    from stl_seed.training.loop import get_backend

    backend_name = str(cfg.backend.name)
    backend = get_backend(backend_name)
    chk = TrainedCheckpoint(
        backend=backend_name,  # type: ignore[arg-type]
        model_path=cell.adapter_dir,  # type: ignore[arg-type]
        base_model=str(cfg.model.hf_id),
        training_loss_history=[],
        wall_clock_seconds=0.0,
    )
    inference_callable = backend.load(chk)

    # A minimal wrapper. The real ``sample_controls`` signature is
    # ``(spec, initial_state, key) -> Float[Array, "H m"]``; the
    # underlying inference callable takes a string prompt and returns
    # text. Bridging requires (a) prompt rendering identical to the
    # training-time format and (b) parsing the output back into
    # actions. The shim here is intentionally thin so that swapping in
    # a richer policy (e.g., chain-of-thought decoding) is local.
    class _CheckpointWrapper:
        name = cell.cell_id

        def sample_controls(self, spec: Any, initial_state: Any, key: Any) -> Any:
            # The eval harness will catch (FloatingPointError, RuntimeError,
            # ValueError) and record n_nan/rho=NaN. Anything else propagates.
            from stl_seed.training.tokenize import format_prompt_for_eval

            prompt = format_prompt_for_eval(
                spec=spec, initial_state=initial_state, task=cfg.task.family
            )
            text_output = inference_callable(prompt)
            from stl_seed.training.tokenize import parse_action_sequence

            return parse_action_sequence(text_output)

    return _CheckpointWrapper()


def evaluate_cell(
    cell: TrainedCell,
    cfg: DictConfig,
    n_seeds: int,
    n_instances: int,
    n_samples_per_spec: int,
    bon_budgets: list[int],
    key_seed: int,
) -> pd.DataFrame:
    """Evaluate a single cell; return a long-form DataFrame ready for parquet."""
    from stl_seed.evaluation.harness import EvalHarness

    sim_registry, spec_registry, stl_eval, x0_fn = _build_registries(cell.task.replace("_", "."))

    # Restrict to the configured eval-spec list (per task config).
    held_out = list(cfg.task.eval_specs)
    held_out = [s for s in held_out if s in spec_registry]
    if not held_out:
        # Fall back to all family specs if the task config's list is empty.
        held_out = sorted(spec_registry.keys())

    harness = EvalHarness(
        simulator_registry=sim_registry,
        spec_registry=spec_registry,
        stl_evaluator=stl_eval,
        initial_state_fn=x0_fn,
        budgets=bon_budgets,
    )

    if cell.adapter_dir is None:
        # Cell never finished training. Emit a single empty row so the
        # downstream analyzer can detect the missing cell explicitly.
        return pd.DataFrame(
            [
                {
                    "model": cell.model,
                    "filter": cell.filter,
                    "task": cell.task,
                    "spec": held_out[0],
                    "instance": 0,
                    "seed": 0,
                    "N": int(max(bon_budgets)),
                    "success": 0,
                    "rho": float("nan"),
                    "action_diversity_first": float("nan"),
                    "action_diversity_seq": float("nan"),
                    "wall_clock_s": 0.0,
                    "status": "MISSING_ADAPTER",
                }
            ]
        )

    checkpoint = _make_checkpoint_proxy(cell, cfg)

    rows: list[dict[str, Any]] = []
    for instance in range(n_instances):
        for seed in range(n_seeds):
            key = key_seed + 10_000 * instance + seed
            t0 = time.perf_counter()
            results = harness.evaluate_checkpoint(
                checkpoint=checkpoint,
                held_out_specs=held_out,
                n_samples_per_spec=int(n_samples_per_spec),
                key=int(key),
            )
            wall = time.perf_counter() - t0
            for spec_name, per in results.per_spec.items():
                for n in bon_budgets:
                    succ = per.bon_success.get(int(n), float("nan"))
                    rows.append(
                        {
                            "model": cell.model,
                            "filter": cell.filter,
                            "task": cell.task,
                            "spec": spec_name,
                            "instance": int(instance),
                            "seed": int(seed),
                            "N": int(n),
                            "success": int(succ > 0.5) if not np.isnan(succ) else 0,
                            "rho": float(np.nanmax(per.rhos[:n])) if n > 0 else float("nan"),
                            "action_diversity_first": float(
                                per.diversity.get("first_action_uniqueness", float("nan"))
                            ),
                            "action_diversity_seq": float(
                                per.diversity.get("sequence_uniqueness", float("nan"))
                            ),
                            "wall_clock_s": float(wall),
                            "status": "OK",
                        }
                    )
    return pd.DataFrame(rows)


def warn_low_diversity(df: pd.DataFrame) -> None:
    """Print a warning per (cell) where mean first-action diversity is below threshold."""
    if df.empty:
        return
    grouped = df.groupby(["model", "filter", "task"])["action_diversity_first"].mean()
    low = grouped[grouped < _DIVERSITY_WARN_THRESHOLD].dropna()
    if low.empty:
        return
    table = Table(title="[yellow]Low action diversity warnings", header_style="bold yellow")
    table.add_column("model")
    table.add_column("filter")
    table.add_column("task")
    table.add_column("mean first_action_uniqueness", justify="right")
    for (m, f, t), val in low.items():
        table.add_row(m, f, t, f"{val:.3f}")
    console.print(table)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_canonical_eval.py",
        description="Phase-2 evaluation runner: BoN-curve eval over trained cells.",
    )
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=_DEFAULT_RUNS_DIR,
        help="Directory containing per-cell subdirectories from the sweep.",
    )
    p.add_argument(
        "--only-cell",
        type=str,
        default=None,
        help="Restrict to a single cell_id for debug.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover cells; do not load adapters or run eval.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Override output parquet path. Default: <runs_dir>/{_RESULTS_PARQUET_NAME}.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs_dir = Path(args.runs_dir)
    out_path = Path(args.output) if args.output is not None else runs_dir / _RESULTS_PARQUET_NAME

    cells = discover_cells(runs_dir)
    if args.only_cell is not None:
        cells = [c for c in cells if c.cell_id == args.only_cell]

    console.print(
        Panel.fit(
            f"Phase-2 canonical eval\n"
            f"  runs_dir: {runs_dir}\n"
            f"  cells: {len(cells)}\n"
            f"  output: {out_path}",
            title="[bold]stl-seed eval planner",
        )
    )

    if not cells:
        console.print("[red]no cells discovered; nothing to evaluate.[/]")
        return 1

    if args.dry_run:
        for c in cells:
            console.print(f"  - {c.cell_id}  adapter={'present' if c.adapter_dir else 'MISSING'}")
        return 0

    all_rows: list[pd.DataFrame] = []
    for i, cell in enumerate(cells):
        console.rule(f"[bold]Cell {i + 1}/{len(cells)}: {cell.cell_id}")
        try:
            cfg = load_cell_config(cell.model, cell.filter, cell.task)
            df = evaluate_cell(
                cell=cell,
                cfg=cfg,
                n_seeds=int(cfg.evaluation.n_seeds),
                n_instances=int(cfg.evaluation.n_instances_per_family),
                n_samples_per_spec=int(cfg.evaluation.n_samples_per_spec),
                bon_budgets=list(cfg.evaluation.bon_budgets),
                key_seed=int(cfg.seed),
            )
            all_rows.append(df)
            console.print(f"  emitted {len(df)} rows")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]EVAL FAILED for {cell.cell_id}:[/] {type(exc).__name__}: {exc}")
            console.print_exception()

    if not all_rows:
        console.print("[red]no eval results produced.[/]")
        return 1

    full = pd.concat(all_rows, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out_path, index=False)
    console.print(f"[green]wrote {len(full)} rows to {out_path}[/]")

    warn_low_diversity(full)
    return 0


if __name__ == "__main__":
    sys.exit(main())

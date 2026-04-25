"""Phase-2 canonical sweep runner: 3 sizes x 3 filters x 2 tasks = 18 cells.

This is the *single command* the user runs on a RunPod 4090 spot pod
once Phase-1 has shipped and credit is loaded. The script:

  1. Loads the Hydra base config (`configs/default.yaml`) and the sweep
     overlay (`configs/sweep_main.yaml`).
  2. Enumerates the 18 (model, filter, task) cells and prints the cost
     estimate. Requires --confirm to proceed past the dry-run gate.
  3. For each cell:
       a. Generate trajectories under pi_ref (or load existing parquet).
       b. Filter trajectories with the requested condition.
       c. Train QLoRA via the bnb backend.
       d. Optionally push the resulting LoRA adapter to HuggingFace Hub.
       e. Write per-cell provenance.json and append to sweep_log.csv.
  4. Resumability: cells with `done.flag` written are skipped, allowing
     spot-interruption recovery. The flag is touched after the adapter
     is on disk AND HF push succeeds (if enabled).
  5. Cost cap: --max-cost-usd aborts the sweep if the running estimate
     plus the next cell's expected cost would exceed budget.
  6. Spot interruption recovery: SIGTERM/SIGINT trigger a clean shutdown
     that writes the current cell's status as INTERRUPTED in the log.

The script is *lazy* about heavy imports (torch / bnb / transformers /
trl / mlx / diffrax). Dry-run and --help work on a CPU-only macOS host
without any of those installed; the heavy stack only loads inside the
training inner loop, after the cost gate has been crossed.

tasks,training,evaluation}` and stdlib + numpy + pandas + omegaconf +

Mock-backend opt-in (Phase-2 dry-run validation)
-------------------------------------------------
Setting ``STL_SEED_USE_MOCK_BACKEND=1`` in the environment causes the
sweep runner to instantiate :class:`~stl_seed.training.backends.mock.MockBNBBackend`
in place of the real :class:`BNBBackend` for every cell. The mock writes
artifacts in the same on-disk layout as the real path but does no actual
training — used by ``scripts/validate_phase2_pipeline.py`` to exercise
the whole pipeline (sweep + eval + analysis) without spending RunPod
GPU time. The substitution happens inside ``run_cell`` via
``stl_seed.training.loop.train_with_filter``'s lazy backend dispatch
(see :mod:`stl_seed.training.backends.mock`).

Usage examples
--------------
Dry run (no GPU, no spend, no training): produces the cell enumeration
and cost forecast, exits cleanly.::

    uv run python scripts/run_canonical_sweep.py --dry-run

Full Phase-2 run on a RunPod 4090::

    uv run python scripts/run_canonical_sweep.py \\
        --config-name sweep_main \\
        --max-cost-usd 25 \\
        --confirm

Resumable single-cell debug (one (model, filter, task) tuple)::

    uv run python scripts/run_canonical_sweep.py \\
        --only-cell qwen3_0.6b__hard__bio_ode_repressilator \\
        --confirm
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Hydra/OmegaConf are part of the core deps; safe to import at module top.
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ---------------------------------------------------------------------------
# Paths and constants.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _REPO_ROOT / "configs"

# Mock-backend / runs-dir env vars (see scripts/validate_phase2_pipeline.py).
import os as _os  # noqa: E402

_MOCK_ENV = "STL_SEED_USE_MOCK_BACKEND"
_RUNS_DIR_ENV = "STL_SEED_RUNS_DIR_OVERRIDE"

_RUNS_DIR = (
    Path(_os.environ[_RUNS_DIR_ENV]).resolve()
    if _os.environ.get(_RUNS_DIR_ENV)
    else _REPO_ROOT / "runs" / "canonical"
)
_SWEEP_LOG = _RUNS_DIR / "sweep_log.csv"


def _mock_backend_enabled() -> bool:
    """Return True iff STL_SEED_USE_MOCK_BACKEND is truthy in the environment."""
    return _os.environ.get(_MOCK_ENV, "").strip() in {"1", "true", "True", "TRUE", "yes"}


# Trajectory store roots searched in priority order. Canonical first
# (Phase-2 full-scale 2,500-traj/task store from
# `scripts/generate_canonical.py`); pilot second (Phase-1 2,000-traj
# fallback from `scripts/generate_pilot.py`); then None to signal
# "regenerate fresh inside the cell" so a stripped-clone reproduction
# still works without prior data prep.
_CANONICAL_DATA_ROOT = _REPO_ROOT / "data" / "canonical"
_PILOT_DATA_ROOT = _REPO_ROOT / "data" / "pilot"

# Per-model expected duration in minutes (from configs/model/*.yaml).
# Used by the cost estimator BEFORE Hydra is invoked, so we can reject a
# nonsensical budget without spending Hydra-init time.
_MODEL_MINUTES = {
    "qwen3_0.6b": 18,
    "qwen3_1.7b": 28,
    "qwen3_4b": 55,
}

console = Console()


# ---------------------------------------------------------------------------
# Cell dataclass.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One (model, filter, task) cell of the canonical sweep."""

    model: str
    filter: str
    task: str

    @property
    def cell_id(self) -> str:
        return f"{self.model}__{self.filter}__{self.task}"

    @property
    def output_dir(self) -> Path:
        return _RUNS_DIR / self.cell_id

    @property
    def done_flag(self) -> Path:
        return self.output_dir / "done.flag"


@dataclass
class CellResult:
    """Per-cell run record persisted to sweep_log.csv."""

    cell_id: str
    model: str
    filter: str
    task: str
    status: str  # "OK" | "SKIPPED" | "FAILED" | "INTERRUPTED" | "DRY_RUN"
    start_iso: str
    end_iso: str
    duration_s: float
    cost_usd: float
    error: str = ""

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Cell enumeration.
# ---------------------------------------------------------------------------


def enumerate_cells(cfg: DictConfig) -> list[Cell]:
    """Return the canonical 18 cells from the sweep overlay.

    Reads `cfg.sweep.{models,filters,tasks}` and produces the Cartesian
    product. The order is deterministic: (model outermost, filter middle,
    task innermost) so resume-by-position is well-defined.
    """
    sweep = cfg.get("sweep", None)
    if sweep is None:
        raise SystemExit("config does not contain a 'sweep' section; pass --config-name sweep_main")
    models = list(sweep.models)
    filters = list(sweep.filters)
    tasks = list(sweep.tasks)
    cells = [Cell(model=m, filter=f, task=t) for m in models for f in filters for t in tasks]
    return cells


# ---------------------------------------------------------------------------
# Trajectory-store resolution (canonical > pilot > regenerate-fresh).
# ---------------------------------------------------------------------------


def _task_to_family(task_cfg_name: str) -> str:
    """Map sweep `task=` slug to the trajectory-store `task` field.

    The Hydra task group uses underscored slugs (`bio_ode_repressilator`)
    while the trajectory-store records the dotted family
    (`bio_ode.repressilator`). Keep this mapping next to the resolver so
    it stays in lockstep with `configs/task/*.yaml#family`.
    """
    return {
        "bio_ode_repressilator": "bio_ode.repressilator",
        "bio_ode_toggle": "bio_ode.toggle",
        "bio_ode_mapk": "bio_ode.mapk",
        "glucose_insulin": "glucose_insulin",
    }.get(task_cfg_name, task_cfg_name)


def resolve_data_root(task_cfg_name: str) -> tuple[Path | None, str]:
    """Return ``(path, source)`` for the trajectory store of ``task``.

    The sweep prefers the canonical (full-scale) store and falls back to
    the pilot, then to a "regenerate fresh" sentinel. The contract is:

    1. ``data/canonical/<dotted_family>/`` if it contains at least one
       ``trajectories-*.parquet`` shard. Source label: ``"canonical"``.
    2. ``data/pilot/`` (the flat Phase-1 store) if any
       ``trajectories-*.parquet`` shard contains at least one row whose
       ``task`` column matches ``dotted_family``. Source label: ``"pilot"``.
    3. ``(None, "regenerate")`` if neither store has the task. The
       caller is then expected to invoke the in-cell generation path
       (slower, paid GPU time).

    The resolver is read-only: it does not touch the parquet payload
    beyond a column-projection scan that loads only the ``task`` column.

    The ``(path, source)`` tuple is what the dry-run summary surfaces, so
    the user can see WHICH source the sweep would consume before
    spending GPU time. Returning ``Path`` (not ``TrajectoryStore``) keeps
    this helper free of heavy imports — the cell-side training driver
    (`stl_seed.training.loop`) is responsible for the actual load.
    """
    family = _task_to_family(task_cfg_name)

    # 1. Canonical: per-task subdir.
    canonical_dir = _CANONICAL_DATA_ROOT / family
    if canonical_dir.is_dir():
        shards = list(canonical_dir.glob("trajectories-*.parquet"))
        if shards:
            return canonical_dir, "canonical"

    # 2. Pilot: flat dir, scan only the `task` column to confirm presence.
    if _PILOT_DATA_ROOT.is_dir():
        pilot_shards = list(_PILOT_DATA_ROOT.glob("trajectories-*.parquet"))
        if pilot_shards:
            try:
                import pyarrow.parquet as pq

                for shard in pilot_shards:
                    tbl = pq.read_table(shard, columns=["task"])
                    if family in {str(t) for t in tbl.column("task").to_pylist()}:
                        return _PILOT_DATA_ROOT, "pilot"
            except (ImportError, FileNotFoundError, OSError):
                # Defensive: if pyarrow misbehaves, fall through to regenerate
                # rather than crash the dry-run forecast.
                pass

    # 3. No prior data — caller must regenerate inside the cell.
    return None, "regenerate"


# ---------------------------------------------------------------------------
# Cost estimator.
# ---------------------------------------------------------------------------


def estimate_cell_cost(cell: Cell, dollars_per_hour: float) -> tuple[float, float]:
    """Return (expected_minutes, expected_dollars) for one cell.

    Uses the per-model lookup; filter/task contribute negligibly to wall
    time at this scale (filter is ~milliseconds; task differs only in
    horizon, which is a small constant in trajectory generation but not
    in SFT).
    """
    minutes = _MODEL_MINUTES.get(cell.model, 30)
    hours = minutes / 60.0
    return float(minutes), float(hours * dollars_per_hour)


def estimate_total_cost(cells: list[Cell], dollars_per_hour: float) -> dict[str, float]:
    """Aggregate cost telemetry across all cells."""
    total_min = 0.0
    total_usd = 0.0
    for c in cells:
        m, d = estimate_cell_cost(c, dollars_per_hour)
        total_min += m
        total_usd += d
    return {
        "n_cells": float(len(cells)),
        "total_minutes": total_min,
        "total_hours": total_min / 60.0,
        "total_usd": total_usd,
    }


# ---------------------------------------------------------------------------
# Hydra config loading.
# ---------------------------------------------------------------------------


def load_config(config_name: str) -> DictConfig:
    """Compose a Hydra config from `configs/<config_name>.yaml`.

    Uses `initialize_config_dir` instead of the decorator pattern so the
    function is callable from a regular script (and from unit tests)
    without re-entering Hydra's main loop.
    """
    abs_config_dir = str(_CONFIG_DIR.resolve())
    with initialize_config_dir(version_base="1.3", config_dir=abs_config_dir):
        cfg = compose(config_name=config_name)
    return cfg


def load_cell_config(cell: Cell) -> DictConfig:
    """Compose a single-cell config: default + per-cell group overrides.

    When the ``STL_SEED_RUNS_DIR_OVERRIDE`` env var is set, the per-cell
    ``run.output_dir`` is also redirected to ``<override>/<cell_id>`` so
    the training-side artifacts land in the same tree as the runner-
    side provenance + sweep log. This keeps the validation pipeline's
    output entirely under one redirectable root.
    """
    abs_config_dir = str(_CONFIG_DIR.resolve())
    overrides = [
        f"model={cell.model}",
        f"filter={cell.filter}",
        f"task={cell.task}",
        f"run.name={cell.cell_id}",
    ]
    runs_dir_override = _os.environ.get(_RUNS_DIR_ENV)
    if runs_dir_override:
        # Absolute path so the override survives Hydra's ${run.name}
        # interpolation (which would otherwise re-prefix with "runs/").
        target = Path(runs_dir_override).resolve() / cell.cell_id
        overrides.append(f"run.output_dir={target}")
    with initialize_config_dir(version_base="1.3", config_dir=abs_config_dir):
        cfg = compose(config_name="default", overrides=overrides)
    return cfg


# ---------------------------------------------------------------------------
# Provenance.
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode()
        return bool(out.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def write_provenance(cell: Cell, cfg: DictConfig, extra: dict[str, Any]) -> None:
    """Write per-cell provenance.json mirroring paper/reproducibility.md §6."""
    cell.output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "cell_id": cell.cell_id,
        "model": cell.model,
        "filter": cell.filter,
        "task": cell.task,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "config_yaml": OmegaConf.to_yaml(cfg),
        "config_hash": hashlib.sha256(OmegaConf.to_yaml(cfg).encode()).hexdigest(),
        "seed": int(cfg.seed),
        "training_date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    payload.update(extra)
    (cell.output_dir / "provenance.json").write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Sweep log.
# ---------------------------------------------------------------------------


def append_sweep_log(result: CellResult) -> None:
    """Append one row to runs/canonical/sweep_log.csv (creates header on first call)."""
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not _SWEEP_LOG.exists()
    fields = list(result.to_row().keys())
    with _SWEEP_LOG.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerow(result.to_row())


# ---------------------------------------------------------------------------
# Per-cell execution.
# ---------------------------------------------------------------------------


def _build_training_config(cell_cfg: DictConfig) -> Any:
    """Build a TrainingConfig from the resolved cell Hydra config.

    Lazy-imports the training package so the dry-run path does not need
    transformers / peft / torch.
    """
    from stl_seed.training.backends.base import TrainingConfig

    t = cell_cfg.model.training
    return TrainingConfig(
        base_model=cell_cfg.model.hf_id,
        learning_rate=float(t.learning_rate),
        lr_schedule=str(t.lr_schedule),
        warmup_ratio=float(t.warmup_ratio),
        num_epochs=int(t.num_epochs),
        batch_size=int(t.batch_size),
        gradient_accumulation_steps=int(t.gradient_accumulation_steps),
        max_seq_length=int(t.max_seq_length),
        lora_rank=int(t.lora_rank),
        lora_alpha=float(t.lora_alpha),
        lora_target_modules=list(t.lora_target_modules),
        lora_dropout=float(t.lora_dropout),
        seed=int(cell_cfg.seed),
        output_dir=Path(cell_cfg.run.output_dir),
        weight_format=str(cell_cfg.backend.weight_format),
        use_8bit_optimizer=bool(cell_cfg.backend.use_8bit_optimizer),
        weight_decay=float(t.weight_decay),
    )


def _maybe_push_to_hub(cell: Cell, cfg: DictConfig, adapter_dir: Path) -> str | None:
    """Optionally push the LoRA adapter to HuggingFace Hub.

    Returns the resolved repo_id on success; None when disabled.
    Failure raises so the cell is marked FAILED in the log.
    """
    if not bool(cfg.hf_hub.enabled):
        return None
    org = cfg.hf_hub.repo_org
    if org is None:
        raise SystemExit("hf_hub.enabled=true but hf_hub.repo_org is null")
    repo_id = f"{org}/{cfg.hf_hub.repo_prefix}-{cell.cell_id}"

    # Lazy import: huggingface_hub is in core deps but `HfApi` triggers
    # the network on first use.
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, private=bool(cfg.hf_hub.private), exist_ok=True)
    api.upload_folder(folder_path=str(adapter_dir), repo_id=repo_id)
    return repo_id


def run_cell(cell: Cell, dry_run: bool = False) -> CellResult:
    """Run a single (model, filter, task) cell end-to-end."""
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    t0 = time.perf_counter()

    if cell.done_flag.exists():
        elapsed = time.perf_counter() - t0
        end_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        return CellResult(
            cell_id=cell.cell_id,
            model=cell.model,
            filter=cell.filter,
            task=cell.task,
            status="SKIPPED",
            start_iso=start_iso,
            end_iso=end_iso,
            duration_s=elapsed,
            cost_usd=0.0,
        )

    cell_cfg = load_cell_config(cell)
    cell.output_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        # In dry-run, write provenance + a placeholder marker so a
        # downstream eval driver can validate cell enumeration.
        write_provenance(
            cell,
            cell_cfg,
            {
                "dry_run": True,
                "expected_minutes": _MODEL_MINUTES.get(cell.model, 30),
            },
        )
        end_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        elapsed = time.perf_counter() - t0
        return CellResult(
            cell_id=cell.cell_id,
            model=cell.model,
            filter=cell.filter,
            task=cell.task,
            status="DRY_RUN",
            start_iso=start_iso,
            end_iso=end_iso,
            duration_s=elapsed,
            cost_usd=0.0,
        )

    # ---- Real training path. Heavy imports are deferred to here. ----
    try:
        from stl_seed.training.loop import train_with_filter

        config = _build_training_config(cell_cfg)
        # Mock-backend opt-in: when STL_SEED_USE_MOCK_BACKEND=1 is set, dispatch
        # to MockBNBBackend instead of the real bnb path. The mock honors the
        # bnb interface so this is a one-line switch (see
        # src/stl_seed/training/backends/mock.py and
        # scripts/validate_phase2_pipeline.py).
        backend_name = "mock_bnb" if _mock_backend_enabled() else str(cell_cfg.backend.name)
        # Filter/task family strings are passed to load_filtered_dataset
        # inside train_with_filter; the function handles the data plumbing.
        checkpoint = train_with_filter(
            filter_condition=str(cell_cfg.filter.name),
            task=str(cell_cfg.task.family),
            model=str(cell_cfg.model.hf_id),
            backend=backend_name,
            config=config,
        )
        adapter_dir = Path(checkpoint.model_path)

        repo_id = _maybe_push_to_hub(cell, cell_cfg, adapter_dir)

        write_provenance(
            cell,
            cell_cfg,
            {
                "training_loss_history": checkpoint.training_loss_history,
                "wall_clock_seconds": checkpoint.wall_clock_seconds,
                "adapter_path": str(adapter_dir),
                "hf_repo_id": repo_id,
            },
        )

        cell.done_flag.write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"))

        end_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        elapsed = time.perf_counter() - t0
        cost_usd = (elapsed / 3600.0) * float(cell_cfg.cost.gpu_dollars_per_hour)
        return CellResult(
            cell_id=cell.cell_id,
            model=cell.model,
            filter=cell.filter,
            task=cell.task,
            status="OK",
            start_iso=start_iso,
            end_iso=end_iso,
            duration_s=elapsed,
            cost_usd=cost_usd,
        )

    except Exception as exc:  # noqa: BLE001
        end_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        elapsed = time.perf_counter() - t0
        # Per CLAUDE.md: do not silently swallow training failures.
        console.print_exception()
        return CellResult(
            cell_id=cell.cell_id,
            model=cell.model,
            filter=cell.filter,
            task=cell.task,
            status="FAILED",
            start_iso=start_iso,
            end_iso=end_iso,
            duration_s=elapsed,
            cost_usd=0.0,
            error=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Signal handling.
# ---------------------------------------------------------------------------


_INTERRUPTED = False


def _install_signal_handlers() -> None:
    """Trap SIGTERM/SIGINT so spot pre-emption logs INTERRUPTED, not crashes."""

    def _handler(signum: int, _frame: Any) -> None:
        global _INTERRUPTED
        _INTERRUPTED = True
        console.print(
            f"[yellow]received signal {signum}; finishing current cell and exiting cleanly[/]"
        )

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_canonical_sweep.py",
        description="Phase-2 canonical training sweep for stl-seed.",
    )
    p.add_argument(
        "--config-name",
        default="sweep_main",
        help="Hydra config name under configs/ (without .yaml). Default: sweep_main.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Enumerate cells, print cost forecast, write provenance stubs; no training.",
    )
    p.add_argument(
        "--confirm",
        action="store_true",
        help="Required to start real training. Without this flag, defaults to dry-run.",
    )
    p.add_argument(
        "--max-cost-usd",
        type=float,
        default=25.0,
        help="Hard ceiling. Sweep aborts if running cost + next-cell estimate exceeds this.",
    )
    p.add_argument(
        "--only-cell",
        type=str,
        default=None,
        help="Run a single cell by cell_id (e.g., qwen3_0.6b__hard__bio_ode_repressilator).",
    )
    p.add_argument(
        "--gpu-dollars-per-hour",
        type=float,
        default=0.34,
        help="Override the per-hour GPU rate used for cost forecasting (default: 0.34 = 4090 spot).",
    )
    return p.parse_args(argv)


def print_cell_table(cells: list[Cell], dollars_per_hour: float) -> None:
    table = Table(
        title=f"[bold]Phase-2 canonical sweep: {len(cells)} cells",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("#", justify="right")
    table.add_column("cell_id")
    table.add_column("expected min", justify="right")
    table.add_column("expected $", justify="right")
    table.add_column("data source")
    table.add_column("done?", justify="center")
    # Cache the data-source resolution per task so we don't re-scan
    # parquet metadata once per cell.
    src_cache: dict[str, tuple[Path | None, str]] = {}
    for i, c in enumerate(cells):
        m, d = estimate_cell_cost(c, dollars_per_hour)
        if c.task not in src_cache:
            src_cache[c.task] = resolve_data_root(c.task)
        _, src = src_cache[c.task]
        src_label = {
            "canonical": "[green]canonical[/]",
            "pilot": "[yellow]pilot (fallback)[/]",
            "regenerate": "[red]regenerate (no prior data)[/]",
        }[src]
        is_done = "[green]Y[/]" if c.done_flag.exists() else "-"
        table.add_row(str(i + 1), c.cell_id, f"{m:.0f}", f"${d:.2f}", src_label, is_done)
    console.print(table)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    cfg = load_config(args.config_name)
    cells = enumerate_cells(cfg)

    if args.only_cell is not None:
        cells = [c for c in cells if c.cell_id == args.only_cell]
        if not cells:
            console.print(f"[red]--only-cell {args.only_cell!r} matched zero cells[/]")
            return 2

    cost_summary = estimate_total_cost(cells, args.gpu_dollars_per_hour)

    console.print(
        Panel.fit(
            "Phase-2 canonical sweep\n"
            f"  config: {args.config_name}\n"
            f"  cells: {int(cost_summary['n_cells'])}\n"
            f"  estimated wall-clock: {cost_summary['total_hours']:.1f} h\n"
            f"  estimated cost: ${cost_summary['total_usd']:.2f}  (cap ${args.max_cost_usd:.2f})\n"
            f"  GPU rate: ${args.gpu_dollars_per_hour:.2f}/h\n"
            f"  output: {_RUNS_DIR}",
            title="[bold]stl-seed sweep planner",
        )
    )
    print_cell_table(cells, args.gpu_dollars_per_hour)

    if cost_summary["total_usd"] > args.max_cost_usd:
        console.print(
            f"[red]ABORT: estimated total cost ${cost_summary['total_usd']:.2f} > "
            f"--max-cost-usd ${args.max_cost_usd:.2f}.[/]\n"
            f"  Reduce sweep scope or raise the cap explicitly."
        )
        return 1

    do_dry_run = args.dry_run or not args.confirm
    if do_dry_run:
        console.print("[yellow]DRY RUN[/] (no training; pass --confirm to actually train)")

    _install_signal_handlers()

    running_cost = 0.0
    n_ok = n_skipped = n_failed = n_dry = n_int = 0

    for i, cell in enumerate(cells):
        if _INTERRUPTED:
            # Record an INTERRUPTED entry so the log distinguishes pre-emption
            # from never-attempted cells.
            iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            r = CellResult(
                cell_id=cell.cell_id,
                model=cell.model,
                filter=cell.filter,
                task=cell.task,
                status="INTERRUPTED",
                start_iso=iso,
                end_iso=iso,
                duration_s=0.0,
                cost_usd=0.0,
            )
            append_sweep_log(r)
            n_int += 1
            continue

        # Cost guard: if this cell's *additional* estimate would push us over
        # the cap, abort cleanly rather than start a doomed cell.
        _, expected_d = estimate_cell_cost(cell, args.gpu_dollars_per_hour)
        if not do_dry_run and (running_cost + expected_d) > args.max_cost_usd:
            console.print(
                f"[red]COST CAP HIT[/] running ${running_cost:.2f} + "
                f"next ${expected_d:.2f} > ${args.max_cost_usd:.2f}; stopping."
            )
            break

        console.rule(f"[bold]Cell {i + 1}/{len(cells)}: {cell.cell_id}")
        result = run_cell(cell, dry_run=do_dry_run)
        append_sweep_log(result)

        if result.status == "OK":
            n_ok += 1
            running_cost += result.cost_usd
        elif result.status == "SKIPPED":
            n_skipped += 1
        elif result.status == "FAILED":
            n_failed += 1
        elif result.status == "DRY_RUN":
            n_dry += 1

        console.print(
            f"  status={result.status}  duration={result.duration_s:.1f}s  "
            f"cost=${result.cost_usd:.4f}  running=${running_cost:.2f}"
        )

    # Final summary banner.
    summary = (
        f"OK={n_ok}  SKIPPED={n_skipped}  FAILED={n_failed}  "
        f"DRY_RUN={n_dry}  INTERRUPTED={n_int}\n"
        f"total cost ~${running_cost:.2f} of ${args.max_cost_usd:.2f} cap"
    )
    border = "green" if n_failed == 0 else "red"
    console.print(Panel(summary, title="Sweep complete", border_style=border))

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

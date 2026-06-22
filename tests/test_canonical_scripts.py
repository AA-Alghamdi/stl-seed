"""Smoke tests for the Phase-2 canonical scripts.

These tests are deliberately lightweight: they verify imports, config
composition, cell enumeration, dry-run end-to-end, and CLI surface.
They do NOT run real training or NumPyro fitting (those require GPUs /
minutes-of-CPU and are exercised by the actual sweep on RunPod).

imports here either.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
CONFIG_DIR = REPO_ROOT / "configs"


def _import_script(name: str):
    """Import a top-level script module by path (scripts/ is not a package)."""
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Hydra config tests
# ---------------------------------------------------------------------------


def test_default_config_loads() -> None:
    """`configs/default.yaml` composes without error and has expected keys."""
    sweep_module = _import_script("run_canonical_sweep")
    cfg = sweep_module.load_config("default")
    assert "model" in cfg
    assert "filter" in cfg
    assert "task" in cfg
    assert "backend" in cfg
    assert "evaluation" in cfg
    assert int(cfg.seed) == 20260424
    # Default selections per default.yaml
    assert cfg.filter.name == "hard"
    assert cfg.backend.name == "bnb"
    # BoN budgets must match theory.md §5.
    assert list(cfg.evaluation.bon_budgets) == [1, 2, 4, 8, 16, 32, 64, 128]


def test_sweep_main_compose_18_cells() -> None:
    """`configs/sweep_main.yaml` enumerates exactly 18 cells with the right axes."""
    sweep_module = _import_script("run_canonical_sweep")
    cfg = sweep_module.load_config("sweep_main")
    cells = sweep_module.enumerate_cells(cfg)
    assert len(cells) == 18
    models = {c.model for c in cells}
    filters = {c.filter for c in cells}
    tasks = {c.task for c in cells}
    assert models == {"qwen3_0.6b", "qwen3_1.7b", "qwen3_4b"}
    assert filters == {"hard", "quantile", "continuous"}
    assert tasks == {"bio_ode_repressilator", "glucose_insulin"}
    # Cell ids are unique and well-formed.
    ids = {c.cell_id for c in cells}
    assert len(ids) == 18
    for cid in ids:
        parts = cid.split("__")
        assert len(parts) == 3, cid


def test_pilot_config_loads() -> None:
    """`configs/pilot.yaml` composes; reduced budgets per pilot scope."""
    sweep_module = _import_script("run_canonical_sweep")
    cfg = sweep_module.load_config("pilot")
    assert cfg.run.name == "pilot"
    assert cfg.backend.name == "mlx"
    assert int(cfg.evaluation.n_seeds) == 2
    assert int(cfg.cost.max_usd) == 1


def test_per_cell_config_resolves() -> None:
    """`load_cell_config(cell)` resolves model + filter + task overrides."""
    sweep_module = _import_script("run_canonical_sweep")
    cell = sweep_module.Cell(model="qwen3_0.6b", filter="continuous", task="glucose_insulin")
    cfg = sweep_module.load_cell_config(cell)
    assert cfg.model.size == "0.6b"
    assert cfg.filter.name == "continuous"
    assert cfg.task.family == "glucose_insulin"
    assert cfg.run.name == cell.cell_id


# ---------------------------------------------------------------------------
# Cost estimator
# ---------------------------------------------------------------------------


def test_estimate_total_cost_under_budget() -> None:
    """Default sweep estimate must fit comfortably under the $25 cap."""
    sweep_module = _import_script("run_canonical_sweep")
    cfg = sweep_module.load_config("sweep_main")
    cells = sweep_module.enumerate_cells(cfg)
    summary = sweep_module.estimate_total_cost(cells, dollars_per_hour=0.34)
    assert int(summary["n_cells"]) == 18
    # 18 cells × ~0.5h average × $0.34 ≈ $3-5; must be < $25.
    assert 0 < summary["total_usd"] < 25.0


# ---------------------------------------------------------------------------
# Dry-run end-to-end (CLI surface)
# ---------------------------------------------------------------------------


def test_run_canonical_sweep_dry_run() -> None:
    """End-to-end dry-run must enumerate all 18 cells without raising."""
    sweep_module = _import_script("run_canonical_sweep")
    rc = sweep_module.main(["--dry-run", "--config-name", "sweep_main"])
    assert rc == 0


def test_run_canonical_sweep_dry_run_subprocess() -> None:
    """Same but via subprocess to verify the script runs as a __main__ entrypoint."""
    rc = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "run_canonical_sweep.py"), "--dry-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rc.returncode == 0, f"stderr:\n{rc.stderr}"
    assert "Phase-2 canonical sweep" in (rc.stdout + rc.stderr)


def test_run_canonical_sweep_only_cell_filters() -> None:
    """--only-cell narrows the sweep to one cell."""
    sweep_module = _import_script("run_canonical_sweep")
    rc = sweep_module.main(
        [
            "--dry-run",
            "--only-cell",
            "qwen3_0.6b__hard__bio_ode_repressilator",
        ]
    )
    assert rc == 0


def test_run_canonical_sweep_unknown_cell_returns_2() -> None:
    sweep_module = _import_script("run_canonical_sweep")
    rc = sweep_module.main(["--dry-run", "--only-cell", "no_such_cell"])
    assert rc == 2


def test_run_canonical_sweep_cost_cap_enforced() -> None:
    """A pathologically-low --max-cost-usd aborts before any cell runs."""
    sweep_module = _import_script("run_canonical_sweep")
    rc = sweep_module.main(["--dry-run", "--max-cost-usd", "0.01"])
    assert rc == 1  # ABORT path


# ---------------------------------------------------------------------------
# Eval / analysis: import + --help (heavier deps live behind lazy imports)
# ---------------------------------------------------------------------------


def test_run_canonical_eval_imports() -> None:
    """`scripts/run_canonical_eval.py` imports cleanly on macOS (no CUDA)."""
    eval_module = _import_script("run_canonical_eval")
    assert hasattr(eval_module, "main")
    assert hasattr(eval_module, "discover_cells")
    assert hasattr(eval_module, "evaluate_cell")


def test_run_canonical_eval_help() -> None:
    rc = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "run_canonical_eval.py"), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rc.returncode == 0
    assert "Phase-2 evaluation runner" in rc.stdout


def test_canonical_analysis_imports() -> None:
    """`scripts/canonical_analysis.py` imports cleanly."""
    analysis_module = _import_script("canonical_analysis")
    assert hasattr(analysis_module, "main")
    assert hasattr(analysis_module, "build_hierarchical_data")
    assert hasattr(analysis_module, "tost_per_cell")


def test_canonical_analysis_help() -> None:
    rc = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "canonical_analysis.py"), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rc.returncode == 0
    assert "hierarchical-Bayes analysis" in rc.stdout


# ---------------------------------------------------------------------------
# Hydra config existence smoke checks (catch typos / missing files early)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rel_path",
    [
        "default.yaml",
        "sweep_main.yaml",
        "pilot.yaml",
        "task/bio_ode_repressilator.yaml",
        "task/bio_ode_toggle.yaml",
        "task/bio_ode_mapk.yaml",
        "task/glucose_insulin.yaml",
        "filter/hard.yaml",
        "filter/quantile.yaml",
        "filter/continuous.yaml",
        "model/qwen3_0.6b.yaml",
        "model/qwen3_1.7b.yaml",
        "model/qwen3_4b.yaml",
        "backend/mlx.yaml",
        "backend/bnb.yaml",
    ],
)
def test_config_file_exists(rel_path: str) -> None:
    assert (CONFIG_DIR / rel_path).is_file(), f"missing config: {rel_path}"

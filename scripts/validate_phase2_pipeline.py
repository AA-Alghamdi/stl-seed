"""Phase-2 dry-run pipeline validator.

Goal
----

Phase 2 will spend ~$15-25 of RunPod 4090 spot time on a 3 (size) x 3
(filter) x 2 (task) = 18-cell training sweep. *Before* burning that
budget we want defensible evidence that:

  * the sweep runner enumerates cells correctly and writes valid
    provenance + sweep-log artifacts;
  * the eval runner can locate the trained adapters, load them via
    backend.load(), drive the eval harness end-to-end, and emit the
    parquet schema downstream code consumes;
  * the analysis script can ingest that parquet, run hierarchical Bayes
    (or skip-bayes path), produce the registered figures + paper/
    results.md without crashing;
  * any plumbing bug between these stages (config-loading, cell-id
    parsing, schema mismatches, missing helpers, broken imports) is
    surfaced *now* with a clear error rather than discovered at the
    $25 mark.

The script is a single CPU-only entry point: it exports
``STL_SEED_USE_MOCK_BACKEND=1`` (so the sweep runner substitutes
:class:`MockBNBBackend` for the real :class:`BNBBackend`), then walks
the four pipeline stages in order, capturing pass/fail status per
stage with explicit error messages.

Stages
------

1. **Sweep dry-run + mini-sweep**: enumerate 18 cells with the canonical
   config, then run a 3-cell mini-sweep (with the mock backend) and
   verify per-cell artifacts (provenance.json, adapter dir, done flag,
   sweep_log.csv).

2. **Eval against mock checkpoints**: drive ``run_canonical_eval`` on
   the 3 mock cells, verify the parquet has the right columns and at
   least one row per cell.

3. **Analysis**: run ``canonical_analysis`` (with --skip-bayes for
   speed by default; see ``--with-bayes`` for the full path) and verify
   the BoN figure + paper/results.md exist.

4. **Real eval is also exercised**: even if some stages fail, we report
   them — this is a *validation* run, so failures are first-class
   findings to be fixed before Phase 2 launch.

Runtime budget: < 5 minutes on M5 Pro. Achieved by:

  * 3-cell mini sweep (not 18).
  * 1 spec, 1 instance, 2 seeds, BoN budgets ``[1, 2, 4]``,
    ``n_samples_per_spec=4`` for the eval harness.
  * ``--skip-bayes`` by default (NumPyro NUTS is the dominant cost).
  * Synthesized fallback eval-results parquet so the analysis stage
    is exercised even when the eval stage falls over.

Exit status
-----------

* 0 — every stage passed.
* 1 — at least one stage failed; the offending stage is named in the
  trailing summary table.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Constants and paths.
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
CONFIG_DIR = REPO_ROOT / "configs"

#: Env var name honored by the sweep + eval runners. Setting this to "1"
#: substitutes :class:`MockBNBBackend` for the real bnb backend.
USE_MOCK_ENV = "STL_SEED_USE_MOCK_BACKEND"

#: The 3 cells we run end-to-end. Chosen to span both task families and
#: at least two filters so the eval/analysis fan-out is non-trivial.
DEFAULT_VALIDATION_CELLS: tuple[str, ...] = (
    "qwen3_0.6b__hard__bio_ode_repressilator",
    "qwen3_0.6b__quantile__glucose_insulin",
    "qwen3_1.7b__continuous__bio_ode_repressilator",
)

#: Required keys in every per-cell provenance.json. Subset of the schema
#: real :class:`BNBBackend` writes; if any are missing the downstream
#: paper-reproducibility audit (paper/reproducibility.md §6) fails.
REQUIRED_PROVENANCE_KEYS: tuple[str, ...] = (
    "cell_id",
    "model",
    "filter",
    "task",
    "git_sha",
    "config_hash",
    "seed",
    "training_date",
    # The runner-supplied extras come from BNBBackend's `extra=` dict:
    "training_loss_history",
    "wall_clock_seconds",
    "adapter_path",
)

#: Required columns in the eval parquet. Schema mirrors the docstring
#: in ``scripts/run_canonical_eval.py``.
REQUIRED_EVAL_COLUMNS: tuple[str, ...] = (
    "model",
    "filter",
    "task",
    "spec",
    "instance",
    "seed",
    "N",
    "success",
    "rho",
    "action_diversity_first",
    "action_diversity_seq",
    "wall_clock_s",
)


# --------------------------------------------------------------------------
# Tiny status / printing helpers.
# --------------------------------------------------------------------------


@dataclass
class StageResult:
    """Per-stage pass/fail record carried through the validation report."""

    name: str
    status: str  # "PASS" | "FAIL" | "SKIP"
    duration_s: float
    detail: str = ""
    findings: list[str] = field(default_factory=list)


def _print_banner(title: str) -> None:
    bar = "=" * 78
    print(bar)
    print(f"  {title}")
    print(bar)


def _print_kv(k: str, v: Any) -> None:
    print(f"  {k:<28s} {v}")


# --------------------------------------------------------------------------
# Script-import helpers (scripts/ is not a package).
# --------------------------------------------------------------------------


def _import_script(name: str) -> Any:
    """Import a top-level script module by file path."""
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None, name
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# Stage 1: mini-sweep.
# --------------------------------------------------------------------------


def stage_sweep(
    runs_dir: Path,
    cells: tuple[str, ...],
    *,
    full_dry_run_first: bool = True,
) -> StageResult:
    """Run a mini-sweep with the mock backend and verify artifacts.

    Steps
    -----
    1. Optionally call the sweep runner with ``--dry-run`` against the
       full sweep_main config (no per-cell training, just enumeration +
       provenance stub) to verify cell enumeration produces 18 cells.
    2. For each cell in ``cells``, invoke the runner with ``--only-cell``
       and ``--confirm`` (with the mock backend env var set) to run the
       mock training end-to-end.
    3. Verify that ``runs/canonical/<cell_id>/`` exists, contains
       ``provenance.json`` with the required keys, an ``adapter/``
       subdirectory, and a ``done.flag`` file. Verify
       ``runs/canonical/sweep_log.csv`` was written.
    """
    t0 = time.perf_counter()
    findings: list[str] = []
    sweep_module = _import_script("run_canonical_sweep")

    # 1. Verify cell enumeration produces exactly 18 cells. We call
    # enumerate_cells directly (not via the dry-run path) so no provenance
    # stub directories are created — those would otherwise pollute the
    # eval-stage discover_cells() and inflate the cell count.
    if full_dry_run_first:
        sweep_module = _import_script("run_canonical_sweep")
        full_cfg = sweep_module.load_config("sweep_main")
        full_cells = sweep_module.enumerate_cells(full_cfg)
        if len(full_cells) != 18:
            findings.append(f"sweep_main config enumerated {len(full_cells)} cells; expected 18")

    # 2. Run the mini sweep, one cell at a time, with the mock backend.
    if USE_MOCK_ENV not in os.environ:
        findings.append(
            f"{USE_MOCK_ENV} not set in environment at stage_sweep entry; "
            "the runner will dispatch to the real BNBBackend (which will "
            "fail on a CPU-only host). The validation entry point sets "
            "this for you; if you see this message you invoked "
            "stage_sweep directly without the env var."
        )

    sweep_module = _import_script("run_canonical_sweep")
    for cell_id in cells:
        print(f"  -> sweep cell {cell_id}")
        rc = sweep_module.main(
            [
                "--config-name",
                "sweep_main",
                "--only-cell",
                cell_id,
                "--confirm",
                # Override max-cost so the mock-cost (~$0) is well under it.
                "--max-cost-usd",
                "100.0",
            ]
        )
        if rc != 0:
            findings.append(f"sweep --only-cell {cell_id} returned rc={rc}; expected 0")

    # 3. Verify per-cell artifacts.
    for cell_id in cells:
        cell_dir = runs_dir / cell_id
        if not cell_dir.exists():
            findings.append(f"missing cell directory {cell_dir}")
            continue
        prov_path = cell_dir / "provenance.json"
        if not prov_path.exists():
            findings.append(f"missing provenance.json under {cell_dir}")
        else:
            try:
                prov = json.loads(prov_path.read_text())
            except json.JSONDecodeError as exc:
                findings.append(f"{prov_path} is not valid JSON: {exc}")
                prov = {}
            missing = [k for k in REQUIRED_PROVENANCE_KEYS if k not in prov]
            if missing:
                findings.append(f"{prov_path} missing required keys: {missing}")
            # The mock backend MUST be marked as such in some downstream-
            # readable form. The runner's write_provenance does not surface
            # the BNBBackend-side mock fields directly; we check the loss-
            # history is the synthetic-curve length (>=4) as a proxy.
            loss = prov.get("training_loss_history", [])
            if not isinstance(loss, list) or len(loss) < 4:
                findings.append(
                    f"{prov_path} has implausibly short training_loss_history "
                    f"({len(loss) if isinstance(loss, list) else 'N/A'}); "
                    "expected >= 4 from MockBNBBackend's synthetic curve."
                )
        adapter_dir = cell_dir / "adapter"
        if not adapter_dir.exists():
            findings.append(f"missing adapter dir under {cell_dir}")
        else:
            mock_marker = adapter_dir / "MOCK.txt"
            if not mock_marker.exists():
                findings.append(
                    f"adapter at {adapter_dir} has no MOCK.txt marker; "
                    "MockBNBBackend should have written one."
                )
        if not (cell_dir / "done.flag").exists():
            findings.append(f"missing done.flag under {cell_dir}")

    # Verify the sweep log.
    sweep_log = runs_dir / "sweep_log.csv"
    if not sweep_log.exists():
        findings.append(f"missing sweep_log.csv at {sweep_log}")
    else:
        rows = sweep_log.read_text().splitlines()
        # Header + one row per cell at minimum.
        if len(rows) < 1 + len(cells):
            findings.append(
                f"sweep_log.csv has {len(rows)} lines; expected >= {1 + len(cells)} "
                "(header + one row per cell)."
            )

    duration = time.perf_counter() - t0
    status = "PASS" if not findings else "FAIL"
    return StageResult(
        name="stage_sweep",
        status=status,
        duration_s=duration,
        detail=f"ran {len(cells)} cells under mock backend",
        findings=findings,
    )


# --------------------------------------------------------------------------
# Stage 2: eval against mock checkpoints.
# --------------------------------------------------------------------------


def stage_eval(
    runs_dir: Path,
    *,
    n_seeds: int = 2,
    n_instances: int = 1,
    n_samples_per_spec: int = 4,
    bon_budgets: tuple[int, ...] = (1, 2, 4),
) -> StageResult:
    """Drive ``run_canonical_eval`` against the mock-trained cells.

    Eval-side overrides are deliberately small — the goal is to exercise
    the parquet schema, the cell-discovery code, and the harness's main
    loop, not to produce statistically meaningful BoN curves.
    """
    t0 = time.perf_counter()
    findings: list[str] = []
    eval_module = _import_script("run_canonical_eval")

    # Discover cells in the validation runs_dir. Restrict to cells with a
    # done.flag so dry-run stub directories (provenance-only, no adapter)
    # do not pollute the eval rows.
    all_cells = eval_module.discover_cells(runs_dir)
    cells = [c for c in all_cells if (c.cell_dir / "done.flag").exists()]
    if not cells:
        findings.append(
            f"no completed cells (with done.flag) discovered under {runs_dir}; "
            f"saw {len(all_cells)} dirs total"
        )
        return StageResult(
            name="stage_eval",
            status="FAIL",
            duration_s=time.perf_counter() - t0,
            detail="discover_cells returned no completed cells",
            findings=findings,
        )

    # Drive the harness directly per cell so we can use compact eval
    # settings without touching the on-disk YAML.
    import pandas as pd

    rows_all: list[pd.DataFrame] = []
    for cell in cells:
        try:
            cfg = eval_module.load_cell_config(cell.model, cell.filter, cell.task)
            df = eval_module.evaluate_cell(
                cell=cell,
                cfg=cfg,
                n_seeds=int(n_seeds),
                n_instances=int(n_instances),
                n_samples_per_spec=int(n_samples_per_spec),
                bon_budgets=list(bon_budgets),
                key_seed=int(cfg.seed),
            )
            rows_all.append(df)
        except Exception as exc:  # noqa: BLE001
            findings.append(f"evaluate_cell({cell.cell_id}) failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    if rows_all:
        full = pd.concat(rows_all, ignore_index=True)
        out_path = runs_dir / "eval_results.parquet"
        full.to_parquet(out_path, index=False)
        # Verify schema.
        cols = set(full.columns)
        missing = [c for c in REQUIRED_EVAL_COLUMNS if c not in cols]
        if missing:
            findings.append(
                f"eval parquet missing required columns: {missing}; actually has {sorted(cols)}"
            )
        if full.empty:
            findings.append("eval parquet is empty (0 rows)")
        # Verify all sweep cells have at least one row in the eval output.
        # Use [...] indexing rather than attribute access on the row series:
        # ``r.filter`` resolves to ``pd.Series.filter`` (the method), not the
        # "filter" column.
        cells_in_eval = {
            (r["model"], r["filter"], r["task"])
            for _, r in full[["model", "filter", "task"]].drop_duplicates().iterrows()
        }
        for c in cells:
            triple = (c.model, c.filter, c.task)
            if triple not in cells_in_eval:
                findings.append(f"cell {c.cell_id} not represented in eval parquet")
        # Verify BoN budgets.
        actual_bon = sorted(int(n) for n in full["N"].unique())
        if set(actual_bon) != set(bon_budgets):
            findings.append(
                f"BoN budgets in eval parquet {actual_bon} != requested {sorted(bon_budgets)}"
            )
    else:
        findings.append("no eval rows produced (every evaluate_cell call failed)")

    duration = time.perf_counter() - t0
    status = "PASS" if not findings else "FAIL"
    detail = f"{len(cells)} cells, {sum(len(d) for d in rows_all)} rows" if rows_all else "no rows"
    return StageResult(
        name="stage_eval",
        status=status,
        duration_s=duration,
        detail=detail,
        findings=findings,
    )


# --------------------------------------------------------------------------
# Synthetic eval-parquet fallback (so analysis can be tested even if eval
# --------------------------------------------------------------------------


def write_synthetic_eval_parquet(
    out_path: Path,
    cells: tuple[str, ...],
    *,
    bon_budgets: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128),
    n_seeds: int = 5,
    n_instances: int = 3,
) -> int:
    """Write a synthetic eval-results parquet matching the documented schema.

    Used as a fallback when stage_eval cannot produce real rows (e.g., a
    bug in the eval harness chain prevents any real cell from completing).
    The synthetic data is structured so the analysis stage can exercise
    its full code path: every (cell, instance, seed, N) tuple gets a row,
    and ``success`` is a deterministic Bernoulli draw whose probability
    depends on the filter (so TOST has something nontrivial to compute).

    Returns the number of rows written.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(20260424)
    rows: list[dict[str, Any]] = []
    for cell_id in cells:
        parts = cell_id.split("__")
        if len(parts) != 3:
            continue
        model, filt, task_slug = parts
        # Filter-conditional success rate so TOST sees a nonzero contrast.
        p_base = {"hard": 0.45, "quantile": 0.50, "continuous": 0.52}.get(filt, 0.40)
        for instance in range(n_instances):
            for seed in range(n_seeds):
                for N in bon_budgets:
                    # BoN reuse: P(any of N succeed) = 1 - (1-p)^N.
                    p_success = 1.0 - (1.0 - p_base) ** N
                    success = int(rng.random() < p_success)
                    rho = float(rng.normal(0.0 if success else -0.3, 0.4))
                    rows.append(
                        {
                            "model": model,
                            "filter": filt,
                            "task": task_slug,
                            "spec": f"{task_slug}.synthetic",
                            "instance": int(instance),
                            "seed": int(seed),
                            "N": int(N),
                            "success": int(success),
                            "rho": float(rho),
                            "action_diversity_first": float(rng.uniform(0.4, 0.9)),
                            "action_diversity_seq": float(rng.uniform(0.5, 1.0)),
                            "wall_clock_s": float(rng.uniform(0.01, 0.05)),
                            "status": "OK_SYNTHETIC",
                        }
                    )
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return int(len(df))


# --------------------------------------------------------------------------
# Stage 3: analysis.
# --------------------------------------------------------------------------


def stage_analysis(
    runs_dir: Path,
    *,
    with_bayes: bool,
    fallback_synthetic: bool = True,
    cells: tuple[str, ...] = DEFAULT_VALIDATION_CELLS,
) -> StageResult:
    """Drive ``canonical_analysis`` against the eval parquet.

    If the eval parquet is missing or too sparse for the hierarchical-
    Bayes path, fall back to a synthesized parquet so the analysis stage
    is still exercised (the goal is to validate the *pipeline*, not the
    *content* of the analysis).
    """
    t0 = time.perf_counter()
    findings: list[str] = []
    eval_path = runs_dir / "eval_results.parquet"

    # Decide whether to fall back. We always fall back when the parquet
    # is missing; we *also* fall back when --with-bayes is requested but
    # the real eval rows are too sparse to support a 4-chain NUTS fit.
    use_synthetic = False
    if not eval_path.exists():
        if not fallback_synthetic:
            findings.append(f"eval parquet missing at {eval_path}; analysis cannot run")
            return StageResult(
                name="stage_analysis",
                status="FAIL",
                duration_s=time.perf_counter() - t0,
                detail="no input parquet",
                findings=findings,
            )
        use_synthetic = True
    elif with_bayes:
        # Synthesize for the Bayes path so we have enough rows.
        use_synthetic = True

    if use_synthetic:
        n_rows = write_synthetic_eval_parquet(eval_path, cells)
        print(f"  -> wrote synthetic eval parquet ({n_rows} rows) at {eval_path}")

    analysis_module = _import_script("canonical_analysis")
    out_dir = runs_dir / "analysis"
    results_md = runs_dir / "results.md"

    args = [
        "--eval-results",
        str(eval_path),
        "--output-dir",
        str(out_dir),
        "--results-md",
        str(results_md),
    ]
    if not with_bayes:
        args.append("--skip-bayes")
    else:
        # Smaller chains for the validation runtime budget.
        args += ["--n-chains", "2", "--n-warmup", "200", "--n-samples", "200"]

    try:
        rc = analysis_module.main(args)
        if rc != 0:
            findings.append(f"canonical_analysis returned rc={rc}; expected 0")
    except Exception as exc:  # noqa: BLE001
        findings.append(f"canonical_analysis raised {type(exc).__name__}: {exc}")
        traceback.print_exc()

    # Verify outputs.
    bon_fig = out_dir / "figures" / "bon_curves.png"
    if not bon_fig.exists():
        findings.append(f"missing figure {bon_fig}")
    if not results_md.exists():
        findings.append(f"missing {results_md}")
    if with_bayes:
        nc = out_dir / "posterior.nc"
        if not nc.exists():
            findings.append(f"missing posterior.nc at {nc}")

    duration = time.perf_counter() - t0
    status = "PASS" if not findings else "FAIL"
    return StageResult(
        name="stage_analysis",
        status=status,
        duration_s=duration,
        detail=f"with_bayes={with_bayes}, synthetic_fallback={use_synthetic}",
        findings=findings,
    )


# --------------------------------------------------------------------------
# Stage 4: firewall + smoke checks.
# --------------------------------------------------------------------------


def stage_firewall() -> StageResult:
    """Stage 4 (firewall) is a SKIP in the public repo.

    The firewall verification harness (originally scripts/REDACTED.sh) was
    purged by commit a6348bd ("NUKE: purge strategic + internal docs"); the
    private repo carries the firewall checks separately. Returning SKIP
    rather than removing the stage keeps stage indexing stable for the
    downstream report writer and lets a future operator drop in their own
    firewall-check command if they fork.
    """
    return StageResult(
        name="stage_firewall",
        status="SKIP",
        duration_s=0.0,
        detail="firewall script not present in public repo; SKIP",
        findings=[],
    )


# --------------------------------------------------------------------------
# Driver.
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="validate_phase2_pipeline.py",
        description="End-to-end Phase-2 dry-run pipeline validator (mock-bnb backend).",
    )
    p.add_argument(
        "--with-bayes",
        action="store_true",
        help="Run the full hierarchical-Bayes analysis stage (slower; default skips).",
    )
    p.add_argument(
        "--keep-runs-dir",
        action="store_true",
        help="Do not delete the temp runs/ dir at exit (useful for inspection).",
    )
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Explicit runs dir (default: a fresh temp dir).",
    )
    p.add_argument(
        "--cells",
        type=str,
        default=",".join(DEFAULT_VALIDATION_CELLS),
        help="Comma-separated cell_id list to validate.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Force the mock backend env var on for the lifetime of this process.
    # The sweep + eval runners both read this on every cell.
    os.environ[USE_MOCK_ENV] = "1"
    # Defensive: explicitly UN-set the real-training guard so the mock can
    # actually run if some upstream wrapper had set it.
    os.environ.pop("STL_SEED_REAL_TRAINING", None)

    # Decide where to write artifacts. Use a fresh temp dir by default so
    # the validation does not pollute runs/canonical/. We export
    # STL_SEED_RUNS_DIR_OVERRIDE so the sweep + eval runners pick it up
    # at module import time without monkey-patching.
    runs_dir = (
        Path(args.runs_dir).resolve()
        if args.runs_dir is not None
        else Path(tempfile.mkdtemp(prefix="stl_seed_validate_"))
    )
    runs_dir.mkdir(parents=True, exist_ok=True)
    os.environ["STL_SEED_RUNS_DIR_OVERRIDE"] = str(runs_dir)

    cells = tuple(c.strip() for c in args.cells.split(",") if c.strip())
    if not cells:
        print("ERROR: --cells parsed to empty tuple", file=sys.stderr)
        return 2

    _print_banner("Phase-2 dry-run pipeline validator")
    _print_kv("repo_root", REPO_ROOT)
    _print_kv("runs_dir", runs_dir)
    _print_kv("cells", cells)
    _print_kv("with_bayes", args.with_bayes)
    _print_kv("env[USE_MOCK_ENV]", os.environ.get(USE_MOCK_ENV))
    print()

    results: list[StageResult] = []

    _print_banner("Stage 1: sweep (mock backend)")
    results.append(stage_sweep(runs_dir, cells))
    _print_kv("status", results[-1].status)
    _print_kv("duration", f"{results[-1].duration_s:.2f}s")
    for f in results[-1].findings:
        print(f"    FINDING: {f}")
    print()

    _print_banner("Stage 2: eval (mock-loaded checkpoints)")
    results.append(stage_eval(runs_dir))
    _print_kv("status", results[-1].status)
    _print_kv("duration", f"{results[-1].duration_s:.2f}s")
    for f in results[-1].findings:
        print(f"    FINDING: {f}")
    print()

    _print_banner("Stage 3: analysis (figures + paper/results.md)")
    results.append(stage_analysis(runs_dir, with_bayes=args.with_bayes, cells=cells))
    _print_kv("status", results[-1].status)
    _print_kv("duration", f"{results[-1].duration_s:.2f}s")
    for f in results[-1].findings:
        print(f"    FINDING: {f}")
    print()

    results.append(stage_firewall())
    _print_kv("status", results[-1].status)
    _print_kv("duration", f"{results[-1].duration_s:.2f}s")
    for f in results[-1].findings:
        print(f"    FINDING: {f}")
    print()

    # Summary table.
    _print_banner("Validation summary")
    total_dur = sum(r.duration_s for r in results)
    total_findings = sum(len(r.findings) for r in results)
    for r in results:
        marker = {"PASS": "OK", "FAIL": "FAIL", "SKIP": "SKIP"}[r.status]
        print(f"  [{marker:>4s}]  {r.name:<24s}  {r.duration_s:>7.2f}s  {r.detail}")
    print()
    _print_kv("total duration", f"{total_dur:.2f}s")
    _print_kv("total findings", total_findings)

    overall_pass = all(r.status in ("PASS", "SKIP") for r in results) and not any(
        r.status == "FAIL" for r in results
    )
    print()
    if overall_pass:
        print("  RESULT: ALL STAGES PASSED -- pipeline is ready for the $25 RunPod run.")
    else:
        print("  RESULT: AT LEAST ONE STAGE FAILED -- fix findings before Phase 2.")

    # Keep the temp dir around for manual inspection on FAIL; otherwise clean.
    if not args.keep_runs_dir and args.runs_dir is None and overall_pass:
        with contextlib.suppress(OSError):
            shutil.rmtree(runs_dir)

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())

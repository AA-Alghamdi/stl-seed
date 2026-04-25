"""Smoke tests for ``scripts/run_unified_comparison.py``.

The unified-comparison script is the artifact's headline visualisation
harness; these tests guard the *contracts* the script must satisfy
without re-running the full empirical sweep:

* ``test_unified_runs_one_seed_per_sampler`` -- end-to-end smoke at
  ``n_seeds=1``: the script returns 0, writes the parquet / figure /
  markdown, and produces one row per cell. This catches import-time
  breakage, sampler-construction errors, and any plotting regressions.
* ``test_results_parquet_schema`` -- the on-disk parquet has the
  contractual columns required by the README hook and downstream
  analysis. Renaming or dropping any column would silently break those
  consumers; the test pins the schema.
* ``test_all_samplers_produce_finite_rho`` -- every (task, sampler,
  seed) cell produces a finite ``final_rho``. NaN/Inf in the harness
  output would propagate into the bootstrap CI and the figure;
  detecting them at this layer is the cheapest place to catch them.
* ``test_aggregate_satisfies_invariants`` -- the bootstrap-CI helper
  satisfies ``ci_lo <= mean_rho <= ci_hi`` (modulo finite-sample
  noise) and the per-cell ``n_seeds`` matches the requested budget.

These tests are smoke tests, not the empirical comparison itself.
The substantive results live in
``paper/unified_comparison_results.md``; here we only verify the
machinery.

"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Importing the harness module directly so we can reuse its helpers and
# CLI without spawning a subprocess. The script is structured so all
# heavy lifting goes through ``main(argv)``.
run_unified_comparison = importlib.import_module("run_unified_comparison")


REQUIRED_COLUMNS: tuple[str, ...] = (
    "task",
    "sampler",
    "seed",
    "final_rho",
    "satisfied",
    "n_steps_changed_by_guidance",
    "wall_clock_s",
)


@pytest.fixture(scope="module")
def harness_outputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Run the harness once with ``n_seeds=1`` and cache the artifacts.

    Module-scope so the per-test invocations share a single
    end-to-end run; each test then asserts a different invariant on
    the same output. This keeps the smoke-test wall-clock under a
    minute even with the full sampler grid.

    The harness writes three artifacts: ``results.parquet``, the
    PNG figure, and the markdown report. We point them all at the
    pytest-managed ``tmp_path`` so the real ``runs/`` and
    ``paper/figures/`` trees are not perturbed by the tests.
    """
    tmp = tmp_path_factory.mktemp("unified_comparison")
    out_dir = tmp / "runs"
    fig_path = tmp / "figures" / "unified_comparison.png"
    md_path = tmp / "unified_comparison_results.md"
    rc = run_unified_comparison.main(
        [
            "--n-seeds",
            "1",
            "--out-dir",
            str(out_dir),
            "--fig-path",
            str(fig_path),
            "--md-path",
            str(md_path),
        ]
    )
    assert rc == 0, f"harness main() returned non-zero: {rc}"
    return {
        "out_dir": out_dir,
        "parquet": out_dir / "results.parquet",
        "figure": fig_path,
        "markdown": md_path,
    }


def test_unified_runs_one_seed_per_sampler(
    harness_outputs: dict[str, Path],
) -> None:
    """Smoke: harness completes end-to-end at n_seeds=1 and writes all three
    artifacts (parquet, PNG, markdown).

    Invariants
    ----------
    * Parquet, figure, and markdown all exist on disk.
    * Parquet has exactly ``n_tasks * n_samplers * 1`` rows
      (one per (task, sampler) cell at seed-budget 1).
    """
    assert harness_outputs["parquet"].exists(), "parquet artifact missing"
    assert harness_outputs["figure"].exists(), "figure artifact missing"
    assert harness_outputs["markdown"].exists(), "markdown artifact missing"

    df = pd.read_parquet(harness_outputs["parquet"])
    expected_rows = (
        len(run_unified_comparison._DEFAULT_TASKS)
        * len(run_unified_comparison._DEFAULT_SAMPLERS)
        * 1
    )
    assert len(df) == expected_rows, (
        f"expected {expected_rows} rows (tasks x samplers x seeds), got {len(df)}"
    )

    # Markdown report has the headline section.
    md_text = harness_outputs["markdown"].read_text()
    assert "Headline" in md_text
    assert "Per-(task, sampler) results" in md_text


def test_results_parquet_schema(
    harness_outputs: dict[str, Path],
) -> None:
    """The on-disk parquet has the contractual columns and dtypes.

    ``REQUIRED_COLUMNS`` is the cross-tier contract; any rename or
    drop would silently break downstream consumers (the figure
    generator, the markdown writer, the README headline hook). We
    pin both the column set and the basic dtype shape:

    * ``task``, ``sampler`` are strings.
    * ``seed``, ``n_steps_changed_by_guidance`` are integers.
    * ``final_rho``, ``wall_clock_s`` are floats.
    * ``satisfied`` is a boolean (or numpy bool).
    """
    df = pd.read_parquet(harness_outputs["parquet"])
    actual_cols = set(df.columns)
    missing = set(REQUIRED_COLUMNS) - actual_cols
    assert not missing, f"parquet missing required columns: {missing}"

    assert df["task"].dtype == object or pd.api.types.is_string_dtype(df["task"])
    assert df["sampler"].dtype == object or pd.api.types.is_string_dtype(df["sampler"])
    assert pd.api.types.is_integer_dtype(df["seed"]), (
        f"seed should be integer, got {df['seed'].dtype}"
    )
    assert pd.api.types.is_integer_dtype(df["n_steps_changed_by_guidance"]), (
        f"n_steps_changed_by_guidance should be integer, got "
        f"{df['n_steps_changed_by_guidance'].dtype}"
    )
    assert pd.api.types.is_float_dtype(df["final_rho"]), (
        f"final_rho should be float, got {df['final_rho'].dtype}"
    )
    assert pd.api.types.is_float_dtype(df["wall_clock_s"]), (
        f"wall_clock_s should be float, got {df['wall_clock_s'].dtype}"
    )
    assert pd.api.types.is_bool_dtype(df["satisfied"]), (
        f"satisfied should be bool, got {df['satisfied'].dtype}"
    )

    # Sanity: every (task, sampler, seed) cell appears exactly once.
    n_unique = df.drop_duplicates(["task", "sampler", "seed"]).shape[0]
    assert n_unique == len(df), "duplicate (task, sampler, seed) rows in parquet"


def test_all_samplers_produce_finite_rho(
    harness_outputs: dict[str, Path],
) -> None:
    """Every cell produces a finite final_rho (no NaN, no Inf).

    A NaN or Inf in the table would propagate into the bootstrap CI
    and the figure error bars; the harness already drops non-finite
    entries from the bootstrap so the only surface symptom would be
    silently shrunken sample sizes. This test forbids the upstream
    fault.
    """
    df = pd.read_parquet(harness_outputs["parquet"])
    nonfinite = df[~np.isfinite(df["final_rho"])]
    assert nonfinite.empty, (
        "non-finite final_rho rows present in parquet:\n"
        f"{nonfinite[['task', 'sampler', 'seed', 'final_rho']].to_string(index=False)}"
    )
    # Wall-clock must be non-negative and finite.
    assert (df["wall_clock_s"] >= 0).all(), "wall_clock_s contains negative entries"
    assert np.isfinite(df["wall_clock_s"]).all(), "wall_clock_s contains non-finite entries"


def test_aggregate_ci_invariants() -> None:
    """The bootstrap CI helper satisfies the ``lo <= mean <= hi`` invariant.

    Properties asserted:
    * For a non-degenerate sample, ``ci_lo <= sample_mean <= ci_hi``
      (within a small numerical tolerance for the percentile estimator).
    * For an all-NaN input, the helper returns ``(nan, nan)``.
    * For a single-element input, the helper returns ``(nan, nan)``
      (bootstrap is undefined with n < 2).
    """
    rng = np.random.default_rng(0)
    sample = rng.normal(loc=3.0, scale=1.5, size=64)
    lo, hi = run_unified_comparison._bootstrap_ci(sample, n_resamples=500, seed=42)
    mean = float(np.mean(sample))
    assert lo <= mean <= hi, f"CI does not bracket the mean: lo={lo}, mean={mean}, hi={hi}"

    nan_input = np.array([np.nan, np.nan])
    nan_lo, nan_hi = run_unified_comparison._bootstrap_ci(nan_input, seed=0)
    assert np.isnan(nan_lo) and np.isnan(nan_hi)

    one_el = np.array([1.0])
    o_lo, o_hi = run_unified_comparison._bootstrap_ci(one_el, seed=0)
    assert np.isnan(o_lo) and np.isnan(o_hi)


def test_aggregate_n_seeds_matches_request() -> None:
    """Per-cell aggregation reports n_seeds equal to the requested budget.

    Constructs a tiny synthetic DataFrame with the harness's row
    schema and verifies the aggregator preserves the count per
    (task, sampler).
    """
    rows = []
    for seed in range(3):
        for task in ("a_task",):
            for samp in ("standard", "gradient_guided"):
                rows.append(
                    {
                        "task": task,
                        "spec_key": "x",
                        "sampler": samp,
                        "seed": seed,
                        "final_rho": float(seed),
                        "satisfied": bool(seed > 0),
                        "n_steps_changed_by_guidance": 0,
                        "wall_clock_s": 0.01,
                    }
                )
    df = pd.DataFrame(rows)
    agg = run_unified_comparison._aggregate(df)
    assert (agg["n_seeds"] == 3).all(), (
        f"n_seeds column not 3 across all aggregated rows: {agg['n_seeds'].tolist()}"
    )
    # Mean rho per cell: 0+1+2 / 3 = 1.0.
    assert (agg["mean_rho"].round(6) == 1.0).all()
    # Satisfaction fraction: 2/3 trajectories satisfied.
    assert np.allclose(agg["sat_frac"], 2.0 / 3.0)

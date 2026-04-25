"""Smoke tests for ``scripts/benchmark_compute_cost.py``.

The cost-benchmark script is the artifact's compute-vs-quality Pareto
visualisation; these tests guard the *contracts* the script must satisfy
without re-running the full N=8 sweep:

* ``test_benchmark_runs_one_sampler_one_seed`` -- end-to-end smoke at
  ``--quick`` settings on a single sampler. Verifies the script returns
  zero, writes the parquet / per_seed parquet / figure / markdown, and
  produces the contractual columns.
* ``test_results_parquet_schema`` -- the on-disk parquet has every
  column promised by the markdown report's table header. Renaming or
  dropping a column would silently break the report; the test pins it.
* ``test_pareto_helper_basic`` -- :func:`_pareto_front` returns the
  known frontier on a hand-built (cost, quality) set.
* ``test_sim_call_proxy_known_values`` -- the analytical
  simulator-call proxy returns the documented values for the canonical
  glucose-insulin task (H=12, K=5).

These tests are smoke tests, not the empirical benchmark itself. The
substantive results live in ``paper/compute_cost_results.md``.

REDACTED firewall. None of these tests import REDACTED / REDACTED /
REDACTED / REDACTED / REDACTED. The tested script is
firewalled (verified by ``scripts/REDACTED.sh``).
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
benchmark_compute_cost = importlib.import_module("benchmark_compute_cost")


REQUIRED_COLUMNS: tuple[str, ...] = (
    "task",
    "sampler",
    "n_seeds",
    "wall_clock_s_cold",
    "wall_clock_s_warm",
    "wall_clock_s_warm_std",
    "peak_rss_delta_mb",
    "mean_rho",
    "std_rho",
    "sem_rho",
    "sat_frac",
    "target_rho",
    "target_hit_frac",
    "n_simulator_calls_proxy",
    "sim_call_formula",
    "time_to_target_s",
)

REQUIRED_PER_SEED_COLUMNS: tuple[str, ...] = (
    "task",
    "spec_key",
    "sampler",
    "seed",
    "seed_index",
    "is_cold",
    "wall_clock_s",
    "rss_pre_mb",
    "rss_post_mb",
    "rss_delta_mb",
    "final_rho",
    "satisfied",
    "hits_target",
)


@pytest.fixture(scope="module")
def harness_outputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Run the harness once at quick-mode settings and cache the artifacts.

    Module-scope so the per-test invocations share a single end-to-end
    run; each test then asserts a different invariant on the same
    output. We restrict to a single cheap sampler (``standard``) and a
    single seed pair so the full smoke wall-clock is sub-15-seconds even
    on CI.
    """
    tmp = tmp_path_factory.mktemp("cost_benchmark")
    out_dir = tmp / "runs"
    fig_path = tmp / "fig.png"
    md_path = tmp / "report.md"

    # Two seeds gives us a separate cold and warm timing; the smallest
    # meaningful smoke configuration. Single sampler keeps wall-clock
    # bounded; ``standard`` is the cheapest cell (~0.1 s on M5 Pro).
    rc = benchmark_compute_cost.main(
        [
            "--n-seeds",
            "2",
            "--tasks",
            "glucose_insulin",
            "--samplers",
            "standard",
            "--out-dir",
            str(out_dir),
            "--fig-path",
            str(fig_path),
            "--md-path",
            str(md_path),
            "--quiet",
        ]
    )
    assert rc == 0, f"benchmark_compute_cost.main returned {rc}"
    return {
        "out_dir": out_dir,
        "parquet": out_dir / "results.parquet",
        "per_seed": out_dir / "per_seed.parquet",
        "fig_path": fig_path,
        "md_path": md_path,
    }


def test_benchmark_runs_one_sampler_one_seed(harness_outputs: dict[str, Path]) -> None:
    """End-to-end: parquet, per_seed, figure, markdown all written."""
    assert harness_outputs["parquet"].exists()
    assert harness_outputs["per_seed"].exists()
    assert harness_outputs["fig_path"].exists()
    assert harness_outputs["md_path"].exists()
    # PNG is non-empty.
    assert harness_outputs["fig_path"].stat().st_size > 1000
    # Markdown is non-trivial.
    md_text = harness_outputs["md_path"].read_text()
    assert "# Compute-cost vs quality Pareto" in md_text
    assert "## 1. Headline" in md_text
    assert "## 3. Per-(task, sampler) results" in md_text


def test_results_parquet_schema(harness_outputs: dict[str, Path]) -> None:
    """All contractual columns are present in the per-cell parquet."""
    df = pd.read_parquet(harness_outputs["parquet"])
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing column {col!r} in results.parquet"
    # One row per (task, sampler) cell.
    assert len(df) == 1, f"Expected 1 row (1 task × 1 sampler), got {len(df)}"
    # Cold timing must be finite and positive.
    assert np.isfinite(df["wall_clock_s_cold"].iloc[0])
    assert df["wall_clock_s_cold"].iloc[0] > 0.0
    # Warm timing exists (n_seeds >= 2).
    assert np.isfinite(df["wall_clock_s_warm"].iloc[0])


def test_per_seed_parquet_schema(harness_outputs: dict[str, Path]) -> None:
    """All contractual columns are present in the per-seed parquet."""
    df = pd.read_parquet(harness_outputs["per_seed"])
    for col in REQUIRED_PER_SEED_COLUMNS:
        assert col in df.columns, f"Missing column {col!r} in per_seed.parquet"
    # Two rows for the two seeds.
    assert len(df) == 2
    # Exactly one cold seed.
    assert int(df["is_cold"].sum()) == 1
    # All rho values finite.
    assert np.isfinite(df["final_rho"]).all()


def test_pareto_helper_basic() -> None:
    """``_pareto_front`` returns the known frontier on a hand-built set."""
    # A: cheap and bad. B: cheap and good (frontier). C: dominated by B
    # (more expensive, same quality). D: expensive and best (frontier).
    points = [
        (1.0, 0.0, "A"),
        (1.0, 5.0, "B"),
        (2.0, 5.0, "C"),
        (10.0, 9.0, "D"),
    ]
    front = benchmark_compute_cost._pareto_front(points)
    assert "B" in front
    assert "D" in front
    assert "A" not in front  # dominated by B (same x, lower y)
    assert "C" not in front  # dominated by B (lower x, same y)


def test_pareto_helper_handles_non_finite() -> None:
    """Non-finite x or y collapses the point off the frontier."""
    points = [
        (1.0, 5.0, "A"),
        (float("inf"), 10.0, "B"),  # inf cost: should NOT be on frontier
        (2.0, float("nan"), "C"),  # nan quality: off frontier
    ]
    front = benchmark_compute_cost._pareto_front(points)
    assert "A" in front
    assert "B" not in front
    assert "C" not in front


def test_sim_call_proxy_known_values() -> None:
    """The analytical sim-call proxy returns the documented values.

    Pinned values for the canonical glucose-insulin task: H=12 control
    points, K=5 vocabulary entries. If any sampler's hyperparameter
    constants change in the script, this test fails loudly so the
    derivation is re-checked.
    """
    H = 12
    K = 5
    expectations = {
        "standard": 1,
        "best_of_n": 8,
        "continuous_bon": 8,
        "gradient_guided": 13,  # H + 1
        "hybrid": 52,  # 4 * (H + 1)
        "horizon_folded": 100,
        "rollout_tree": 97,  # H * 8 + 1
        "cmaes_gradient": 670,  # 32 * 20 + 30
        "beam_search_warmstart": 510,  # H * 8 * K + 30
    }
    for samp, expected in expectations.items():
        got = benchmark_compute_cost._sim_call_proxy(samp, horizon=H, k_vocab=K)
        assert got == expected, f"{samp}: expected {expected}, got {got}"


def test_sim_call_proxy_unknown_sampler_raises() -> None:
    with pytest.raises(ValueError, match="Unknown sampler"):
        benchmark_compute_cost._sim_call_proxy("not_a_sampler", horizon=12, k_vocab=5)


def test_main_rejects_unknown_sampler(tmp_path: Path) -> None:
    """CLI returns non-zero when given an invalid sampler name."""
    rc = benchmark_compute_cost.main(
        [
            "--n-seeds",
            "1",
            "--tasks",
            "glucose_insulin",
            "--samplers",
            "not_a_real_sampler",
            "--out-dir",
            str(tmp_path / "runs"),
            "--fig-path",
            str(tmp_path / "fig.png"),
            "--md-path",
            str(tmp_path / "report.md"),
            "--quiet",
        ]
    )
    assert rc == 2


def test_main_rejects_unknown_task(tmp_path: Path) -> None:
    """CLI returns non-zero when given an invalid task name."""
    rc = benchmark_compute_cost.main(
        [
            "--n-seeds",
            "1",
            "--tasks",
            "not_a_real_task",
            "--samplers",
            "standard",
            "--out-dir",
            str(tmp_path / "runs"),
            "--fig-path",
            str(tmp_path / "fig.png"),
            "--md-path",
            str(tmp_path / "report.md"),
            "--quiet",
        ]
    )
    assert rc == 2

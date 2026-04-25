"""Tests for ``stl_seed.evaluation``.

Covers the metrics primitives and the harness end-to-end against
synthetic stand-ins for the simulator and STL evaluator. The harness
is intentionally protocol-based so we can exercise it without the
sibling A8/A9 deliverables (concrete simulators / STL evaluator).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stl_seed.evaluation import (
    EvalHarness,
    action_diversity,
    bon_success,
    bon_success_curve,
    goodhart_gap,
    rho_margin,
    success_rate,
)
from stl_seed.evaluation.runner import (
    DIVERSITY_WARNING_THRESHOLD,
    EvalRunner,
    RunnerConfig,
    stringify_aggregate,
)

# ---------------------------------------------------------------------------
# Metrics: success_rate
# ---------------------------------------------------------------------------


def test_success_rate_basic() -> None:
    """ρ = [-1, 0.5, 1] → success_rate = 2/3."""
    assert success_rate(np.array([-1.0, 0.5, 1.0])) == pytest.approx(2.0 / 3.0)


def test_success_rate_jax_array() -> None:
    rhos = jnp.array([-0.1, 0.0, 0.1, 1.0])
    # ρ > 0 strict — both 0.0 and -0.1 fail; only 0.1 and 1.0 pass.
    assert success_rate(rhos) == pytest.approx(0.5)


def test_success_rate_drops_nan() -> None:
    rhos = np.array([1.0, np.nan, -0.1, np.inf, -np.inf])
    # Finite subset: [1.0, -0.1] → success rate 1/2.
    assert success_rate(rhos) == pytest.approx(0.5)


def test_success_rate_empty() -> None:
    assert np.isnan(success_rate(np.array([])))
    assert np.isnan(success_rate(np.array([np.nan, np.inf])))


# ---------------------------------------------------------------------------
# Metrics: bon_success
# ---------------------------------------------------------------------------


def test_bon_success_monotone_in_n() -> None:
    """BoN success is non-decreasing in N (theory.md §5; max(·) over a
    growing set can only increase)."""
    rng = np.random.default_rng(0)
    n_seeds, k_max = 25, 64
    rhos = rng.normal(loc=-0.2, scale=1.0, size=(n_seeds, k_max))
    budgets = (1, 2, 4, 8, 16, 32, 64)
    curve = bon_success_curve(rhos, budgets=budgets)
    vals = [curve[n] for n in budgets]
    for i in range(len(vals) - 1):
        assert vals[i] <= vals[i + 1] + 1e-12, (
            f"BoN-{budgets[i]} > BoN-{budgets[i + 1]}: {vals[i]:.4f} vs {vals[i + 1]:.4f}"
        )


def test_bon_success_all_positive() -> None:
    """If every sample is ρ > 0, BoN success is 1.0 at every budget."""
    rhos = np.ones((10, 8), dtype=np.float64)
    curve = bon_success_curve(rhos, budgets=(1, 2, 4, 8))
    for v in curve.values():
        assert v == pytest.approx(1.0)


def test_bon_success_all_negative() -> None:
    """If every sample is ρ < 0, BoN success is 0.0 at every budget."""
    rhos = -np.ones((5, 16), dtype=np.float64)
    curve = bon_success_curve(rhos, budgets=(1, 2, 4, 8, 16))
    for v in curve.values():
        assert v == pytest.approx(0.0)


def test_bon_success_invalid_budget() -> None:
    rhos = np.zeros((3, 4))
    with pytest.raises(ValueError):
        bon_success(rhos, n=0)
    with pytest.raises(ValueError):
        bon_success(rhos, n=5)


def test_bon_success_requires_2d() -> None:
    with pytest.raises(ValueError):
        bon_success(np.array([1.0, 2.0]), n=1)


# ---------------------------------------------------------------------------
# Metrics: rho_margin and goodhart_gap
# ---------------------------------------------------------------------------


def test_rho_margin_known() -> None:
    rhos = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    mean, iqr = rho_margin(rhos)
    assert mean == pytest.approx(2.0)
    # numpy linear interp: Q1 = 1.0, Q3 = 3.0 → IQR = 2.0
    assert iqr == pytest.approx(2.0)


def test_goodhart_gap_zero_when_specs_match() -> None:
    """ρ_proxy ≡ ρ_gold → gap = 0 exactly."""
    rng = np.random.default_rng(0)
    rhos = rng.normal(size=200)
    assert goodhart_gap(rhos, rhos) == pytest.approx(0.0, abs=1e-12)


def test_goodhart_gap_positive_when_proxy_larger() -> None:
    """If proxy ρ is uniformly larger than gold ρ, gap is positive
    (proxy over-rewards relative to the tightened gold spec)."""
    rng = np.random.default_rng(1)
    base = rng.normal(size=300)
    # Tighten by 0.2 → gold rho is base - 0.2.
    proxy = base
    gold = base - 0.5
    gap = goodhart_gap(proxy, gold, kappa=1.0)
    assert gap > 0.05


def test_goodhart_gap_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        goodhart_gap(np.array([1.0, 2.0]), np.array([1.0]))


# ---------------------------------------------------------------------------
# Harness end-to-end with stand-ins
# ---------------------------------------------------------------------------


@dataclass
class _StubTraj:
    states: jnp.ndarray
    actions: jnp.ndarray
    times: jnp.ndarray


class _StubSim:
    """Minimal Simulator stub: records (state_dim, action_dim, horizon)
    and returns a trivial trajectory whose state norm = ‖controls‖."""

    state_dim = 2
    action_dim = 1
    horizon = 4

    def simulate(self, initial_state, control_sequence, key):  # noqa: ARG002
        ctrl = jnp.asarray(control_sequence)
        # Trajectory states track the control magnitude
        states = jnp.tile(jnp.array([float(jnp.linalg.norm(ctrl)), 0.0]), (4, 1))
        return _StubTraj(states=states, actions=ctrl, times=jnp.arange(4.0))


class _GoodCheckpoint:
    """Always emits controls with positive norm → ρ > 0 (with stub
    evaluator below)."""

    name = "good"

    def sample_controls(self, spec, initial_state, key):  # noqa: ARG002
        return jnp.ones((4, 1)) * 0.5  # ‖.‖ = 1.0 → ρ = 0.5 > 0


class _BadCheckpoint:
    """Always emits zero controls → ρ ≤ 0 (with stub evaluator)."""

    name = "bad"

    def sample_controls(self, spec, initial_state, key):  # noqa: ARG002
        return jnp.zeros((4, 1))


class _RandomCheckpoint:
    """Emits Gaussian controls keyed by ``key`` so seeds are deterministic."""

    name = "random"

    def sample_controls(self, spec, initial_state, key):  # noqa: ARG002
        return jax.random.normal(key, shape=(4, 1)) * 0.5


def _stub_evaluator(spec, trajectory) -> float:  # noqa: ARG001
    """ρ := ‖states[0]‖ − 0.3, so ‖controls‖ > 0.3 → ρ > 0."""
    return float(jnp.linalg.norm(trajectory.states[0]) - 0.3)


def _const_x0(spec_name, key):  # noqa: ARG001
    return jnp.zeros((2,))


def _make_harness(specs=("toggle.spec", "lv.spec")) -> EvalHarness:
    sims = {s: _StubSim() for s in specs}
    return EvalHarness(
        simulator_registry=sims,
        spec_registry={s: object() for s in specs},
        stl_evaluator=_stub_evaluator,
        initial_state_fn=_const_x0,
        budgets=(1, 2, 4, 8),
    )


def test_harness_good_checkpoint_full_success() -> None:
    h = _make_harness()
    res = h.evaluate_checkpoint(
        checkpoint=_GoodCheckpoint(),
        held_out_specs=["toggle.spec", "lv.spec"],
        n_samples_per_spec=8,
        key=42,
    )
    assert res.checkpoint_name == "good"
    assert set(res.per_spec.keys()) == {"toggle.spec", "lv.spec"}
    for s, r in res.per_spec.items():
        assert r.success_rate_marginal == pytest.approx(1.0), s
        for n, v in r.bon_success.items():
            assert v == pytest.approx(1.0), (s, n)


def test_harness_bad_checkpoint_zero_success() -> None:
    h = _make_harness()
    res = h.evaluate_checkpoint(
        checkpoint=_BadCheckpoint(),
        held_out_specs=["toggle.spec"],
        n_samples_per_spec=8,
        key=0,
    )
    r = res.per_spec["toggle.spec"]
    assert r.success_rate_marginal == pytest.approx(0.0)
    for v in r.bon_success.values():
        assert v == pytest.approx(0.0)


def test_harness_bon_monotone_random_policy() -> None:
    """BoN curve from a random policy should be non-decreasing across
    budgets within a single (cell, seed)."""
    h = _make_harness()
    res = h.evaluate_checkpoint(
        checkpoint=_RandomCheckpoint(),
        held_out_specs=["toggle.spec"],
        n_samples_per_spec=8,
        key=7,
    )
    r = res.per_spec["toggle.spec"]
    budgets = sorted(r.bon_success.keys())
    vals = [r.bon_success[n] for n in budgets]
    for i in range(len(vals) - 1):
        assert vals[i] <= vals[i + 1] + 1e-12, (budgets[i], vals[i], vals[i + 1])


def test_harness_rejects_too_small_budget_pool() -> None:
    h = _make_harness()
    with pytest.raises(ValueError, match="must be >="):
        h.evaluate_checkpoint(
            checkpoint=_GoodCheckpoint(),
            held_out_specs=["toggle.spec"],
            n_samples_per_spec=4,  # < max(budgets)=8
            key=0,
        )


def test_harness_unknown_spec_raises() -> None:
    h = _make_harness()
    with pytest.raises(KeyError):
        h.evaluate_checkpoint(
            checkpoint=_GoodCheckpoint(),
            held_out_specs=["nonexistent"],
            n_samples_per_spec=8,
            key=0,
        )


def test_harness_aggregate_bon_matches_per_spec_mean() -> None:
    h = _make_harness()
    res = h.evaluate_checkpoint(
        checkpoint=_RandomCheckpoint(),
        held_out_specs=["toggle.spec", "lv.spec"],
        n_samples_per_spec=8,
        key=11,
    )
    agg = res.aggregate_bon()
    for n in (1, 2, 4, 8):
        per_spec_vals = [r.bon_success[n] for r in res.per_spec.values()]
        assert agg[n] == pytest.approx(np.mean(per_spec_vals))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def test_runner_smoke(tmp_path) -> None:
    """End-to-end: runner evaluates 2 checkpoints, writes outputs."""
    sims = {"s": _StubSim()}
    cfg = RunnerConfig(
        n_samples_per_spec=8,
        budgets=(1, 2, 4, 8),
        output_dir=tmp_path,
        seed_base=100,
    )
    runner = EvalRunner(
        simulator_registry=sims,
        spec_registry={"s": object()},
        stl_evaluator=_stub_evaluator,
        initial_state_fn=_const_x0,
        config=cfg,
    )
    records = runner.run(
        checkpoints=[_GoodCheckpoint(), _BadCheckpoint()],
        held_out_specs=["s"],
    )
    assert len(records) == 2
    assert {r.checkpoint_name for r in records} == {"good", "bad"}
    for r in records:
        assert r.success
        assert r.output_path is not None and r.output_path.exists()
    table = stringify_aggregate(records)
    assert "good" in table and "bad" in table


def test_runner_resumes_existing(tmp_path) -> None:
    """If output exists and overwrite=False, the runner does not re-run."""
    sims = {"s": _StubSim()}
    cfg = RunnerConfig(
        n_samples_per_spec=8,
        budgets=(1, 2, 4, 8),
        output_dir=tmp_path,
        seed_base=0,
    )
    runner = EvalRunner(
        simulator_registry=sims,
        spec_registry={"s": object()},
        stl_evaluator=_stub_evaluator,
        initial_state_fn=_const_x0,
        config=cfg,
    )
    # First run writes
    r1 = runner.run([_GoodCheckpoint()], ["s"])
    assert r1[0].success and not r1[0].extras.get("resumed", False)
    # Second run reads
    r2 = runner.run([_GoodCheckpoint()], ["s"])
    assert r2[0].success and r2[0].extras.get("resumed") is True


# ---------------------------------------------------------------------------
# action_diversity (Fix 3 — paper/REDACTED.md §"Issues encountered")
# ---------------------------------------------------------------------------


def test_action_diversity_all_distinct() -> None:
    """N distinct first actions → first_action_uniqueness == 1.0."""
    rng = np.random.default_rng(0)
    actions = rng.normal(size=(5, 4, 2))
    out = action_diversity(actions)
    assert out["first_action_uniqueness"] == pytest.approx(1.0)
    assert out["sequence_uniqueness"] == pytest.approx(1.0)
    assert out["first_action_pairwise_distance"] > 0.0


def test_action_diversity_all_identical_first_action() -> None:
    """Every prompt's first action is the same → first_action_uniqueness ==
    1/n_prompts (the A15 memorization signature)."""
    a = np.zeros((5, 4, 2), dtype=np.float64)
    # Different tails so sequence_uniqueness is 1.0 but first action is shared
    rng = np.random.default_rng(1)
    a[:, 1:, :] = rng.normal(size=(5, 3, 2))
    a[:, 0, :] = np.array([12.34, -1.0])  # identical first action
    out = action_diversity(a)
    assert out["first_action_uniqueness"] == pytest.approx(1.0 / 5.0)
    assert out["sequence_uniqueness"] == pytest.approx(1.0)
    # Only one distinct first action → mean pairwise dist = 0
    assert out["first_action_pairwise_distance"] == pytest.approx(0.0)


def test_action_diversity_quantization_collapses_near_duplicates() -> None:
    """Two near-identical first actions (diff < q) hash to the same key."""
    a = np.zeros((3, 2, 1), dtype=np.float64)
    a[0, 0, 0] = 1.0
    a[1, 0, 0] = 1.0 + 1e-9  # below default q=1e-6
    a[2, 0, 0] = 2.0
    out = action_diversity(a)
    # rows 0 and 1 collapse → 2 distinct first actions out of 3
    assert out["first_action_uniqueness"] == pytest.approx(2.0 / 3.0)


def test_action_diversity_handles_nan_rows() -> None:
    """NaN rows do not crash and do not contribute to pairwise distance."""
    a = np.zeros((4, 2, 1), dtype=np.float64)
    a[0] = np.nan
    a[1, 0, 0] = 1.0
    a[2, 0, 0] = 2.0
    a[3, 0, 0] = 3.0
    out = action_diversity(a)
    # First-action keys: ("nan",), (1,), (2,), (3,) → 4 distinct
    assert out["first_action_uniqueness"] == pytest.approx(1.0)
    # Pairwise distances over distinct *finite* first actions only:
    # |1-2|=1, |1-3|=2, |2-3|=1 → mean = 4/3
    assert out["first_action_pairwise_distance"] == pytest.approx(4.0 / 3.0)


def test_action_diversity_empty_returns_nan() -> None:
    out = action_diversity(np.zeros((0, 2, 1)))
    assert np.isnan(out["first_action_uniqueness"])
    assert np.isnan(out["sequence_uniqueness"])
    assert np.isnan(out["first_action_pairwise_distance"])


def test_action_diversity_wrong_shape_raises() -> None:
    with pytest.raises(ValueError, match="n_prompts, H, m"):
        action_diversity(np.zeros((4, 2)))


def test_action_diversity_invalid_quantization_raises() -> None:
    with pytest.raises(ValueError, match="quantization"):
        action_diversity(np.zeros((2, 2, 1)), quantization=0.0)


# ---------------------------------------------------------------------------
# Harness diversity wiring + runner [DIVERSITY WARNING] flag
# ---------------------------------------------------------------------------


class _MemorizingCheckpoint:
    """Always emits the same control sequence regardless of prompt — this is
    the A15 failure mode the diversity warning exists to catch."""

    name = "memorizing"

    def sample_controls(self, spec, initial_state, key):  # noqa: ARG002
        return jnp.ones((4, 1)) * 0.7


def test_harness_records_diversity_per_spec() -> None:
    h = _make_harness()
    res = h.evaluate_checkpoint(
        checkpoint=_RandomCheckpoint(),
        held_out_specs=["toggle.spec", "lv.spec"],
        n_samples_per_spec=8,
        key=42,
    )
    for spec_name, per_spec in res.per_spec.items():
        assert "first_action_uniqueness" in per_spec.diversity, spec_name
        assert "sequence_uniqueness" in per_spec.diversity, spec_name
        assert "first_action_pairwise_distance" in per_spec.diversity, spec_name


def test_harness_memorizing_checkpoint_low_diversity() -> None:
    """A constant-output policy must score first_action_uniqueness = 1/N."""
    h = _make_harness()
    res = h.evaluate_checkpoint(
        checkpoint=_MemorizingCheckpoint(),
        held_out_specs=["toggle.spec"],
        n_samples_per_spec=8,
        key=0,
    )
    per_spec = res.per_spec["toggle.spec"]
    assert per_spec.diversity["first_action_uniqueness"] == pytest.approx(1.0 / 8.0)
    assert per_spec.diversity["first_action_uniqueness"] < DIVERSITY_WARNING_THRESHOLD


def test_runner_flags_diversity_warning_in_stringified_output(tmp_path) -> None:
    """A memorizing checkpoint should trigger a [DIVERSITY WARNING] tag."""
    sims = {"s": _StubSim()}
    cfg = RunnerConfig(
        n_samples_per_spec=8,
        budgets=(1, 2, 4, 8),
        output_dir=tmp_path,
        seed_base=0,
    )
    runner = EvalRunner(
        simulator_registry=sims,
        spec_registry={"s": object()},
        stl_evaluator=_stub_evaluator,
        initial_state_fn=_const_x0,
        config=cfg,
    )
    records = runner.run([_MemorizingCheckpoint(), _RandomCheckpoint()], ["s"])
    by_name = {r.checkpoint_name: r for r in records}
    assert "s" in by_name["memorizing"].diversity_warnings
    assert "s" not in by_name["random"].diversity_warnings
    table = stringify_aggregate(records)
    # The memorizing line carries the warning tag; the random line does not.
    mem_line = next(line for line in table.splitlines() if "memorizing" in line)
    rand_line = next(line for line in table.splitlines() if "random" in line)
    assert "DIVERSITY WARNING" in mem_line
    assert "DIVERSITY WARNING" not in rand_line


def test_per_spec_result_as_dict_includes_diversity() -> None:
    """JSON round-trip must carry the diversity dict (so paper figures
    can read it directly off the eval artifact)."""
    h = _make_harness()
    res = h.evaluate_checkpoint(
        checkpoint=_MemorizingCheckpoint(),
        held_out_specs=["toggle.spec"],
        n_samples_per_spec=8,
        key=1,
    )
    payload = res.per_spec["toggle.spec"].as_dict()
    assert "diversity" in payload
    assert "first_action_uniqueness" in payload["diversity"]

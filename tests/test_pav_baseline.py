"""Tests for the PAV process-reward-model baseline.

The PAV implementation lives in :mod:`stl_seed.baselines.pav` and is the
learned counterpart to the formal STL-rho verifier in
:mod:`stl_seed.stl.evaluator`.

What we cover
-------------
* Initialization: the MLP wires state_dim and hidden_dim correctly, and
  rejects degenerate constructor arguments.
* Training: synthetic (state, advantage) pairs cause the training loss to
  decrease over epochs.
* Scoring: ``score(trajectory)`` returns a finite scalar for any valid
  trajectory and zero for an unfit model.
* Sample-efficiency monotonicity: AUC of PAV (trained on a separable
  synthetic problem) improves --- on average --- with more training
  trajectories.
* Asymmetry: STL-rho scores arbitrary trajectories with no training, by
  design; this is the structural advantage we are claiming in the paper.
* End-to-end: ``compare_pav_vs_stl`` returns a well-formed
  :class:`ComparisonResult`.

the in-repo STL evaluator, the in-repo Trajectory type, and the in-repo
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stl_seed.baselines.comparison import ComparisonResult, compare_pav_vs_stl
from stl_seed.baselines.pav import (
    _MLP,
    PAVProcessRewardModel,
    compute_per_step_mc_labels,
)
from stl_seed.specs import (
    Always,
    Interval,
    STLSpec,
)
from stl_seed.specs.bio_ode_specs import _gt
from stl_seed.tasks._trajectory import Trajectory, TrajectoryMeta

# ---------------------------------------------------------------------------
# Synthetic-data helpers (private to this test module).
# ---------------------------------------------------------------------------


def _make_traj(
    states: np.ndarray,
    actions: np.ndarray,
    times: np.ndarray,
) -> Trajectory:
    meta = TrajectoryMeta(
        n_nan_replacements=jnp.asarray(0.0),
        final_solver_result=jnp.asarray(0.0),
        used_stiff_fallback=jnp.asarray(0.0),
    )
    return Trajectory(
        states=jnp.asarray(states),
        actions=jnp.asarray(actions),
        times=jnp.asarray(times),
        meta=meta,
    )


def _separable_trajectories(
    n: int,
    state_dim: int,
    horizon: int,
    T: int,
    success_offset: float,
    seed: int,
) -> tuple[list[Trajectory], np.ndarray]:
    """Build a corpus where success-class trajectories live in a different
    region of state space (offset by ``success_offset`` on every channel).

    Returns ``(trajectories, terminal_success)``.
    """
    rng = np.random.default_rng(seed)
    trajs: list[Trajectory] = []
    succ_list: list[int] = []
    for i in range(n):
        is_succ = int(i % 2)
        offset = success_offset if is_succ else 0.0
        states = rng.normal(loc=offset, scale=0.5, size=(T, state_dim))
        actions = rng.normal(size=(horizon, 1))
        times = np.linspace(0.0, 1.0, T)
        trajs.append(_make_traj(states, actions, times))
        succ_list.append(is_succ)
    return trajs, np.asarray(succ_list, dtype=np.float64)


def _bio_like_spec() -> STLSpec:
    """A simple Always[0,1] (x[0] >= 0.5) spec on a 2-channel signal."""
    pred = _gt("x", 0, 0.5)  # Returns a Predicate already.
    formula = Always(inner=pred, interval=Interval(0.0, 1.0))
    return STLSpec(
        name="test.synthetic.always_x0_gt_half",
        formula=formula,
        signal_dim=2,
        horizon_minutes=1.0,
        description="synthetic test spec",
        citations=("test_pav_baseline.py",),
        formula_text="G_[0,1] (x_0 >= 0.5)",
    )


# ---------------------------------------------------------------------------
# Initialization tests.
# ---------------------------------------------------------------------------


def test_pav_initializes_with_correct_dims() -> None:
    """state_dim, hidden_dim wire correctly through the MLP."""
    pav = PAVProcessRewardModel(state_dim=4, hidden=64, dropout=0.2)
    assert pav.state_dim == 4
    assert pav.hidden == 64
    assert pav.dropout == 0.2
    assert pav.is_fit is False
    # Score before fit should return 0.0 (well-formed default).
    states = jnp.zeros((10, 4))
    actions = jnp.zeros((3, 1))
    times = jnp.linspace(0.0, 1.0, 10)
    traj = _make_traj(np.asarray(states), np.asarray(actions), np.asarray(times))
    assert pav.score(traj) == 0.0


def test_pav_mlp_layer_shapes() -> None:
    """The internal _MLP has the expected three Linear layers."""
    mlp = _MLP(state_dim=5, hidden=16, dropout=0.0, key=jax.random.PRNGKey(0))
    assert len(mlp.layers) == 3
    # First Linear: 5 -> 16
    assert mlp.layers[0].in_features == 5
    assert mlp.layers[0].out_features == 16
    # Hidden Linear: 16 -> 16
    assert mlp.layers[1].in_features == 16
    assert mlp.layers[1].out_features == 16
    # Output Linear: 16 -> 1
    assert mlp.layers[2].in_features == 16
    assert mlp.layers[2].out_features == 1
    # Forward pass returns scalar.
    x = jnp.zeros((5,))
    y = mlp(x, training=False)
    assert y.shape == ()


def test_pav_mlp_rejects_invalid_args() -> None:
    """The _MLP constructor catches degenerate sizes and bad dropout."""
    key = jax.random.PRNGKey(0)
    with pytest.raises(ValueError):
        _MLP(state_dim=0, hidden=4, dropout=0.0, key=key)
    with pytest.raises(ValueError):
        _MLP(state_dim=4, hidden=0, dropout=0.0, key=key)
    with pytest.raises(ValueError):
        _MLP(state_dim=4, hidden=4, dropout=1.0, key=key)
    with pytest.raises(ValueError):
        _MLP(state_dim=4, hidden=4, dropout=-0.1, key=key)


# ---------------------------------------------------------------------------
# Training tests.
# ---------------------------------------------------------------------------


def test_pav_fits_synthetic_data() -> None:
    """Train on synthetic separable data; verify training loss decreases."""
    trajs, ts = _separable_trajectories(
        n=40, state_dim=2, horizon=4, T=11, success_offset=2.0, seed=0
    )
    pav = PAVProcessRewardModel(state_dim=2, hidden=32, dropout=0.0)
    history = pav.fit(
        trajs,
        ts,
        n_epochs=30,
        lr=1e-2,
        key=jax.random.PRNGKey(0),
        k_neighbors=5,
    )
    assert pav.is_fit is True
    assert "train_loss" in history
    assert len(history["train_loss"]) == 30
    # Loss should drop materially over 30 epochs on separable data.
    loss_start = float(history["train_loss"][0])
    loss_end = float(history["train_loss"][-1])
    assert loss_end < loss_start, f"Train loss did not decrease: start={loss_start} end={loss_end}"
    # Wall-time recorded.
    assert history["wall_time_s"] > 0.0
    # Pair-count recorded.
    assert history["n_train_pairs"] > 0


def test_pav_score_well_formed() -> None:
    """Score returns a finite scalar for any trajectory after a fit."""
    trajs, ts = _separable_trajectories(
        n=20, state_dim=3, horizon=4, T=11, success_offset=1.5, seed=1
    )
    pav = PAVProcessRewardModel(state_dim=3, hidden=16, dropout=0.0)
    pav.fit(trajs, ts, n_epochs=10, lr=1e-2, key=jax.random.PRNGKey(0), k_neighbors=4)
    for traj in trajs:
        s = pav.score(traj)
        assert isinstance(s, float)
        assert np.isfinite(s), f"PAV score not finite: {s}"


def test_pav_fit_requires_min_two_trajectories() -> None:
    """fit() raises if the corpus has < 2 trajectories (kNN needs a pool)."""
    trajs, ts = _separable_trajectories(
        n=1, state_dim=2, horizon=2, T=5, success_offset=1.0, seed=2
    )
    pav = PAVProcessRewardModel(state_dim=2, hidden=8, dropout=0.0)
    with pytest.raises(ValueError):
        pav.fit(trajs, ts, n_epochs=2, lr=1e-2, key=jax.random.PRNGKey(0))


def test_compute_per_step_mc_labels_shapes_and_terminal_anchor() -> None:
    """MC labels have the right shapes and the terminal-step MC equals the success label."""
    trajs, ts = _separable_trajectories(
        n=10, state_dim=2, horizon=3, T=11, success_offset=1.0, seed=3
    )
    ds = compute_per_step_mc_labels(trajs, ts, k_neighbors=3)
    # H steps per trajectory => N * H rows.
    assert ds.states.shape == (10 * 3, 2)
    assert ds.advantages.shape == (10 * 3,)
    assert ds.traj_ids.shape == (10 * 3,)
    assert ds.step_ids.shape == (10 * 3,)
    # All advantages finite.
    assert np.all(np.isfinite(ds.advantages))
    # traj_ids range over 0..N-1, step_ids over 0..H-1.
    assert set(ds.traj_ids.tolist()) == set(range(10))
    assert set(ds.step_ids.tolist()) == set(range(3))


# ---------------------------------------------------------------------------
# Sample-efficiency / AUC tests.
# ---------------------------------------------------------------------------


def test_pav_predicts_terminal_better_with_more_data() -> None:
    """On a well-separated synthetic problem, PAV's held-out AUC improves
    --- on average --- as the train set grows from 30 to 200 trajectories.

    This is a noisy stochastic test, so we average over a couple of seeds and
    require the larger-train-set average to exceed the smaller one rather
    than monotonic at every single seed.
    """
    test_trajs, test_ts = _separable_trajectories(
        n=80, state_dim=2, horizon=4, T=11, success_offset=2.5, seed=999
    )

    aucs_small: list[float] = []
    aucs_large: list[float] = []
    for seed in (0, 1, 2):
        small_trajs, small_ts = _separable_trajectories(
            n=30, state_dim=2, horizon=4, T=11, success_offset=2.5, seed=10 + seed
        )
        large_trajs, large_ts = _separable_trajectories(
            n=200, state_dim=2, horizon=4, T=11, success_offset=2.5, seed=100 + seed
        )

        pav_small = PAVProcessRewardModel(state_dim=2, hidden=32, dropout=0.0)
        pav_small.fit(
            small_trajs,
            small_ts,
            n_epochs=20,
            lr=1e-2,
            key=jax.random.PRNGKey(seed),
            k_neighbors=5,
        )
        pav_large = PAVProcessRewardModel(state_dim=2, hidden=32, dropout=0.0)
        pav_large.fit(
            large_trajs,
            large_ts,
            n_epochs=20,
            lr=1e-2,
            key=jax.random.PRNGKey(seed),
            k_neighbors=10,
        )

        from stl_seed.baselines.comparison import _roc_auc

        scores_small = pav_small.score_batch(test_trajs)
        scores_large = pav_large.score_batch(test_trajs)
        aucs_small.append(_roc_auc(test_ts, scores_small))
        aucs_large.append(_roc_auc(test_ts, scores_large))

    avg_small = float(np.nanmean(aucs_small))
    avg_large = float(np.nanmean(aucs_large))
    assert avg_large > avg_small, (
        f"PAV AUC did not improve with more data: small={avg_small:.3f} "
        f"large={avg_large:.3f} (per-seed small={aucs_small} large={aucs_large})"
    )


# ---------------------------------------------------------------------------
# STL baseline asymmetry test.
# ---------------------------------------------------------------------------


def test_stl_baseline_doesnt_need_training() -> None:
    """STL-rho scoring works on a single trajectory with zero "training".

    This is the structural asymmetry the paper claims: STL-rho is a
    closed-form verifier with no learnable parameters; PAV requires a
    corpus of labeled trajectories and an MLP fit. The test confirms the
    asymmetry concretely: invoking ``evaluate_robustness`` on a fresh
    single trajectory returns a finite signed margin without any prior
    state.
    """
    from stl_seed.stl.evaluator import evaluate_robustness

    spec = _bio_like_spec()
    # One trajectory; scores well above the 0.5 threshold throughout.
    states = np.full((11, 2), 1.0)
    actions = np.zeros((4, 1))
    times = np.linspace(0.0, 1.0, 11)
    traj = _make_traj(states, actions, times)
    rho = float(evaluate_robustness(spec, traj))
    assert np.isfinite(rho)
    # x = 1.0 throughout; threshold 0.5 => margin 0.5 (within float epsilon).
    assert rho == pytest.approx(0.5, abs=1e-6)

    # And one that violates: x = 0.0 throughout => margin -0.5.
    states = np.full((11, 2), 0.0)
    traj_neg = _make_traj(states, actions, times)
    rho_neg = float(evaluate_robustness(spec, traj_neg))
    assert rho_neg == pytest.approx(-0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Comparison-result shape test.
# ---------------------------------------------------------------------------


def test_compare_returns_well_formed() -> None:
    """``compare_pav_vs_stl`` returns the expected dataclass shape."""
    spec = _bio_like_spec()
    # Build a corpus where successes have x[0] > 0.5 at every save-time
    # (so STL ρ is positive for the success class) and failures have x[0] < 0.5.
    rng = np.random.default_rng(7)
    n = 40
    H = 3
    T = 11
    trajs = []
    succ = []
    for i in range(n):
        is_succ = int(i % 2)
        x0 = rng.uniform(0.7, 1.0, size=(T,)) if is_succ else rng.uniform(0.0, 0.3, size=(T,))
        x1 = rng.normal(size=(T,))
        states = np.stack([x0, x1], axis=1)
        actions = rng.normal(size=(H, 1))
        times = np.linspace(0.0, 1.0, T)
        trajs.append(_make_traj(states, actions, times))
        succ.append(is_succ)
    succ = np.asarray(succ, dtype=np.float64)

    result = compare_pav_vs_stl(
        trajectories=trajs,
        terminal_success=succ,
        spec=spec,
        n_train=20,
        n_test=15,
        seed=0,
        sample_efficiency_grid=[10, 20],
        pav_n_epochs=10,
        pav_lr=1e-2,
        pav_hidden=16,
        pav_dropout=0.0,
        k_neighbors=4,
        task_name="test.synthetic",
    )
    assert isinstance(result, ComparisonResult)
    assert result.task == "test.synthetic"
    assert result.spec_key == spec.name
    assert result.n_train == 20
    assert result.n_test == 15
    assert result.pav_scores.shape == (15,)
    assert result.stl_scores.shape == (15,)
    assert result.test_terminal_success.shape == (15,)
    assert np.isfinite(result.pav_auc)
    assert np.isfinite(result.stl_auc)
    # On this perfectly separated synthetic dataset, STL ρ MUST achieve
    # AUC close to 1.0 (it is the optimal threshold for this spec).
    assert result.stl_auc >= 0.95, f"STL AUC unexpectedly low: {result.stl_auc}"
    # Sample efficiency points are recorded.
    assert len(result.sample_efficiency) == 2
    for pt in result.sample_efficiency:
        assert pt.n_train > 0
        assert pt.n_test == 15
    # Either crossover_n_train is None or it's one of our grid points.
    if result.crossover_n_train is not None:
        assert result.crossover_n_train in {pt.n_train for pt in result.sample_efficiency}


def test_compare_rejects_oversized_split() -> None:
    """``compare_pav_vs_stl`` raises if n_train + n_test exceeds corpus size."""
    spec = _bio_like_spec()
    trajs, succ = _separable_trajectories(
        n=10, state_dim=2, horizon=3, T=11, success_offset=2.0, seed=0
    )
    with pytest.raises(ValueError):
        compare_pav_vs_stl(
            trajectories=trajs,
            terminal_success=succ,
            spec=spec,
            n_train=20,
            n_test=20,
            seed=0,
            pav_n_epochs=2,
        )


def test_compare_with_spec_key_string() -> None:
    """``compare_pav_vs_stl`` accepts a registry key string for ``spec``."""
    from stl_seed.specs import REGISTRY

    # Pick any registered spec; we just need it not to crash. We use a
    # synthetic corpus with the right state_dim for the chosen spec.
    spec_key = "bio_ode.repressilator.easy"
    spec_obj = REGISTRY[spec_key]
    rng = np.random.default_rng(0)
    n = 30
    state_dim = spec_obj.signal_dim
    H = 4
    T = 21
    trajs = []
    succ = []
    for i in range(n):
        states = rng.normal(size=(T, state_dim))
        actions = rng.normal(size=(H, 3))
        times = np.linspace(0.0, 200.0, T)
        trajs.append(_make_traj(states, actions, times))
        succ.append(int(i % 2))
    succ = np.asarray(succ, dtype=np.float64)
    result = compare_pav_vs_stl(
        trajectories=trajs,
        terminal_success=succ,
        spec=spec_key,
        n_train=15,
        n_test=10,
        seed=0,
        sample_efficiency_grid=[15],
        pav_n_epochs=5,
        pav_lr=1e-2,
        pav_hidden=8,
        pav_dropout=0.0,
        k_neighbors=3,
        task_name="bio_ode.repressilator",
    )
    assert isinstance(result, ComparisonResult)
    assert result.spec_key == spec_key
    assert result.n_test == 10


# ---------------------------------------------------------------------------
# V2 surface tests: model selection + AdamW + early stopping.
# ---------------------------------------------------------------------------


def test_pav_fit_supports_weight_decay() -> None:
    """fit(...) accepts weight_decay and trains without crashing."""
    trajs, ts = _separable_trajectories(
        n=30, state_dim=2, horizon=4, T=11, success_offset=2.0, seed=0
    )
    pav = PAVProcessRewardModel(state_dim=2, hidden=16, dropout=0.0)
    history = pav.fit(
        trajs,
        ts,
        n_epochs=10,
        lr=1e-2,
        key=jax.random.PRNGKey(0),
        k_neighbors=4,
        weight_decay=1e-3,
    )
    assert pav.is_fit is True
    # AdamW should still drive the training loss down on separable data.
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert "best_epoch" in history
    assert history["best_epoch"] >= 1


def test_pav_fit_early_stopping_returns_best() -> None:
    """early_stopping_patience triggers stop and reports best_epoch."""
    trajs, ts = _separable_trajectories(
        n=20, state_dim=2, horizon=4, T=11, success_offset=2.0, seed=1
    )
    pav = PAVProcessRewardModel(state_dim=2, hidden=8, dropout=0.0)
    # Use a *very* short patience so the test exits early on plateau.
    history = pav.fit(
        trajs,
        ts,
        n_epochs=200,
        lr=1e-2,
        key=jax.random.PRNGKey(2),
        k_neighbors=4,
        early_stopping_patience=2,
    )
    # Either we ran <200 epochs (early stopped) or we ran 200 with no improvement.
    assert len(history["train_loss"]) <= 200
    assert isinstance(history["stopped_early"], bool)
    if history["stopped_early"]:
        assert len(history["train_loss"]) < 200
        # best_epoch must be <= the last epoch run.
        assert history["best_epoch"] <= len(history["train_loss"])


def test_pav_fit_with_selection_picks_lowest_val_mse() -> None:
    """fit_with_selection sweeps the grid and returns the best (h, wd)."""
    trajs, ts = _separable_trajectories(
        n=60, state_dim=2, horizon=4, T=11, success_offset=2.0, seed=3
    )
    pav, report = PAVProcessRewardModel.fit_with_selection(
        trajectories=trajs,
        terminal_success=ts,
        state_dim=2,
        hidden_grid=(8, 16),
        weight_decay_grid=(0.0, 1e-3),
        dropout=0.0,
        n_epochs=20,
        lr=1e-2,
        key=jax.random.PRNGKey(4),
        val_frac=0.3,
        k_neighbors=5,
        early_stopping_patience=3,
    )
    assert pav.is_fit is True
    assert report["best_hidden"] in {8, 16}
    assert report["best_weight_decay"] in {0.0, 1e-3}
    # Grid has 2x2 = 4 cells; all should have a finite val MSE.
    assert len(report["grid"]) == 4
    val_mses = [c["best_val_mse"] for c in report["grid"]]
    assert all(np.isfinite(m) for m in val_mses)
    # The reported "best" must equal the min over the grid.
    assert report["best_val_mse"] == min(val_mses)


def test_compare_v2_with_knn_label_source() -> None:
    """compare_pav_v2_vs_stl runs end-to-end with label_source='knn'."""
    from stl_seed.baselines.comparison import (
        ComparisonResultV2,
        compare_pav_v2_vs_stl,
    )

    spec = _bio_like_spec()
    rng = np.random.default_rng(11)
    n = 50
    H = 3
    T = 11
    trajs = []
    succ = []
    for i in range(n):
        is_succ = int(i % 2)
        x0 = rng.uniform(0.7, 1.0, size=(T,)) if is_succ else rng.uniform(0.0, 0.3, size=(T,))
        x1 = rng.normal(size=(T,))
        states = np.stack([x0, x1], axis=1)
        actions = rng.normal(size=(H, 1))
        times = np.linspace(0.0, 1.0, T)
        trajs.append(_make_traj(states, actions, times))
        succ.append(is_succ)
    succ = np.asarray(succ, dtype=np.float64)

    result = compare_pav_v2_vs_stl(
        trajectories=trajs,
        terminal_success=succ,
        spec=spec,
        task="synthetic_test",  # arbitrary; rollout path not exercised
        n_train=30,
        n_test=15,
        seed=0,
        label_source="knn",
        hidden_grid=(8, 16),
        weight_decay_grid=(0.0, 1e-3),
        pav_n_epochs=10,
        pav_lr=1e-2,
        pav_dropout=0.0,
        early_stopping_patience=3,
        val_frac=0.3,
        k_neighbors=4,
        task_name="synthetic_test",
    )
    assert isinstance(result, ComparisonResultV2)
    assert result.label_source == "knn"
    assert result.n_train == 30
    assert result.n_test == 15
    assert np.isfinite(result.stl_auc)
    assert result.stl_auc >= 0.95
    # The best cell must be one of the four grid corners.
    assert result.pav_best_hidden in {8, 16}
    assert result.pav_best_weight_decay in {0.0, 1e-3}
    assert len(result.selection_grid) == 4
    # No on-policy simulations were charged for the kNN path.
    assert result.n_onpolicy_simulations == 0

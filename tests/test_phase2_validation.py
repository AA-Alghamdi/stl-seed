"""Tests for the Phase-2 dry-run validation pipeline.

Coverage:

* :class:`MockBNBBackend` instantiates cleanly on any platform (no GPU).
* :meth:`MockBNBBackend.train` produces a monotonically decreasing
  synthetic loss curve and writes a BNB-schema-compatible provenance.json
  + adapter directory + MOCK.txt sentinel.
* :meth:`MockBNBBackend.load` returns a callable that emits parseable
  ``<state>...</state><action>...</action>`` blocks, round-tripping
  through :func:`stl_seed.training.tokenize.parse_action_sequence`.
* The mock refuses to run when ``STL_SEED_REAL_TRAINING=1`` is set.
* The full validation pipeline (sweep + eval + analysis + firewall)
  passes end-to-end under the mock backend.

REDACTED firewall: no REDACTED / REDACTED / REDACTED imports.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from stl_seed.training.backends.base import TrainedCheckpoint, TrainingConfig
from stl_seed.training.backends.mock import (
    REAL_TRAINING_ENV,
    USE_MOCK_ENV,
    MockBNBBackend,
    is_mock_enabled,
)

# Helpers ----------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _import_script(name: str):
    """Import a top-level script module by file path (scripts/ is not a package)."""
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Mock backend basics
# ---------------------------------------------------------------------------


class TestMockBackendBasics:
    def test_mock_backend_initializes(self):
        """MockBNBBackend instantiates on any platform without a GPU."""
        backend = MockBNBBackend()
        assert backend.name == "bnb"  # impersonates bnb at the dispatch layer
        assert backend.is_mock is True

    def test_mock_backend_get_backend_dispatch(self):
        """get_backend('mock_bnb') returns a MockBNBBackend instance."""
        from stl_seed.training.loop import get_backend

        backend = get_backend("mock_bnb")
        assert backend.name == "bnb"
        assert isinstance(backend, MockBNBBackend)

    def test_is_mock_enabled_default_false(self, monkeypatch):
        monkeypatch.delenv(USE_MOCK_ENV, raising=False)
        assert is_mock_enabled() is False

    def test_is_mock_enabled_true_for_truthy_values(self, monkeypatch):
        for val in ["1", "true", "True", "TRUE", "yes"]:
            monkeypatch.setenv(USE_MOCK_ENV, val)
            assert is_mock_enabled() is True, val


# ---------------------------------------------------------------------------
# Train: synthetic loss curve and artifact layout
# ---------------------------------------------------------------------------


class TestMockBackendTrain:
    def test_train_returns_decreasing_loss_curve(self, tmp_path):
        backend = MockBNBBackend()
        cfg = TrainingConfig(output_dir=tmp_path)
        ckpt = backend.train(
            base_model="Qwen/Qwen3-0.6B-Instruct",
            dataset=[{"messages": []}] * 32,
            config=cfg,
            output_dir=tmp_path,
        )
        loss = ckpt.training_loss_history
        assert isinstance(loss, list)
        assert len(loss) >= 4, f"loss history too short: {len(loss)}"
        # Monotonically non-increasing in the expected sense (we enforce it
        # in _synthetic_loss_curve).
        for prev, cur in zip(loss[:-1], loss[1:], strict=False):
            assert cur <= prev, f"loss not monotonic: {prev} -> {cur}"
        # Final loss must be above the asymptote (~0.55) and below the init
        # (~4.5 for 0.6b). This pins the curve shape so a regression in
        # _synthetic_loss_curve fires loudly.
        assert 0.4 < loss[-1] < 5.5, f"final loss out of band: {loss[-1]}"

    def test_train_writes_valid_checkpoint(self, tmp_path):
        backend = MockBNBBackend()
        cfg = TrainingConfig(output_dir=tmp_path)
        ckpt = backend.train(
            base_model="Qwen/Qwen3-0.6B-Instruct",
            dataset=[{"messages": []}] * 16,
            config=cfg,
            output_dir=tmp_path,
        )
        # Adapter directory + MOCK.txt sentinel.
        adapter_dir = Path(ckpt.model_path)
        assert adapter_dir.exists() and adapter_dir.is_dir()
        assert (adapter_dir / "adapter_config.json").exists()
        assert (adapter_dir / "MOCK.txt").exists()
        adapter_cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
        assert adapter_cfg.get("mock") is True
        # Provenance.json with the BNB-schema fields the runner consumes.
        prov_path = tmp_path / "provenance.json"
        assert prov_path.exists()
        prov = json.loads(prov_path.read_text())
        for key in [
            "backend",
            "base_model",
            "n_examples",
            "config",
            "wall_clock_seconds",
            "n_loss_points",
            "mock",
            "mock_backend",
        ]:
            assert key in prov, f"provenance missing {key!r}: {sorted(prov.keys())}"
        assert prov["mock"] is True
        assert prov["backend"] == "bnb"
        assert prov["n_examples"] == 16

    def test_train_loss_curve_size_scales_with_model(self, tmp_path):
        """4B should converge slower (larger tau → loss higher at the same step)."""
        backend = MockBNBBackend()
        cfg = TrainingConfig(output_dir=tmp_path)
        # Identical dataset; only base_model differs.
        ck_06 = backend.train(
            base_model="Qwen/Qwen3-0.6B-Instruct",
            dataset=[{"messages": []}] * 80,
            config=cfg,
            output_dir=tmp_path / "0.6b",
        )
        ck_4b = backend.train(
            base_model="Qwen/Qwen3-4B-Instruct",
            dataset=[{"messages": []}] * 80,
            config=cfg,
            output_dir=tmp_path / "4b",
        )
        # At the *same* step index, the 4B's loss should be >= the 0.6B's.
        # Compare in the middle of the curve (early steps are dominated by
        # init noise; late steps may both have hit asymptote).
        n = min(len(ck_06.training_loss_history), len(ck_4b.training_loss_history))
        if n >= 4:
            mid = n // 2
            assert ck_4b.training_loss_history[mid] >= ck_06.training_loss_history[mid] - 0.5, (
                f"4B loss ({ck_4b.training_loss_history[mid]:.3f}) should be >= "
                f"0.6B ({ck_06.training_loss_history[mid]:.3f}) at step {mid}; "
                "decay-rate-by-model is a load-bearing property of the mock."
            )


# ---------------------------------------------------------------------------
# Load: callable returns parseable text
# ---------------------------------------------------------------------------


class TestMockBackendLoad:
    def test_load_returns_callable(self, tmp_path):
        backend = MockBNBBackend()
        ckpt = backend.train(
            base_model="Qwen/Qwen3-0.6B-Instruct",
            dataset=[{"messages": []}] * 8,
            config=TrainingConfig(output_dir=tmp_path),
            output_dir=tmp_path,
        )
        gen = backend.load(ckpt)
        assert callable(gen)

    def test_load_callable_emits_parseable_output(self, tmp_path):
        from stl_seed.training.tokenize import parse_action_sequence

        backend = MockBNBBackend()
        ckpt = backend.train(
            base_model="Qwen/Qwen3-0.6B-Instruct",
            dataset=[{"messages": []}] * 8,
            config=TrainingConfig(output_dir=tmp_path),
            output_dir=tmp_path,
        )
        gen = backend.load(ckpt)
        prompt = (
            "system: ...\n\n"
            "Initial state: <state>1.0,2.0,3.0</state>\n"
            "Specification: G_[0,10] (x_1 > 0.5)\n"
            "Emit exactly 10 (state, action) blocks."
        )
        out = gen(prompt)
        assert isinstance(out, str)
        actions = parse_action_sequence(out)
        assert actions.shape == (10, 3)
        # Same prompt + same seed -> deterministic.
        assert out == gen(prompt)


# ---------------------------------------------------------------------------
# Real-training guard
# ---------------------------------------------------------------------------


class TestRealTrainingGuard:
    def test_construction_refused_when_real_training_env_set(self, monkeypatch):
        """MockBNBBackend() raises if STL_SEED_REAL_TRAINING=1."""
        monkeypatch.setenv(REAL_TRAINING_ENV, "1")
        with pytest.raises(RuntimeError, match=REAL_TRAINING_ENV):
            MockBNBBackend()

    def test_train_refused_when_real_training_env_set(self, monkeypatch, tmp_path):
        """If the env var is flipped on after construction, train() also refuses."""
        backend = MockBNBBackend()
        monkeypatch.setenv(REAL_TRAINING_ENV, "1")
        with pytest.raises(RuntimeError, match=REAL_TRAINING_ENV):
            backend.train(
                base_model="Qwen/Qwen3-0.6B-Instruct",
                dataset=[],
                config=TrainingConfig(output_dir=tmp_path),
                output_dir=tmp_path,
            )

    def test_load_refused_when_real_training_env_set(self, monkeypatch, tmp_path):
        backend = MockBNBBackend()
        ckpt = backend.train(
            base_model="Qwen/Qwen3-0.6B-Instruct",
            dataset=[{"messages": []}] * 4,
            config=TrainingConfig(output_dir=tmp_path),
            output_dir=tmp_path,
        )
        monkeypatch.setenv(REAL_TRAINING_ENV, "1")
        with pytest.raises(RuntimeError, match=REAL_TRAINING_ENV):
            backend.load(ckpt)


# ---------------------------------------------------------------------------
# tokenize helpers used by the eval pipeline
# ---------------------------------------------------------------------------


class TestTokenizeEvalHelpers:
    def test_format_prompt_for_eval_basic(self):
        from stl_seed.specs import REGISTRY
        from stl_seed.training.tokenize import format_prompt_for_eval

        spec = REGISTRY["bio_ode.repressilator.easy"]
        out = format_prompt_for_eval(
            spec=spec,
            initial_state=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            task="bio_ode.repressilator",
            horizon=10,
        )
        assert isinstance(out, str)
        assert "<state>" in out
        assert "Emit exactly 10 (state, action) blocks." in out

    def test_format_prompt_for_eval_accepts_underscored_task(self):
        from stl_seed.specs import REGISTRY
        from stl_seed.training.tokenize import format_prompt_for_eval

        spec = REGISTRY["bio_ode.repressilator.easy"]
        out_dotted = format_prompt_for_eval(
            spec=spec,
            initial_state=np.zeros(6),
            task="bio_ode.repressilator",
            horizon=5,
        )
        out_underscore = format_prompt_for_eval(
            spec=spec,
            initial_state=np.zeros(6),
            task="bio_ode_repressilator",
            horizon=5,
        )
        assert out_dotted == out_underscore

    def test_parse_action_sequence_round_trip(self):
        from stl_seed.training.tokenize import (
            parse_action_sequence,
            serialize_assistant_turn,
        )

        states = np.zeros((4, 2))
        actions = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        serialized = serialize_assistant_turn(states, actions)
        parsed = parse_action_sequence(serialized)
        np.testing.assert_allclose(parsed, actions, rtol=1e-3)

    def test_parse_action_sequence_empty_raises(self):
        from stl_seed.training.tokenize import parse_action_sequence

        with pytest.raises(ValueError, match="No <action>"):
            parse_action_sequence("not a model output")

    def test_parse_action_sequence_inconsistent_widths_raises(self):
        from stl_seed.training.tokenize import parse_action_sequence

        bad = "<state>0</state><action>1.0,2.0</action>\n<state>0</state><action>3.0</action>"
        with pytest.raises(ValueError, match="Inconsistent"):
            parse_action_sequence(bad)


# ---------------------------------------------------------------------------
# End-to-end pipeline validation
# ---------------------------------------------------------------------------


class TestValidationPipeline:
    @pytest.mark.slow
    def test_validate_phase2_pipeline_3_cells(self, tmp_path, monkeypatch):
        """Full sweep + eval + analysis pipeline must pass under the mock backend.

        Marked slow: ~3-5s on M5 Pro (mock train is instant; the bottleneck
        is the bio-ode simulator JIT-compile + the synthetic-loss curve
        write loop). Still easily under the 5-minute CI budget.
        """
        # Set env vars for the validator before importing it. The validator
        # also sets these in main(), but tests can't rely on subprocess
        # state — set explicitly.
        monkeypatch.setenv(USE_MOCK_ENV, "1")
        monkeypatch.delenv(REAL_TRAINING_ENV, raising=False)

        # Use a fixed runs_dir under tmp_path so the validator does not
        # delete it on success (we want to inspect outputs).
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        validator = _import_script("validate_phase2_pipeline")
        rc = validator.main(
            [
                "--runs-dir",
                str(runs_dir),
                # Use only 2 cells (one per task) so the test stays fast.
                "--cells",
                "qwen3_0.6b__hard__bio_ode_repressilator,qwen3_0.6b__quantile__glucose_insulin",
            ]
        )
        assert rc == 0, "validator reported failure; see captured stdout"

        # Verify the artifacts the validator promised.
        sweep_log = runs_dir / "sweep_log.csv"
        assert sweep_log.exists()
        eval_parquet = runs_dir / "eval_results.parquet"
        assert eval_parquet.exists()
        results_md = runs_dir / "results.md"
        assert results_md.exists()
        bon_fig = runs_dir / "analysis" / "figures" / "bon_curves.png"
        assert bon_fig.exists()

    def test_validate_pipeline_help_works(self):
        """The validator's --help must succeed without imports failing."""
        validator = _import_script("validate_phase2_pipeline")
        with pytest.raises(SystemExit) as excinfo:
            validator.parse_args(["--help"])
        # argparse exits with code 0 for --help.
        assert excinfo.value.code == 0


# ---------------------------------------------------------------------------
# Sweep + eval runner integration with the mock backend
# ---------------------------------------------------------------------------


class TestRunnerIntegration:
    def test_sweep_runner_honors_mock_env(self, tmp_path, monkeypatch):
        """run_canonical_sweep dispatches to MockBNBBackend when env var is set."""
        monkeypatch.setenv(USE_MOCK_ENV, "1")
        monkeypatch.setenv("STL_SEED_RUNS_DIR_OVERRIDE", str(tmp_path))
        monkeypatch.delenv(REAL_TRAINING_ENV, raising=False)

        sweep_module = _import_script("run_canonical_sweep")
        # _RUNS_DIR is read at import time from the env var.
        assert Path(sweep_module._RUNS_DIR).resolve() == tmp_path.resolve()
        assert sweep_module._mock_backend_enabled() is True

        rc = sweep_module.main(
            [
                "--config-name",
                "sweep_main",
                "--only-cell",
                "qwen3_0.6b__hard__bio_ode_repressilator",
                "--confirm",
                "--max-cost-usd",
                "100",
            ]
        )
        assert rc == 0
        cell_dir = tmp_path / "qwen3_0.6b__hard__bio_ode_repressilator"
        assert cell_dir.exists()
        assert (cell_dir / "done.flag").exists()
        assert (cell_dir / "adapter" / "MOCK.txt").exists()

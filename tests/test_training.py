"""Unit tests for ``stl_seed.training`` (subphase 1.3 A10).

Coverage:

* Default :class:`TrainingConfig` values match SERA's ``unsloth_qwen3_moe_qlora.yaml``
  (paper/REDACTED.md §C.3) where applicable.
* :func:`format_trajectory_as_text` produces the documented format and
  round-trips through a regex parser.
* Each Jinja2 prompt template renders without errors.
* ``MLXBackend`` and ``BNBBackend`` instantiate cleanly on any platform
  (no native imports happen until ``train`` is called).
* Calling ``MLXBackend.train`` on a non-Apple-Silicon host raises a clear
  ``ImportError`` — never crashes with a low-level mlx import error.
* Calling ``BNBBackend.train`` on a non-CUDA host raises a clear
  ``ImportError`` — never crashes with a low-level bnb error.
* :func:`train_with_filter` dispatches to the correct backend by name.

No actual SFT runs in CI — that is gated behind ``@pytest.mark.mlx`` /
``@pytest.mark.cuda``.
"""

from __future__ import annotations

import platform
import re
from pathlib import Path
from unittest import mock

import jax.numpy as jnp
import numpy as np
import pytest

from stl_seed.specs import REGISTRY
from stl_seed.tasks._trajectory import Trajectory, TrajectoryMeta
from stl_seed.training import TrainedCheckpoint, TrainingConfig, train_with_filter
from stl_seed.training.backends.base import TrainingBackend
from stl_seed.training.loop import get_backend
from stl_seed.training.prompts import (
    GLUCOSE_INSULIN_SYSTEM_PROMPT,
    MAPK_SYSTEM_PROMPT,
    REPRESSILATOR_SYSTEM_PROMPT,
    TOGGLE_SYSTEM_PROMPT,
    list_tasks,
    render_system_prompt,
)
from stl_seed.training.tokenize import (
    format_for_chat,
    format_trajectory_as_text,
    serialize_assistant_turn,
    trajectory_to_record,
)

# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


def _make_trajectory(T: int = 10, n: int = 3, H: int = 5, m: int = 2) -> Trajectory:
    """Construct a synthetic Trajectory pytree for testing."""
    states = jnp.asarray(np.linspace(0.1, 1.0, T * n).reshape(T, n).astype(np.float32))
    actions = jnp.asarray(np.linspace(-0.5, 0.5, H * m).reshape(H, m).astype(np.float32))
    times = jnp.linspace(0.0, 60.0, T)
    meta = TrajectoryMeta(
        n_nan_replacements=jnp.asarray(0.0),
        final_solver_result=jnp.asarray(0.0),
        used_stiff_fallback=jnp.asarray(0.0),
    )
    return Trajectory(states=states, actions=actions, times=times, meta=meta)


# ---------------------------------------------------------------------------
# TrainingConfig defaults.
# ---------------------------------------------------------------------------


class TestTrainingConfigDefaults:
    """Defaults must mirror SERA's QLoRA YAML (paper/REDACTED.md §C.3)."""

    def test_lora_rank_matches_sera_qlora(self):
        # paper/REDACTED.md §C.3: lora_r: 32
        assert TrainingConfig().lora_rank == 32

    def test_lora_alpha_matches_sera_qlora(self):
        # paper/REDACTED.md §C.3: lora_alpha: 128
        assert TrainingConfig().lora_alpha == 128.0

    def test_lora_alpha_to_rank_ratio_is_4x(self):
        # SERA QLoRA YAML: alpha/r = 4 (header comment in unsloth file).
        cfg = TrainingConfig()
        assert cfg.lora_alpha / cfg.lora_rank == pytest.approx(4.0)

    def test_learning_rate_matches_sera_qlora(self):
        # paper/REDACTED.md §C.3: learning_rate: 5e-5 (LoRA rule of thumb)
        assert TrainingConfig().learning_rate == 5e-5

    def test_lr_schedule_matches_sera_qlora(self):
        # paper/REDACTED.md §C.3: lr_scheduler_type: cosine
        assert TrainingConfig().lr_schedule == "cosine"

    def test_warmup_ratio_matches_sera_qlora(self):
        # paper/REDACTED.md §C.3: warmup_ratio: 0.1
        assert TrainingConfig().warmup_ratio == 0.1

    def test_num_epochs_matches_sera_qlora(self):
        # paper/REDACTED.md §C.3: num_train_epochs: 3
        assert TrainingConfig().num_epochs == 3

    def test_batch_size_matches_sera_qlora(self):
        # paper/REDACTED.md §C.3: per_device_train_batch_size: 1
        assert TrainingConfig().batch_size == 1

    def test_grad_accum_matches_sera_qlora(self):
        # paper/REDACTED.md §C.3: gradient_accumulation_steps: 4
        assert TrainingConfig().gradient_accumulation_steps == 4

    def test_weight_decay_matches_sera_qlora(self):
        # paper/REDACTED.md §C.3: weight_decay: 0.01
        assert TrainingConfig().weight_decay == 0.01

    def test_seed_matches_sera_qlora(self):
        # paper/REDACTED.md §C.3: seed: 42
        assert TrainingConfig().seed == 42

    def test_lora_dropout_matches_sera_qlora(self):
        # paper/REDACTED.md §C.3: lora_dropout: 0.0 (FA-compat).
        assert TrainingConfig().lora_dropout == 0.0

    def test_weight_format_default_is_nf4(self):
        # paper/REDACTED.md §C.3: load_in_4bit: true (NF4).
        # Dettmers QLoRA paper (arXiv:2305.14314 §3) confirms NF4 + double-quant.
        assert TrainingConfig().weight_format == "nf4"

    def test_use_8bit_optimizer_default_true(self):
        # paper/REDACTED.md §C.3: optim: adamw_8bit.
        assert TrainingConfig().use_8bit_optimizer is True

    def test_lora_target_modules_cover_attention_and_mlp(self):
        # paper/REDACTED.md §C.3: target_modules = q/k/v/o + gate/up/down.
        # Names use the local-relative form (self_attn.* / mlp.*) required
        # by mlx_lm's linear_to_lora_layers — see paper/REDACTED.md
        # §"Issues encountered" for the silent-no-op rationale.
        cfg = TrainingConfig()
        targets = set(cfg.lora_target_modules)
        assert {
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        }.issubset(targets)
        assert {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}.issubset(targets)
        # Defaults must NOT contain bare names — they would trigger the
        # MLX naming-convention warning emitted in __post_init__.
        for t in targets:
            assert "." in t, f"default target {t!r} is bare; would no-op under MLX"

    def test_max_seq_length_is_smaller_than_sera(self):
        # SERA uses 32768 for code; stl-seed control trajectories are shorter
        # (paper/REDACTED.md STEP-3 mapping table, "sequence_len: ADAPT").
        assert TrainingConfig().max_seq_length <= 32768

    def test_default_base_model_is_qwen3_smallest(self):
        # theory.md §1: smallest in 3×3×2 sweep is Qwen3-0.6B.
        assert "Qwen3-0.6B" in TrainingConfig().base_model

    def test_invalid_learning_rate_raises(self):
        with pytest.raises(ValueError, match="learning_rate"):
            TrainingConfig(learning_rate=-1.0)

    def test_invalid_warmup_ratio_raises(self):
        with pytest.raises(ValueError, match="warmup_ratio"):
            TrainingConfig(warmup_ratio=1.5)

    def test_invalid_lora_rank_raises(self):
        with pytest.raises(ValueError, match="lora_rank"):
            TrainingConfig(lora_rank=0)


# ---------------------------------------------------------------------------
# Tokenizer.
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_serialize_assistant_turn_basic(self):
        states = np.array([[0.1, 0.2], [0.3, 0.4]])
        actions = np.array([[1.0], [2.0]])
        out = serialize_assistant_turn(states, actions)
        assert out.count("<state>") == 2
        assert out.count("<action>") == 2
        assert "1.000e+00" in out

    def test_serialize_assistant_turn_shape_mismatch(self):
        states = np.zeros((3, 2))
        actions = np.zeros((4, 1))
        with pytest.raises(ValueError, match="matching horizon"):
            serialize_assistant_turn(states, actions)

    def test_format_trajectory_as_text_keys(self):
        traj = _make_trajectory()
        spec = REGISTRY["bio_ode.repressilator.easy"]
        out = format_trajectory_as_text(traj, spec, task_name="repressilator")
        assert set(out.keys()) == {"system", "user", "assistant"}
        assert "<state>" in out["assistant"]
        assert "<action>" in out["assistant"]

    def test_format_trajectory_round_trips_actions(self):
        """The action vectors must appear verbatim (4-sig-fig) in assistant turn."""
        traj = _make_trajectory(T=20, n=3, H=4, m=2)
        spec = REGISTRY["bio_ode.repressilator.easy"]
        out = format_trajectory_as_text(traj, spec, task_name="repressilator")
        action_pattern = re.compile(r"<action>([^<]+)</action>")
        matches = action_pattern.findall(out["assistant"])
        assert len(matches) == 4  # H control steps
        # Parse first action and compare to original.
        parsed = [float(x) for x in matches[0].split(",")]
        original = np.asarray(traj.actions[0]).tolist()
        for p, o in zip(parsed, original, strict=True):
            assert p == pytest.approx(o, rel=1e-3)

    def test_format_trajectory_unknown_task_raises(self):
        traj = _make_trajectory()
        spec = REGISTRY["bio_ode.repressilator.easy"]
        with pytest.raises(KeyError, match="Unknown task family"):
            format_trajectory_as_text(traj, spec, task_name="not_a_task")

    def test_format_for_chat_message_shape(self):
        conv = {"system": "S", "user": "U", "assistant": "A"}
        msgs = format_for_chat(conv)
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
        assert [m["content"] for m in msgs] == ["S", "U", "A"]

    def test_trajectory_to_record_columns(self):
        traj = _make_trajectory()
        spec = REGISTRY["bio_ode.repressilator.easy"]
        rec = trajectory_to_record(traj, spec, task_name="repressilator", weight=0.7)
        assert set(rec.keys()) >= {
            "messages",
            "prompt",
            "completion",
            "weight",
            "task",
        }
        assert rec["weight"] == pytest.approx(0.7)
        assert rec["task"] == "repressilator"


# ---------------------------------------------------------------------------
# Prompts.
# ---------------------------------------------------------------------------


class TestPrompts:
    def test_list_tasks_returns_all_four_families(self):
        assert set(list_tasks()) == {
            "repressilator",
            "toggle",
            "mapk",
            "glucose_insulin",
        }

    @pytest.mark.parametrize(
        "task,template",
        [
            ("repressilator", REPRESSILATOR_SYSTEM_PROMPT),
            ("toggle", TOGGLE_SYSTEM_PROMPT),
            ("mapk", MAPK_SYSTEM_PROMPT),
            ("glucose_insulin", GLUCOSE_INSULIN_SYSTEM_PROMPT),
        ],
    )
    def test_each_template_has_required_variables(self, task, template):
        assert "{{ spec_text }}" in template
        assert "{{ horizon }}" in template
        assert "{{ duration_minutes }}" in template

    @pytest.mark.parametrize("task", ["repressilator", "toggle", "mapk", "glucose_insulin"])
    def test_render_smoke(self, task):
        out = render_system_prompt(
            task=task,
            spec_text="G_[0,10] (x_1 > 0.5)",
            horizon=20,
            duration_minutes=60.0,
        )
        assert "G_[0,10] (x_1 > 0.5)" in out
        assert "20" in out
        assert "60.0" in out

    def test_render_unknown_task_raises(self):
        with pytest.raises(KeyError, match="Unknown task family"):
            render_system_prompt(
                task="not_a_task",
                spec_text="...",
                horizon=10,
                duration_minutes=30.0,
            )

    def test_render_missing_var_raises(self):
        # StrictUndefined: missing template var should fail loudly.
        # We simulate by registering a custom template with an extra var that
        # render() does not pass through; render_system_prompt does not allow
        # that, so we instead test the underlying jinja env directly by
        # constructing a string with an undefined var.
        import jinja2

        from stl_seed.training.prompts import _ENV

        tmpl = _ENV.from_string("Hello {{ missing_var }}")
        with pytest.raises(jinja2.UndefinedError):
            tmpl.render()


# ---------------------------------------------------------------------------
# Backend instantiation + guards.
# ---------------------------------------------------------------------------


class TestBackendGuards:
    def test_mlx_module_imports_on_any_platform(self):
        """Importing the mlx backend module must not crash on non-Apple-Silicon."""
        # The module-level import at top of this file already covers this
        # implicitly, but exercise it here as an explicit assertion.
        from stl_seed.training.backends import mlx as mlx_module

        assert mlx_module.MLXBackend.__name__ == "MLXBackend"

    def test_bnb_module_imports_on_any_platform(self):
        """Importing the bnb backend module must not crash on a CPU-only machine."""
        from stl_seed.training.backends import bnb as bnb_module

        assert bnb_module.BNBBackend.__name__ == "BNBBackend"

    def test_mlx_construct_on_any_platform(self):
        from stl_seed.training.backends.mlx import MLXBackend

        backend = MLXBackend()
        assert backend.name == "mlx"
        assert isinstance(backend, TrainingBackend)

    def test_bnb_construct_on_any_platform(self):
        from stl_seed.training.backends.bnb import BNBBackend

        backend = BNBBackend()
        assert backend.name == "bnb"
        assert isinstance(backend, TrainingBackend)

    def test_mlx_train_raises_clear_error_on_non_apple_silicon(self):
        """If we are NOT on Apple Silicon, MLXBackend.train must raise ImportError."""
        from stl_seed.training.backends.mlx import MLXBackend, _check_apple_silicon

        is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
        if is_apple_silicon:
            # On Apple Silicon the platform check passes; this test is N/A.
            pytest.skip("On Apple Silicon, platform check passes.")

        backend = MLXBackend()
        with pytest.raises(ImportError, match="Apple Silicon"):
            _check_apple_silicon()
        with pytest.raises(ImportError, match="Apple Silicon"):
            backend.train(
                base_model="Qwen/Qwen3-0.6B-Instruct",
                dataset=[],
                config=TrainingConfig(),
                output_dir=Path("/tmp/stl_seed_test"),
            )

    def test_bnb_train_raises_clear_error_on_non_cuda(self):
        """If torch.cuda.is_available() is False, BNBBackend.train must raise ImportError."""
        import contextlib

        from stl_seed.training.backends.bnb import BNBBackend, _check_cuda

        # If torch is not installed at all, _check_cuda raises ImportError on
        # the import itself, which is also a valid "not on CUDA" path.
        with contextlib.suppress(ImportError):
            import torch  # noqa: F401

        cuda_available = False
        with contextlib.suppress(ImportError):
            import torch as _torch

            cuda_available = _torch.cuda.is_available()

        if cuda_available:
            pytest.skip("CUDA is available; the no-CUDA guard cannot be tested here.")

        backend = BNBBackend()
        with pytest.raises(ImportError, match="(CUDA|PyTorch)"):
            _check_cuda()
        with pytest.raises(ImportError, match="(CUDA|PyTorch)"):
            backend.train(
                base_model="Qwen/Qwen3-0.6B-Instruct",
                dataset=[],
                config=TrainingConfig(),
                output_dir=Path("/tmp/stl_seed_test"),
            )


# ---------------------------------------------------------------------------
# Hardware-gated smoke tests.
# ---------------------------------------------------------------------------


@pytest.mark.mlx
def test_mlx_backend_initializes_on_apple_silicon():
    """On Apple Silicon, MLXBackend can run its platform check."""
    from stl_seed.training.backends.mlx import MLXBackend, _check_apple_silicon

    backend = MLXBackend()
    assert backend.name == "mlx"
    _check_apple_silicon()  # must not raise


@pytest.mark.cuda
def test_bnb_backend_initializes_on_cuda():
    """On CUDA, BNBBackend can run its platform check."""
    from stl_seed.training.backends.bnb import BNBBackend, _check_cuda

    backend = BNBBackend()
    assert backend.name == "bnb"
    _check_cuda()  # must not raise


# ---------------------------------------------------------------------------
# Loop dispatch.
# ---------------------------------------------------------------------------


class TestLoopDispatch:
    def test_get_backend_mlx(self):
        backend = get_backend("mlx")
        assert backend.name == "mlx"

    def test_get_backend_bnb(self):
        backend = get_backend("bnb")
        assert backend.name == "bnb"

    def test_get_backend_unknown(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("not_a_backend")  # type: ignore[arg-type]

    def test_train_with_filter_invalid_filter_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown filter_condition"):
            train_with_filter(
                filter_condition="not_a_filter",
                task="repressilator",
                model="Qwen/Qwen3-0.6B-Instruct",
                backend="mlx",
                config=TrainingConfig(output_dir=tmp_path),
                dataset=[],
            )

    def test_train_with_filter_dispatches_to_mlx(self, tmp_path):
        """train_with_filter must select the MLX backend when backend='mlx'."""
        cfg = TrainingConfig(output_dir=tmp_path)
        fake_ckpt = TrainedCheckpoint(
            backend="mlx",
            model_path=tmp_path,
            base_model="Qwen/Qwen3-0.6B-Instruct",
            training_loss_history=[1.0, 0.5],
            wall_clock_seconds=1.0,
        )

        with mock.patch(
            "stl_seed.training.backends.mlx.MLXBackend.train",
            return_value=fake_ckpt,
        ) as patched:
            ckpt = train_with_filter(
                filter_condition="hard",
                task="repressilator",
                model="Qwen/Qwen3-0.6B-Instruct",
                backend="mlx",
                config=cfg,
                dataset=[{"messages": [], "weight": 1.0}],
            )
            patched.assert_called_once()
            assert ckpt.backend == "mlx"

    def test_train_with_filter_dispatches_to_bnb(self, tmp_path):
        """train_with_filter must select the bnb backend when backend='bnb'."""
        cfg = TrainingConfig(output_dir=tmp_path)
        fake_ckpt = TrainedCheckpoint(
            backend="bnb",
            model_path=tmp_path,
            base_model="Qwen/Qwen3-0.6B-Instruct",
            training_loss_history=[2.0, 1.5],
            wall_clock_seconds=2.0,
        )

        with mock.patch(
            "stl_seed.training.backends.bnb.BNBBackend.train",
            return_value=fake_ckpt,
        ) as patched:
            ckpt = train_with_filter(
                filter_condition="continuous",
                task="toggle",
                model="Qwen/Qwen3-0.6B-Instruct",
                backend="bnb",
                config=cfg,
                dataset=[{"messages": [], "weight": 0.5}],
            )
            patched.assert_called_once()
            assert ckpt.backend == "bnb"

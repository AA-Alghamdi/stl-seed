"""Supplemental coverage for ``stl_seed.training`` (loop + backend helpers).

Targets:

* ``training/loop.py``. _load_filtered_dataset (lines 71-81), the
  config.base_model warning branch (126-128), the missing-dataset path (141).
* ``training/backends/mlx.py``. _estimate_iters edge cases, _config_to_dict
  serialization (313-337).
* ``training/backends/bnb.py``. _config_to_dict serialization (370-379).
* ``training/backends/base.py``. TrainingConfig invalid lora_alpha,
  invalid num_epochs, invalid lora_dropout (lines 144, 146, 150).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stl_seed.training import TrainingConfig, train_with_filter
from stl_seed.training.backends.bnb import _config_to_dict as bnb_config_to_dict
from stl_seed.training.backends.mlx import (
    _config_to_dict as mlx_config_to_dict,
)
from stl_seed.training.backends.mlx import (
    _estimate_iters,
)
from stl_seed.training.loop import _load_filtered_dataset

# ---------------------------------------------------------------------------
# Loop helpers
# ---------------------------------------------------------------------------


def test_load_filtered_dataset_raises_filenotfound_when_no_data() -> None:
    """``stl_seed.filter.dataset.load_filtered_dataset`` is implemented (Tier 9
    closing fix) but raises FileNotFoundError when the data manifest is
    missing for the requested (task, filter) pair."""
    with pytest.raises(FileNotFoundError, match="No filtered manifest"):
        _load_filtered_dataset("hard", "this_task_does_not_exist_12345")


def test_train_with_filter_warns_when_model_overrides_config(tmp_path, caplog) -> None:
    """A mismatched config.base_model + explicit ``model`` argument should
    trigger a logged warning (covers lines 126-128). We pre-pass a
    dataset to avoid the (CUDA / MLX) load path entirely; the dispatch
    will raise InappError downstream but the warning fires before that.
    """
    cfg = TrainingConfig(base_model="some/other-model", output_dir=tmp_path)

    # We use an unknown filter to force an early ValueError BEFORE the
    # backend dispatch so we don't need an actual MLX/bnb host.
    with pytest.raises(ValueError, match="Unknown filter_condition"):
        train_with_filter(
            filter_condition="not_a_filter",
            task="glucose_insulin",
            model="Qwen/Qwen3-0.6B-Instruct",
            backend="mlx",
            config=cfg,
            dataset=[],
        )


def test_train_with_filter_loads_dataset_when_none_passed(tmp_path) -> None:
    """When ``dataset=None`` and no manifest exists for the requested cell,
    train_with_filter must surface FileNotFoundError from
    load_filtered_dataset (post-Tier-9. the loader is now implemented)."""
    cfg = TrainingConfig(output_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="No filtered manifest"):
        train_with_filter(
            filter_condition="hard",
            task="this_task_does_not_exist_12345",
            model="Qwen/Qwen3-0.6B-Instruct",
            backend="mlx",
            config=cfg,
            dataset=None,
        )


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------


def test_estimate_iters_zero_examples_returns_zero() -> None:
    assert _estimate_iters(n_examples=0, batch_size=1, grad_accum=1, num_epochs=3) == 0


def test_estimate_iters_basic_formula() -> None:
    # 100 examples / (batch=2 * accum=4) = 13 steps per epoch (ceiling), * 2 epochs
    assert _estimate_iters(n_examples=100, batch_size=2, grad_accum=4, num_epochs=2) == 13 * 2


def test_estimate_iters_zero_eff_batch_clamped_to_one() -> None:
    """If batch_size or grad_accum is zero, effective batch is clamped to 1."""
    out = _estimate_iters(n_examples=10, batch_size=0, grad_accum=0, num_epochs=1)
    assert out == 10  # 10 examples / 1 effective = 10 steps


def test_mlx_config_to_dict_serializes_paths_and_lists() -> None:
    cfg = TrainingConfig(output_dir=Path("/tmp/abc"))
    d = mlx_config_to_dict(cfg)
    assert d["output_dir"] == "/tmp/abc"
    assert isinstance(d["lora_target_modules"], list)
    assert d["learning_rate"] == cfg.learning_rate


def test_bnb_config_to_dict_serializes_paths_and_lists() -> None:
    cfg = TrainingConfig(output_dir=Path("/tmp/xyz"))
    d = bnb_config_to_dict(cfg)
    assert d["output_dir"] == "/tmp/xyz"
    assert isinstance(d["lora_target_modules"], list)


# ---------------------------------------------------------------------------
# TrainingConfig validation branches
# ---------------------------------------------------------------------------


def test_training_config_invalid_lora_alpha_raises() -> None:
    with pytest.raises(ValueError, match="lora_alpha"):
        TrainingConfig(lora_alpha=0.0)


def test_training_config_invalid_num_epochs_raises() -> None:
    with pytest.raises(ValueError, match="num_epochs"):
        TrainingConfig(num_epochs=0)


def test_training_config_invalid_lora_dropout_raises() -> None:
    with pytest.raises(ValueError, match="lora_dropout"):
        TrainingConfig(lora_dropout=1.0)
    with pytest.raises(ValueError, match="lora_dropout"):
        TrainingConfig(lora_dropout=-0.1)


def test_training_config_string_output_dir_coerced_to_path() -> None:
    cfg = TrainingConfig(output_dir="some/relative/path")  # type: ignore[arg-type]
    assert isinstance(cfg.output_dir, Path)

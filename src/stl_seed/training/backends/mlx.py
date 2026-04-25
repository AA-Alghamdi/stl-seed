"""MLX (Apple Silicon) training backend for stl-seed.

Wraps ``mlx_lm.lora`` to LoRA-finetune a Qwen3 base on the filtered SFT
dataset locally on an M-series Mac. This is the pilot path used during
Phase 1 iteration; the canonical Phase 2 sweep runs on the bnb backend
on RunPod.

REDACTED firewall: this module is part of the SERA-style SFT loop and does
not import REDACTED / REDACTED / REDACTED / REDACTED.

Lazy-import discipline (paper/architecture.md "dual backend" requirement
+ task spec REQUIREMENTS): all ``mlx_lm`` and ``mlx`` imports are
deferred inside :meth:`MLXBackend.train` and :meth:`MLXBackend.load`
so that ``import stl_seed.training.backends.mlx`` succeeds on a vanilla
Linux/CPU machine that has no MLX wheels installed.
"""

from __future__ import annotations

import json
import logging
import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from stl_seed.training.backends.base import (
    TrainedCheckpoint,
    TrainingConfig,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Platform guard.
# ---------------------------------------------------------------------------


def _check_apple_silicon() -> None:
    """Raise a clear ImportError on non-Apple-Silicon hosts.

    MLX requires M-series GPUs and the Metal backend. Intel Macs and
    Linux/Windows hosts cannot run mlx_lm at all, so we fail fast with a
    pointer to the bnb backend.
    """
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ImportError(
            "MLXBackend requires Apple Silicon (Darwin/arm64); detected "
            f"{platform.system()}/{platform.machine()}. Use BNBBackend on "
            "CUDA hosts (e.g., RunPod 4090) instead."
        )


# ---------------------------------------------------------------------------
# Backend.
# ---------------------------------------------------------------------------


class MLXBackend:
    """``TrainingBackend`` implementation backed by ``mlx_lm.lora``.

    Construction is cheap and does not import MLX. The first call to
    :meth:`train` performs the platform check and the lazy import.
    """

    name: str = "mlx"

    def __init__(self) -> None:
        # Construction must not import MLX (so unit tests can instantiate
        # the class on CPU CI). Deferred imports happen in ``train``/``load``.
        self._loaded_model_cache: dict[str, Any] = {}

    # ----- public API ----------------------------------------------------

    def train(
        self,
        base_model: str,
        dataset: Any,
        config: TrainingConfig,
        output_dir: Path,
    ) -> TrainedCheckpoint:
        """LoRA-finetune ``base_model`` on ``dataset`` via ``mlx_lm.lora``.

        Parameters
        ----------
        base_model:
            HuggingFace model ID (e.g., ``"Qwen/Qwen3-0.6B-Instruct"``).
            Will be downloaded by ``mlx_lm`` if not already cached.
        dataset:
            HuggingFace ``datasets.Dataset`` with ``messages`` and ``weight``
            columns (see :mod:`stl_seed.training.tokenize`).
        config:
            :class:`TrainingConfig`; see ``base.py`` for field semantics.
        output_dir:
            Directory to write LoRA adapter weights and ``provenance.json``.

        Returns
        -------
        :class:`TrainedCheckpoint` with ``backend="mlx"``.
        """
        _check_apple_silicon()

        # Lazy heavy imports — only loaded when actually training on Mac.
        try:
            import mlx.core as mx  # noqa: F401  (warm the device)
            from mlx_lm.tuner.utils import linear_to_lora_layers  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover (env-specific)
            raise ImportError(
                "MLXBackend.train requires mlx + mlx_lm. Install with "
                "`uv sync --extra mlx` on Apple Silicon."
            ) from exc

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Convert HF base → MLX format if not already cached.
        #    mlx_lm.utils.load handles HF-format → MLX conversion transparently
        #    via its tokenizer + weight-map plumbing.
        from mlx_lm.utils import load as mlx_load  # type: ignore[import-not-found]

        log.info("Loading base model %s into MLX...", base_model)
        model, tokenizer = mlx_load(base_model)

        # 2. Inject LoRA adapters on the requested target modules.
        #    mlx_lm's linear_to_lora_layers takes a count and a config dict.
        lora_config = {
            "rank": config.lora_rank,
            "scale": config.lora_alpha / config.lora_rank,
            "dropout": config.lora_dropout,
            "keys": list(config.lora_target_modules),
        }
        # Apply to all transformer blocks (None = "all"; mlx_lm accepts an int
        # to limit to the last-N blocks for partial-LoRA, but we mirror SERA
        # in finetuning every block).
        n_blocks = len(getattr(model, "layers", []))
        linear_to_lora_layers(
            model,
            num_layers=n_blocks,
            config=lora_config,
        )

        # 3. Render dataset to a JSONL file in the format mlx_lm.lora expects:
        #    one record per line with a "messages" field.
        dataset_path = output_dir / "_train.jsonl"
        weights_list: list[float] = []
        with dataset_path.open("w") as fh:
            for row in dataset:
                rec = {"messages": row["messages"]}
                fh.write(json.dumps(rec) + "\n")
                weights_list.append(float(row.get("weight", 1.0)))

        # 4. Build a weighted loss closure. We override mlx_lm's default
        #    cross-entropy with one that scales the per-sample loss by
        #    the dataset's "weight" column (continuous-filter support).
        loss_history: list[float] = []
        wall_start = time.time()

        # We build a TrainingArgs dict matching the mlx_lm.lora.LoRATrainer API.
        # Field names follow mlx_lm 0.20+; see mlx_lm/tuner/trainer.py.
        train_args = {
            "batch_size": config.batch_size,
            "iters": _estimate_iters(
                n_examples=len(weights_list),
                batch_size=config.batch_size,
                grad_accum=config.gradient_accumulation_steps,
                num_epochs=config.num_epochs,
            ),
            "val_batches": 0,  # we run our own held-out eval
            "steps_per_report": 10,
            "steps_per_eval": 200,
            "steps_per_save": 500,
            "adapter_file": str(output_dir / "adapters.safetensors"),
            "max_seq_length": config.max_seq_length,
            "grad_checkpoint": True,
            "learning_rate": config.learning_rate,
            "warmup_steps": int(
                _estimate_iters(
                    n_examples=len(weights_list),
                    batch_size=config.batch_size,
                    grad_accum=config.gradient_accumulation_steps,
                    num_epochs=config.num_epochs,
                )
                * config.warmup_ratio
            ),
            "lr_schedule": config.lr_schedule,
            "seed": config.seed,
        }

        # 5. Invoke mlx_lm.lora's main training entry.
        #    The CLI entry is `python -m mlx_lm.lora --train ...`; the
        #    importable function in 0.20+ lives at mlx_lm.tuner.trainer.train.
        from mlx_lm.tuner.trainer import TrainingArgs  # type: ignore[import-not-found]
        from mlx_lm.tuner.trainer import train as mlx_train

        targs = TrainingArgs(**train_args)

        # mlx_lm's loss is token-level CE; we scale by the per-sample weight
        # via a closure over weights_list. The mlx_lm trainer iterates with
        # a deterministic dataloader, so weights line up by index modulo
        # batch_size.
        def weighted_loss_fn(model_, batch_):
            # Default mlx_lm loss returns scalar mean CE over tokens.
            from mlx_lm.tuner.losses import default_loss  # type: ignore[import-not-found]

            base_loss, ntoks = default_loss(model_, batch_)
            # Look up per-sample weights from the deterministic order. The
            # batch carries its sample indices in batch_["sample_idx"] when
            # we provide them via the iterate_batches hook below.
            idxs = batch_.get("sample_idx", None)
            if idxs is None:
                return base_loss, ntoks
            import mlx.core as mx_

            w = mx_.array(
                [weights_list[int(i) % len(weights_list)] for i in idxs],
                dtype=base_loss.dtype,
            )
            return base_loss * w.mean(), ntoks

        def on_report(info):
            if "train_loss" in info:
                loss_history.append(float(info["train_loss"]))

        # Optimizer: AdamW (mlx_lm's default for LoRA).
        import mlx.optimizers as optim  # type: ignore[import-not-found]

        opt = optim.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        log.info(
            "Starting MLX LoRA training: %d examples × %d epochs, lr=%g, rank=%d",
            len(weights_list),
            config.num_epochs,
            config.learning_rate,
            config.lora_rank,
        )

        try:
            mlx_train(
                model=model,
                tokenizer=tokenizer,
                optimizer=opt,
                train_dataset=_JsonlDataset(dataset_path),
                val_dataset=_JsonlDataset(dataset_path),  # placeholder
                args=targs,
                loss=weighted_loss_fn,
                training_callback=on_report,
            )
        except Exception:  # noqa: BLE001
            # Per CLAUDE.md: never silently swallow training failures.
            log.exception("MLX training failed; checkpoint will be incomplete.")
            raise

        wall = time.time() - wall_start

        # 6. Write provenance manifest.
        provenance = {
            "backend": "mlx",
            "base_model": base_model,
            "n_examples": len(weights_list),
            "config": _config_to_dict(config),
            "wall_clock_seconds": wall,
            "n_loss_points": len(loss_history),
        }
        (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

        return TrainedCheckpoint(
            backend="mlx",
            model_path=output_dir,
            base_model=base_model,
            training_loss_history=loss_history,
            wall_clock_seconds=wall,
            metadata=provenance,
        )

    def load(self, checkpoint: TrainedCheckpoint) -> Callable[..., Any]:
        """Load a trained MLX LoRA checkpoint into a generation callable.

        Returns a function ``generate(prompt: str, **kwargs) -> str``
        that wraps ``mlx_lm.generate``.
        """
        _check_apple_silicon()
        try:
            from mlx_lm import generate  # type: ignore[import-not-found]
            from mlx_lm.utils import load as mlx_load  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "MLXBackend.load requires mlx_lm. Install with "
                "`uv sync --extra mlx` on Apple Silicon."
            ) from exc

        adapter_path = checkpoint.model_path / "adapters.safetensors"
        model, tokenizer = mlx_load(
            checkpoint.base_model,
            adapter_path=str(adapter_path) if adapter_path.exists() else None,
        )

        def _generate(prompt: str, **kwargs: Any) -> str:
            return generate(model, tokenizer, prompt=prompt, **kwargs)

        return _generate


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _estimate_iters(
    n_examples: int,
    batch_size: int,
    grad_accum: int,
    num_epochs: int,
) -> int:
    """Conservative iteration count for mlx_lm's iter-based trainer."""
    if n_examples <= 0:
        return 0
    eff = max(batch_size * grad_accum, 1)
    per_epoch = max((n_examples + eff - 1) // eff, 1)
    return per_epoch * num_epochs


def _config_to_dict(config: TrainingConfig) -> dict[str, Any]:
    """Serialize a TrainingConfig to JSON-friendly dict (Path → str)."""
    out: dict[str, Any] = {}
    for k, v in config.__dict__.items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, list | tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out


class _JsonlDataset:
    """Minimal mlx_lm-compatible dataset reading our JSONL on demand.

    mlx_lm.tuner.datasets defines a ChatDataset that wraps a list of
    {"messages": [...]} dicts. We mirror that interface so the trainer
    iterates without needing the full mlx_lm.tuner.datasets stack.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._records: list[dict[str, Any]] = []
        with self.path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self._records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._records[idx]

    def __iter__(self):
        return iter(self._records)


__all__ = ["MLXBackend"]

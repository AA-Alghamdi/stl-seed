"""MLX (Apple Silicon) training backend for stl-seed.

Wraps ``mlx_lm.tuner.trainer.train`` to LoRA-finetune a Qwen3 base on the
filtered SFT dataset locally on an M-series Mac. This is the pilot path
used during Phase 1 iteration; the canonical Phase 2 sweep runs on the
bnb backend on RunPod.

REDACTED firewall: this module is part of the SERA-style SFT loop and does
not import REDACTED / REDACTED / REDACTED / REDACTED.

mlx_lm 0.31 API contract (post-A15 patch)
-----------------------------------------

The pre-A15 wrapper was pinned to mlx_lm <= 0.20 and broke under 0.31:

* ``TrainingArgs`` no longer accepts ``learning_rate`` / ``lr_schedule``
  / ``warmup_steps`` / ``seed`` (the schedule lives on the optimizer;
  the seed lives on ``iterate_batches``).
* ``train(...)`` no longer takes a ``tokenizer`` argument — the
  tokenizer is consumed by the dataset wrapper instead.
* ``mlx_lm.tuner.utils.linear_to_lora_layers`` matches keys against
  module names *relative to each TransformerBlock* (e.g.
  ``"self_attn.q_proj"``). Bare names (e.g. ``"q_proj"``) silently
  match zero modules, producing a 0-trainable-parameter LoRA.
* The dataset passed to ``train(...)`` must be a processed
  ``CacheDataset`` wrapping a ``ChatDataset`` — the raw ``ChatDataset``
  returns ``dict`` records for which ``iterate_batches`` raises
  ``KeyError: 0`` when it tries ``len(dataset[idx][0])``.
* Reload via ``mlx_lm.load(adapter_path=...)`` requires
  ``adapter_config.json`` next to ``adapters.safetensors``; the trainer
  writes only the safetensors, so we synthesize the JSON here.

The post-patch shape mirrors ``scripts/smoke_test_mlx.py::_run_mlx_training``
which was validated end-to-end in A15 (see ``paper/REDACTED.md``).

Lazy-import discipline
----------------------

All ``mlx_lm`` and ``mlx`` imports are deferred inside :meth:`MLXBackend.train`
and :meth:`MLXBackend.load` so that ``import stl_seed.training.backends.mlx``
succeeds on a vanilla Linux/CPU machine that has no MLX wheels installed.
"""

from __future__ import annotations

import contextlib
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
    """``TrainingBackend`` implementation backed by ``mlx_lm.tuner.trainer.train``.

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
        """LoRA-finetune ``base_model`` on ``dataset`` via ``mlx_lm``.

        Parameters
        ----------
        base_model:
            HuggingFace or mlx-community model ID (e.g.,
            ``"mlx-community/Qwen3-0.6B-bf16"`` or a HF id like
            ``"Qwen/Qwen3-0.6B"`` which mlx_lm will convert on the fly).
        dataset:
            Iterable of records with at minimum a ``messages`` field
            (chat-format list of role/content dicts) and optionally a
            ``weight`` field (recorded in the manifest; mlx_lm 0.31's
            default loss does not consume per-sample weights — for true
            weighted SFT the bnb backend is the canonical path). A
            HuggingFace ``datasets.Dataset`` works directly because it
            satisfies the iterable + indexable protocol used here.
        config:
            :class:`TrainingConfig`; see ``base.py`` for field semantics.
            ``config.lora_target_modules`` MUST use the locally-prefixed
            naming form (e.g. ``"self_attn.q_proj"``); see the warning
            emitted by ``TrainingConfig.__post_init__``.
        output_dir:
            Directory to write LoRA adapter weights (``adapters.safetensors``),
            ``adapter_config.json`` (for round-trip via :meth:`load`),
            ``_train.jsonl`` (the rendered chat dataset), and
            ``provenance.json``.

        Returns
        -------
        :class:`TrainedCheckpoint` with ``backend="mlx"``.
        """
        _check_apple_silicon()

        # Lazy heavy imports — only loaded when actually training on Mac.
        try:
            import mlx.core as mx
            import mlx.optimizers as optim  # type: ignore[import-not-found]
            from mlx_lm import load as mlx_load  # type: ignore[import-not-found]
            from mlx_lm.tuner.datasets import ChatDataset  # type: ignore[import-not-found]
            from mlx_lm.tuner.trainer import (  # type: ignore[import-not-found]
                CacheDataset,
                TrainingArgs,
                default_loss,
                iterate_batches,
            )
            from mlx_lm.tuner.trainer import (
                train as mlx_train,
            )
            from mlx_lm.tuner.utils import (  # type: ignore[import-not-found]
                linear_to_lora_layers,
                print_trainable_parameters,
            )
        except ImportError as exc:  # pragma: no cover (env-specific)
            raise ImportError(
                "MLXBackend.train requires mlx + mlx_lm. Install with "
                "`uv sync --extra mlx` on Apple Silicon."
            ) from exc

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1) Render dataset to JSONL for repro + future inspection. We collect
        #    weights into a sidecar manifest only — mlx_lm 0.31's default loss
        #    does not consume per-sample weights, and we do not silently
        #    pretend it does. (For weighted SFT use the bnb backend.)
        dataset_path = output_dir / "_train.jsonl"
        weights_list: list[float] = []
        with dataset_path.open("w") as fh:
            for row in dataset:
                # Accept both dict-style and HuggingFace-row-style indexing.
                msgs = row["messages"]
                fh.write(json.dumps({"messages": msgs}) + "\n")
                w = 1.0
                if isinstance(row, dict) and "weight" in row:
                    w = float(row["weight"])
                else:
                    with contextlib.suppress(AttributeError, TypeError):
                        w = float(row.get("weight", 1.0))  # type: ignore[union-attr]
                weights_list.append(w)

        # 2) Load base model (mlx_lm handles HF → MLX conversion transparently).
        log.info("Loading base model %s into MLX...", base_model)
        model, tokenizer = mlx_load(base_model)

        # 3) Inject LoRA adapters on the requested target modules.
        #    Per the docstring at the top of this module: keys must be
        #    locally-prefixed (e.g. "self_attn.q_proj") for MLX to find them.
        n_blocks = len(getattr(model, "layers", []))
        if n_blocks == 0:
            raise RuntimeError(
                f"Loaded model {base_model!r} has no .layers attribute; "
                "cannot determine LoRA injection scope."
            )
        model.freeze()
        lora_config = {
            "rank": config.lora_rank,
            "scale": config.lora_alpha / config.lora_rank,
            "dropout": config.lora_dropout,
            "keys": list(config.lora_target_modules),
        }
        linear_to_lora_layers(
            model,
            num_layers=n_blocks,
            config=lora_config,
            use_dora=False,
        )
        # print_trainable_parameters is the canonical MLX-side guard against
        # the silent-no-op LoRA path. We log a clear error if zero params
        # are trainable (this happens when target keys do not match).
        try:
            print_trainable_parameters(model)
        except Exception:  # noqa: BLE001
            log.warning("print_trainable_parameters raised; continuing.")
        # Recursive count over the trainable_parameters tree (which is a
        # nested dict of {name: array | dict | list}; mlx_lm uses
        # mlx.utils.tree_flatten internally for the % calculation).
        n_trainable = _count_trainable(model) if hasattr(model, "trainable_parameters") else None
        if n_trainable is not None and n_trainable == 0:
            raise RuntimeError(
                "LoRA produced 0 trainable parameters. This means "
                f"config.lora_target_modules={config.lora_target_modules} "
                "did not match any modules under MLX's linear_to_lora_layers "
                "(which scopes match within each TransformerBlock). Use the "
                "locally-prefixed naming form, e.g. 'self_attn.q_proj'. "
                "See paper/REDACTED.md §'Issues encountered'."
            )

        # 4) Build the chat dataset. mlx_lm.iterate_batches expects each
        #    element to be a processed (token_ids, prompt_offset) tuple,
        #    not the raw chat dict — so we MUST wrap ChatDataset with
        #    CacheDataset (CacheDataset.__getitem__ runs .process(...) on
        #    first access). Without this wrap, iterate_batches raises
        #    KeyError: 0 because the raw dict has no integer key 0.
        records = []
        with dataset_path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if not records:
            raise ValueError("MLXBackend.train received an empty dataset.")
        chat_dataset = CacheDataset(ChatDataset(records, tokenizer, mask_prompt=False))
        # We supply the same dataset for val so mlx_lm doesn't crash if it
        # tries to evaluate; we set val_batches=0 below to skip validation.
        val_dataset = CacheDataset(ChatDataset(records, tokenizer, mask_prompt=False))

        # 5) Optimizer with linear-warmup → cosine-decay schedule. The
        #    schedule is attached to the optimizer (mlx_lm 0.31 contract);
        #    TrainingArgs no longer carries learning_rate/lr_schedule.
        n_iters = _estimate_iters(
            n_examples=len(weights_list),
            batch_size=config.batch_size,
            grad_accum=config.gradient_accumulation_steps,
            num_epochs=config.num_epochs,
        )
        if n_iters <= 0:
            raise ValueError(
                f"Estimated iters={n_iters} from "
                f"n_examples={len(weights_list)}, batch_size={config.batch_size}, "
                f"grad_accum={config.gradient_accumulation_steps}, "
                f"num_epochs={config.num_epochs}. Cannot train on empty dataset."
            )
        warmup_steps = max(1, int(round(n_iters * config.warmup_ratio)))
        warmup_steps = min(warmup_steps, max(n_iters - 1, 1))
        if config.lr_schedule == "cosine":
            warmup = optim.linear_schedule(0.0, config.learning_rate, warmup_steps)
            decay = optim.cosine_decay(config.learning_rate, n_iters - warmup_steps)
            schedule = optim.join_schedules([warmup, decay], [warmup_steps])
        elif config.lr_schedule == "linear":
            warmup = optim.linear_schedule(0.0, config.learning_rate, warmup_steps)
            # Decay back to 0 over the remainder.
            decay = optim.linear_schedule(config.learning_rate, 0.0, max(n_iters - warmup_steps, 1))
            schedule = optim.join_schedules([warmup, decay], [warmup_steps])
        else:  # constant
            schedule = config.learning_rate  # mlx optimizers accept a float.

        opt = optim.AdamW(learning_rate=schedule, weight_decay=config.weight_decay)

        # 6) Build TrainingArgs (no learning_rate / lr_schedule / warmup /
        #    seed fields in 0.31). Disable validation and intermediate saves;
        #    we save the final adapter via the trainer's adapter_file slot.
        train_args = TrainingArgs(
            batch_size=config.batch_size,
            iters=n_iters,
            val_batches=0,
            steps_per_report=max(1, n_iters // 10),
            steps_per_eval=10**9,
            steps_per_save=10**9,
            max_seq_length=config.max_seq_length,
            adapter_file=str(output_dir / "adapters.safetensors"),
            grad_checkpoint=True,
            grad_accumulation_steps=config.gradient_accumulation_steps,
        )

        loss_history: list[float] = []

        class _Callback:
            def on_train_loss_report(self, train_info: dict) -> None:
                v = train_info.get("train_loss", float("nan"))
                with contextlib.suppress(TypeError, ValueError):
                    loss_history.append(float(v))

            def on_val_loss_report(self, val_info: dict) -> None:  # noqa: ARG002
                pass

        log.info(
            "Starting MLX LoRA training: %d examples × %d epochs (%d iters), "
            "lr=%g, rank=%d, targets=%s",
            len(weights_list),
            config.num_epochs,
            n_iters,
            config.learning_rate,
            config.lora_rank,
            list(config.lora_target_modules),
        )

        wall_start = time.time()
        try:
            mlx_train(
                model=model,
                optimizer=opt,
                train_dataset=chat_dataset,
                val_dataset=val_dataset,
                args=train_args,
                loss=default_loss,
                iterate_batches=iterate_batches,
                training_callback=_Callback(),
            )
        except Exception:  # noqa: BLE001
            # Per CLAUDE.md: never silently swallow training failures.
            log.exception("MLX training failed; checkpoint will be incomplete.")
            raise
        wall = time.time() - wall_start

        # 7) Write adapter_config.json next to adapters.safetensors so that
        #    mlx_lm.load(adapter_path=...) can rebuild the LoRA wrapper at
        #    eval time. Schema follows mlx_lm 0.31's load_adapters: it reads
        #    num_layers, lora_parameters (rank, scale, dropout, optional
        #    keys), and fine_tune_type.
        adapter_config = {
            "fine_tune_type": "lora",
            "num_layers": n_blocks,
            "lora_parameters": {
                "rank": config.lora_rank,
                "scale": config.lora_alpha / config.lora_rank,
                "dropout": config.lora_dropout,
                "keys": list(config.lora_target_modules),
            },
        }
        (output_dir / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2))

        # 8) Provenance manifest.
        provenance = {
            "backend": "mlx",
            "base_model": base_model,
            "n_examples": len(weights_list),
            "config": _config_to_dict(config),
            "wall_clock_seconds": wall,
            "n_iters": n_iters,
            "warmup_steps": warmup_steps,
            "n_loss_points": len(loss_history),
            "loss_history": loss_history,
            "mlx_metal_available": bool(mx.metal.is_available()),
            "weights_min": min(weights_list) if weights_list else None,
            "weights_max": max(weights_list) if weights_list else None,
            "weights_mean": (sum(weights_list) / len(weights_list) if weights_list else None),
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

        ``mlx_lm.load(adapter_path=...)`` expects a *directory* containing
        both ``adapters.safetensors`` and ``adapter_config.json`` (per
        mlx_lm 0.31 ``load_adapters``); we wrote both in :meth:`train`,
        so we point the loader at ``checkpoint.model_path`` directly.
        """
        _check_apple_silicon()
        try:
            from mlx_lm import generate  # type: ignore[import-not-found]
            from mlx_lm import load as mlx_load  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "MLXBackend.load requires mlx_lm. Install with "
                "`uv sync --extra mlx` on Apple Silicon."
            ) from exc

        adapter_dir = Path(checkpoint.model_path)
        safetensors_path = adapter_dir / "adapters.safetensors"
        adapter_config_path = adapter_dir / "adapter_config.json"
        if safetensors_path.exists() and adapter_config_path.exists():
            model, tokenizer = mlx_load(checkpoint.base_model, adapter_path=str(adapter_dir))
        else:
            log.warning(
                "MLXBackend.load: adapter artifacts not found at %s "
                "(safetensors=%s, config=%s); loading base model only.",
                adapter_dir,
                safetensors_path.exists(),
                adapter_config_path.exists(),
            )
            model, tokenizer = mlx_load(checkpoint.base_model)

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


def _count_trainable(model: Any) -> int:
    """Count scalar trainable parameters in an MLX model.

    ``model.trainable_parameters()`` returns a nested
    ``dict[str, array | dict | list]`` (mlx Module-tree shape). We flatten
    via ``mlx.utils.tree_flatten`` to get a flat list of (name, array)
    leaves, then sum ``.size``.
    """
    try:
        from mlx.utils import tree_flatten  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover
        return 0
    try:
        params = model.trainable_parameters()
    except Exception:  # noqa: BLE001
        return 0
    leaves = tree_flatten(params)
    total = 0
    for _name, leaf in leaves:
        sz = getattr(leaf, "size", None)
        if isinstance(sz, int):
            total += sz
    return total


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


__all__ = ["MLXBackend"]

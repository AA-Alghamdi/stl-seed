"""bitsandbytes + TRL training backend for stl-seed (CUDA / RunPod).

Wraps ``trl.SFTTrainer`` on a ``bitsandbytes`` 4-bit (NF4) quantized base
model with LoRA adapters from PEFT. This is the canonical Phase 2
backend; the Phase 1 pilot uses ``MLXBackend`` on the M5 Pro.

Hyperparameters mirror SERA's ``unsloth_qwen3_moe_qlora.yaml`` field-for-
field where possible (paper/REDACTED.md §C.3): NF4 + bf16 compute,
double-quant, LoRA rank 32, alpha 128, lr 5e-5, cosine + 0.1 warmup,
3 epochs, weight_decay 0.01, adamw_8bit optimizer.

Per-sample weighting (continuous-filter support): TRL's ``SFTTrainer``
inherits from ``transformers.Trainer``. We override ``compute_loss`` via
a thin :class:`_WeightedSFTTrainer` subclass that scales the per-token
cross-entropy by the per-sample ``weight`` column from the dataset. This
matches the formalism in ``paper/theory.md`` §2 (continuous-weighted
condition: w_i = softmax(ρ_i / β)).

REDACTED firewall: imports nothing from REDACTED / REDACTED / REDACTED.

Lazy-import discipline: the heavy imports (``torch``, ``bitsandbytes``,
``transformers``, ``trl``, ``peft``) are deferred inside :meth:`train`
so that ``import stl_seed.training.backends.bnb`` succeeds on a
no-CUDA, no-bnb laptop.
"""

from __future__ import annotations

import json
import logging
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


def _check_cuda() -> None:
    """Raise a clear ImportError if no CUDA device is visible.

    bitsandbytes' 4-bit kernels require CUDA. CPU-only inference / training
    via bnb is not supported (the bnb fallback path on CPU is decoder-only
    and not what we want for SFT).
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "BNBBackend requires PyTorch. Install with `uv sync --extra cuda` on a CUDA host."
        ) from exc

    if not torch.cuda.is_available():
        raise ImportError(
            "BNBBackend requires CUDA. `torch.cuda.is_available()` is False. "
            "Use MLXBackend on Apple Silicon or run on a GPU host (RunPod)."
        )


# ---------------------------------------------------------------------------
# Backend.
# ---------------------------------------------------------------------------


class BNBBackend:
    """``TrainingBackend`` implementation backed by TRL + bitsandbytes."""

    name: str = "bnb"

    def __init__(self) -> None:
        # Construction must not import torch/bnb. Tests on CPU instantiate
        # this class to verify the protocol.
        self._loaded_model_cache: dict[str, Any] = {}

    # ----- public API ----------------------------------------------------

    def train(
        self,
        base_model: str,
        dataset: Any,
        config: TrainingConfig,
        output_dir: Path,
    ) -> TrainedCheckpoint:
        """QLoRA-finetune ``base_model`` on ``dataset`` via TRL.

        Parameters mirror :meth:`MLXBackend.train`. See ``base.py`` for
        ``TrainingConfig`` semantics.
        """
        _check_cuda()

        # Lazy heavy imports.
        try:
            import torch
            from peft import LoraConfig, prepare_model_for_kbit_training
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                TrainerCallback,
            )
            from trl import SFTConfig
        except ImportError as exc:  # pragma: no cover (env-specific)
            raise ImportError(
                "BNBBackend.train requires bitsandbytes + trl + peft + transformers. "
                "Install with `uv sync --extra cuda` on a CUDA host."
            ) from exc

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. BitsAndBytes 4-bit config (SERA QLoRA: NF4, double-quant, bf16 compute).
        if config.weight_format == "nf4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif config.weight_format == "int8":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        elif config.weight_format == "fp16":
            bnb_config = None  # vanilla load
        else:  # pragma: no cover (defended by TrainingConfig validation)
            raise ValueError(f"Unsupported weight_format: {config.weight_format}")

        # 2. Load base + tokenizer.
        log.info("Loading base model %s with weight_format=%s", base_model, config.weight_format)
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        if bnb_config is not None:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=True,
            )

        # 3. LoRA config (SERA QLoRA YAML field-for-field).
        peft_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=list(config.lora_target_modules),
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # 4. SFTConfig (TRL's HF-Trainer-args wrapper).
        sft_args = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            lr_scheduler_type=config.lr_schedule,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            optim="adamw_8bit" if config.use_8bit_optimizer else "adamw_torch",
            bf16=True,
            tf32=False,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=1,
            seed=config.seed,
            data_seed=config.seed,
            report_to=[],  # disable wandb/etc by default; project opts in via env
            max_seq_length=config.max_seq_length,
            packing=False,  # per-sample weighting requires no packing
            remove_unused_columns=False,  # we need the "weight" column intact
        )

        # 5. Loss-history callback.
        loss_history: list[float] = []

        class _LossLogger(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ARG002
                if logs and "loss" in logs:
                    loss_history.append(float(logs["loss"]))

        # 6. Construct the trainer with per-sample weighting.
        trainer = _WeightedSFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            peft_config=peft_config,
            args=sft_args,
            callbacks=[_LossLogger()],
        )

        # 7. Train.
        wall_start = time.time()
        try:
            trainer.train()
        except Exception:  # noqa: BLE001
            log.exception("BNB training failed; checkpoint will be incomplete.")
            raise
        wall = time.time() - wall_start

        # 8. Save LoRA adapters only (NOT the merged base — keeps artifact small).
        adapter_dir = output_dir / "adapter"
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        # 9. Provenance manifest.
        provenance = {
            "backend": "bnb",
            "base_model": base_model,
            "n_examples": len(dataset),
            "config": _config_to_dict(config),
            "wall_clock_seconds": wall,
            "n_loss_points": len(loss_history),
        }
        (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

        return TrainedCheckpoint(
            backend="bnb",
            model_path=adapter_dir,
            base_model=base_model,
            training_loss_history=loss_history,
            wall_clock_seconds=wall,
            metadata=provenance,
        )

    def load(self, checkpoint: TrainedCheckpoint) -> Callable[..., Any]:
        """Load a trained bnb LoRA checkpoint as a generation callable."""
        _check_cuda()
        try:
            import torch
            from peft import PeftModel
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                pipeline,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "BNBBackend.load requires bitsandbytes + transformers + peft."
            ) from exc

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            checkpoint.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base, str(checkpoint.model_path))
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint.model_path))
        gen = pipeline("text-generation", model=model, tokenizer=tokenizer)

        def _generate(prompt: str, **kwargs: Any) -> str:
            out = gen(prompt, **kwargs)
            if isinstance(out, list) and out and "generated_text" in out[0]:
                return out[0]["generated_text"]
            return str(out)

        return _generate


# ---------------------------------------------------------------------------
# Weighted SFTTrainer subclass.
# ---------------------------------------------------------------------------


def _build_weighted_sft_trainer_cls():  # pragma: no cover (CUDA-only)
    """Build the weighted SFTTrainer subclass lazily.

    The subclass closes over ``trl.SFTTrainer`` which we cannot import at
    module top-level (CPU-only test machines do not have ``trl`` /
    ``bitsandbytes`` available). We construct the class on first use.
    """
    from trl import SFTTrainer

    class WeightedSFTTrainer(SFTTrainer):
        """SFTTrainer that scales per-sample loss by the dataset's ``weight`` column.

        Implements the continuous-filter loss from theory.md §2:

            L_v(θ) = - Σ_i w_i · log p_θ(completion_i | prompt_i)

        For ``v ∈ {hard, quantile}`` every ``w_i = 1.0`` so this reduces
        exactly to the standard SFT loss.
        """

        def compute_loss(
            self,
            model,
            inputs,
            return_outputs: bool = False,
            num_items_in_batch=None,
        ):
            # ``inputs["weight"]`` is a per-sample tensor ferried by the
            # default collator (we set remove_unused_columns=False).
            weights = inputs.pop("weight", None)
            outputs = model(**inputs)
            # Token-level CE with reduction='none' per sample.
            from torch.nn import functional as torch_functional

            logits = outputs.logits
            labels = inputs.get("labels")
            if labels is None:
                # Fallback to TRL's default; weight is unused.
                loss = outputs.loss
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                per_tok = torch_functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="none",
                )
                # Reshape to (batch, seq-1) and average tokens within each sample.
                per_tok = per_tok.view(shift_labels.size())
                mask = (shift_labels != -100).float()
                per_sample = (per_tok * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                if weights is not None:
                    w = weights.to(per_sample.dtype).to(per_sample.device)
                    # Normalize weights per batch so the gradient scale is
                    # invariant to the choice of softmax temperature β.
                    w = w / w.sum().clamp_min(1e-8) * w.numel()
                    loss = (per_sample * w).mean()
                else:
                    loss = per_sample.mean()
            return (loss, outputs) if return_outputs else loss

    return WeightedSFTTrainer


def _make_weighted_sft_trainer(*args, **kwargs):  # pragma: no cover (CUDA-only)
    """Factory that constructs the weighted SFTTrainer subclass on demand."""
    cls = _build_weighted_sft_trainer_cls()
    return cls(*args, **kwargs)


# Public-ish alias kept for the in-module callsite. The PascalCase form
# documents that the call returns a Trainer-shaped object even though the
# binding is a function (the actual class is built inside the factory).
_WeightedSFTTrainer = _make_weighted_sft_trainer  # noqa: N816


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _config_to_dict(config: TrainingConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in config.__dict__.items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, list | tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out


__all__ = ["BNBBackend"]

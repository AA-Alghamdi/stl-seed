"""Abstract ``TrainingBackend`` protocol + shared dataclasses.

This module is the schema-level handshake between the training loop
(``stl_seed.training.loop``) and the concrete backends (``mlx``, ``bnb``).
Both backends construct :class:`TrainedCheckpoint` from the same
:class:`TrainingConfig` so the rest of the pipeline (eval harness,
hierarchical-Bayes analysis) can consume artifacts uniformly.

Defaults are sourced as follows (every numerical field is cited):

* QLoRA hyperparameters (``learning_rate``, ``lora_rank``, ``lora_alpha``,
  ``lora_dropout``, ``warmup_ratio``, ``num_epochs``, ``weight_decay``,
  ``gradient_accumulation_steps``, ``batch_size``, ``seed``, ``weight_format``)
  mirror SERA's ``unsloth_qwen3_moe_qlora.yaml`` (see
  ``paper/REDACTED.md`` §C.3). That file is the closest SERA analog to
  stl-seed's consumer-budget single-GPU envelope.
* ``lora_target_modules`` mirrors SERA's all-attention + all-MLP target
  set: the q/k/v/o projections plus the gate/up/down projections of the
  Qwen3 MLP block (paper/REDACTED.md §C.3).
* ``max_seq_length`` is reduced from SERA's 32768 to 8192 because control
  trajectories are far shorter than codebases (paper/REDACTED.md
  STEP-3 mapping table, "sequence_len: ADAPT").
* ``base_model`` defaults to ``Qwen/Qwen3-0.6B-Instruct`` — the smallest
  model in stl-seed's 3×3×2 sweep (theory.md §1).

Where SERA does not state a value (or where stl-seed deliberately diverges),
we fall back to QLoRA-paper defaults [Dettmers et al., arXiv:2305.14314]:
NF4 quantization, double-quant, bf16 compute. These are flagged in inline
comments.

This module imports nothing heavier than the standard library plus
``typing_extensions`` (for ``Protocol``). It must remain importable on every
target platform.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Configuration dataclass.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingConfig:
    """All hyperparameters consumed by a :class:`TrainingBackend`.

    Defaults track ``unsloth_qwen3_moe_qlora.yaml`` from SERA where possible
    (paper/REDACTED.md §C.3); stl-seed-specific deviations are commented
    inline. Every field is immutable so that a config object can be hashed
    and used as a Hydra-resolved ``omegaconf.DictConfig`` snapshot.
    """

    # --- Base model ----------------------------------------------------------
    # Default to the smallest Qwen3 in the 3×3×2 sweep (theory.md §1).
    base_model: str = "Qwen/Qwen3-0.6B-Instruct"

    # --- Optimizer schedule --------------------------------------------------
    # SERA QLoRA YAML: learning_rate: 5e-5 (5× higher than full-SFT 1e-5;
    # standard LoRA rule of thumb).
    learning_rate: float = 5e-5
    # SERA QLoRA YAML: lr_scheduler_type: cosine.
    lr_schedule: Literal["constant", "cosine", "linear"] = "cosine"
    # SERA QLoRA YAML: warmup_ratio: 0.1.
    warmup_ratio: float = 0.1
    # SERA QLoRA YAML: num_train_epochs: 3.
    num_epochs: int = 3
    # SERA QLoRA YAML: per_device_train_batch_size: 1.
    batch_size: int = 1
    # SERA QLoRA YAML: gradient_accumulation_steps: 4.
    gradient_accumulation_steps: int = 4

    # --- Sequence length -----------------------------------------------------
    # SERA QLoRA YAML uses 32768 for code; stl-seed control trajectories are
    # far shorter (paper/REDACTED.md STEP-3 mapping, "sequence_len: ADAPT").
    # 8192 is comfortable for H ≤ 200 control horizons with an 8-token-per-step
    # serialization budget.
    max_seq_length: int = 8192

    # --- LoRA ----------------------------------------------------------------
    # SERA QLoRA YAML: lora_r: 32, lora_alpha: 128 (alpha/r = 4×).
    lora_rank: int = 32
    lora_alpha: float = 128.0
    # SERA QLoRA YAML: target_modules = q/k/v/o + gate/up/down projections.
    # The exact attribute names are Qwen3-family canonical (validated against
    # the Qwen3 modeling file in transformers ≥ 4.45).
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    # SERA QLoRA YAML: lora_dropout: 0.0 (disabled for FlashAttention compat).
    lora_dropout: float = 0.0

    # --- Reproducibility -----------------------------------------------------
    # SERA QLoRA YAML: seed: 42.
    seed: int = 42

    # --- I/O -----------------------------------------------------------------
    output_dir: Path = field(default_factory=lambda: Path("runs/default"))

    # --- Quantization --------------------------------------------------------
    # SERA QLoRA YAML: load_in_4bit: true, compute bfloat16. The
    # ``weight_format`` field selects the on-disk / on-GPU representation:
    #
    # * ``"nf4"`` (default) — bitsandbytes NF4 with double-quant, bf16
    #   compute. Matches SERA's MoE-QLoRA recipe and the Dettmers QLoRA
    #   paper (arXiv:2305.14314 §3, "NF4").
    # * ``"int8"`` — bitsandbytes int8 (LLM.int8). Used when NF4 kernels
    #   are unavailable on a particular Ada/Hopper driver combo.
    # * ``"fp16"`` — no quantization; LoRA on top of fp16 base. Used by
    #   the MLX backend, which does not currently expose NF4 kernels on
    #   M-series GPUs (mlx_lm 0.20).
    weight_format: Literal["fp16", "nf4", "int8"] = "nf4"

    # --- Optimizer-state quantization ----------------------------------------
    # SERA QLoRA YAML: optim: adamw_8bit. Surfaced as a flag so the bnb
    # backend can pass it through to the HF Trainer.
    use_8bit_optimizer: bool = True

    # --- Weight decay --------------------------------------------------------
    # SERA QLoRA YAML: weight_decay: 0.01.
    weight_decay: float = 0.01

    def __post_init__(self) -> None:
        # Coerce a str output_dir into Path for Hydra-resolved configs.
        if isinstance(self.output_dir, str):
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.lora_rank <= 0:
            raise ValueError(f"lora_rank must be positive, got {self.lora_rank}")
        if self.lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {self.lora_alpha}")
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        if not (0.0 <= self.warmup_ratio <= 1.0):
            raise ValueError(f"warmup_ratio must be in [0, 1], got {self.warmup_ratio}")
        if not (0.0 <= self.lora_dropout < 1.0):
            raise ValueError(f"lora_dropout must be in [0, 1), got {self.lora_dropout}")


# ---------------------------------------------------------------------------
# Output dataclass.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainedCheckpoint:
    """A trained LoRA adapter on disk plus minimal training diagnostics.

    The ``model_path`` is whatever the backend wrote to disk. For the bnb
    backend this is a directory of LoRA adapter weights (NOT the base
    model); for the mlx backend this is the directory ``mlx_lm.lora``
    populates. Both backends record only the adapter, not the base model
    — the base is re-downloaded from HuggingFace at eval time.

    ``training_loss_history`` is the per-step training loss, length
    ``ceil(num_train_examples * num_epochs / (batch_size * grad_accum))``.
    Used for the loss curves in ``paper/figures/``.
    """

    backend: Literal["mlx", "bnb"]
    model_path: Path
    base_model: str
    training_loss_history: list[float]
    wall_clock_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocol.
# ---------------------------------------------------------------------------


@runtime_checkable
class TrainingBackend(Protocol):
    """Protocol defined verbatim from ``paper/architecture.md``.

    Both backends consume a HuggingFace ``Dataset`` of records with at minimum
    ``prompt: str``, ``completion: str``, and ``weight: float`` columns. The
    weight is consumed in the SFT loss as a per-sample scalar (see the
    backend implementations for the per-framework realization — bnb uses a
    custom collator + reduction='none' loss, mlx uses a custom loss closure).
    """

    name: str  # "mlx" or "bnb"

    def train(
        self,
        base_model: str,
        dataset: Any,  # datasets.Dataset, kept Any to defer the import
        config: TrainingConfig,
        output_dir: Path,
    ) -> TrainedCheckpoint:
        """Run SFT and return the resulting checkpoint."""
        ...

    def load(self, checkpoint: TrainedCheckpoint) -> Callable[..., Any]:
        """Load a checkpoint into an inference-ready callable.

        The returned callable is backend-specific (an ``mlx_lm`` generator
        for the mlx backend; a ``transformers.pipeline`` for the bnb
        backend). Eval code wraps both behind a uniform ``Policy`` shim
        in ``stl_seed.generation.policies``.
        """
        ...


__all__ = [
    "TrainingConfig",
    "TrainedCheckpoint",
    "TrainingBackend",
]

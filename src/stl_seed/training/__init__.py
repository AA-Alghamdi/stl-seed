"""SFT training subsystem for stl-seed.

Two interchangeable backends implement the same ``TrainingBackend`` protocol
(see ``paper/architecture.md`` §"Training backend interface"):

* :class:`stl_seed.training.backends.mlx.MLXBackend` — Apple Silicon (M-series)
  via ``mlx_lm.lora``. Used for local iteration on the M5 Pro pilot.
* :class:`stl_seed.training.backends.bnb.BNBBackend` — CUDA via
  ``trl.SFTTrainer`` + ``bitsandbytes`` 4-bit (NF4 + bf16 compute). Used for
  the canonical Phase 2 sweep on RunPod 4090 spot.

Both backends accept the same :class:`TrainingConfig` dataclass and produce a
:class:`TrainedCheckpoint`. Defaults mirror SERA's ``unsloth_qwen3_moe_qlora.yaml``
where applicable (the consumer-budget QLoRA path is the closest SERA analog
for stl-seed's hardware envelope; see ``paper/REDACTED.md`` §C.3).

The backend modules import their heavy native dependencies lazily inside
``train(...)``; importing this package on a CPU-only laptop without
``mlx`` or ``bitsandbytes`` installed must succeed without error.
"""

from __future__ import annotations

from stl_seed.training.backends.base import (
    TrainedCheckpoint,
    TrainingBackend,
    TrainingConfig,
)
from stl_seed.training.loop import train_with_filter

__all__ = [
    "TrainingBackend",
    "TrainingConfig",
    "TrainedCheckpoint",
    "train_with_filter",
]

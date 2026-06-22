"""Backend registry for stl-seed SFT training.

Backends are imported lazily by ``stl_seed.training.loop.train_with_filter``;
this module deliberately avoids eager imports of ``mlx`` or ``bitsandbytes``
so that the package remains importable on machines without either installed.
"""

from __future__ import annotations

from stl_seed.training.backends.base import (
    TrainedCheckpoint,
    TrainingBackend,
    TrainingConfig,
)

__all__ = [
    "TrainingBackend",
    "TrainingConfig",
    "TrainedCheckpoint",
]

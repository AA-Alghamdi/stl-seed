"""STL filter conditions for stl-seed (Subphase 1.3, A8).

Three `FilterCondition` implementations from `paper/theory.md` §2:

* `HardFilter`        — keep ρ > threshold; weights = 1.0
* `QuantileFilter`    — keep top-K%        ; weights = 1.0
* `ContinuousWeightedFilter` — keep all     ; weights = N · softmax(ρ / β)

Plus `build_sft_dataset(...)` in `dataset.py` that converts a filtered
trajectory list + weights into a HuggingFace `Dataset`.

is touched.
"""

from __future__ import annotations

from stl_seed.filter.conditions import (
    ContinuousWeightedFilter,
    FilterError,
    HardFilter,
    QuantileFilter,
)
from stl_seed.filter.dataset import build_sft_dataset

__all__ = [
    "ContinuousWeightedFilter",
    "FilterError",
    "HardFilter",
    "QuantileFilter",
    "build_sft_dataset",
]

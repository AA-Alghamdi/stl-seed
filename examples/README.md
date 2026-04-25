# `stl-seed` examples

Three runnable examples, each fully self-contained, that walk you from a
single ODE integration through filtered SFT training. Read in order.

| File | Reads | Writes | Wall clock |
|---|---|---|---|
| [`01_basic_simulation.py`](01_basic_simulation.py) | nothing | stdout | ~5 s |
| [`02_stl_filtering.py`](02_stl_filtering.py) | nothing | stdout | ~30 s |
| [`03_mlx_training_minimal.py`](03_mlx_training_minimal.py) | nothing | `runs/example_03/` (adapter, ~5 MB) + HF cache (~1.2 GB on first run) | ~1-2 min on M5 Pro |

Run them from the repository root:

```bash
uv run python examples/01_basic_simulation.py
uv run python examples/02_stl_filtering.py
uv run python examples/03_mlx_training_minimal.py
```

`01_basic_simulation.py` introduces the `Simulator` + `Trajectory` +
STL evaluator triple on the glucose-insulin family. It runs three
open-loop schedules (zero infusion, constant 1.5 U/h, fasting baseline)
against the ADA 2024 Time-in-Range spec and prints the signed
robustness margin for each.

`02_stl_filtering.py` builds on it: generates 50 trajectories under a
random + heuristic policy mix, scores them with the same spec, then
applies all three filter conditions (`HardFilter`, `QuantileFilter`,
`ContinuousWeightedFilter`) and prints the per-filter rho summary +
weight statistics.

`03_mlx_training_minimal.py` extends 02 to the SFT loop. It renders the
filtered trajectories as chat conversations, runs 10 LoRA iterations on
`mlx-community/Qwen3-0.6B-bf16`, saves the adapter, then reloads and
decodes one sample. Apple Silicon only — fails with a clear error on
non-arm64 hosts. For a longer (50-iter, parse-checked) version, see
[`scripts/smoke_test_mlx.py`](../scripts/smoke_test_mlx.py).

Once you have read all three, the next stop is
[`docs/api_reference.md`](../docs/api_reference.md) for the per-module
API surface, and [`paper/architecture.md`](../paper/architecture.md)
for the locked module layout and shared interfaces.

# Examples

Three runnable scripts, in order. Run from the repo root.

| File                         | Wall clock         | Writes                                                               |
| ---------------------------- | ------------------ | -------------------------------------------------------------------- |
| `01_basic_simulation.py`     | ~5 s               | stdout                                                               |
| `02_stl_filtering.py`        | ~30 s              | stdout                                                               |
| `03_mlx_training_minimal.py` | ~1-2 min on M5 Pro | `runs/example_03/` adapter (~5 MB) + HF cache (~1.2 GB on first run) |

```bash
uv run python examples/01_basic_simulation.py
uv run python examples/02_stl_filtering.py
uv run python examples/03_mlx_training_minimal.py
```

`01` runs three open-loop schedules on the Bergman/Dalla Man glucose-insulin model and scores each under the ADA 2024 Time-in-Range spec. `02` generates 50 trajectories, scores them, and applies all three filter conditions side by side. `03` filters with `HardFilter`, renders chat samples, runs 10 LoRA iterations on `mlx-community/Qwen3-0.6B-bf16`, and reloads the adapter to decode one sample. Apple Silicon only. non-arm64 hosts get a clear error before any download.

For the longer (50-iter, parse-checked) MLX path, see [`scripts/smoke_test_mlx.py`](../scripts/smoke_test_mlx.py).

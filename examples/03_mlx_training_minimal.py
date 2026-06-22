"""Example 03. Minimal MLX QLoRA training pipeline (Apple Silicon only).

End-to-end loop: generate 50 glucose-insulin trajectories, filter with
`HardFilter`, render as chat, run 10 LoRA iterations on
`mlx-community/Qwen3-0.6B-bf16`, reload + decode one sample.

Apple Silicon only (Darwin / arm64). Non-arm64 hosts get a clear error
before any model download. Wall clock ~1-2 min on M5 Pro; disk ~1.2 GB
HF cache + ~5 MB adapter under `runs/example_03/`. Run from the repo
root:

    uv run python examples/03_mlx_training_minimal.py
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import jax
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUN_DIR = _REPO_ROOT / "runs" / "example_03"

_MODEL_ID = "mlx-community/Qwen3-0.6B-bf16"
_SEED = 20260424
_N_TRAJECTORIES = 50
_SPEC_KEY = "glucose_insulin.tir.easy"
_TASK_NAME = "glucose_insulin"

_ITERS = 10
_BATCH_SIZE = 1
_GRAD_ACCUM = 4
_MAX_SEQ_LEN = 2048
_LORA_RANK = 8
_LORA_ALPHA = 16.0
_LORA_DROPOUT = 0.0
_LR = 2e-4
_LORA_TARGETS = ["self_attn.q_proj", "self_attn.v_proj"]


def _require_apple_silicon() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        sys.stderr.write(
            "[example 03] Requires Apple Silicon (Darwin / arm64); "
            f"detected {platform.system()} / {platform.machine()}.\n"
            "On Linux / CUDA, use the BNBBackend instead. see "
            "docs/README.md and src/stl_seed/training/backends/bnb.py for the equivalent path.\n"
        )
        raise SystemExit(1)
    try:
        import mlx_lm  # noqa: F401
    except ImportError as err:
        sys.stderr.write(
            "[example 03] The `mlx` extra is not installed.\nRun: uv sync --extra mlx --extra dev\n"
        )
        raise SystemExit(1) from err


def _generate_corpus():
    """Generate 50 trajectories and filter to the rho > 0 subset."""
    from stl_seed.filter.conditions import HardFilter
    from stl_seed.generation.runner import TrajectoryRunner
    from stl_seed.specs import REGISTRY
    from stl_seed.tasks.glucose_insulin import (
        BergmanParams,
        GlucoseInsulinSimulator,
        MealSchedule,
        default_normal_subject_initial_state,
    )

    sim = GlucoseInsulinSimulator()
    params = BergmanParams()
    runner = TrajectoryRunner(
        simulator=sim,
        spec_registry={_SPEC_KEY: REGISTRY[_SPEC_KEY]},
        output_store=None,
        initial_state=default_normal_subject_initial_state(params),
        horizon=sim.n_control_points,
        action_dim=1,
        aux={"meal_schedule": MealSchedule.empty()},
        sim_params=params,
    )
    print(f"  generating N={_N_TRAJECTORIES} trajectories ...")
    key = jax.random.key(_SEED)
    trajectories, metadata = runner.generate_trajectories(
        task=_TASK_NAME,
        n=_N_TRAJECTORIES,
        policy_mix={"random": 0.5, "heuristic": 0.5},
        key=key,
        spec_key=_SPEC_KEY,
    )
    rhos = np.array([m["robustness"] for m in metadata], dtype=np.float64)
    print(
        f"  generated {len(trajectories)} (rho range "
        f"[{rhos.min():+.3f}, {rhos.max():+.3f}], sat={(rhos > 0).mean():.0%})."
    )
    filt = HardFilter(rho_threshold=0.0, min_kept=1)
    kept_traj, weights = filt.filter(trajectories, rhos)
    print(f"  HardFilter kept {len(kept_traj)} trajectories (weights uniform).")
    return kept_traj, np.asarray(weights), REGISTRY[_SPEC_KEY]


def _render_chat(kept_traj, spec) -> list[dict]:
    """Convert each kept trajectory to a `{"messages": [...]}` chat record."""
    from stl_seed.training.tokenize import format_trajectory_as_text

    samples: list[dict] = []
    for traj in kept_traj:
        conv = format_trajectory_as_text(traj, spec, _TASK_NAME)
        samples.append(
            {
                "messages": [
                    {"role": "system", "content": conv["system"]},
                    {"role": "user", "content": conv["user"]},
                    {"role": "assistant", "content": conv["assistant"]},
                ]
            }
        )
    return samples


def _run_lora(samples: list[dict]) -> tuple[Path, list[float]]:
    """Run 10 LoRA iterations and write the adapter under _RUN_DIR."""
    import mlx.optimizers as optim
    from mlx_lm import load as mlx_load
    from mlx_lm.tuner.datasets import ChatDataset
    from mlx_lm.tuner.trainer import (
        CacheDataset,
        TrainingArgs,
        default_loss,
        iterate_batches,
        train,
    )
    from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters

    _RUN_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = _RUN_DIR / "_train.jsonl"
    with dataset_path.open("w") as fh:
        for s in samples:
            fh.write(json.dumps(s) + "\n")

    print(f"  loading {_MODEL_ID} ...")
    t0 = time.perf_counter()
    model, tokenizer = mlx_load(_MODEL_ID)
    print(f"  loaded in {time.perf_counter() - t0:.1f} s")

    n_blocks = len(getattr(model, "layers", []))
    model.freeze()
    lora_config = {
        "rank": _LORA_RANK,
        "scale": _LORA_ALPHA / _LORA_RANK,
        "dropout": _LORA_DROPOUT,
        "keys": list(_LORA_TARGETS),
    }
    linear_to_lora_layers(model, num_layers=n_blocks, config=lora_config, use_dora=False)
    print_trainable_parameters(model)

    samples_list = [
        json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()
    ]
    train_dataset = CacheDataset(ChatDataset(samples_list, tokenizer, mask_prompt=False))
    val_dataset = CacheDataset(ChatDataset(samples_list, tokenizer, mask_prompt=False))

    schedule = optim.cosine_decay(_LR, _ITERS)
    opt = optim.AdamW(learning_rate=schedule, weight_decay=0.01)
    args = TrainingArgs(
        batch_size=_BATCH_SIZE,
        iters=_ITERS,
        val_batches=0,
        steps_per_report=2,
        steps_per_eval=10**9,
        steps_per_save=10**9,
        max_seq_length=_MAX_SEQ_LEN,
        adapter_file=str(_RUN_DIR / "adapters.safetensors"),
        grad_checkpoint=True,
        grad_accumulation_steps=_GRAD_ACCUM,
    )

    losses: list[float] = []

    class _Callback:
        def on_train_loss_report(self, info):
            losses.append(float(info.get("train_loss", float("nan"))))

        def on_val_loss_report(self, info):
            pass

    print(f"  training for {_ITERS} iters ...")
    t0 = time.perf_counter()
    train(
        model=model,
        optimizer=opt,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        args=args,
        loss=default_loss,
        iterate_batches=iterate_batches,
        training_callback=_Callback(),
    )
    wall = time.perf_counter() - t0
    print(f"  training done in {wall:.1f} s")

    adapter_config = {
        "fine_tune_type": "lora",
        "num_layers": n_blocks,
        "lora_parameters": {
            "rank": _LORA_RANK,
            "scale": _LORA_ALPHA / _LORA_RANK,
            "dropout": _LORA_DROPOUT,
            "keys": list(_LORA_TARGETS),
        },
    }
    (_RUN_DIR / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2))
    print(f"  adapter saved to {_RUN_DIR}")
    return _RUN_DIR, losses


def _decode_one(adapter_dir: Path, samples: list[dict]) -> str:
    """Reload adapter and decode the first held-out sample (greedy)."""
    from mlx_lm import generate
    from mlx_lm import load as mlx_load
    from mlx_lm.sample_utils import make_sampler

    print(f"  reloading model + adapter from {adapter_dir} ...")
    model, tokenizer = mlx_load(_MODEL_ID, adapter_path=str(adapter_dir))
    sampler = make_sampler(temp=0.0)
    prompt = tokenizer.apply_chat_template(
        samples[0]["messages"][:-1], add_generation_prompt=True, tokenize=False
    )
    output = generate(
        model, tokenizer, prompt=prompt, max_tokens=400, sampler=sampler, verbose=False
    )
    return str(output)


def main() -> int:
    print("Example 03. minimal MLX QLoRA training pipeline")
    print(f"  device       : {platform.system()} / {platform.machine()}")
    print(f"  base model   : {_MODEL_ID}")
    print(f"  spec         : {_SPEC_KEY}")
    print(f"  iters        : {_ITERS}, batch={_BATCH_SIZE}, accum={_GRAD_ACCUM}")
    print()

    _require_apple_silicon()
    np.random.seed(_SEED)

    print("Step 1. generate + filter trajectories")
    kept_traj, _weights, spec = _generate_corpus()
    if not kept_traj:
        sys.stderr.write("No trajectories survived the filter; cannot train.\n")
        return 1

    print("\nStep 2. render chat samples")
    samples = _render_chat(kept_traj, spec)
    print(f"  {len(samples)} chat samples ready.")

    print("\nStep 3. LoRA training")
    adapter_dir, losses = _run_lora(samples)
    if losses:
        print(f"  loss curve: first={losses[0]:.4f}  last={losses[-1]:.4f}  min={min(losses):.4f}")
    else:
        print("  no loss reports captured (iters may have been too few).")

    print("\nStep 4. reload + decode one sample")
    output = _decode_one(adapter_dir, samples)
    preview = output[:200].replace("\n", " ")
    print(f"  decoded preview: {preview}{'...' if len(output) > 200 else ''}")

    print("\nDone. Adapter under runs/example_03/. For the full pilot")
    print("smoke test (50 iters + parse-eval), see scripts/smoke_test_mlx.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

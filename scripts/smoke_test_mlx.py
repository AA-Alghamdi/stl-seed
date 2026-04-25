"""A15 — MLX QLoRA smoke test on Qwen3-0.6B (Subphase 1.4 HARD CHECKPOINT).

Goal: prove the SFT training loop works end-to-end on M5 Pro / 48 GB.
NOT a convergence run — 50 iterations, rank-8 LoRA on q/v projections only.

Pass criteria (gates the hard checkpoint):
  (a) Training runs without crash.
  (b) Loss decreases (mean of last 5 iters < mean of first 5 iters).
  (c) No NaN/Inf in loss history.
  (d) >= 1 of 5 held-out generations parses as
        <state>v1,...,vn</state><action>u1,...,um</action>.

Pipeline:
  1. Load filtered glucose-insulin trajectory IDs/weights from
     `data/pilot/filtered_glucose_insulin_hard.parquet` (built by
     `scripts/filter_pilot.py`, A14). Rejoin with the source
     `TrajectoryStore` to recover full state/action arrays.
  2. Subsample to 100 trajectories (95 train + 5 held-out for parse-eval),
     deterministically via `numpy.random.default_rng(seed)`.
  3. Render each (trajectory, spec, task_name) into a chat-format
     `{"messages": [...]}` record using
     `stl_seed.training.tokenize.format_trajectory_as_text`.
  4. Write a JSONL the mlx_lm.tuner.datasets.ChatDataset can consume.
  5. Load `mlx-community/Qwen3-0.6B-bf16` via `mlx_lm.load`, attach LoRA
     (rank=8, alpha=16, q_proj+v_proj, on every transformer block) via
     `mlx_lm.tuner.utils.linear_to_lora_layers`.
  6. Drive `mlx_lm.tuner.trainer.train` with iters=50,
     grad_accumulation_steps=4, max_seq_length=2048, AdamW(lr=2e-4)
     wrapped in a 5-step linear warmup + 45-step cosine decay schedule.
     Capture per-report training loss in a `TrainingCallback`.
  7. Reload checkpoint with adapter via `mlx_lm.load(adapter_path=...)`,
     `mlx_lm.generate` on the 5 held-out user prompts, regex-check the
     output for at least one valid <state>...</state><action>...</action>
     block.
  8. Write `paper/REDACTED.md` with verdict + diagnostics.

Why we bypass `MLXBackend`:
  mlx_lm 0.31.3 changed `TrainingArgs` (no `learning_rate`, no
  `lr_schedule`, no `seed`, no `warmup_steps` — schedule lives on the
  optimizer; seed on `iterate_batches`) and `train()`'s signature (no
  `tokenizer` argument; the tokenizer is consumed by the dataset wrapper
  instead). The wrapper at `src/stl_seed/training/backends/mlx.py`
  expects the pre-0.20 API and will need a follow-up patch — we file
  that as a Phase-2 followup at the bottom of the report.

REDACTED firewall: imports only from `stl_seed.{filter,generation,specs,
tasks,training}` plus mlx / mlx_lm / numpy / pyarrow / rich / re / json.
No REDACTED / REDACTED / REDACTED / REDACTED.

Usage:
    cd /Users/abdullahalghamdi/stl-seed
    uv run python scripts/smoke_test_mlx.py 2>&1 | tee scripts/smoke_test_mlx.log
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ---------------------------------------------------------------------------
# Paths and constants.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "data" / "pilot"
_FILTERED_PARQUET = _DATA_DIR / "filtered_glucose_insulin_hard.parquet"
_RUNS_DIR = _REPO_ROOT / "runs" / "smoke_test_mlx"
_REPORT_PATH = _REPO_ROOT / "paper" / "REDACTED.md"

_SEED = 20260424
_N_TOTAL = 100  # 95 train + 5 held-out
_N_HELDOUT = 5
_ITERS = 50
_BATCH_SIZE = 1
_GRAD_ACCUM = 4
_MAX_SEQ_LEN = 2048
_LORA_RANK = 8
_LORA_ALPHA = 16.0
_LORA_DROPOUT = 0.0
_LR = 2e-4
_WARMUP_STEPS = 5  # 10% of 50 iters
_WEIGHT_DECAY = 0.01
# mlx_lm's `linear_to_lora_layers` matches keys against module names *relative
# to the TransformerBlock*, which for Qwen3-family models means the attention
# projections live at `self_attn.{q,k,v,o}_proj`. The bare key form (e.g.
# `"q_proj"`) — which the REDACTED firewall doc and the architecture spec lean on —
# does NOT match here; that's a known mlx_lm idiom.
_LORA_TARGETS = ["self_attn.q_proj", "self_attn.v_proj"]

# mlx-community ships pre-converted MLX-format weights; bf16 variant retains
# full precision (we do not need 4-bit for a 0.6B smoke test on 48GB unified
# memory). The non-bf16 variants are quantized — fine to LoRA on top of, but
# we want maximum gradient signal at this tiny scale.
_MODEL_ID = "mlx-community/Qwen3-0.6B-bf16"

# Spec for the held-out trajectories. Same one A13 used to score them.
_SPEC_KEY = "glucose_insulin.tir.easy"
_TASK_NAME = "glucose_insulin"

# Wall-clock guard — abort if training exceeds this budget.
_WALL_CLOCK_BUDGET_S = 600.0

console = Console()


# ---------------------------------------------------------------------------
# 0) Environment / determinism.
# ---------------------------------------------------------------------------


def _check_apple_silicon() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit(
            f"This smoke test requires Apple Silicon (Darwin/arm64); detected "
            f"{platform.system()}/{platform.machine()}."
        )


def _seed_everything(seed: int) -> None:
    """Deterministic seeding for numpy + mlx."""
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import mlx.core as mx

        mx.random.seed(seed)
    except ImportError:  # pragma: no cover (environment guard)
        pass


# ---------------------------------------------------------------------------
# 1) Build the SFT dataset.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Sample:
    messages: list[dict[str, str]]
    user: str
    assistant: str
    weight: float
    traj_id: str


def _load_filtered_glucose_dataset(
    n_total: int,
    rng: np.random.Generator,
) -> list[_Sample]:
    """Load filtered glucose-insulin trajectories and render to chat samples.

    The pipeline is:
      filtered_glucose_insulin_hard.parquet  (id + weight + rho)
        --(traj_id join)-->  TrajectoryStore  (full states/actions/times)
        --(format_trajectory_as_text + spec)-->  chat messages

    Returns
    -------
    list of _Sample of length `n_total`, sampled deterministically from the
    filtered set without replacement.
    """
    from stl_seed.generation.store import TrajectoryStore
    from stl_seed.specs import REGISTRY
    from stl_seed.training.tokenize import format_trajectory_as_text

    if not _FILTERED_PARQUET.exists():
        raise SystemExit(
            f"Missing filtered manifest at {_FILTERED_PARQUET}. Run scripts/filter_pilot.py first."
        )

    filtered_tbl = pq.read_table(_FILTERED_PARQUET)
    traj_ids = filtered_tbl.column("traj_id").to_pylist()
    weights = filtered_tbl.column("weight").to_pylist()
    rhos = filtered_tbl.column("robustness").to_pylist()
    n_filtered = len(traj_ids)
    console.print(
        f"  filtered manifest: {n_filtered:,} trajectories pass HardFilter "
        f"(min ρ kept = {min(rhos):+.3e})"
    )

    if n_total > n_filtered:
        raise SystemExit(f"Requested {n_total} samples but only {n_filtered} pass the filter.")

    # Deterministic without-replacement subsample.
    chosen_idx = rng.choice(n_filtered, size=n_total, replace=False)
    chosen_ids = [traj_ids[int(i)] for i in chosen_idx]
    chosen_weights = [float(weights[int(i)]) for i in chosen_idx]

    # Pull full trajectories from the source store.
    store = TrajectoryStore(_DATA_DIR)
    spec = REGISTRY[_SPEC_KEY]
    samples: list[_Sample] = []
    for tid, w in zip(chosen_ids, chosen_weights, strict=True):
        record = store.get_by_id(tid)
        if record is None:
            raise SystemExit(f"Trajectory id {tid} not found in store.")
        traj, _meta = record
        conv = format_trajectory_as_text(traj, spec, _TASK_NAME)
        messages = [
            {"role": "system", "content": conv["system"]},
            {"role": "user", "content": conv["user"]},
            {"role": "assistant", "content": conv["assistant"]},
        ]
        samples.append(
            _Sample(
                messages=messages,
                user=conv["user"],
                assistant=conv["assistant"],
                weight=w,
                traj_id=tid,
            )
        )
    return samples


def _write_chat_jsonl(path: Path, samples: list[_Sample]) -> None:
    """Write chat-format JSONL consumable by mlx_lm.tuner.datasets.ChatDataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for s in samples:
            fh.write(json.dumps({"messages": s.messages}) + "\n")


# ---------------------------------------------------------------------------
# 2) MLX training.
# ---------------------------------------------------------------------------


class _LossCollector:
    """TrainingCallback that captures per-report train losses in memory."""

    def __init__(self) -> None:
        self.train: list[tuple[int, float]] = []
        self.val: list[tuple[int, float]] = []

    def on_train_loss_report(self, train_info: dict) -> None:
        it = int(train_info.get("iteration", train_info.get("step", -1)))
        loss = float(train_info.get("train_loss", float("nan")))
        self.train.append((it, loss))

    def on_val_loss_report(self, val_info: dict) -> None:
        it = int(val_info.get("iteration", val_info.get("step", -1)))
        loss = float(val_info.get("val_loss", float("nan")))
        self.val.append((it, loss))


def _run_mlx_training(
    train_samples: list[_Sample],
    output_dir: Path,
) -> tuple[_LossCollector, float, dict[str, Any]]:
    """Drive mlx_lm.tuner.trainer.train with the smoke-test config.

    Returns
    -------
    (callback, wall_clock_seconds, env_info)
    """
    import mlx.core as mx
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

    output_dir.mkdir(parents=True, exist_ok=True)

    # Render dataset to JSONL for repro + future inspection.
    dataset_path = output_dir / "_train.jsonl"
    _write_chat_jsonl(dataset_path, train_samples)

    console.rule(f"[bold]Loading {_MODEL_ID}")
    t_load_start = time.perf_counter()
    model, tokenizer = mlx_load(_MODEL_ID)
    t_load_end = time.perf_counter()
    console.print(f"  model loaded in {t_load_end - t_load_start:.1f} s")

    n_blocks = len(getattr(model, "layers", []))
    console.print(
        f"  attaching LoRA (rank={_LORA_RANK}, alpha={_LORA_ALPHA}, "
        f"targets={_LORA_TARGETS}, blocks={n_blocks})"
    )
    # Freeze the base model first, then linear_to_lora_layers swaps target
    # nn.Linear with a LoRA-wrapped version whose adapter weights are
    # trainable. mlx_lm conventions: scale = alpha / rank.
    model.freeze()
    lora_config = {
        "rank": _LORA_RANK,
        "scale": _LORA_ALPHA / _LORA_RANK,
        "dropout": _LORA_DROPOUT,
        "keys": list(_LORA_TARGETS),
    }
    linear_to_lora_layers(
        model,
        num_layers=n_blocks,
        config=lora_config,
        use_dora=False,
    )
    print_trainable_parameters(model)

    # Build chat dataset (mlx_lm tokenizes via apply_chat_template; we do not
    # mask the prompt — the entire conversation is loss-bearing, matching
    # SERA's chat-template / per-message-train convention with all-true mask
    # for this 0.6B smoke test).
    samples_list = []
    with dataset_path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                samples_list.append(json.loads(line))
    # mlx_lm's `iterate_batches` expects each dataset element to be a
    # processed (token_ids, prompt_offset) tuple, not the raw chat dict.
    # `ChatDataset.process(...)` performs the apply_chat_template tokenization;
    # `CacheDataset` wraps any dataset with a `.process` method and lazily
    # applies it on first __getitem__ access. Without the CacheDataset wrap,
    # iterate_batches' `len(dataset[idx][0])` raises `KeyError: 0` because the
    # dict has no integer key 0.
    chat_dataset = CacheDataset(ChatDataset(samples_list, tokenizer, mask_prompt=False))
    val_dataset = CacheDataset(ChatDataset(samples_list, tokenizer, mask_prompt=False))

    # Optimizer with linear warmup → cosine decay schedule.
    warmup = optim.linear_schedule(0.0, _LR, _WARMUP_STEPS)
    cosine = optim.cosine_decay(_LR, _ITERS - _WARMUP_STEPS)
    schedule = optim.join_schedules([warmup, cosine], [_WARMUP_STEPS])
    opt = optim.AdamW(learning_rate=schedule, weight_decay=_WEIGHT_DECAY)

    train_args = TrainingArgs(
        batch_size=_BATCH_SIZE,
        iters=_ITERS,
        val_batches=0,  # we run our own held-out parse eval after training
        steps_per_report=5,
        steps_per_eval=10**9,  # disabled
        steps_per_save=10**9,  # we save manually at end via mx.save_safetensors
        max_seq_length=_MAX_SEQ_LEN,
        adapter_file=str(output_dir / "adapters.safetensors"),
        grad_checkpoint=True,
        grad_accumulation_steps=_GRAD_ACCUM,
    )

    callback = _LossCollector()
    console.rule("[bold]Starting MLX training")
    t_train_start = time.perf_counter()
    try:
        train(
            model=model,
            optimizer=opt,
            train_dataset=chat_dataset,
            val_dataset=val_dataset,
            args=train_args,
            loss=default_loss,
            iterate_batches=iterate_batches,
            training_callback=callback,
        )
    except Exception:
        # Per CLAUDE.md: never silently swallow training failures.
        console.print_exception()
        raise
    t_train_end = time.perf_counter()
    wall = t_train_end - t_train_start
    console.print(f"[green]training finished in {wall:.1f} s ({wall / 60:.2f} min)[/]")

    # Write `adapter_config.json` next to `adapters.safetensors` so that
    # `mlx_lm.load(adapter_path=...)` can rebuild the LoRA wrapper at eval
    # time. This mirrors what `mlx_lm.lora.train_model` does at
    # mlx_lm/lora.py:257 (save_config). Schema follows mlx_lm 0.31's
    # `load_adapters`: it reads `num_layers`, `lora_parameters` (rank, scale,
    # dropout, optional keys), and `fine_tune_type` from this JSON.
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
    (output_dir / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2))

    # Persist the LoRA adapter + a tiny manifest for the eval step.
    # The trainer already wrote `adapters.safetensors`; we add a manifest
    # describing what is in this run.
    manifest = {
        "base_model": _MODEL_ID,
        "lora_rank": _LORA_RANK,
        "lora_alpha": _LORA_ALPHA,
        "lora_targets": _LORA_TARGETS,
        "iters": _ITERS,
        "batch_size": _BATCH_SIZE,
        "grad_accumulation_steps": _GRAD_ACCUM,
        "max_seq_length": _MAX_SEQ_LEN,
        "lr": _LR,
        "warmup_steps": _WARMUP_STEPS,
        "weight_decay": _WEIGHT_DECAY,
        "seed": _SEED,
        "n_train_examples": len(train_samples),
        "wall_clock_seconds": wall,
        "loss_history": callback.train,
    }
    (output_dir / "provenance.json").write_text(json.dumps(manifest, indent=2))

    env_info = {
        "mlx_metal_available": bool(mx.metal.is_available()),
    }
    return callback, wall, env_info


# ---------------------------------------------------------------------------
# 3) Loss-decrease sanity.
# ---------------------------------------------------------------------------


def _loss_decrease_check(
    callback: _LossCollector,
) -> tuple[bool, dict[str, Any]]:
    """Apply pass criteria (b) and (c) on the captured loss history."""
    losses = [v for _, v in callback.train]
    diag: dict[str, Any] = {
        "n_reports": len(losses),
        "first_loss": losses[0] if losses else float("nan"),
        "last_loss": losses[-1] if losses else float("nan"),
        "min_loss": min(losses) if losses else float("nan"),
        "max_loss": max(losses) if losses else float("nan"),
    }
    if not losses:
        diag["reason"] = "no loss reports captured"
        return False, diag

    if any(math.isnan(x) or math.isinf(x) for x in losses):
        diag["reason"] = "NaN/Inf in loss history"
        return False, diag

    # The trainer reports every 5 iters; with 50 iters / 5-step report we
    # expect ~10 reports. We compare mean of first 2 vs last 2 (5 steps each
    # iteration, so 10 iterations of warmup smoothing). If the loss history
    # has fewer than 4 points we degrade to first vs last directly.
    if len(losses) >= 4:
        n = max(2, len(losses) // 5)
        first_mean = float(np.mean(losses[:n]))
        last_mean = float(np.mean(losses[-n:]))
        diag["first_window_mean"] = first_mean
        diag["last_window_mean"] = last_mean
        diag["window_size"] = n
        decreased = last_mean < first_mean
    else:
        decreased = losses[-1] < losses[0]

    diag["loss_decreased"] = decreased
    if not decreased:
        diag["reason"] = "loss did not decrease (mean last window >= mean first window)"
    return decreased, diag


def _print_loss_table(callback: _LossCollector) -> None:
    table = Table(
        title="[bold]Training loss curve",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("iter")
    table.add_column("train_loss", justify="right")
    for it, loss in callback.train:
        table.add_row(str(it), f"{loss:.4f}")
    console.print(table)


# ---------------------------------------------------------------------------
# 4) Held-out parse evaluation.
# ---------------------------------------------------------------------------


# Each block is `<state>...</state><action>...</action>`. Numbers may be in
# 4-sig-fig scientific notation per `tokenize._fmt_scalar` (e.g. `1.234e-01`)
# or, more loosely from a still-noisy 0.6B at 50 steps, plain decimals.
_BLOCK_RE = re.compile(
    r"<state>\s*([+\-0-9.eE,\s]+)\s*</state>\s*<action>\s*([+\-0-9.eE,\s]+)\s*</action>"
)


@dataclass
class _ParseResult:
    parses: bool
    n_blocks: int
    first_action_values: list[float]
    raw_output: str
    error: str | None = None


def _parse_output(text: str) -> _ParseResult:
    """Search for at least one <state>..</state><action>..</action> block.

    Records the first action vector for downstream stats.
    """
    matches = _BLOCK_RE.findall(text)
    if not matches:
        return _ParseResult(
            parses=False,
            n_blocks=0,
            first_action_values=[],
            raw_output=text,
            error="no <state>...</state><action>...</action> block found",
        )
    # Collect first-block action values; tolerate parse failures on individual
    # numbers without claiming a parse-success.
    state_str, action_str = matches[0]
    action_values: list[float] = []
    parse_error: str | None = None
    for tok in action_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            action_values.append(float(tok))
        except ValueError:
            parse_error = f"non-numeric token in action: {tok!r}"
            break
    parses = parse_error is None and len(action_values) >= 1
    return _ParseResult(
        parses=parses,
        n_blocks=len(matches),
        first_action_values=action_values,
        raw_output=text,
        error=parse_error,
    )


def _heldout_eval(
    held_samples: list[_Sample],
    output_dir: Path,
) -> list[_ParseResult]:
    """Generate completions for `held_samples` from the LoRA-adapted model
    and parse each output for at least one valid (state, action) block."""
    from mlx_lm import generate
    from mlx_lm import load as mlx_load
    from mlx_lm.sample_utils import make_sampler

    # mlx_lm.load(adapter_path=...) expects a *directory* containing both
    # `adapters.safetensors` and `adapter_config.json` (the latter is what
    # mlx_lm/tuner/utils.py:load_adapters reads to rebuild the LoRA wrapper
    # before loading the weights). We point it at the run directory which
    # _run_mlx_training writes both files into.
    safetensors_path = output_dir / "adapters.safetensors"
    config_path = output_dir / "adapter_config.json"
    if not safetensors_path.exists() or not config_path.exists():
        raise SystemExit(
            f"missing adapter artifacts at {output_dir} "
            f"(safetensors={safetensors_path.exists()}, config={config_path.exists()})"
        )
    console.rule("[bold]Loading trained model for held-out generation")
    model, tokenizer = mlx_load(_MODEL_ID, adapter_path=str(output_dir))

    # Conservative sampling: temperature=0.0 (greedy) to make the parse-eval
    # depend on what the model actually learned, not on lucky sampling.
    sampler = make_sampler(temp=0.0)

    results: list[_ParseResult] = []
    for i, sample in enumerate(held_samples):
        # Render only the system + user turns; let the model fill in the
        # assistant turn. We rebuild the chat template with
        # add_generation_prompt=True so the model sees `<|im_start|>assistant\n`.
        prompt_messages = sample.messages[:-1]
        prompt = tokenizer.apply_chat_template(
            prompt_messages, add_generation_prompt=True, tokenize=False
        )
        # Cap generation length: each (state, action) block is ~80-120
        # tokens; a horizon of ~12 control steps × 100 tokens = 1200 tokens
        # covers it. Keep this conservative for smoke-test speed.
        try:
            output = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=1200,
                sampler=sampler,
                verbose=False,
            )
        except Exception as e:  # noqa: BLE001
            console.print(f"  [red]heldout-{i}[/] generate raised: {e}")
            results.append(
                _ParseResult(
                    parses=False,
                    n_blocks=0,
                    first_action_values=[],
                    raw_output="",
                    error=f"{type(e).__name__}: {e}",
                )
            )
            continue
        result = _parse_output(output)
        console.print(
            f"  heldout-{i}: parses={result.parses}  "
            f"blocks={result.n_blocks}  "
            f"first_action={result.first_action_values[:3]}"
        )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# 5) Disk-usage accounting.
# ---------------------------------------------------------------------------


def _bytes_under(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    import contextlib

    total = 0
    for sub in path.rglob("*"):
        if sub.is_file():
            with contextlib.suppress(OSError):
                total += sub.stat().st_size
    return total


def _hf_cache_root() -> Path:
    # Honour HF_HOME / HF_HUB_CACHE if set; otherwise default location.
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
    if cache:
        return Path(cache).expanduser()
    return Path.home() / ".cache" / "huggingface" / "hub"


def _model_cache_size() -> int:
    """Best-effort size of the downloaded base model in the HF cache."""
    root = _hf_cache_root()
    # mlx-community/Qwen3-0.6B-bf16 → models--mlx-community--Qwen3-0.6B-bf16
    safe = _MODEL_ID.replace("/", "--")
    for prefix in ("models--", ""):
        cand = root / f"{prefix}{safe}"
        if cand.exists():
            return _bytes_under(cand)
    return 0


def _human_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.2f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.2f} GiB"


# ---------------------------------------------------------------------------
# 6) Report writer.
# ---------------------------------------------------------------------------


def _write_report(
    *,
    verdict: str,
    callback: _LossCollector,
    wall: float,
    setup_seconds: float,
    decrease_diag: dict[str, Any],
    parse_results: list[_ParseResult],
    issues: list[str],
    followups: list[str],
    train_samples: list[_Sample],
    held_samples: list[_Sample],
    env_info: dict[str, Any],
) -> None:
    n_parse = sum(1 for r in parse_results if r.parses)
    first_actions: list[float] = []
    for r in parse_results:
        if r.parses and r.first_action_values:
            first_actions.append(r.first_action_values[0])
    if first_actions:
        action_mean = float(np.mean(first_actions))
        action_min = float(np.min(first_actions))
        action_max = float(np.max(first_actions))
    else:
        action_mean = action_min = action_max = float("nan")

    losses = [v for _, v in callback.train]
    if losses:
        n_window = max(2, len(losses) // 5)
        first_mean = float(np.mean(losses[:n_window]))
        last_mean = float(np.mean(losses[-n_window:]))
    else:
        n_window = 0
        first_mean = last_mean = float("nan")

    model_size = _model_cache_size()
    ckpt_size = _bytes_under(_RUNS_DIR)

    lines: list[str] = []
    lines.append("# A15 — MLX QLoRA smoke test report")
    lines.append("")
    lines.append("**Date:** 2026-04-24")
    lines.append(
        f"**Host:** Apple Silicon ({platform.machine()}, {platform.system()} {platform.release()})"
    )
    lines.append(f"**MLX Metal available:** {env_info.get('mlx_metal_available', 'unknown')}")
    lines.append(f"**Verdict:** **{verdict}**")
    lines.append("")
    lines.append("## Model")
    lines.append("")
    lines.append(f"- **Identifier:** `{_MODEL_ID}`")
    lines.append(
        "- **Why this identifier:** `mlx-community/Qwen3-0.6B-bf16` ships pre-converted MLX-format "
        "weights at full bf16 precision. The smoke test's purpose is to verify the SFT loop end-to-end on "
        "M5 Pro / 48 GB unified memory, so we keep the base in bf16 (no quantization) to maximize gradient "
        "signal at this tiny scale. The original task brief mentioned `Qwen/Qwen3-0.6B` as a fallback path; "
        "in practice mlx_lm's `load()` would re-convert that on the fly, which is wasteful when an "
        "MLX-native variant exists. Note: the MLX backend's `weight_format` field documents that NF4 kernels "
        'are not exposed on M-series GPUs in mlx_lm 0.20+, so this is also the canonical "fp16/bf16 LoRA on '
        'top of base" path the wrapper anticipates.'
    )
    lines.append("")
    lines.append("## Hyperparameters")
    lines.append("")
    lines.append("| Field | Value | Source |")
    lines.append("|---|---|---|")
    lines.append(f"| iters | {_ITERS} | smoke-test cap |")
    lines.append(f"| batch_size | {_BATCH_SIZE} | SERA QLoRA YAML §C.3 |")
    lines.append(f"| gradient_accumulation_steps | {_GRAD_ACCUM} | SERA QLoRA YAML §C.3 |")
    lines.append(f"| max_seq_length | {_MAX_SEQ_LEN} | task brief override |")
    lines.append(f"| lora_rank | {_LORA_RANK} | smoke-test (smaller than SERA's 32) |")
    lines.append(f"| lora_alpha | {_LORA_ALPHA} | smoke-test (alpha/rank = 2×) |")
    lines.append(
        f"| lora_target_modules | {_LORA_TARGETS} | smoke-test (q_proj+v_proj only for speed) |"
    )
    lines.append(f"| lora_dropout | {_LORA_DROPOUT} | SERA QLoRA YAML §C.3 |")
    lines.append(f"| learning_rate | {_LR} | task brief override |")
    lines.append("| lr schedule | linear warmup → cosine | matches SERA cosine |")
    lines.append(f"| warmup_steps | {_WARMUP_STEPS} | 10% of iters |")
    lines.append(f"| weight_decay | {_WEIGHT_DECAY} | SERA QLoRA YAML §C.3 |")
    lines.append(f"| seed | {_SEED} | A13/A14 consistency |")
    lines.append("| optimizer | AdamW | mlx.optimizers default for LoRA |")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(
        f"- **Source:** `{_FILTERED_PARQUET.relative_to(_REPO_ROOT)}` (built by A14 / `scripts/filter_pilot.py`)."
    )
    lines.append(f"- **Spec:** `{_SPEC_KEY}`.")
    lines.append(
        f"- **Subsample:** {len(train_samples) + len(held_samples)} of 1344 filtered glucose-insulin trajectories (deterministic, seed={_SEED})."
    )
    lines.append(
        f"- **Split:** {len(train_samples)} train + {len(held_samples)} held-out for parse-eval."
    )
    lines.append("")
    lines.append("## Loss curve")
    lines.append("")
    lines.append(f"- **Initial loss (mean of first {n_window} reports):** {first_mean:.4f}")
    lines.append(f"- **Final loss (mean of last {n_window} reports):** {last_mean:.4f}")
    lines.append(f"- **Loss decreased:** {'YES' if decrease_diag.get('loss_decreased') else 'NO'}")
    lines.append("")
    lines.append("| iter | train_loss |")
    lines.append("|---:|---:|")
    for it, loss in callback.train:
        lines.append(f"| {it} | {loss:.4f} |")
    lines.append("")
    lines.append("## Held-out parse evaluation")
    lines.append("")
    lines.append(
        f"Trained model run on {len(held_samples)} held-out user prompts; greedy "
        f"decoding (temp=0.0); 1200-token max; output regex-checked for at least one "
        f"`<state>...</state><action>...</action>` block."
    )
    lines.append("")
    lines.append(f"- **Parse-success rate:** {n_parse} / {len(held_samples)}")
    lines.append(
        f"- **Mean first-action insulin rate (parsed):** {action_mean:.4e}  (range [{action_min:.4e}, {action_max:.4e}])"
    )
    lines.append("")
    lines.append("| held-out idx | parses | n_blocks | first_action |")
    lines.append("|---:|:---:|---:|:---|")
    for i, r in enumerate(parse_results):
        first_action_str = (
            ",".join(f"{x:.4e}" for x in r.first_action_values[:3])
            if r.first_action_values
            else "—"
        )
        lines.append(f"| {i} | {'Y' if r.parses else 'N'} | {r.n_blocks} | {first_action_str} |")
    lines.append("")
    lines.append("## Wall clock and disk usage")
    lines.append("")
    lines.append(
        f"- **Setup (mlx install + model download + dataset build):** {setup_seconds:.1f} s"
    )
    lines.append(f"- **Training (50 iters, batch=1×accum=4):** {wall:.1f} s ({wall / 60:.2f} min)")
    lines.append(f"- **HF cache size for {_MODEL_ID}:** {_human_bytes(model_size)}")
    lines.append(
        f"- **Run directory `{_RUNS_DIR.relative_to(_REPO_ROOT)}`:** {_human_bytes(ckpt_size)}"
    )
    lines.append("")
    lines.append("## Hard-checkpoint pass criteria")
    lines.append("")
    lines.append("| criterion | met? | notes |")
    lines.append("|---|:---:|---|")
    lines.append(
        f"| (a) training runs without crash | {'YES' if losses else 'NO'} | {len(losses)} loss reports captured |"
    )
    lines.append(
        f"| (b) final loss < initial loss | {'YES' if decrease_diag.get('loss_decreased') else 'NO'} | {first_mean:.4f} → {last_mean:.4f} |"
    )
    lines.append(
        f"| (c) no NaN/Inf in loss history | {'YES' if not any(math.isnan(x) or math.isinf(x) for x in losses) else 'NO'} | min={decrease_diag.get('min_loss', float('nan')):.4f}, max={decrease_diag.get('max_loss', float('nan')):.4f} |"
    )
    lines.append(
        f"| (d) parse-success rate ≥ 1/5 | {'YES' if n_parse >= 1 else 'NO'} | {n_parse}/{len(held_samples)} |"
    )
    lines.append("")
    lines.append(f"**VERDICT: {verdict}**")
    lines.append("")
    lines.append("## Issues encountered")
    lines.append("")
    if issues:
        for it in issues:
            lines.append(f"- {it}")
    else:
        lines.append("- None blocking the smoke test.")
    lines.append("")
    lines.append("## Phase-2 followups")
    lines.append("")
    if followups:
        for it in followups:
            lines.append(f"- {it}")
    else:
        lines.append("- None identified during this run.")
    lines.append("")

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text("\n".join(lines))
    console.print(f"[green]wrote {_REPORT_PATH.relative_to(_REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> int:
    console.print(
        Panel.fit(
            "A15 — MLX QLoRA smoke test on Qwen3-0.6B-bf16\n"
            f"Model: {_MODEL_ID}\n"
            f"Iters: {_ITERS}  | batch×accum: {_BATCH_SIZE}×{_GRAD_ACCUM}  | "
            f"LoRA rank/alpha: {_LORA_RANK}/{_LORA_ALPHA}  | targets: {_LORA_TARGETS}",
            title="[bold]Subphase 1.4 hard checkpoint",
        )
    )

    _check_apple_silicon()
    _seed_everything(_SEED)

    setup_start = time.perf_counter()

    # 1) Build dataset.
    console.rule("[bold]Step 1 — Build SFT dataset")
    rng = np.random.default_rng(_SEED)
    samples = _load_filtered_glucose_dataset(_N_TOTAL, rng)
    console.print(f"  built {len(samples)} chat samples; example user-turn preview:")
    console.print(f"    {samples[0].user[:160]}...")
    console.print("  example assistant-turn preview:")
    console.print(f"    {samples[0].assistant[:160]}...")

    # Reserve last 5 deterministically-shuffled samples as held-out.
    held_samples = samples[-_N_HELDOUT:]
    train_samples = samples[: len(samples) - _N_HELDOUT]
    console.print(f"  split: {len(train_samples)} train / {len(held_samples)} held-out")
    setup_seconds = time.perf_counter() - setup_start

    # 2) Train.
    console.rule("[bold]Step 2 — MLX QLoRA fine-tuning")
    issues: list[str] = []
    followups: list[str] = []
    try:
        callback, wall, env_info = _run_mlx_training(train_samples, _RUNS_DIR)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]TRAINING FAILED:[/] {type(e).__name__}: {e}")
        callback = _LossCollector()
        wall = 0.0
        env_info = {}
        issues.append(f"Training crash: {type(e).__name__}: {e}")
        verdict = "FAIL"
        # Write partial report so the next agent has something to read.
        _write_report(
            verdict=verdict,
            callback=callback,
            wall=wall,
            setup_seconds=setup_seconds,
            decrease_diag={"reason": "training crashed before any reports"},
            parse_results=[],
            issues=issues,
            followups=[
                "MLXBackend wrapper at src/stl_seed/training/backends/mlx.py "
                "expects pre-0.20 mlx_lm API (TrainingArgs.{learning_rate,"
                "lr_schedule,warmup_steps,seed} no longer exist; train() no "
                "longer takes tokenizer). Patch the wrapper before RunPod "
                "training, OR scope the wrapper as bnb-only and keep the "
                "smoke-test path in scripts/.",
            ],
            train_samples=train_samples,
            held_samples=held_samples,
            env_info=env_info,
        )
        return 1
    if wall > _WALL_CLOCK_BUDGET_S:
        issues.append(f"Training exceeded {_WALL_CLOCK_BUDGET_S:.0f}s budget (took {wall:.1f}s)")

    # 3) Loss-decrease check.
    console.rule("[bold]Step 3 — Loss decrease verification")
    decreased, decrease_diag = _loss_decrease_check(callback)
    _print_loss_table(callback)
    console.print(f"  loss decreased: {decreased}  diag: {decrease_diag}")

    # 4) Held-out parse evaluation.
    console.rule("[bold]Step 4 — Held-out parse evaluation")
    parse_results = _heldout_eval(held_samples, _RUNS_DIR)
    n_parse = sum(1 for r in parse_results if r.parses)
    console.print(f"  parse-success rate: {n_parse}/{len(parse_results)}")

    # 5) Verdict.
    losses = [v for _, v in callback.train]
    crit_a = bool(losses)
    crit_b = decreased
    crit_c = not any(math.isnan(x) or math.isinf(x) for x in losses)
    crit_d = n_parse >= 1
    verdict = "PASS" if (crit_a and crit_b and crit_c and crit_d) else "FAIL"
    if not crit_a:
        issues.append("Criterion (a) failed: no loss reports captured")
    if not crit_b:
        issues.append(
            f"Criterion (b) failed: {decrease_diag.get('reason', 'loss did not decrease')}"
        )
    if not crit_c:
        issues.append("Criterion (c) failed: NaN/Inf in loss history")
    if not crit_d:
        issues.append(f"Criterion (d) failed: parse-success {n_parse}/{len(parse_results)} < 1")

    # Phase-2 followups (always recorded, regardless of pass/fail).
    followups.append(
        "MLXBackend wrapper at src/stl_seed/training/backends/mlx.py is "
        "pinned to pre-0.20 mlx_lm: it constructs TrainingArgs with "
        "`learning_rate`/`lr_schedule`/`warmup_steps`/`seed` (removed in "
        "0.31), and passes `tokenizer` to train() (removed). The wrapper "
        "needs a follow-up patch to pass the schedule on the optimizer and "
        "build a ChatDataset before calling train(). Scope: ~50 LOC in "
        "MLXBackend.train; the smoke test demonstrates the post-patch "
        "shape end-to-end."
    )
    followups.append(
        "RunPod canonical training (Phase 2) uses the bnb backend, not MLX, "
        "so the MLX wrapper bug above does not block Phase 2 — but A16 (the "
        "next subphase 1.4 agent) should validate the bnb path with the "
        "same dataset shape produced here."
    )
    followups.append(
        "Add an integration test under tests/training/ that builds a tiny "
        "synthetic dataset, runs 5 mlx_lm iterations, and asserts loss "
        "decreased; same shape as this smoke test but with a 2-layer model "
        "stub for CPU CI."
    )

    _write_report(
        verdict=verdict,
        callback=callback,
        wall=wall,
        setup_seconds=setup_seconds,
        decrease_diag=decrease_diag,
        parse_results=parse_results,
        issues=issues,
        followups=followups,
        train_samples=train_samples,
        held_samples=held_samples,
        env_info=env_info,
    )

    # Final console verdict.
    color = "green" if verdict == "PASS" else "red"
    console.print(
        Panel(
            f"[bold {color}]VERDICT: {verdict}[/]\n"
            f"loss: {decrease_diag.get('first_window_mean', 'n/a')} → "
            f"{decrease_diag.get('last_window_mean', 'n/a')}\n"
            f"parse-success: {n_parse}/{len(parse_results)}\n"
            f"wall-clock: {wall:.1f} s",
            title="A15 hard-checkpoint result",
            border_style=color,
        )
    )
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

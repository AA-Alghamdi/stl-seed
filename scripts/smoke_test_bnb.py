"""A23 — bitsandbytes 4-bit QLoRA smoke test on Qwen3-0.6B (RunPod-only).

Mirror of ``scripts/smoke_test_mlx.py`` for the canonical Phase-2 backend
(``stl_seed.training.backends.bnb``). Designed to be the single command
the user runs on a fresh RunPod 4090 pod to confirm the bnb path is
functional before kicking off the 18-cell sweep.

Pass criteria (gates the bnb backend for Phase 2):

  (a) Training runs without crash.
  (b) Loss decreases (mean of last window < mean of first window).
  (c) No NaN/Inf in loss history.
  (d) >= 1 of 5 held-out generations parses as
      ``<state>v1,...,vn</state><action>u1,...,um</action>``.

Pipeline (intentionally identical in shape to the MLX smoke test so the
two backends produce comparable artifacts):

  1. Load filtered glucose-insulin trajectory IDs/weights from
     ``data/pilot/filtered_glucose_insulin_hard.parquet`` (built by
     ``scripts/filter_pilot.py``, A14). Rejoin with the source
     ``TrajectoryStore`` to recover full state/action arrays.
  2. Subsample 100 trajectories (95 train + 5 held-out for parse-eval),
     deterministically via ``numpy.random.default_rng(seed)``.
  3. Render each trajectory into a chat-format ``{"messages": [...]}``
     record using ``stl_seed.training.tokenize.format_trajectory_as_text``.
  4. Materialize a ``datasets.Dataset`` with ``messages`` + ``weight``
     columns. The bnb backend's ``_WeightedSFTTrainer`` consumes the
     weight column.
  5. Build a :class:`stl_seed.training.backends.base.TrainingConfig`
     scaled down for a smoke test (50 train steps via
     ``num_epochs=1``, batch=1, grad-accum=4, rank-8 LoRA on q/v
     projections, NF4 quantization, bf16 compute, lr 2e-4, 5-step
     warmup, cosine decay). All knobs cited in the report.
  6. Drive ``BNBBackend.train(...)``; capture the loss history.
  7. Reload the LoRA adapter with ``BNBBackend.load(...)``; greedy-decode
     the 5 held-out user prompts; parse for at least one valid block.
  8. Write ``paper/smoke_test_bnb_report.md`` with verdict + diagnostics.

Why this script raises on non-CUDA hosts:
  bitsandbytes 4-bit kernels require CUDA (see
  ``stl_seed.training.backends.bnb._check_cuda``). The script is designed
  to run on RunPod, NOT on the user's M5 Pro. We import the heavy stack
  lazily so the file imports cleanly during CI and during the
  pre-flight check on macOS, but the moment ``main()`` runs it asserts
  CUDA is available and exits with a helpful error.

REDACTED firewall: imports only from
``stl_seed.{filter,generation,specs,tasks,training}`` plus numpy /
pyarrow / rich / re / json / datasets. No REDACTED / REDACTED /
REDACTED / REDACTED.

Usage (on RunPod):
    cd /workspace/stl-seed
    uv run python scripts/smoke_test_bnb.py 2>&1 | tee scripts/smoke_test_bnb.log

Cost estimate (4090 spot @ $0.34/hr): ~5 min wall-clock end-to-end (~30s
model download + ~3 min training + ~1 min held-out gen) ≈ $0.03. The
30-min budget cited in docker/runpod_README.md includes the first-run
HF-cache warm-up.
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
_RUNS_DIR = _REPO_ROOT / "runs" / "smoke_test_bnb"
_REPORT_PATH = _REPO_ROOT / "paper" / "smoke_test_bnb_report.md"

_SEED = 20260424
_N_TOTAL = 100
_N_HELDOUT = 5

# Hyperparameters — chosen to mirror the MLX smoke test (A15) so the two
# backends produce directly comparable artifacts. The bnb backend computes
# total training steps from num_epochs * len(dataset) / (batch * accum), so
# we scale num_epochs to land near 50 optimizer steps on a 95-row train
# set: 50 * (1 * 4) / 95 ≈ 2.1, so num_epochs=2 → 47 steps.
_NUM_EPOCHS = 2
_BATCH_SIZE = 1
_GRAD_ACCUM = 4
_MAX_SEQ_LEN = 2048
_LORA_RANK = 8
_LORA_ALPHA = 16.0
_LORA_DROPOUT = 0.0
_LORA_TARGETS = ["q_proj", "v_proj"]  # bnb/peft wants bare names, not dotted
_LR = 2e-4
_WARMUP_RATIO = 0.10
_WEIGHT_DECAY = 0.01

# Canonical CUDA student. Falls back to mlx-community variant if the user
# pre-cached only that one — but the bnb path needs the HF-format weights
# (the mlx-community variant ships .safetensors but in MLX layout, which
# transformers cannot consume).
_PRIMARY_MODEL = "Qwen/Qwen3-0.6B-Instruct"
_FALLBACK_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

_SPEC_KEY = "glucose_insulin.tir.easy"
_TASK_NAME = "glucose_insulin"

# Wall-clock guard — abort if training exceeds this budget (RunPod cost cap).
_WALL_CLOCK_BUDGET_S = 1800.0

console = Console()


# ---------------------------------------------------------------------------
# 0) Environment / determinism / CUDA gate.
# ---------------------------------------------------------------------------


def _require_cuda() -> None:
    """Assert CUDA is visible. Helpful error message if not.

    This script intentionally does NOT run on macOS — bitsandbytes 4-bit
    kernels are CUDA-only. On the user's M5 Pro, the corresponding
    ``smoke_test_mlx.py`` is the right tool.
    """
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "scripts/smoke_test_bnb.py requires the cuda extra "
            "(`uv sync --extra cuda`). PyTorch is not importable.\n"
            f"  underlying error: {exc}"
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit(
            "scripts/smoke_test_bnb.py requires CUDA, but "
            "torch.cuda.is_available() is False.\n"
            f"  detected platform: {platform.system()}/{platform.machine()}\n"
            "  on Apple Silicon, run scripts/smoke_test_mlx.py instead.\n"
            "  on Linux, verify nvidia-smi and that the cuda extra installed "
            "the GPU build of torch (uv sync --extra cuda)."
        )


def _seed_everything(seed: int) -> None:
    """Deterministic seeding for numpy + torch + python's hash randomization.

    PyTorch's CUDA kernels are not bit-deterministic by default
    (``torch.use_deterministic_algorithms(True)`` would force it but at a
    large speed cost and with extra constraints — e.g. it disallows
    non-deterministic matmul kernels in cuBLAS). We document this in
    paper/reproducibility.md as a known source of non-determinism.
    """
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover (handled by _require_cuda)
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
    """Load filtered glucose-insulin trajectories, render to chat samples.

    Identical to ``smoke_test_mlx.py`` so the dataset shape is held
    constant across backends.
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

    chosen_idx = rng.choice(n_filtered, size=n_total, replace=False)
    chosen_ids = [traj_ids[int(i)] for i in chosen_idx]
    chosen_weights = [float(weights[int(i)]) for i in chosen_idx]

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


def _build_hf_dataset(samples: list[_Sample], tokenizer: Any):
    """Build a ``datasets.Dataset`` shaped for trl.SFTTrainer with weights.

    Each row carries:
      * ``text``: the rendered chat (apply_chat_template with all turns).
      * ``weight``: per-sample scalar passed to _WeightedSFTTrainer.

    We pre-render with ``apply_chat_template`` so SFTTrainer's internal
    formatting becomes a no-op (TRL >= 0.12 detects the ``text`` column
    and uses it directly when no ``formatting_func`` is supplied).
    """
    from datasets import Dataset

    rows: list[dict[str, Any]] = []
    for s in samples:
        text = tokenizer.apply_chat_template(
            s.messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        rows.append({"text": text, "weight": float(s.weight)})
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# 2) bnb training.
# ---------------------------------------------------------------------------


def _resolve_model_id() -> str:
    """Pick the smallest available HF-format student.

    Honours ``STL_SEED_SMOKE_MODEL`` env override; otherwise tries the
    primary, then the fallback. We do a HEAD-style probe via
    ``huggingface_hub.repo_info`` to fail fast with a clear error if both
    are unreachable (e.g. user has no HF token and model is gated).
    """
    override = os.environ.get("STL_SEED_SMOKE_MODEL")
    if override:
        console.print(f"  model: {override} (from STL_SEED_SMOKE_MODEL)")
        return override

    from huggingface_hub import repo_info
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    for candidate in (_PRIMARY_MODEL, _FALLBACK_MODEL):
        try:
            repo_info(candidate, repo_type="model")
            console.print(f"  model: {candidate} (resolved)")
            return candidate
        except (RepositoryNotFoundError, HfHubHTTPError) as exc:
            console.print(f"  [yellow]model {candidate} unreachable: {exc}[/]")
            continue
    raise SystemExit(
        f"Neither {_PRIMARY_MODEL} nor {_FALLBACK_MODEL} is reachable. Check network / HF auth."
    )


def _run_bnb_training(
    train_samples: list[_Sample],
    output_dir: Path,
    model_id: str,
) -> tuple[list[float], float, dict[str, Any]]:
    """Drive BNBBackend.train with the smoke-test config.

    Returns
    -------
    (loss_history, wall_clock_seconds, env_info)
    """
    import torch
    from transformers import AutoTokenizer

    from stl_seed.training.backends.base import TrainingConfig
    from stl_seed.training.backends.bnb import BNBBackend

    output_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer first — we need its chat template to materialize the dataset
    # before handing off to the backend.
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    console.rule("[bold]Building HF Dataset")
    hf_dataset = _build_hf_dataset(train_samples, tokenizer)
    console.print(f"  built dataset of {len(hf_dataset)} rows")

    config = TrainingConfig(
        base_model=model_id,
        learning_rate=_LR,
        lr_schedule="cosine",
        warmup_ratio=_WARMUP_RATIO,
        num_epochs=_NUM_EPOCHS,
        batch_size=_BATCH_SIZE,
        gradient_accumulation_steps=_GRAD_ACCUM,
        max_seq_length=_MAX_SEQ_LEN,
        lora_rank=_LORA_RANK,
        lora_alpha=_LORA_ALPHA,
        lora_target_modules=_LORA_TARGETS,
        lora_dropout=_LORA_DROPOUT,
        seed=_SEED,
        output_dir=output_dir,
        weight_format="nf4",
        use_8bit_optimizer=True,
        weight_decay=_WEIGHT_DECAY,
    )

    backend = BNBBackend()
    console.rule("[bold]Starting bnb QLoRA training")
    t_start = time.perf_counter()
    try:
        checkpoint = backend.train(
            base_model=model_id,
            dataset=hf_dataset,
            config=config,
            output_dir=output_dir,
        )
    except Exception:
        # Per CLAUDE.md: never silently swallow training failures.
        console.print_exception()
        raise
    wall = time.perf_counter() - t_start
    console.print(f"[green]training finished in {wall:.1f} s ({wall / 60:.2f} min)[/]")

    env_info = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": torch.cuda.get_device_capability(0),
    }

    # Persist a smoke-test manifest alongside the backend's provenance.json.
    manifest = {
        "base_model": model_id,
        "lora_rank": _LORA_RANK,
        "lora_alpha": _LORA_ALPHA,
        "lora_targets": _LORA_TARGETS,
        "num_epochs": _NUM_EPOCHS,
        "batch_size": _BATCH_SIZE,
        "grad_accumulation_steps": _GRAD_ACCUM,
        "max_seq_length": _MAX_SEQ_LEN,
        "lr": _LR,
        "warmup_ratio": _WARMUP_RATIO,
        "weight_decay": _WEIGHT_DECAY,
        "seed": _SEED,
        "n_train_examples": len(train_samples),
        "wall_clock_seconds": wall,
        "loss_history": checkpoint.training_loss_history,
        "env": env_info,
    }
    (output_dir / "smoke_test_manifest.json").write_text(json.dumps(manifest, indent=2))

    return list(checkpoint.training_loss_history), wall, env_info


# ---------------------------------------------------------------------------
# 3) Loss-decrease sanity.
# ---------------------------------------------------------------------------


def _loss_decrease_check(losses: list[float]) -> tuple[bool, dict[str, Any]]:
    """Apply pass criteria (b) and (c) on the captured loss history."""
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
        diag["reason"] = "loss did not decrease"
    return decreased, diag


def _print_loss_table(losses: list[float]) -> None:
    table = Table(
        title="[bold]Training loss curve",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("report#")
    table.add_column("train_loss", justify="right")
    for i, loss in enumerate(losses):
        table.add_row(str(i), f"{loss:.4f}")
    console.print(table)


# ---------------------------------------------------------------------------
# 4) Held-out parse evaluation.
# ---------------------------------------------------------------------------


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
    matches = _BLOCK_RE.findall(text)
    if not matches:
        return _ParseResult(
            parses=False,
            n_blocks=0,
            first_action_values=[],
            raw_output=text,
            error="no <state>...</state><action>...</action> block found",
        )
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
    model_id: str,
) -> list[_ParseResult]:
    """Generate completions for held-out prompts; parse each output."""
    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    adapter_dir = output_dir / "adapter"
    if not adapter_dir.exists():
        raise SystemExit(f"Adapter directory not found at {adapter_dir}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results: list[_ParseResult] = []
    for i, sample in enumerate(held_samples):
        prompt_messages = sample.messages[:-1]
        prompt = tokenizer.apply_chat_template(
            prompt_messages, add_generation_prompt=True, tokenize=False
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=1200,
                    do_sample=False,  # greedy; matches MLX smoke test
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
            output = tokenizer.decode(new_ids, skip_special_tokens=True)
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
# 5) Report writer.
# ---------------------------------------------------------------------------


def _write_report(
    *,
    verdict: str,
    losses: list[float],
    wall: float,
    setup_seconds: float,
    decrease_diag: dict[str, Any],
    parse_results: list[_ParseResult],
    issues: list[str],
    followups: list[str],
    train_samples: list[_Sample],
    held_samples: list[_Sample],
    env_info: dict[str, Any],
    model_id: str,
) -> None:
    n_parse = sum(1 for r in parse_results if r.parses)

    if losses:
        n_window = max(2, len(losses) // 5)
        first_mean = float(np.mean(losses[:n_window]))
        last_mean = float(np.mean(losses[-n_window:]))
    else:
        n_window = 0
        first_mean = last_mean = float("nan")

    lines: list[str] = []
    lines.append("# A23 — bnb QLoRA smoke test report")
    lines.append("")
    lines.append("**Date:** generated at run-time")
    lines.append(
        f"**Host:** {platform.system()} {platform.machine()} "
        f"({env_info.get('device_name', 'no-GPU')})"
    )
    lines.append(
        f"**torch:** {env_info.get('torch_version', 'n/a')} "
        f"(CUDA {env_info.get('cuda_version', 'n/a')})"
    )
    lines.append(f"**Verdict:** **{verdict}**")
    lines.append("")
    lines.append("## Model")
    lines.append("")
    lines.append(f"- **Identifier:** `{model_id}`")
    lines.append(
        "- **Why this identifier:** the bnb backend consumes HF-format weights "
        f"directly; `{_PRIMARY_MODEL}` is the canonical Phase-2 student. The "
        f"fallback `{_FALLBACK_MODEL}` is used only if the primary is "
        "temporarily unreachable. Override with the `STL_SEED_SMOKE_MODEL` "
        "env var if you want to validate a different base."
    )
    lines.append("")
    lines.append("## Hyperparameters")
    lines.append("")
    lines.append("| Field | Value | Source |")
    lines.append("|---|---|---|")
    lines.append(f"| num_epochs | {_NUM_EPOCHS} | smoke-test cap (~50 steps) |")
    lines.append(f"| batch_size | {_BATCH_SIZE} | SERA QLoRA YAML §C.3 |")
    lines.append(f"| gradient_accumulation_steps | {_GRAD_ACCUM} | SERA QLoRA YAML §C.3 |")
    lines.append(f"| max_seq_length | {_MAX_SEQ_LEN} | smoke-test override |")
    lines.append(f"| lora_rank | {_LORA_RANK} | smoke-test (smaller than SERA's 32) |")
    lines.append(f"| lora_alpha | {_LORA_ALPHA} | smoke-test (alpha/rank = 2x) |")
    lines.append(
        f"| lora_target_modules | {_LORA_TARGETS} | smoke-test (q_proj+v_proj only for speed) |"
    )
    lines.append(f"| lora_dropout | {_LORA_DROPOUT} | SERA QLoRA YAML §C.3 |")
    lines.append(f"| learning_rate | {_LR} | smoke-test override (matches A15) |")
    lines.append("| lr schedule | cosine + linear warmup | matches SERA cosine |")
    lines.append(f"| warmup_ratio | {_WARMUP_RATIO} | SERA QLoRA YAML §C.3 |")
    lines.append(f"| weight_decay | {_WEIGHT_DECAY} | SERA QLoRA YAML §C.3 |")
    lines.append(f"| seed | {_SEED} | A13/A14/A15 consistency |")
    lines.append("| optimizer | adamw_8bit | SERA QLoRA YAML §C.3 |")
    lines.append("| weight_format | nf4 (double-quant, bf16 compute) | SERA QLoRA YAML §C.3 |")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(
        f"- **Source:** `{_FILTERED_PARQUET.relative_to(_REPO_ROOT)}` "
        "(built by A14 / `scripts/filter_pilot.py`)."
    )
    lines.append(f"- **Spec:** `{_SPEC_KEY}`.")
    lines.append(
        f"- **Subsample:** {len(train_samples) + len(held_samples)} of 1344 "
        f"filtered glucose-insulin trajectories (deterministic, seed={_SEED})."
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
    lines.append("| report# | train_loss |")
    lines.append("|---:|---:|")
    for i, loss in enumerate(losses):
        lines.append(f"| {i} | {loss:.4f} |")
    lines.append("")
    lines.append("## Held-out parse evaluation")
    lines.append("")
    lines.append(
        f"Trained model run on {len(held_samples)} held-out user prompts; "
        "greedy decoding (do_sample=False); 1200-token max; output regex-checked "
        "for at least one `<state>...</state><action>...</action>` block."
    )
    lines.append("")
    lines.append(f"- **Parse-success rate:** {n_parse} / {len(held_samples)}")
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
    lines.append("## Wall clock")
    lines.append("")
    lines.append(f"- **Setup (deps + model download + dataset build):** {setup_seconds:.1f} s")
    lines.append(f"- **Training:** {wall:.1f} s ({wall / 60:.2f} min)")
    lines.append("")
    lines.append("## Hard-checkpoint pass criteria")
    lines.append("")
    lines.append("| criterion | met? | notes |")
    lines.append("|---|:---:|---|")
    lines.append(
        f"| (a) training runs without crash | "
        f"{'YES' if losses else 'NO'} | {len(losses)} loss reports captured |"
    )
    lines.append(
        f"| (b) final loss < initial loss | "
        f"{'YES' if decrease_diag.get('loss_decreased') else 'NO'} | "
        f"{first_mean:.4f} → {last_mean:.4f} |"
    )
    lines.append(
        f"| (c) no NaN/Inf in loss history | "
        f"{'YES' if not any(math.isnan(x) or math.isinf(x) for x in losses) else 'NO'} | "
        f"min={decrease_diag.get('min_loss', float('nan')):.4f}, "
        f"max={decrease_diag.get('max_loss', float('nan')):.4f} |"
    )
    lines.append(
        f"| (d) parse-success rate >= 1/5 | "
        f"{'YES' if n_parse >= 1 else 'NO'} | {n_parse}/{len(held_samples)} |"
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
            "A23 — bnb 4-bit QLoRA smoke test on Qwen3-0.6B-Instruct\n"
            f"Iters target: ~50 ({_NUM_EPOCHS} epochs over 95 rows / "
            f"{_BATCH_SIZE}x{_GRAD_ACCUM} effective batch)\n"
            f"LoRA rank/alpha: {_LORA_RANK}/{_LORA_ALPHA}  | "
            f"targets: {_LORA_TARGETS}",
            title="[bold]Subphase 1.7 — Phase-2 readiness",
        )
    )

    _require_cuda()
    _seed_everything(_SEED)

    setup_start = time.perf_counter()

    # Resolve the model identifier first so we fail fast on network issues.
    console.rule("[bold]Step 0 — Resolve model identifier")
    model_id = _resolve_model_id()

    # 1) Build dataset.
    console.rule("[bold]Step 1 — Build SFT dataset")
    rng = np.random.default_rng(_SEED)
    samples = _load_filtered_glucose_dataset(_N_TOTAL, rng)
    held_samples = samples[-_N_HELDOUT:]
    train_samples = samples[: len(samples) - _N_HELDOUT]
    console.print(f"  split: {len(train_samples)} train / {len(held_samples)} held-out")
    setup_seconds = time.perf_counter() - setup_start

    # 2) Train.
    console.rule("[bold]Step 2 — bnb QLoRA fine-tuning")
    issues: list[str] = []
    followups: list[str] = []
    try:
        losses, wall, env_info = _run_bnb_training(train_samples, _RUNS_DIR, model_id)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]TRAINING FAILED:[/] {type(e).__name__}: {e}")
        losses = []
        wall = 0.0
        env_info = {}
        issues.append(f"Training crash: {type(e).__name__}: {e}")
        verdict = "FAIL"
        _write_report(
            verdict=verdict,
            losses=losses,
            wall=wall,
            setup_seconds=setup_seconds,
            decrease_diag={"reason": "training crashed before any reports"},
            parse_results=[],
            issues=issues,
            followups=followups,
            train_samples=train_samples,
            held_samples=held_samples,
            env_info=env_info,
            model_id=model_id,
        )
        return 1
    if wall > _WALL_CLOCK_BUDGET_S:
        issues.append(f"Training exceeded {_WALL_CLOCK_BUDGET_S:.0f}s budget (took {wall:.1f}s)")

    # 3) Loss-decrease check.
    console.rule("[bold]Step 3 — Loss decrease verification")
    decreased, decrease_diag = _loss_decrease_check(losses)
    _print_loss_table(losses)
    console.print(f"  loss decreased: {decreased}  diag: {decrease_diag}")

    # 4) Held-out parse evaluation.
    console.rule("[bold]Step 4 — Held-out parse evaluation")
    try:
        parse_results = _heldout_eval(held_samples, _RUNS_DIR, model_id)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]HELD-OUT EVAL FAILED:[/] {type(e).__name__}: {e}")
        parse_results = []
        issues.append(f"Held-out eval crash: {type(e).__name__}: {e}")
    n_parse = sum(1 for r in parse_results if r.parses)
    console.print(f"  parse-success rate: {n_parse}/{len(parse_results)}")

    # 5) Verdict.
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

    followups.append(
        "On bnb the lr=2e-4 from the MLX smoke test may be aggressive given "
        "the 8-bit AdamW optimizer; if the curve is jittery, try lr=5e-5 "
        "(SERA QLoRA YAML default) and re-run."
    )
    followups.append(
        "Validate that the same dataset shape lands the bnb run within "
        "+/- 0.1 of the MLX smoke-test final loss (1.484 -> 0.466) — large "
        "divergence indicates a tokenization-template mismatch between the "
        "two backends."
    )
    followups.append(
        "Before kicking off the canonical 18-cell sweep, raise lora_rank to "
        "32 and lora_target_modules to the full q/k/v/o + gate/up/down set "
        "(SERA QLoRA YAML §C.3)."
    )

    _write_report(
        verdict=verdict,
        losses=losses,
        wall=wall,
        setup_seconds=setup_seconds,
        decrease_diag=decrease_diag,
        parse_results=parse_results,
        issues=issues,
        followups=followups,
        train_samples=train_samples,
        held_samples=held_samples,
        env_info=env_info,
        model_id=model_id,
    )

    color = "green" if verdict == "PASS" else "red"
    console.print(
        Panel(
            f"[bold {color}]VERDICT: {verdict}[/]\n"
            f"loss: {decrease_diag.get('first_window_mean', 'n/a')} → "
            f"{decrease_diag.get('last_window_mean', 'n/a')}\n"
            f"parse-success: {n_parse}/{len(parse_results)}\n"
            f"wall-clock: {wall:.1f} s",
            title="A23 hard-checkpoint result",
            border_style=color,
        )
    )
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

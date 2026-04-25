"""Mock training backend for Phase-2 dry-run pipeline validation.

This module provides :class:`MockBNBBackend`, a *no-CUDA, no-network*
simulation of the canonical :class:`~stl_seed.training.backends.bnb.BNBBackend`.
It is the workhorse of ``scripts/validate_phase2_pipeline.py``: it lets the
end-to-end sweep + eval + analysis pipeline be exercised against the same
artifacts the real backend would produce, without spending a dollar of
RunPod GPU time.

What the mock does
------------------

* :meth:`MockBNBBackend.train` synthesizes a deterministic, monotonically
  decreasing training-loss curve whose decay rate, asymptote, and noise are
  config-dependent (so the 0.6B "trains" faster than the 4B). It then writes
  a directory layout indistinguishable on the surface from
  :class:`BNBBackend`'s output:

  * ``<output_dir>/adapter/`` contains a stub ``adapter_config.json``,
    ``MOCK.txt`` sentinel, and a tokenizer-pretrained-style placeholder.
  * ``<output_dir>/provenance.json`` mirrors :class:`BNBBackend`'s
    schema field-for-field (``backend``, ``base_model``, ``n_examples``,
    ``config``, ``wall_clock_seconds``, ``n_loss_points``) so downstream
    consumers see the same keys. The mock adds ``"mock": true`` and a
    ``"mock_*"`` prefix on every synthetic field so that no human reading
    the manifest can ever mistake it for a real run.

* :meth:`MockBNBBackend.load` returns a callable ``generate(prompt, **kw)``
  that decodes a synthetic, parser-compatible
  ``<state>...</state><action>...</action>`` block per control step. The
  output respects the spec horizon (read off the prompt) and the action
  dimensionality (heuristically inferred from the initial-state block in
  the prompt). This lets the eval harness run end-to-end against the mock
  checkpoint and record real-shaped (n_seeds, N) success matrices — even
  though every sample is "deterministic random" by construction.

Safety rails
------------

* The mock impersonates the bnb backend by setting ``name = "bnb"`` so that
  ``stl_seed.training.loop.get_backend("bnb")`` can return it transparently
  when ``STL_SEED_USE_MOCK_BACKEND=1`` is set in the environment. Without
  the env var, callers get the real :class:`BNBBackend` as before.
* If ``STL_SEED_REAL_TRAINING=1`` is set, the mock refuses to run — both
  :meth:`train` and :meth:`load` raise :class:`RuntimeError`. This guards
  against accidentally booting the mock inside a real Phase-2 RunPod
  invocation (the user's .bashrc on RunPod is expected to set the
  REAL_TRAINING flag).
* :attr:`MockBNBBackend.is_mock` is ``True``, distinguishing it from a real
  backend at runtime.

Why mirror BNBBackend (and not MLXBackend)
------------------------------------------

The Phase-2 sweep runs on RunPod via the bnb backend; the mock's purpose
is to validate that pipeline. MLXBackend writes a different on-disk shape
(``adapters.safetensors`` + ``adapter_config.json``) consumed by
``mlx_lm.load(adapter_path=...)``, which is irrelevant to the Phase-2
artifact contract.

style SFT loop scaffolding.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from stl_seed.training.backends.base import (
    TrainedCheckpoint,
    TrainingConfig,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Env-var contract.
# ---------------------------------------------------------------------------

#: Setting this to ``"1"`` opts a sweep into the mock backend (the runner
#: substitutes :class:`MockBNBBackend` for :class:`BNBBackend`). Documented
#: alongside the runner switch in ``scripts/run_canonical_sweep.py``.
USE_MOCK_ENV: str = "STL_SEED_USE_MOCK_BACKEND"

#: Setting this to ``"1"`` declares the current process is doing real
#: Phase-2 training. The mock refuses to run if this flag is on, even when
#: :data:`USE_MOCK_ENV` is also set, to make accidental mock-on-RunPod
#: misconfiguration impossible.
REAL_TRAINING_ENV: str = "STL_SEED_REAL_TRAINING"


def is_mock_enabled() -> bool:
    """Return ``True`` iff :data:`USE_MOCK_ENV` is truthy in the environment."""
    return os.environ.get(USE_MOCK_ENV, "").strip() in {"1", "true", "True", "TRUE", "yes"}


def _refuse_if_real_training() -> None:
    """Raise if the real-training guard is set.

    The :class:`MockBNBBackend` should never run inside a process that has
    declared itself a real Phase-2 training run. This is a defense-in-depth
    check — the env var is the second line of defense after the explicit
    user opt-in via :data:`USE_MOCK_ENV`.
    """
    if os.environ.get(REAL_TRAINING_ENV, "").strip() in {"1", "true", "True", "TRUE", "yes"}:
        raise RuntimeError(
            f"MockBNBBackend refused to run because {REAL_TRAINING_ENV}=1 is set. "
            "This guard prevents the mock from polluting a real Phase-2 RunPod run. "
            "Unset the env var, or use the real BNBBackend explicitly."
        )


# ---------------------------------------------------------------------------
# Synthetic loss curve.
# ---------------------------------------------------------------------------


def _synthetic_loss_curve(
    n_steps: int,
    *,
    base_model_hint: str,
    seed: int,
    asymptote: float = 0.55,
) -> list[float]:
    """Generate a deterministic, monotonically decreasing loss curve.

    The curve has the shape::

        loss[t] = exp(-t / tau) * (1 + epsilon[t]) + asymptote

    where ``tau`` and the noise scale depend on the base model size (so a
    0.6B "converges" faster than a 4B), and ``epsilon[t]`` is a small,
    seed-controlled perturbation that does not flip the monotonic trend.

    Parameters
    ----------
    n_steps:
        Number of loss-log points to synthesize. Mirrors the real
        :class:`BNBBackend`'s ``n_loss_points`` — the SFTTrainer logs every
        ``logging_steps=10`` steps, so ``n_steps`` is roughly the total
        number of optimizer updates / 10 for the real path.
    base_model_hint:
        The HuggingFace model id (e.g. ``"Qwen/Qwen3-1.7B-Instruct"``).
        Substring-matched against ``"0.6B"``, ``"1.7B"``, ``"4B"`` to set
        the decay timescale.
    seed:
        Seed for the noise; the same ``(n_steps, base_model_hint, seed)``
        triple deterministically reproduces the same curve.
    asymptote:
        The plateau the synthetic loss decays to. ~0.55 is a plausible NLL
        for SFT on next-token prediction over a small action vocabulary
        (well above zero, well below the random-init ~6).
    """
    if n_steps <= 0:
        return []

    hint = base_model_hint.lower()
    if "4b" in hint or "4_b" in hint:
        tau = max(n_steps / 2.0, 1.0)
        noise_scale = 0.06
        init_loss = 6.5
    elif "1.7b" in hint or "1_7b" in hint:
        tau = max(n_steps / 3.0, 1.0)
        noise_scale = 0.05
        init_loss = 5.5
    else:  # default: smallest model, fastest decay
        tau = max(n_steps / 4.5, 1.0)
        noise_scale = 0.04
        init_loss = 4.5

    rng = random.Random(int(seed) ^ hash(base_model_hint) & 0xFFFF_FFFF)
    losses: list[float] = []
    last_smoothed = init_loss
    for t in range(n_steps):
        decay = math.exp(-t / tau)
        eps = (rng.random() - 0.5) * 2.0 * noise_scale  # [-noise_scale, +noise_scale]
        raw = (init_loss - asymptote) * decay * (1.0 + eps) + asymptote
        # Enforce strict monotonicity in expectation by running a one-step
        # exponential moving average against the previous smoothed value.
        # This keeps the curve realistically noisy step-to-step but
        # monotonically decreasing in trend, which matches what the real
        # SFT loop produces over the small SERA recipe step counts.
        smoothed = min(raw, last_smoothed - 1e-4) if t > 0 else raw
        last_smoothed = smoothed
        losses.append(float(smoothed))
    return losses


# ---------------------------------------------------------------------------
# Synthetic generation callable.
# ---------------------------------------------------------------------------


_INITIAL_STATE_RE = re.compile(r"<state>([^<]+)</state>")
_HORIZON_RE = re.compile(r"Emit exactly\s+(\d+)\s+\(state, action\) blocks")


def _infer_horizon_and_dim(
    prompt: str, default_horizon: int = 50, default_action_dim: int = 1
) -> tuple[int, int]:
    """Parse the (horizon, action_dim) tuple out of an eval-time prompt.

    The eval harness's prompt template ends with ``Emit exactly H (state,
    action) blocks`` (see :func:`stl_seed.training.tokenize.format_prompt_for_eval`),
    and the initial-state block carries the state dimension. We take the
    state dim as a proxy for the action dim — it is correct for the
    glucose-insulin family (1-D action) and for the bio-ode families where
    we set the action dim to 1; the mock does not need exact dimensionality
    to satisfy the parser-shape contract, only consistent shape across
    samples.
    """
    horizon_match = _HORIZON_RE.search(prompt)
    horizon = int(horizon_match.group(1)) if horizon_match else default_horizon

    state_match = _INITIAL_STATE_RE.search(prompt)
    if state_match is not None:
        try:
            n_components = len(state_match.group(1).split(","))
        except (AttributeError, ValueError):
            n_components = default_action_dim
    else:
        n_components = default_action_dim
    return int(horizon), int(max(1, min(n_components, 8)))


def _make_mock_generation_callable(
    seed: int,
    base_model: str,
    *,
    default_horizon: int = 50,
    default_action_dim: int = 1,
) -> Callable[..., str]:
    """Return a deterministic ``generate(prompt, **kw)`` that emits parseable text.

    The synthesized output is a newline-joined sequence of
    ``<state>...</state><action>...</action>`` blocks that
    :func:`stl_seed.training.tokenize.parse_action_sequence` can decode
    into an ``(H, m)`` numpy array. The state values are fixed at zero
    (the simulator only consumes the *action* sequence at eval time, so
    the assistant-side state values are cosmetic); the action values are
    drawn from a tiny seeded RNG so different prompts get different
    rollouts (otherwise BoN would collapse to the same single sample N
    times — exactly the action-diversity regression mode flagged in
    """
    base_seed = (hash(base_model) ^ int(seed)) & 0xFFFF_FFFF

    def _generate(prompt: str, **kwargs: Any) -> str:
        # Per-call seed: prompt hash + base seed + a per-call counter via
        # kwargs (some callers pass `seed=...`; we honor it for determinism).
        call_seed = (base_seed ^ hash(prompt) ^ int(kwargs.get("seed", 0))) & 0xFFFF_FFFF
        rng = random.Random(call_seed)

        horizon, action_dim = _infer_horizon_and_dim(
            prompt,
            default_horizon=default_horizon,
            default_action_dim=default_action_dim,
        )

        # Bound horizon so accidentally huge values from a malformed prompt
        # cannot blow up the validation script's runtime.
        horizon = max(1, min(horizon, 256))
        action_dim = max(1, min(action_dim, 8))

        blocks: list[str] = []
        for _h in range(horizon):
            state_vals = [0.0] * action_dim
            action_vals = [rng.uniform(-1.0, 1.0) for _ in range(action_dim)]
            state_str = ",".join(f"{v:.3e}" for v in state_vals)
            action_str = ",".join(f"{v:.3e}" for v in action_vals)
            blocks.append(f"<state>{state_str}</state><action>{action_str}</action>")
        return "\n".join(blocks)

    return _generate


# ---------------------------------------------------------------------------
# Mock backend.
# ---------------------------------------------------------------------------


class MockBNBBackend:
    """No-GPU, no-network mock of :class:`BNBBackend` for pipeline validation.

    Attributes
    ----------
    name:
        Set to ``"bnb"`` so the mock impersonates the canonical Phase-2
        backend at the dispatch layer. The runner is expected to swap
        :class:`BNBBackend` -> :class:`MockBNBBackend` only when
        :data:`USE_MOCK_ENV` is set.
    is_mock:
        Always ``True``. Distinguishes the mock from the real
        :class:`BNBBackend` at runtime. The provenance manifest also
        records ``"mock": true`` so the artifact is unambiguously
        marked even after the in-memory object is gone.
    """

    name: str = "bnb"  # impersonate bnb so get_backend("bnb") works transparently
    is_mock: bool = True

    def __init__(self, *, mock_train_seconds: float = 0.05) -> None:
        """Construct the mock.

        Parameters
        ----------
        mock_train_seconds:
            How long :meth:`train` should pretend to take. We do *not*
            ``time.sleep`` — instead, we record this as ``wall_clock_seconds``
            in the provenance to make downstream cost-accounting code
            exercise its arithmetic. Default 0.05s keeps the validation
            script under the 5-minute CI ceiling even with 18 cells.
        """
        _refuse_if_real_training()
        self.mock_train_seconds = float(mock_train_seconds)

    # ----- public API ----------------------------------------------------

    def train(
        self,
        base_model: str,
        dataset: Any,
        config: TrainingConfig,
        output_dir: Path,
    ) -> TrainedCheckpoint:
        """Pretend to QLoRA-finetune ``base_model`` and write a mock checkpoint.

        Mirrors :meth:`BNBBackend.train` field-for-field where it matters
        for downstream consumers; deviates only by skipping the actual
        training. The on-disk layout written under ``output_dir`` is:

            <output_dir>/
                adapter/
                    adapter_config.json     (PEFT-style stub, mock-flagged)
                    MOCK.txt                (sentinel — humans cannot miss)
                    tokenizer_config.json   (empty stub)
                provenance.json             (BNB-schema-compatible)

        Parameters and return value mirror :meth:`BNBBackend.train`.
        """
        _refuse_if_real_training()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir = output_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        # Defensive dataset length probe. Real BNBBackend records
        # ``len(dataset)`` directly; we mirror that, but tolerate a missing
        # ``__len__`` (e.g., a generator-based dataset stub) so callers can
        # pass anything iterable.
        try:
            n_examples = int(len(dataset))
        except TypeError:
            try:
                n_examples = sum(1 for _ in iter(dataset))
            except TypeError:
                n_examples = 0
        # Estimate the per-step count the real path would log: a SERA-recipe
        # SFT run with batch_size=1, grad_accum=4, num_epochs=3 logs every
        # 10 steps. n_loss_points ≈ ceil(n_examples * num_epochs /
        # (batch_size * grad_accum)) / 10. We clamp to [4, 200] so the
        # synthetic curve is plausible for any dataset size.
        eff_batch = max(int(config.batch_size) * int(config.gradient_accumulation_steps), 1)
        total_steps = max(int(math.ceil(n_examples * config.num_epochs / eff_batch)), 1)
        n_loss_points = max(4, min(200, total_steps // 10 + 1))

        loss_history = _synthetic_loss_curve(
            n_loss_points,
            base_model_hint=base_model,
            seed=int(config.seed),
        )

        # Write a tiny adapter_config.json. The schema below mirrors PEFT's
        # adapter_config.json keys at the field level (peft_type, r,
        # lora_alpha, target_modules, task_type) so a future loader that
        # introspects the JSON would not crash. The 'mock_*' fields make the
        # artifact unmistakable.
        adapter_config = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": int(config.lora_rank),
            "lora_alpha": float(config.lora_alpha),
            "lora_dropout": float(config.lora_dropout),
            "target_modules": list(config.lora_target_modules),
            "bias": "none",
            "base_model_name_or_path": base_model,
            "mock": True,
            "mock_backend": "MockBNBBackend",
            "mock_warning": (
                "This adapter contains NO real weights. It is a pipeline-validation "
                "stub written by stl_seed.training.backends.mock.MockBNBBackend."
            ),
        }
        (adapter_dir / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2))
        (adapter_dir / "MOCK.txt").write_text(
            "This directory is a MOCK adapter written by MockBNBBackend.\n"
            "It contains no real weights. See provenance.json for the recipe.\n"
        )
        # Empty tokenizer placeholder so any downstream consumer that
        # touches tokenizer files does not bare-trip on a missing file.
        (adapter_dir / "tokenizer_config.json").write_text("{}\n")

        # Provenance schema MUST mirror BNBBackend's so the runner's
        # write_provenance() merges cleanly. Extra mock_* keys are additive.
        provenance: dict[str, Any] = {
            "backend": "bnb",
            "base_model": base_model,
            "n_examples": int(n_examples),
            "config": _config_to_dict(config),
            "wall_clock_seconds": float(self.mock_train_seconds),
            "n_loss_points": int(len(loss_history)),
            # ---- mock-only fields (additive) ----
            "mock": True,
            "mock_backend": "MockBNBBackend",
            "mock_loss_curve_seed": int(config.seed),
            "mock_synthetic_loss_first": loss_history[0] if loss_history else None,
            "mock_synthetic_loss_last": loss_history[-1] if loss_history else None,
            "mock_warning": (
                "Generated by MockBNBBackend (no GPU, no training). "
                "Use only for pipeline validation."
            ),
        }
        (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

        log.info(
            "MockBNBBackend.train: wrote stub adapter to %s "
            "(n_examples=%d, n_loss_points=%d, last_loss=%.4f)",
            adapter_dir,
            n_examples,
            len(loss_history),
            loss_history[-1] if loss_history else float("nan"),
        )

        return TrainedCheckpoint(
            backend="bnb",  # impersonate to match the runner's expectations
            model_path=adapter_dir,
            base_model=base_model,
            training_loss_history=loss_history,
            wall_clock_seconds=float(self.mock_train_seconds),
            metadata=provenance,
        )

    def load(self, checkpoint: TrainedCheckpoint) -> Callable[..., Any]:
        """Return a deterministic ``generate(prompt, **kw) -> str`` callable.

        The returned callable emits a newline-joined sequence of
        ``<state>...</state><action>...</action>`` blocks parseable by
        :func:`stl_seed.training.tokenize.parse_action_sequence`. The
        horizon and action dimension are inferred from the prompt itself,
        so the callable is task-agnostic.
        """
        _refuse_if_real_training()
        # Use the checkpoint's recorded base_model and a derived seed so two
        # `load` calls on the same checkpoint produce the same generator
        # (matches the real path: PEFT.from_pretrained is deterministic).
        seed = int(checkpoint.metadata.get("mock_loss_curve_seed", 20260424))
        return _make_mock_generation_callable(
            seed=seed,
            base_model=checkpoint.base_model,
        )


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _config_to_dict(config: TrainingConfig) -> dict[str, Any]:
    """Serialize a :class:`TrainingConfig` to a JSON-friendly dict.

    Mirrors :func:`stl_seed.training.backends.bnb._config_to_dict` so the
    provenance manifest's ``config`` block is byte-identical between the
    real and mock backends for non-mock fields.
    """
    out: dict[str, Any] = {}
    for k, v in config.__dict__.items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, list | tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out


__all__ = [
    "MockBNBBackend",
    "USE_MOCK_ENV",
    "REAL_TRAINING_ENV",
    "is_mock_enabled",
]

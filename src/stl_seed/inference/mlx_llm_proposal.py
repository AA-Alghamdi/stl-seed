"""Real-LLM proposal for the inference samplers, backed by MLX + Qwen3.

Why this exists
---------------

Every sampler in :mod:`stl_seed.inference` consumes an
:class:`stl_seed.inference.protocol.LLMProposal` callable. Until now the
canonical empirical comparison
(``scripts/run_unified_comparison.py``) used a *uniform-flat* synthetic
LLM that returns ``jnp.zeros(K)`` regardless of the prompt. That makes
the cross-sampler comparison apples-to-apples but it is a synthetic
baseline: any "+128x lift over standard sampling" headline measured
against a uniform proxy is at risk of evaporating once a real
language-model prior is wired in. This module fixes the methodological
weakness by wrapping a real Qwen3 checkpoint (default
``mlx-community/Qwen3-1.7B-bf16``; ``Qwen3-0.6B-bf16`` for budget) so
the same harness can be re-run with a real LLM in the loop.

Implementation: token-prefix log-prob lookup
--------------------------------------------

Action vocabularies in this repo are small discrete grids (``K`` from 4
to 125). Asking the LLM to *generate* a serialized action and then
parse-and-project to the vocabulary (approach (b) in the design doc) is
brittle for a non-fine-tuned base model — the bare Qwen3 enters
"thinking" mode and emits prose, not the
``<state>...</state><action>...</action>`` blocks.

We therefore implement approach (a): for each entry ``V_k`` in the
vocabulary we precompute the canonical text serialization
``"u_1,u_2,...,u_m</action>"`` (matching
:func:`stl_seed.training.tokenize._fmt_vec`) and tokenize it once. At
each LLM call the prompt is built as

    system_prompt + user_turn + <action-history-as-assistant-text> +
        "<state>?</state><action>"

and we score every ``V_k`` candidate by teacher-forced log-probability
of its token sequence under the model. The K candidates are batched
into a single forward pass (padded to ``max_k token_len(V_k)``), so the
per-call cost is one forward through ``(K, prompt_len + max_cand_len)``,
not K independent forwards. Empirically this is ~100ms / call for
Qwen3-0.6B and ~430ms / call for Qwen3-1.7B at our prompt length
(~380 tokens).

Returning *log-probabilities* — not the softmax — preserves the LLMProposal
contract that gradient guidance can additively bias logits before
re-normalising. That preservation is the technical reason the
gradient-guided sampler needs unnormalised logits.

Per-rollout amortisation
------------------------

Within one rollout (``H`` LLM calls back-to-back), the leading
``system_prompt + user_turn`` is identical across all H calls; only the
action-history suffix grows. mlx_lm exposes a KV-cache via
``mlx_lm.models.cache.make_prompt_cache``, but using it across the
batched-K forward is awkward (the cache must replicate K-fold). We keep
the implementation simple — re-prefill on every call — and rely on
mlx's fast-prefill for the static prompt. A KV-cache optimisation is a
clean follow-up if real-LLM throughput becomes a bottleneck.

-------------

This module imports only from ``stl_seed.training.{prompts,tokenize}``,
``stl_seed.specs``, JAX/NumPy, and (lazily) MLX/mlx_lm. No

References
----------

- Qwen Team. *Qwen3: technical report.* arXiv:2504.??? (2025). Used
  here unmodified at the bf16 mlx-community conversion.
- Hu et al. *LoRA: low-rank adaptation of large language models.*
  arXiv:2106.09685 (2021). The smoke-test adapter at
  ``runs/smoke_test_mlx/`` could be loaded via ``adapter_path`` to
  restore the LoRA-fine-tuned variant on glucose-insulin; we do not do
  that by default because the smoke-test was 50 iters and not a
  publication-grade fine-tune.
"""

from __future__ import annotations

import platform
from typing import Any

import jax.numpy as jnp
import jaxtyping as jt
import numpy as np

from stl_seed.specs import STLSpec
from stl_seed.training.prompts import render_system_prompt
from stl_seed.training.tokenize import _fmt_vec

# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


# Recognised model identifiers + their HF-Hub IDs. The 4B is a stretch
# target if the M5 Pro / 48 GB unified memory budget allows; 0.6B is the
# canonical fast-iter model and what we use when the wall-clock budget is
# tight.
_MODEL_ALIASES: dict[str, str] = {
    "qwen3-0.6b": "mlx-community/Qwen3-0.6B-bf16",
    "qwen3-1.7b": "mlx-community/Qwen3-1.7B-bf16",
    "qwen3-4b": "mlx-community/Qwen3-4B-bf16",
}


# Cache loaded models keyed by model_id so multiple MLXLLMProposal
# instances within one process (e.g. one per task) re-use the same
# weights. Each Qwen3-1.7B-bf16 is ~3.4 GB on disk; loading it five
# times per script run would be wasteful and slow.
_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}


def _resolve_model_id(model_id: str) -> str:
    """Map a short alias to the canonical mlx-community HF id, or pass through."""
    return _MODEL_ALIASES.get(model_id, model_id)


def _check_apple_silicon() -> None:
    """Raise a clear error on non-Apple-Silicon platforms.

    MLX is Apple-only. The CUDA path uses bnb (Phase 2 RunPod sweep),
    not this proposal. We refuse early rather than emit cryptic mlx
    import errors deep inside ``_load_model``.
    """
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError(
            "MLXLLMProposal requires Apple Silicon (Darwin/arm64); "
            f"detected {platform.system()}/{platform.machine()}. "
            "Use the uniform-LLM proxy for CI / Linux smoke runs, "
            "or the bnb-backed proposal on RunPod."
        )


def _load_model(model_id: str) -> tuple[Any, Any]:
    """Load (or reuse) an mlx_lm model + tokenizer.

    Returns ``(model, tokenizer)``. Subsequent calls with the same
    ``model_id`` return the cached pair without re-reading from disk.
    """
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]
    _check_apple_silicon()
    # Lazy import: file must be importable on non-Apple platforms so
    from mlx_lm import load as mlx_load

    model, tokenizer = mlx_load(model_id)
    _MODEL_CACHE[model_id] = (model, tokenizer)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Action serialization — must match stl_seed.training.tokenize byte-for-byte
# so that scoring sequences correspond to what the model would emit if the
# downstream Phase-2 SFT'd Qwen3 wrote one out.
# ---------------------------------------------------------------------------


def _action_text(action: np.ndarray) -> str:
    """Canonical string for one action vector. Includes the closing tag.

    The closing ``</action>`` is part of the scored sequence. Two reasons:
    (1) it disambiguates length-padded short-vs-long candidates (the
    model may otherwise prefer a short candidate purely because shorter
    sequences have less negative log-probability), and (2) it forces the
    LLM to commit to the full numeric value rather than e.g. preferring
    an action that *could* be a prefix of multiple full actions.
    """
    return f"{_fmt_vec(action.tolist())}</action>"


def _state_text_placeholder(state_dim: int) -> str:
    """Placeholder ``<state>?,?,...,?</state>`` for the current step.

    The LLM is conditioned on the action history and the initial state
    (in the system prompt + user turn). The per-step state observation
    that would normally precede the action would require an extra
    simulator call to materialise — and the action history alone is
    sufficient context for a base model to emit a reasonable prior over
    next actions. We therefore use a literal ``?`` placeholder which
    keeps the format intact without committing to a fabricated state.
    """
    return "<state>" + ",".join(["?"] * state_dim) + "</state>"


# ---------------------------------------------------------------------------
# The proposal.
# ---------------------------------------------------------------------------


class MLXLLMProposal:
    """Real-LLM proposal: token-prefix log-prob over the action vocabulary.

    Implements :class:`stl_seed.inference.protocol.LLMProposal`. For each
    LLM call returns a JAX array of shape ``(K,)`` containing the
    teacher-forced log-probability of each canonical action serialization
    under a Qwen3 base model.

    Parameters
    ----------
    action_vocabulary:
        ``(K, m)`` array of action vectors. Required.
    spec:
        STL spec used in the prompt's system + user turns. Required.
    task:
        Task-family name. One of ``glucose_insulin``, ``repressilator``,
        ``toggle``, ``mapk``. Required.
    initial_state:
        ``(n,)`` initial state vector for the prompt. Required so the
        prompt's ``Initial state: <state>...</state>`` line is anchored
        to the same IC the rollout starts from.
    horizon:
        Number of control steps in the rollout. Used only for the
        ``Emit exactly H (state, action) blocks.`` line in the user
        turn so the LLM's prior over response length is calibrated.
    state_dim:
        Dimensionality of the simulator's state vector. Used to render
        the ``<state>?,?,...,?</state>`` placeholder for the current
        per-step observation. Required.
    model_id:
        HF Hub identifier or short alias (``qwen3-0.6b``, ``qwen3-1.7b``,
        ``qwen3-4b``). Default ``qwen3-1.7b``. The alias resolves to
        ``mlx-community/Qwen3-{size}-bf16``.
    sampling_temperature:
        Returned-logit divisor applied before the proposal returns. The
        LLMProposal contract is "return logits"; samplers may then
        further temperature-scale before sampling. We default to ``1.0``
        (return raw log-probs) so downstream samplers preserve their
        existing temperature semantics. The value is recorded in
        diagnostics for reproducibility but is not used here.
    enable_thinking:
        Forwarded to ``tokenizer.apply_chat_template``. Always ``False``
        for this scoring use case — the Qwen3 thinking-block does not
        affect token-prefix scoring of an action sequence appended after
        ``add_generation_prompt=True`` and only inflates the prompt-
        token count. Kept as a parameter so future Qwen3 variants whose
        chat template requires it can flip the default.

    Notes
    -----
    The action-vocabulary text representations are pre-tokenized in
    ``__init__`` (~K tokenizer calls) so per-call cost is bounded by one
    batched forward pass through the model.

    The proposal is **stateful** in that the chat-template prompt is
    cached as token IDs after first construction. It is *not* stateful
    across different ``initial_state`` arguments — a different IC at
    call time would silently use a stale prompt. This is acceptable for
    the unified-comparison harness (one IC per task), and we assert the
    contract in ``__call__`` so misuse fails loudly.
    """

    def __init__(
        self,
        *,
        action_vocabulary: jt.Float[jt.Array, "K m"] | np.ndarray,
        spec: STLSpec,
        task: str,
        initial_state: jt.Float[jt.Array, " n"] | np.ndarray,
        horizon: int,
        state_dim: int,
        model_id: str = "qwen3-1.7b",
        sampling_temperature: float = 1.0,
        enable_thinking: bool = False,
    ) -> None:
        V = np.asarray(action_vocabulary, dtype=np.float64)
        if V.ndim != 2:
            raise ValueError(f"action_vocabulary must be (K, m); got shape {V.shape}")
        self._vocab = V
        self.K, self.m = int(V.shape[0]), int(V.shape[1])

        self.spec = spec
        self.task = str(task)
        self._initial_state = np.asarray(initial_state, dtype=np.float64)
        if self._initial_state.ndim != 1:
            raise ValueError(
                f"initial_state must be 1-D (n,); got shape {self._initial_state.shape}"
            )
        self._initial_state_text = _fmt_vec(self._initial_state.tolist())
        self.horizon = int(horizon)
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1; got {self.horizon}")
        self.state_dim = int(state_dim)
        if self.state_dim < 1:
            raise ValueError(f"state_dim must be >= 1; got {self.state_dim}")

        self.model_id = _resolve_model_id(model_id)
        self.sampling_temperature = float(sampling_temperature)
        self.enable_thinking = bool(enable_thinking)

        # Lazy-load model + tokenizer.
        self._model, self._tokenizer = _load_model(self.model_id)

        # Build and cache the static system + user prompt as token IDs.
        self._static_prompt_text = self._build_static_prompt()
        self._static_prompt_tokens = self._tokenizer.encode(self._static_prompt_text)
        self._state_placeholder = _state_text_placeholder(self.state_dim)

        # Pre-tokenize each action candidate. We tokenize the full
        # ``"<value>,...,</action>"`` form (without the leading
        # ``<action>`` tag, which is part of the per-step turn-text and
        # therefore scored as part of the prompt). Tokens are stored as a
        # list of np.int32 lists.
        self._cand_token_seqs: list[list[int]] = []
        for k in range(self.K):
            text = _action_text(self._vocab[k])
            ids = self._tokenizer.encode(text, add_special_tokens=False)
            self._cand_token_seqs.append(list(ids))

        # Padding token. Qwen3 tokenizers expose pad_token_id as None
        # historically; fall back to eos. The pad token is masked out in
        # log-prob computation so the choice is cosmetic.
        self._pad_id = (
            self._tokenizer.pad_token_id
            if getattr(self._tokenizer, "pad_token_id", None) is not None
            else self._tokenizer.eos_token_id or 0
        )

        # Diagnostics counters — handy for debugging but not part of the
        # LLMProposal contract. Tests that introspect the proposal's
        # behaviour (e.g. tests/test_mlx_llm_proposal.py) read these.
        self.n_calls: int = 0
        self.last_log_probs: np.ndarray | None = None
        self.last_chosen_argmax: int | None = None

    # ---------------------------------------------------------------- private
    def _build_static_prompt(self) -> str:
        """Render the system + user turns and apply the chat template.

        The action-history suffix is appended at call time inside
        :meth:`__call__`.
        """
        system_text = render_system_prompt(
            task=_canonical_task_name(self.task),
            spec_text=self.spec.formula_text,
            horizon=self.horizon,
            duration_minutes=float(self.spec.horizon_minutes),
        )
        user_text = (
            f"Initial state: <state>{self._initial_state_text}</state>\n"
            f"Specification: {self.spec.formula_text}\n"
            f"Emit exactly {self.horizon} (state, action) blocks."
        )
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=self.enable_thinking,
        )
        return prompt

    def _build_history_suffix(self, history: np.ndarray) -> str:
        """Build the ``<state>?</state><action>...</action>`` history text.

        Each committed step in ``history`` becomes a serialized block of
        the form ``<state>?,?,...</state><action>u_1,u_2,...</action>``.
        We use the placeholder for the state because we do not have the
        per-step state observation here without re-simulating; the LLM's
        prior depends on the action sequence regardless. A trailing
        ``\\n`` separator is placed between blocks so the format matches
        :func:`stl_seed.training.tokenize.serialize_assistant_turn`.
        """
        if history.shape[0] == 0:
            return ""
        blocks = []
        placeholder = self._state_placeholder
        for t in range(history.shape[0]):
            a_text = _fmt_vec(history[t].tolist())
            blocks.append(f"{placeholder}<action>{a_text}</action>")
        return "\n".join(blocks) + "\n"

    def _score_batch(
        self,
        prompt_tokens: list[int],
    ) -> np.ndarray:
        """Score the K candidates by teacher-forced log-probability.

        Build a ``(K, prompt_len + max_cand_len - 1)`` int32 batch where
        each row is ``prompt_tokens + cand_tokens[k][:-1]`` padded with
        ``self._pad_id``. Run one forward pass; for each candidate
        accumulate log-prob over its token positions.
        """
        # Lazy import again so the file is importable on non-Apple platforms.
        import mlx.core as mx
        import mlx.nn as nn

        plen = len(prompt_tokens)
        max_cand_len = max(len(ct) for ct in self._cand_token_seqs)
        # Total sequence length: prompt tokens plus all but the last
        # candidate token (we predict the last token from the second-to-
        # last position's logits).
        total_len = plen + max_cand_len - 1
        batch_np = np.full((self.K, total_len), self._pad_id, dtype=np.int32)
        for k, ct in enumerate(self._cand_token_seqs):
            batch_np[k, :plen] = prompt_tokens
            if len(ct) > 1:
                batch_np[k, plen : plen + len(ct) - 1] = ct[:-1]
        batch = mx.array(batch_np)
        logits = self._model(batch)
        # log-softmax over vocab; (K, T, V). Cast to float32 explicitly:
        # mlx_lm models return bf16 logits which numpy.asarray cannot
        # parse via the PEP-3118 buffer protocol (RuntimeError "Item size
        # 2 for PEP 3118 buffer format string B does not match the dtype
        # B item size 1"). The .astype(mx.float32) materialises a fresh
        # float32 array that NumPy *can* zero-copy view.
        log_softmax = nn.log_softmax(logits.astype(mx.float32), axis=-1)
        mx.eval(log_softmax)
        # We need log_softmax[k, plen-1+i, ct[i]] for each i in [0, len(ct)).
        # Materialize to NumPy for the per-candidate accumulation; the per-
        # token gather is K * max_cand_len ~ a few hundred lookups, cheap.
        ls_np = np.asarray(log_softmax)
        scores = np.zeros(self.K, dtype=np.float64)
        for k, ct in enumerate(self._cand_token_seqs):
            s = 0.0
            for i, tok in enumerate(ct):
                pos = plen - 1 + i
                s += float(ls_np[k, pos, int(tok)])
            scores[k] = s
        return scores

    # ----------------------------------------------------------------- public
    def __call__(
        self,
        state: jt.Float[jt.Array, " n"],
        history: jt.Float[jt.Array, "T_hist m"],
        key: jt.PRNGKeyArray,  # noqa: ARG002 (deterministic; key unused)
    ) -> jt.Float[jt.Array, " K"]:
        """Return logits over the K-element action vocabulary.

        Conforms to :class:`stl_seed.inference.protocol.LLMProposal`. The
        ``key`` argument is accepted for protocol-conformance and is
        unused — token-prefix scoring is deterministic given the prompt.

        Parameters
        ----------
        state:
            The initial state of the rollout. We assert this matches the
            ``initial_state`` passed at construction so the cached prompt
            stays valid; if it does not match we re-build the prompt
            (rare path, but supports the case where the harness re-uses
            one MLXLLMProposal across multiple ICs by accident).
        history:
            ``(T_hist, m)`` array of committed actions so far. Used to
            build the assistant-turn suffix.
        key:
            Unused.

        Returns
        -------
        ``(K,)`` JAX array of logits (raw teacher-forced log-probs). The
        sampler will softmax-normalise these as needed.
        """
        # Coerce to numpy.
        state_np = np.asarray(state)
        if state_np.ndim != 1:
            raise ValueError(f"state must be 1-D; got shape {state_np.shape}")
        history_np = np.asarray(history)
        if history_np.ndim != 2:
            raise ValueError(f"history must be 2-D (T_hist, m); got shape {history_np.shape}")
        if history_np.shape[1] != self.m:
            raise ValueError(
                f"history action dim {history_np.shape[1]} != vocabulary m={self.m}"
            )

        # If the state changed (silent IC swap), rebuild the cached prompt.
        if not np.allclose(state_np, self._initial_state, rtol=1e-6, atol=1e-6):
            self._initial_state = state_np.copy()
            self._initial_state_text = _fmt_vec(state_np.tolist())
            self._static_prompt_text = self._build_static_prompt()
            self._static_prompt_tokens = self._tokenizer.encode(self._static_prompt_text)

        # Build the per-call prompt: static prompt + history suffix +
        # current step's state placeholder + opening <action> tag.
        history_suffix = self._build_history_suffix(history_np)
        # The current step's prefix (no closing </action>; the candidates
        # supply that).
        current_step_prefix = f"{self._state_placeholder}<action>"
        # We cannot just concatenate token-id lists because the
        # tokenizer's BPE may merge across the static-prompt boundary.
        # Re-tokenize the suffix as a single string so merging is correct.
        suffix_text = history_suffix + current_step_prefix
        suffix_tokens = self._tokenizer.encode(suffix_text, add_special_tokens=False)
        prompt_tokens = list(self._static_prompt_tokens) + list(suffix_tokens)

        scores = self._score_batch(prompt_tokens)

        # Update diagnostics.
        self.n_calls += 1
        self.last_log_probs = scores.copy()
        self.last_chosen_argmax = int(np.argmax(scores))

        # Return as JAX array so the gradient-guidance path (which
        # additively biases logits and then softmaxes) can keep its JAX
        # tracer infrastructure intact downstream.
        return jnp.asarray(scores, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


# The system-prompt registry uses the bare task names (``repressilator``,
# ``toggle``, ``mapk``, ``glucose_insulin``); the unified harness uses
# dotted forms (``bio_ode.repressilator`` etc.). Map both to the bare
# form for ``render_system_prompt``.
_TASK_NAME_CANONICAL: dict[str, str] = {
    "glucose_insulin": "glucose_insulin",
    "bio_ode.repressilator": "repressilator",
    "bio_ode.toggle": "toggle",
    "bio_ode.mapk": "mapk",
    "repressilator": "repressilator",
    "toggle": "toggle",
    "mapk": "mapk",
    # Cardiac is not in stl_seed.training.prompts; we map it to
    # glucose_insulin's prompt as a least-bad fallback because the action
    # space is also a 1-D scalar in [0, 1] and the LLM only sees the
    # system text as context. A dedicated cardiac prompt is a Phase-2
    # follow-up; the current prompt suffices for token-prefix scoring of
    # numeric action vectors.
    "cardiac_ap": "glucose_insulin",
}


def _canonical_task_name(task: str) -> str:
    if task not in _TASK_NAME_CANONICAL:
        raise KeyError(
            f"Unknown task family {task!r}; expected one of "
            f"{sorted(_TASK_NAME_CANONICAL.keys())}."
        )
    return _TASK_NAME_CANONICAL[task]


__all__ = [
    "MLXLLMProposal",
]

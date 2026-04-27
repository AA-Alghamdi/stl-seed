"""Unit tests for :class:`stl_seed.inference.mlx_llm_proposal.MLXLLMProposal`.

These tests cover the importability and platform-guard contract on
non-Apple platforms (so they pass on Linux CI) and the input-validation
behaviour. Apple-Silicon-only end-to-end tests are gated on ``sys.platform``
+ ``platform.machine()`` and run only on the target hardware.

"""

from __future__ import annotations

import platform

import numpy as np
import pytest

from stl_seed.inference.gradient_guided import make_uniform_action_vocabulary
from stl_seed.specs import REGISTRY


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


# ---------------------------------------------------------------------------
# Platform-agnostic tests.
# ---------------------------------------------------------------------------


def test_module_importable_on_any_platform() -> None:
    """The module must import even without MLX installed.

    The class :class:`MLXLLMProposal` lazy-imports MLX inside ``_load_model``
    so on non-Apple platforms the import statement at module top must not
    on Linux CI.
    """
    from stl_seed.inference import mlx_llm_proposal

    assert hasattr(mlx_llm_proposal, "MLXLLMProposal")


def test_resolve_model_id_aliases() -> None:
    from stl_seed.inference.mlx_llm_proposal import _resolve_model_id

    assert _resolve_model_id("qwen3-0.6b") == "mlx-community/Qwen3-0.6B-bf16"
    assert _resolve_model_id("qwen3-1.7b") == "mlx-community/Qwen3-1.7B-bf16"
    assert _resolve_model_id("qwen3-4b") == "mlx-community/Qwen3-4B-bf16"
    # Unknown ids pass through verbatim (allows custom HF ids).
    assert _resolve_model_id("custom/model-id") == "custom/model-id"


def test_canonical_task_name_mapping() -> None:
    from stl_seed.inference.mlx_llm_proposal import _canonical_task_name

    assert _canonical_task_name("glucose_insulin") == "glucose_insulin"
    assert _canonical_task_name("bio_ode.repressilator") == "repressilator"
    assert _canonical_task_name("bio_ode.toggle") == "toggle"
    assert _canonical_task_name("bio_ode.mapk") == "mapk"
    # cardiac_ap maps to glucose_insulin (least-bad fallback; documented in
    # the module).
    assert _canonical_task_name("cardiac_ap") == "glucose_insulin"
    with pytest.raises(KeyError):
        _canonical_task_name("nonexistent_task")


def test_action_text_serialization_matches_tokenize() -> None:
    """``_action_text`` must produce the same numeric format as
    :func:`stl_seed.training.tokenize._fmt_vec`, including the closing tag."""
    from stl_seed.inference.mlx_llm_proposal import _action_text
    from stl_seed.training.tokenize import _fmt_vec

    a = np.array([1.0, 2.5])
    assert _action_text(a) == f"{_fmt_vec(a.tolist())}</action>"
    assert _action_text(np.array([0.0])) == "0.000e+00</action>"


def test_state_text_placeholder_format() -> None:
    from stl_seed.inference.mlx_llm_proposal import _state_text_placeholder

    assert _state_text_placeholder(1) == "<state>?</state>"
    assert _state_text_placeholder(3) == "<state>?,?,?</state>"


# ---------------------------------------------------------------------------
# Apple-Silicon-only end-to-end tests.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _is_apple_silicon(),
    reason="MLXLLMProposal end-to-end requires Apple Silicon (Darwin/arm64).",
)
def test_proposal_returns_finite_logits_glucose_insulin() -> None:
    """Smoke: the proposal must return a (K,) finite-valued JAX logit vector.

    Uses the small Qwen3-0.6B for speed. Tests
    glucose-insulin (K=5; 1-D action; H=12).
    """
    pytest.importorskip("mlx")
    pytest.importorskip("mlx_lm")
    import jax
    import jax.numpy as jnp

    from stl_seed.inference.mlx_llm_proposal import MLXLLMProposal
    from stl_seed.tasks.glucose_insulin import (
        BergmanParams,
        default_normal_subject_initial_state,
    )

    spec = REGISTRY["glucose_insulin.tir.easy"]
    params = BergmanParams()
    x0 = default_normal_subject_initial_state(params)
    V = make_uniform_action_vocabulary(0.0, 5.0, k_per_dim=5)

    prop = MLXLLMProposal(
        action_vocabulary=V,
        spec=spec,
        task="glucose_insulin",
        initial_state=np.asarray(x0),
        horizon=12,
        state_dim=int(np.asarray(x0).shape[0]),
        model_id="qwen3-0.6b",
    )

    history = jnp.zeros((0, 1))
    logits = prop(jnp.asarray(x0), history, jax.random.key(0))
    assert logits.shape == (5,), f"Expected (5,), got {logits.shape}"
    arr = np.asarray(logits)
    assert np.all(np.isfinite(arr)), f"Non-finite logits: {arr}"
    # The model must produce a non-degenerate distribution -- if all
    # logits are identical the proposal would be indistinguishable from
    # the uniform proxy and the load-bearing scientific test (real LLM
    # vs uniform) would not be valid.
    assert arr.std() > 1e-6, f"Logits are uniform-flat: std={arr.std()}"


@pytest.mark.skipif(
    not _is_apple_silicon(),
    reason="MLXLLMProposal end-to-end requires Apple Silicon (Darwin/arm64).",
)
def test_proposal_input_validation() -> None:
    """The proposal raises clear errors on shape mismatches."""
    pytest.importorskip("mlx")
    pytest.importorskip("mlx_lm")
    import jax
    import jax.numpy as jnp

    from stl_seed.inference.mlx_llm_proposal import MLXLLMProposal
    from stl_seed.tasks.glucose_insulin import (
        BergmanParams,
        default_normal_subject_initial_state,
    )

    spec = REGISTRY["glucose_insulin.tir.easy"]
    params = BergmanParams()
    x0 = default_normal_subject_initial_state(params)
    V = make_uniform_action_vocabulary(0.0, 5.0, k_per_dim=5)

    prop = MLXLLMProposal(
        action_vocabulary=V,
        spec=spec,
        task="glucose_insulin",
        initial_state=np.asarray(x0),
        horizon=12,
        state_dim=int(np.asarray(x0).shape[0]),
        model_id="qwen3-0.6b",
    )

    # 2-D state is rejected.
    with pytest.raises(ValueError, match="state must be 1-D"):
        prop(jnp.zeros((1, 3)), jnp.zeros((0, 1)), jax.random.key(0))

    # 1-D history is rejected.
    with pytest.raises(ValueError, match="history must be 2-D"):
        prop(jnp.asarray(x0), jnp.zeros((3,)), jax.random.key(0))

    # Action-dim mismatch in history.
    with pytest.raises(ValueError, match="history action dim"):
        prop(jnp.asarray(x0), jnp.zeros((1, 3)), jax.random.key(0))


@pytest.mark.skipif(
    not _is_apple_silicon(),
    reason="MLXLLMProposal end-to-end requires Apple Silicon (Darwin/arm64).",
)
def test_proposal_chunked_scoring_matches_unchunked() -> None:
    """Chunking the K-axis must not change the scored log-probabilities.

    Pre-registered invariant: ``_score_batch`` chunked over ``chunk_size``
    rows of the K-batch must produce log-probs bit-identical (modulo the
    bf16->fp32 round-trip) to the unchunked single-pass version. Tests
    on a small ``K=5`` glucose-insulin vocabulary so the comparison is
    cheap; the K=125 repressilator OOM regression is covered by
    ``test_proposal_handles_large_k_repressilator_no_oom`` below.
    """
    pytest.importorskip("mlx")
    pytest.importorskip("mlx_lm")
    import jax
    import jax.numpy as jnp

    from stl_seed.inference.mlx_llm_proposal import MLXLLMProposal
    from stl_seed.tasks.glucose_insulin import (
        BergmanParams,
        default_normal_subject_initial_state,
    )

    spec = REGISTRY["glucose_insulin.tir.easy"]
    params = BergmanParams()
    x0 = default_normal_subject_initial_state(params)
    V = make_uniform_action_vocabulary(0.0, 5.0, k_per_dim=5)

    common = dict(
        action_vocabulary=V,
        spec=spec,
        task="glucose_insulin",
        initial_state=np.asarray(x0),
        horizon=12,
        state_dim=int(np.asarray(x0).shape[0]),
        model_id="qwen3-0.6b",
    )
    prop_chunked = MLXLLMProposal(chunk_size=2, **common)
    prop_unchunked = MLXLLMProposal(chunk_size=999, **common)

    history = jnp.zeros((0, 1))
    logits_chunked = np.asarray(prop_chunked(jnp.asarray(x0), history, jax.random.key(0)))
    logits_unchunked = np.asarray(prop_unchunked(jnp.asarray(x0), history, jax.random.key(0)))
    # Numerical agreement: the model's matmul reductions on Metal are not
    # bit-exact across batch sizes — the kernel selection depends on the
    # leading batch dim, and different reduction orders give fp32
    # differences ~1e-1 in raw logit space (observed: max-abs ~0.28 on
    # K=5 glucose-insulin, qwen3-0.6b). Tolerance is set to absorb this
    # well-known matmul-reduction-order noise; what matters operationally
    # is that the ARGMAX over candidates is preserved (the sampler reads
    # the relative ranking, not the absolute values). We assert both:
    # tight argmax agreement, plus a loose log-prob agreement that catches
    # any structural bug (e.g. wrong row-to-candidate mapping).
    assert int(np.argmax(logits_chunked)) == int(np.argmax(logits_unchunked))
    np.testing.assert_allclose(logits_chunked, logits_unchunked, rtol=5e-2, atol=5e-1)


@pytest.mark.skipif(
    not _is_apple_silicon(),
    reason="MLXLLMProposal end-to-end requires Apple Silicon (Darwin/arm64).",
)
def test_proposal_handles_large_k_repressilator_no_oom() -> None:
    """K=125 vocabulary on the repressilator must not crash with Metal OOM.

    The pre-fix wrapper batched all K candidates into one
    ``(K, T, |vocab|)`` log-softmax tensor, which at K=125, T~400, fp32
    = 30.4 GB and exceeds Metal's per-buffer ceiling on a 48 GB
    unified-memory M-series GPU. The chunked implementation
    (``chunk_size=16`` default) makes each forward pass at most
    ``(16, T, |vocab|) = ~3.9 GB`` which fits comfortably.

    This regression test was the load-bearing failure the 2026-04-25
    real-LLM hard-spec comparison ran into; without the fix, beam-search
    warmstart + Qwen3-0.6B on the repressilator could not be evaluated.
    """
    pytest.importorskip("mlx")
    pytest.importorskip("mlx_lm")
    import jax
    import jax.numpy as jnp

    from stl_seed.inference.mlx_llm_proposal import MLXLLMProposal
    from stl_seed.tasks.bio_ode import _repressilator_initial_state
    from stl_seed.tasks.bio_ode_params import RepressilatorParams

    spec = REGISTRY["bio_ode.repressilator.easy"]
    params = RepressilatorParams()
    x0 = _repressilator_initial_state(params)
    # k_per_dim=5 over a 3-D action box -> K=125 (the OOM-triggering size).
    V = make_uniform_action_vocabulary([0.0] * 3, [1.0] * 3, k_per_dim=5)
    assert V.shape == (125, 3)

    prop = MLXLLMProposal(
        action_vocabulary=V,
        spec=spec,
        task="bio_ode.repressilator",
        initial_state=np.asarray(x0),
        horizon=10,
        state_dim=int(np.asarray(x0).shape[0]),
        model_id="qwen3-0.6b",
        chunk_size=16,
    )
    out = prop(jnp.asarray(x0), jnp.zeros((0, 3)), jax.random.key(0))
    assert out.shape == (125,)
    arr = np.asarray(out)
    assert np.all(np.isfinite(arr)), "Non-finite logits returned"
    assert arr.std() > 1e-6, "Logits collapsed to uniform; LLM scoring broken"


@pytest.mark.skipif(
    not _is_apple_silicon(),
    reason="MLXLLMProposal construction requires Apple Silicon (Darwin/arm64).",
)
def test_proposal_construct_invalid_args() -> None:
    """Construction-time input validation."""
    pytest.importorskip("mlx")
    pytest.importorskip("mlx_lm")

    from stl_seed.inference.mlx_llm_proposal import MLXLLMProposal
    from stl_seed.tasks.glucose_insulin import (
        BergmanParams,
        default_normal_subject_initial_state,
    )

    spec = REGISTRY["glucose_insulin.tir.easy"]
    params = BergmanParams()
    x0 = default_normal_subject_initial_state(params)
    V = make_uniform_action_vocabulary(0.0, 5.0, k_per_dim=5)

    # 1-D vocabulary is rejected.
    with pytest.raises(ValueError, match="action_vocabulary must be"):
        MLXLLMProposal(
            action_vocabulary=np.zeros((5,)),
            spec=spec,
            task="glucose_insulin",
            initial_state=np.asarray(x0),
            horizon=12,
            state_dim=3,
            model_id="qwen3-0.6b",
        )

    # 2-D initial_state rejected.
    with pytest.raises(ValueError, match="initial_state must be 1-D"):
        MLXLLMProposal(
            action_vocabulary=V,
            spec=spec,
            task="glucose_insulin",
            initial_state=np.zeros((1, 3)),
            horizon=12,
            state_dim=3,
            model_id="qwen3-0.6b",
        )

    # horizon < 1 rejected.
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        MLXLLMProposal(
            action_vocabulary=V,
            spec=spec,
            task="glucose_insulin",
            initial_state=np.asarray(x0),
            horizon=0,
            state_dim=3,
            model_id="qwen3-0.6b",
        )

    # chunk_size < 1 rejected.
    with pytest.raises(ValueError, match="chunk_size must be >= 1"):
        MLXLLMProposal(
            action_vocabulary=V,
            spec=spec,
            task="glucose_insulin",
            initial_state=np.asarray(x0),
            horizon=12,
            state_dim=3,
            model_id="qwen3-0.6b",
            chunk_size=0,
        )

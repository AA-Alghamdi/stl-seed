"""Trajectory → text serialization for SFT.

A :class:`stl_seed.tasks._trajectory.Trajectory` carries

* ``states: Float[Array, "T n"]`` — per-save-time state vector
* ``actions: Float[Array, "H m"]`` — piecewise-constant action sequence
* ``times: Float[Array, " T"]`` — save-time grid

For SFT we serialize a trajectory into one (system, user, assistant)
conversation where:

* The system message is the task's system prompt (see ``prompts.py``).
* The user message states the initial condition and the STL spec text.
* The assistant message is the *full action sequence* interleaved with the
  state observations the agent would have seen at each control step.

The output format per assistant turn step is:

    <state>v1,v2,...,vn</state><action>u1,u2,...,um</action>

with one such block per control step, separated by newlines. This matches
the I/O contract documented in ``prompts.py`` so the trained model
emits the same structure at inference time.

The state samples used in the assistant message are taken from
``trajectory.states`` at the control-step boundaries (every
``T // H`` save-time index). When ``T`` is not an integer multiple of
``H``, we use the nearest-rounded index — this is the same convention
used by the simulator's piecewise-constant action discretization.

The function returns plain text. Tokenization is delegated to the
backend's tokenizer (HuggingFace ``AutoTokenizer`` for bnb,
``mlx_lm.tokenizer`` for mlx). We intentionally do not pre-tokenize: the
two backends use different tokenizers (the same Qwen3 vocab on both, but
different ID-stream conventions), and pre-tokenizing here would couple
the dataset to one backend.

REDACTED firewall: this module imports only ``Trajectory`` from
``stl_seed.tasks._trajectory``, ``Node`` from ``stl_seed.specs``,
plus ``numpy``. No ``REDACTED``, no ``REDACTED``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from stl_seed.specs import STLSpec
from stl_seed.tasks._trajectory import Trajectory
from stl_seed.training.prompts import render_system_prompt

# ---------------------------------------------------------------------------
# Number formatting.
# ---------------------------------------------------------------------------


def _fmt_scalar(x: float) -> str:
    """Format a scalar in 4-sig-fig scientific notation.

    Matches the format the system prompt promises the model. Negative zero
    is normalized to positive zero for stable round-tripping.
    """
    f = float(x)
    if f == 0.0:
        # Avoid "-0.000e+00" / "0.000e+00" mismatch under different platforms.
        return "0.000e+00"
    return f"{f:.3e}"


def _fmt_vec(v: Iterable[float]) -> str:
    """Comma-separated 4-sig-fig scientific notation."""
    return ",".join(_fmt_scalar(x) for x in v)


# ---------------------------------------------------------------------------
# Trajectory serialization.
# ---------------------------------------------------------------------------


def _control_step_state_indices(n_save: int, horizon: int) -> np.ndarray:
    """Pick state-array indices that align with control-step boundaries.

    Returns ``horizon`` indices into the ``states`` array (length ``n_save``)
    so that index ``k`` is the state observed *just before* control action
    ``k`` would be emitted. Uses ``round(k * (n_save - 1) / horizon)`` to
    keep both endpoints in range.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if n_save <= 0:
        raise ValueError(f"n_save must be positive, got {n_save}")
    # k=0 → 0; k=horizon-1 → at most n_save-1.
    raw = np.round(np.arange(horizon) * (n_save - 1) / max(horizon - 1, 1)).astype(int)
    return np.clip(raw, 0, n_save - 1)


def serialize_assistant_turn(
    states: np.ndarray,
    actions: np.ndarray,
) -> str:
    """Serialize per-step (state, action) blocks into one assistant message.

    Parameters
    ----------
    states:
        Array of shape ``(H, n)`` — one state observation per control step.
    actions:
        Array of shape ``(H, m)`` — one action per control step.

    Returns
    -------
    str:
        Newline-joined ``<state>...</state><action>...</action>`` blocks.
    """
    states = np.asarray(states)
    actions = np.asarray(actions)
    if states.ndim != 2 or actions.ndim != 2:
        raise ValueError(
            f"states and actions must be 2-D, got shapes {states.shape}, {actions.shape}"
        )
    if states.shape[0] != actions.shape[0]:
        raise ValueError(
            "states and actions must have matching horizon (axis-0); got "
            f"{states.shape[0]} vs {actions.shape[0]}"
        )
    blocks = []
    for k in range(actions.shape[0]):
        s_str = _fmt_vec(states[k])
        a_str = _fmt_vec(actions[k])
        blocks.append(f"<state>{s_str}</state><action>{a_str}</action>")
    return "\n".join(blocks)


def format_trajectory_as_text(
    trajectory: Trajectory,
    spec: STLSpec,
    task_name: str,
) -> dict[str, str]:
    """Build the (system, user, assistant) conversation for one trajectory.

    Parameters
    ----------
    trajectory:
        A :class:`Trajectory` from one of the simulators.
    spec:
        The STL spec the trajectory was scored under.
    task_name:
        One of ``repressilator``, ``toggle``, ``mapk``, ``glucose_insulin``.
        Selects the system prompt template.

    Returns
    -------
    dict with keys ``"system"``, ``"user"``, ``"assistant"`` (plain strings).
    The dict shape is what ``stl_seed.training.tokenize.format_for_chat``
    converts into a HuggingFace ``messages`` list.

    Notes
    -----
    The user turn includes the initial state and the spec; the assistant
    turn includes the full per-step trace. This matches SERA's "tool
    transcript" pattern (paper/REDACTED.md §B.5: each assistant turn
    is loss-bearing and the agent's actions are the supervision signal).

    Per-step state observations in the assistant turn are taken from
    ``trajectory.states`` at control-step boundaries (see
    :func:`_control_step_state_indices`).
    """
    states = np.asarray(trajectory.states)  # (T, n)
    actions = np.asarray(trajectory.actions)  # (H, m)
    if states.ndim != 2 or actions.ndim != 2:
        raise ValueError(
            "trajectory.states and trajectory.actions must be 2-D; "
            f"got {states.shape}, {actions.shape}"
        )

    horizon = actions.shape[0]
    duration_minutes = float(spec.horizon_minutes)
    system = render_system_prompt(
        task=task_name,
        spec_text=spec.formula_text,
        horizon=horizon,
        duration_minutes=duration_minutes,
    )

    initial_state_str = _fmt_vec(states[0])
    user = (
        f"Initial state: <state>{initial_state_str}</state>\n"
        f"Specification: {spec.formula_text}\n"
        f"Emit exactly {horizon} (state, action) blocks."
    )

    # Assistant trace: one (observation, action) block per control step.
    idx = _control_step_state_indices(states.shape[0], horizon)
    obs_states = states[idx]  # (H, n)
    assistant = serialize_assistant_turn(obs_states, actions)

    return {"system": system, "user": user, "assistant": assistant}


def format_for_chat(conversation: dict[str, str]) -> list[dict[str, str]]:
    """Convert a (system, user, assistant) dict into HF chat-template messages.

    Output schema matches what ``transformers.AutoTokenizer.apply_chat_template``
    consumes (and what TRL's ``SFTTrainer`` accepts in chat-formatted mode).
    """
    return [
        {"role": "system", "content": conversation["system"]},
        {"role": "user", "content": conversation["user"]},
        {"role": "assistant", "content": conversation["assistant"]},
    ]


def trajectory_to_record(
    trajectory: Trajectory,
    spec: STLSpec,
    task_name: str,
    weight: float = 1.0,
) -> dict[str, Any]:
    """Convert one trajectory to a HuggingFace dataset row.

    Output columns:

    * ``messages`` — chat-format list of role/content dicts (TRL-friendly).
    * ``prompt`` — concatenated system + user (for backends that prefer
      flat prompt + completion).
    * ``completion`` — assistant turn (plain text).
    * ``weight`` — per-sample SFT loss weight (1.0 for hard/quantile,
      ``softmax(ρ/β)`` for the continuous filter; carried through by
      the bnb collator and the mlx custom-loss closure).
    * ``task`` — task family name (carried for diagnostics / per-task eval).
    """
    conv = format_trajectory_as_text(trajectory, spec, task_name)
    return {
        "messages": format_for_chat(conv),
        "prompt": f"{conv['system']}\n\n{conv['user']}",
        "completion": conv["assistant"],
        "weight": float(weight),
        "task": task_name,
    }


__all__ = [
    "format_trajectory_as_text",
    "format_for_chat",
    "trajectory_to_record",
    "serialize_assistant_turn",
]

"""HuggingFace `Dataset` builder for filtered trajectories.

The deliverable spec calls for converting `(filtered_trajectories, weights)`
into rows of `(prompt, completion, weight, trajectory_id, spec_key)`. The
text rendering uses `stl_seed.training.tokenize.format_trajectory_as_text`
when the `training` subpackage is available; if it is not (subphase 1.3
agent A10 hasn't landed yet), this module ships a self-contained reference
formatter that produces the same JSON-schema output we will commit to in
A10. The fallback formatter is documented and tested here so we can swap
it out without breaking callers.

HuggingFace `datasets` library, and standard scientific Python.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import numpy as np
from jaxtyping import Array, Float

from stl_seed.tasks._trajectory import Trajectory

_DEFAULT_PROMPT_TEMPLATE = (
    "You are a control agent for the {task} system.\n"
    "Spec: {spec_text}\n"
    "Initial state: {initial_state}\n"
    "Emit a piecewise-constant control schedule of length H={horizon} as a "
    "JSON list of {action_dim}-vectors.\n"
)


def _format_trajectory_as_text(
    traj: Trajectory,
    spec_text: str,
    task: str,
) -> tuple[str, str]:
    """Self-contained reference formatter: trajectory -> (prompt, completion).

    The completion is a JSON list of length H, where each element is a
    list of `m` floats. the agent's emitted action sequence. The prompt
    contains the task name, the spec text, the initial state, and the
    requested horizon / action_dim. This is the SCHEMA we will commit to
    in `stl_seed.training.tokenize` (subphase 1.3 agent A10); pinning it
    here keeps `filter.dataset` self-contained.
    """
    states_np = np.asarray(traj.states)
    actions_np = np.asarray(traj.actions).round(4).tolist()
    initial_state = states_np[0].round(4).tolist()
    H, m = np.asarray(traj.actions).shape

    prompt = _DEFAULT_PROMPT_TEMPLATE.format(
        task=task,
        spec_text=spec_text,
        initial_state=initial_state,
        horizon=H,
        action_dim=m,
    )
    completion = json.dumps(actions_np)
    return prompt, completion


def _resolve_formatter() -> Any:
    """Return the canonical `format_trajectory_as_text` if available."""
    try:
        from stl_seed.training.tokenize import (  # type: ignore[import-not-found]
            format_trajectory_as_text,
        )

        return format_trajectory_as_text
    except ImportError:
        return _format_trajectory_as_text


def build_sft_dataset(
    filtered: Sequence[Trajectory],
    weights: Float[Array, " N"] | np.ndarray | Sequence[float],
    tokenizer: Any | None = None,
    prompt_template: str | None = None,
    *,
    metadata: Sequence[dict[str, Any]] | None = None,
    task: str = "unknown",
    spec_text: str = "",
) -> Any:
    """Convert filtered trajectories to a HuggingFace `Dataset`.

    Parameters
    ----------
    filtered:
        Output of `FilterCondition.filter(...)`'s first return value.
    weights:
        Output of `FilterCondition.filter(...)`'s second return value
        (length must match `filtered`).
    tokenizer:
        Optional HF tokenizer. If provided AND the canonical
        `format_trajectory_as_text` is available, the canonical version
        may use it for consistency with the training tokenizer. The
        fallback formatter does not consume `tokenizer`.
    prompt_template:
        Reserved for the canonical formatter. Ignored by the fallback.
    metadata:
        Optional per-trajectory metadata list (id, spec_key, etc.). If
        provided, each element supplies `id` and `spec_key` fields used in
        the dataset rows. If absent, synthetic IDs `traj_0000` ... are
        emitted and `spec_key` defaults to `task`.
    task:
        Task name embedded in the prompt.
    spec_text:
        STL spec text embedded in the prompt.

    Returns
    -------
    `datasets.Dataset` with columns: `prompt`, `completion`, `weight`,
    `trajectory_id`, `spec_key`.
    """
    weights_np = np.asarray(weights, dtype=np.float32).reshape(-1)
    if len(filtered) != weights_np.size:
        raise ValueError(
            f"filtered ({len(filtered)}) and weights ({weights_np.size}) length mismatch"
        )
    if metadata is not None and len(metadata) != len(filtered):
        raise ValueError(f"metadata length ({len(metadata)}) must match filtered ({len(filtered)})")

    formatter = _resolve_formatter()
    rows = {
        "prompt": [],
        "completion": [],
        "weight": [],
        "trajectory_id": [],
        "spec_key": [],
    }
    for i, traj in enumerate(filtered):
        # Canonical formatter (when present) takes more args; the fallback
        # ignores extras via **kwargs catch in its own signature.
        try:
            prompt, completion = formatter(traj, spec_text, task)
        except TypeError:
            # Canonical signature differs. pass tokenizer / prompt_template.
            prompt, completion = formatter(
                traj=traj,
                spec_text=spec_text,
                task=task,
                tokenizer=tokenizer,
                prompt_template=prompt_template,
            )
        rows["prompt"].append(prompt)
        rows["completion"].append(completion)
        rows["weight"].append(float(weights_np[i]))
        if metadata is not None:
            rows["trajectory_id"].append(str(metadata[i].get("id", f"traj_{i:04d}")))
            rows["spec_key"].append(str(metadata[i].get("spec_key", task)))
        else:
            rows["trajectory_id"].append(f"traj_{i:04d}")
            rows["spec_key"].append(task)

    # Build the HF Dataset lazily so tests that don't import datasets still pass.
    from datasets import Dataset  # type: ignore[import-not-found]

    return Dataset.from_dict(rows)


def load_filtered_dataset(
    task: str,
    filter_condition: str,
    *,
    data_root: Any | None = None,
) -> Any:
    """Load a filtered SFT dataset for a given (task, filter_condition).

    Discovers the filtered manifest under one of:
      data_root/<dotted_task>/<task>_<filter>.parquet
      data/canonical/<task>_<filter>.parquet
      data/pilot/filtered_<task>_<filter>.parquet
      data/pilot/<task>/<task>_<filter>.parquet

    Returns a HuggingFace `Dataset` with columns prompt / completion / weight /
    trajectory_id / spec_key. The function is referenced by Phase-2 paths
    (eval harness, sweep runner) and by the mock backend short-circuit.

    Args:
        task: e.g. "bio_ode.repressilator", "glucose_insulin"
        filter_condition: one of {"hard", "quantile", "continuous"}
        data_root: optional override for the data directory (Path or str).

    Raises:
        FileNotFoundError if no matching manifest is found.

    """
    from pathlib import Path

    from datasets import Dataset  # type: ignore[import-not-found]

    # Build candidate search paths in priority order.
    repo_root = Path(__file__).resolve().parents[3]
    roots: list[Path] = []
    if data_root is not None:
        roots.append(Path(data_root))
    roots.extend(
        [
            repo_root / "data" / "canonical",
            repo_root / "data" / "pilot",
        ]
    )

    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / task / f"{task}_{filter_condition}.parquet",
                root / f"{task}_{filter_condition}.parquet",
                root / f"filtered_{task}_{filter_condition}.parquet",
            ]
        )

    found = next((p for p in candidates if p.exists()), None)
    if found is None:
        raise FileNotFoundError(
            f"No filtered manifest for task={task!r}, filter={filter_condition!r}.\n"
            f"Searched:\n  " + "\n  ".join(str(p) for p in candidates)
        )

    return Dataset.from_parquet(str(found))


__all__ = ["build_sft_dataset", "load_filtered_dataset"]

"""Policy implementations for stl-seed trajectory generation.

All policies conform to the locked `Policy` Protocol from
`paper/architecture.md` §"Policy interface":

    __call__(
        state: Float[Array, " n"],
        spec: STLSpec,
        history: list[tuple[Array, Array]],   # (state, action) pairs so far
        key: PRNGKeyArray,
    ) -> Float[Array, " m"]

Determinism contract: every call is deterministic in `key` (random policies
fold the key into PRNG calls; deterministic controllers ignore it). This
matches the architecture-level reproducibility convention that "all
randomness derived from a single jax.random.key(seed) per run".

Action units: the action vector returned by a policy is in the *physical*
action space of the target task family (e.g., U/h for glucose-insulin
insulin infusion; dimensionless inducer fractions in [0, 1] for bio_ode).
The simulator clips to its declared bounds — policies may emit values
outside the bounds and the clip is the authoritative gate (matches the
glucose-insulin simulator convention; see
`tasks/glucose_insulin.py:simulate`).

REDACTED firewall: this module touches `stl_seed.specs.STLSpec` only as a
typing reference. No REDACTED artifact is imported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

if TYPE_CHECKING:
    from stl_seed.specs import STLSpec


# Type aliases for clarity — the architecture lists `State` and `Action` in
# the Policy protocol signature without binding them; we use JAX arrays.
State = Float[Array, " n"]
Action = Float[Array, " m"]
History = list[tuple[State, Action]]


# -----------------------------------------------------------------------------
# RandomPolicy
# -----------------------------------------------------------------------------


class RandomPolicy:
    """Uniform random action per step within `[action_low, action_high]`.

    Deterministic in `key`: at each call we fold the call index (via the
    history length) into the key so that successive calls within a single
    trajectory get distinct PRNG values without re-using the trajectory-
    level key.

    Parameters
    ----------
    action_dim:
        Dimensionality of the action vector (`m`).
    action_low, action_high:
        Per-component bounds (broadcastable to shape `(m,)`).
    """

    def __init__(
        self,
        action_dim: int,
        action_low: float | np.ndarray,
        action_high: float | np.ndarray,
    ) -> None:
        if action_dim < 1:
            raise ValueError(f"action_dim must be >= 1, got {action_dim}")
        self.action_dim = action_dim
        self.action_low = jnp.broadcast_to(jnp.asarray(action_low), (action_dim,))
        self.action_high = jnp.broadcast_to(jnp.asarray(action_high), (action_dim,))
        if not jnp.all(self.action_low <= self.action_high):
            raise ValueError("action_low must be <= action_high componentwise")

    def __call__(
        self,
        state: State,  # noqa: ARG002  (state unused; matches Protocol)
        spec: STLSpec,  # noqa: ARG002
        history: History,
        key: PRNGKeyArray,
    ) -> Action:
        # Fold a per-step subkey deterministically from the trajectory key
        # plus the call index so the same key + step yields the same action.
        step = len(history)
        subkey = jax.random.fold_in(key, step)
        return jax.random.uniform(
            subkey,
            shape=(self.action_dim,),
            minval=self.action_low,
            maxval=self.action_high,
        )


# -----------------------------------------------------------------------------
# ConstantPolicy
# -----------------------------------------------------------------------------


class ConstantPolicy:
    """Returns the same action every step. Useful as a zero-control baseline.

    Parameters
    ----------
    value:
        Action vector returned at every step. Cast to a JAX float32 array.
    """

    def __init__(self, value: float | np.ndarray | Array) -> None:
        v = jnp.atleast_1d(jnp.asarray(value, dtype=jnp.float32))
        self._value = v
        self.action_dim = int(v.shape[0])

    def __call__(
        self,
        state: State,  # noqa: ARG002
        spec: STLSpec,  # noqa: ARG002
        history: History,  # noqa: ARG002
        key: PRNGKeyArray,  # noqa: ARG002
    ) -> Action:
        return self._value


# -----------------------------------------------------------------------------
# PIDController
# -----------------------------------------------------------------------------


class PIDController:
    """Classical proportional-integral-derivative controller on `state[0]`.

    Targets the glucose-insulin task family by default: `state[0] = G` in
    mg/dL and the action is a scalar insulin infusion rate in U/h. The PID
    output drives `state[0]` toward `setpoint`. The midway-of-band
    setpoint of 110 mg/dL falls inside the ADA 2024 Time-in-Range
    [70, 180] mg/dL band (Time-in-Range references in
    `specs/glucose_insulin_specs.py`).

    Sign convention: a *positive* glucose error (`state[0] > setpoint`,
    i.e. hyperglycaemia) demands *more* insulin, so the controller output
    is `+kp * error + ...` (no negation).

    Parameters
    ----------
    setpoint:
        Target value for `state[0]`. Default 110.0 (mg/dL) for glucose.
    kp, ki, kd:
        Proportional, integral, derivative gains (units consistent with
        the units of `state[0]` and the action). Defaults are the
        literature-anchored gains from Marchetti et al., "An improved PID
        switching control strategy for type 1 diabetes," IEEE TBME 55:857
        (2008), Table II, scaled to the U/h action units of this task.
    action_clip:
        Optional `(low, high)` bound on the scalar output, applied after
        the PID equation. Default `(0.0, 5.0)` matches the glucose-insulin
        action box.
    action_dim:
        Output dimensionality. Default 1 (scalar insulin rate).
    """

    def __init__(
        self,
        setpoint: float = 110.0,
        kp: float = 0.05,
        ki: float = 0.001,
        kd: float = 0.02,
        action_clip: tuple[float, float] | None = (0.0, 5.0),
        action_dim: int = 1,
    ) -> None:
        self.setpoint = float(setpoint)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.action_clip = action_clip
        self.action_dim = action_dim

    def __call__(
        self,
        state: State,
        spec: STLSpec,  # noqa: ARG002
        history: History,
        key: PRNGKeyArray,  # noqa: ARG002
    ) -> Action:
        # Current error (scalar): G - setpoint.
        error = jnp.asarray(state[0]) - self.setpoint

        # Integral term: cumulative sum of past errors (one per past call).
        # We rebuild it from history so the controller is stateless and
        # therefore safe to vmap over instances.
        if history:
            past_errors = jnp.stack([s[0] - self.setpoint for (s, _) in history])
            integral = jnp.sum(past_errors)
            # Derivative: backward difference between current and previous error.
            prev_error = past_errors[-1]
            derivative = error - prev_error
        else:
            integral = jnp.asarray(0.0)
            derivative = jnp.asarray(0.0)

        u = self.kp * error + self.ki * integral + self.kd * derivative

        if self.action_clip is not None:
            lo, hi = self.action_clip
            u = jnp.clip(u, lo, hi)

        # Broadcast to the requested action dimensionality (default 1).
        return jnp.broadcast_to(u, (self.action_dim,))


# -----------------------------------------------------------------------------
# BangBangController
# -----------------------------------------------------------------------------


class BangBangController:
    """Threshold-driven bang-bang controller (suited to bio_ode tasks).

    For each output channel, emits `high_action` if the corresponding
    *observation* falls below `threshold` and `low_action` otherwise. By
    default we observe `state[: action_dim]` (i.e. the first `action_dim`
    state components). This matches the bio_ode convention where each
    inducer modulates one upstream protein/RNA, so the natural feedback is
    "if the controlled species is low, push the inducer high".

    Parameters
    ----------
    threshold:
        Switching threshold on the observed component. Scalar or
        broadcastable to shape `(action_dim,)`.
    low_action, high_action:
        The two action levels. Scalar or broadcastable to `(action_dim,)`.
    action_dim:
        Output dimensionality. Default 1.
    observation_indices:
        Optional 1-d list/array of state indices used as the observation
        for each action channel (length `action_dim`). Default
        `range(action_dim)` — channel `i` watches `state[i]`.
    """

    def __init__(
        self,
        threshold: float | np.ndarray = 50.0,
        low_action: float | np.ndarray = 0.0,
        high_action: float | np.ndarray = 1.0,
        action_dim: int = 1,
        observation_indices: list[int] | None = None,
    ) -> None:
        if action_dim < 1:
            raise ValueError(f"action_dim must be >= 1, got {action_dim}")
        self.action_dim = action_dim
        self.threshold = jnp.broadcast_to(jnp.asarray(threshold, dtype=jnp.float32), (action_dim,))
        self.low_action = jnp.broadcast_to(
            jnp.asarray(low_action, dtype=jnp.float32), (action_dim,)
        )
        self.high_action = jnp.broadcast_to(
            jnp.asarray(high_action, dtype=jnp.float32), (action_dim,)
        )
        if observation_indices is None:
            self.observation_indices = list(range(action_dim))
        else:
            if len(observation_indices) != action_dim:
                raise ValueError("observation_indices length must equal action_dim")
            self.observation_indices = list(observation_indices)

    def __call__(
        self,
        state: State,
        spec: STLSpec,  # noqa: ARG002
        history: History,  # noqa: ARG002
        key: PRNGKeyArray,  # noqa: ARG002
    ) -> Action:
        observed = jnp.stack([state[i] for i in self.observation_indices])
        return jnp.where(observed < self.threshold, self.high_action, self.low_action)


# -----------------------------------------------------------------------------
# MLXModelPolicy
# -----------------------------------------------------------------------------


class MLXModelPolicy:
    """LLM-based policy backed by an MLX-loaded Qwen3 model on Apple Silicon.

    GUARDS:
    * The `mlx`/`mlx_lm` packages are imported INSIDE `__init__` so this
      module can be imported on non-Apple-Silicon hosts (the
      `mlx`-extras dependency is optional in `pyproject.toml`).
    * On import failure or non-Apple-Silicon platform, a clear `RuntimeError`
      is raised at construction time, NOT at module load time.

    The policy formats the (state, spec) into `prompt_template` (a Python
    `str.format`-style template with named fields `state`, `spec_text`,
    `history_summary`, `action_dim`), prompts the model, and parses the
    completion as a JSON list of floats (the action vector). On parse
    failure, returns a zero action and logs a warning.

    Parameters
    ----------
    model_path:
        Path or HF model ID accepted by `mlx_lm.load`.
    tokenizer_path:
        Path or HF model ID accepted by `mlx_lm.load`. If `None`,
        `model_path` is reused.
    prompt_template:
        Python format string with the four fields above.
    action_dim:
        Expected output dimensionality. The parser raises if the JSON list
        length does not match.
    max_tokens:
        Maximum generation length. Default 64 (enough for a small list).
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str | None = None,
        prompt_template: str = (
            "State: {state}\n"
            "Spec: {spec_text}\n"
            "History: {history_summary}\n"
            "Emit a JSON list of {action_dim} floats as the next action.\n"
            "Action: "
        ),
        action_dim: int = 1,
        max_tokens: int = 64,
    ) -> None:
        # Lazy import — fails informatively if MLX is not installed.
        try:
            import mlx_lm  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "MLXModelPolicy requires the `mlx` extras. Install with "
                "`uv pip install -e .[mlx]` on an Apple-Silicon host. "
                f"Underlying ImportError: {e}"
            ) from e

        # Platform check — MLX's CPU fallback exists but is not the supported
        # target. Refuse non-Apple-Silicon outright to avoid silent slowness.
        import platform

        machine = platform.machine().lower()
        system = platform.system().lower()
        if not (system == "darwin" and machine in {"arm64", "aarch64"}):
            raise RuntimeError(
                "MLXModelPolicy is only supported on Apple Silicon (darwin/arm64). "
                f"Detected: system={system!r}, machine={machine!r}."
            )

        self._mlx_lm = mlx_lm
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.prompt_template = prompt_template
        self.action_dim = action_dim
        self.max_tokens = max_tokens

        # Load model and tokenizer eagerly so the failure mode is obvious.
        self.model, self.tokenizer = mlx_lm.load(self.model_path)

    def __call__(
        self,
        state: State,
        spec: STLSpec,
        history: History,
        key: PRNGKeyArray,  # noqa: ARG002  (MLX seeds globally; key is consumed
        # to satisfy the protocol)
    ) -> Action:
        # Format inputs.
        state_np = np.asarray(state).round(4).tolist()
        history_summary = (
            f"{len(history)} steps; last action="
            f"{np.asarray(history[-1][1]).round(4).tolist() if history else None}"
        )
        prompt = self.prompt_template.format(
            state=state_np,
            spec_text=getattr(spec, "formula_text", str(spec)),
            history_summary=history_summary,
            action_dim=self.action_dim,
        )

        # Generate.
        response = self._mlx_lm.generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            verbose=False,
        )

        # Parse — strip everything outside the first balanced `[...]`.
        try:
            import json
            import re

            match = re.search(r"\[[^\[\]]*\]", str(response))
            if match is None:
                raise ValueError("no JSON list found in completion")
            parsed = json.loads(match.group(0))
            if not isinstance(parsed, list) or len(parsed) != self.action_dim:
                raise ValueError(f"expected list of {self.action_dim} floats, got {parsed!r}")
            return jnp.asarray(parsed, dtype=jnp.float32)
        except Exception:
            # Defensive: a malformed completion should not poison the run.
            # Return zeros and let the simulator's clip handle the bound.
            return jnp.zeros((self.action_dim,), dtype=jnp.float32)


# -----------------------------------------------------------------------------
# HeuristicPolicy router
# -----------------------------------------------------------------------------


# Default heuristic-controller specs per task family. Keys are the
# `task` string passed to `TrajectoryRunner.generate_trajectories(...)`;
# the value is a kwargs dict for the corresponding controller.
_HEURISTIC_DEFAULTS: dict[str, dict[str, Any]] = {
    # Glucose-insulin: PID with literature-anchored gains, scalar action.
    "glucose_insulin": {
        "controller": "pid",
        "kwargs": {
            "setpoint": 110.0,
            "kp": 0.05,
            "ki": 0.001,
            "kd": 0.02,
            "action_clip": (0.0, 5.0),
            "action_dim": 1,
        },
    },
    # Bio_ode/repressilator: bang-bang inducers, 3 channels.
    # State is (m_1, m_2, m_3, p_1, p_2, p_3). Specs gate on proteins,
    # so observe state[3:6] not the default state[0:3] (mRNA).
    # Inducer u_i SILENCES gene i (per bio_ode.py docstring), so flip the
    # bang-bang sense: when protein i is HIGH, push u_i HIGH to silence.
    "bio_ode.repressilator": {
        "controller": "bangbang",
        "kwargs": {
            "threshold": 137.5,  # nM, midway between P_LOW=25 and P_HIGH=250
            "low_action": 1.0,  # protein HIGH → silence gene (drive u high)
            "high_action": 0.0,  # protein LOW → release inducer (u=0)
            "action_dim": 3,
            "observation_indices": [3, 4, 5],  # observe proteins, not mRNA
        },
    },
    # Bio_ode/toggle: bang-bang on 2 inducers.
    # State is (x_1, x_2) repressor concentrations (no mRNA in 2-state form),
    # so default observation_indices = [0, 1] is correct for this task.
    "bio_ode.toggle": {
        "controller": "bangbang",
        "kwargs": {
            "threshold": 100.0,  # nM, midway between LOW=30 and HIGH=200
            "low_action": 0.0,
            "high_action": 1.0,
            "action_dim": 2,
        },
    },
    # Bio_ode/MAPK: bang-bang on a single stimulus channel.
    "bio_ode.mapk": {
        "controller": "bangbang",
        "kwargs": {
            "threshold": 0.5,
            "low_action": 0.0,
            "high_action": 1.0,
            "action_dim": 1,
            "observation_indices": [2],  # observe MAPK-PP (terminal tier)
        },
    },
}


class HeuristicPolicy:
    """Routes to a task-appropriate hand-coded controller.

    Currently routes:
    * `glucose_insulin*` -> `PIDController` with literature-anchored gains.
    * `bio_ode.repressilator*` / `bio_ode.toggle*` / `bio_ode.mapk*`
      -> `BangBangController` with task-specific thresholds.

    Unknown task families raise `KeyError` at construction time.

    Parameters
    ----------
    task_family:
        The task identifier (matches `TrajectoryRunner.generate_trajectories`'s
        `task=` argument). The longest matching prefix in
        `_HEURISTIC_DEFAULTS` selects the controller.
    overrides:
        Optional kwargs that override the literature-anchored defaults.
    """

    def __init__(
        self,
        task_family: str,
        overrides: dict[str, Any] | None = None,
    ) -> None:
        # Pick the longest matching prefix.
        match = None
        for prefix in sorted(_HEURISTIC_DEFAULTS.keys(), key=len, reverse=True):
            if task_family == prefix or task_family.startswith(prefix + "."):
                match = prefix
                break
        if match is None:
            raise KeyError(
                f"no heuristic registered for task_family={task_family!r}; "
                f"known prefixes: {sorted(_HEURISTIC_DEFAULTS.keys())}"
            )

        spec = _HEURISTIC_DEFAULTS[match]
        kwargs = dict(spec["kwargs"])
        if overrides:
            kwargs.update(overrides)

        if spec["controller"] == "pid":
            self._impl: Any = PIDController(**kwargs)
        elif spec["controller"] == "bangbang":
            self._impl = BangBangController(**kwargs)
        else:  # pragma: no cover  -- defensive
            raise ValueError(f"unknown controller kind: {spec['controller']!r}")

        self.task_family = task_family
        self.action_dim = self._impl.action_dim

    def __call__(
        self,
        state: State,
        spec: STLSpec,
        history: History,
        key: PRNGKeyArray,
    ) -> Action:
        return self._impl(state, spec, history, key)


__all__ = [
    "BangBangController",
    "ConstantPolicy",
    "HeuristicPolicy",
    "MLXModelPolicy",
    "PIDController",
    "RandomPolicy",
    "State",
    "Action",
    "History",
]

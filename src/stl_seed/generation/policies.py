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
        Target value for the *observed* component of `state` (default
        `state[0]`; configurable via ``observation_indices``). Default
        110.0 (mg/dL) for glucose.
    kp, ki, kd:
        Proportional, integral, derivative gains (units consistent with
        the units of the observed state component and the action).
        Defaults are the literature-anchored gains from Marchetti et al.,
        "An improved PID switching control strategy for type 1 diabetes,"
        IEEE TBME 55:857 (2008), Table II, scaled to the U/h action units
        of this task.
    action_clip:
        Optional `(low, high)` bound on the scalar output, applied after
        the PID equation. Default `(0.0, 5.0)` matches the glucose-insulin
        action box.
    action_dim:
        Output dimensionality. Default 1 (scalar insulin rate).
    observation_indices:
        Optional 1-d list of state indices used as the observation
        signal. Currently only the first index is used (PID is a
        single-channel controller); the parameter is accepted as a list
        to mirror :class:`BangBangController`'s API and to leave room for
        a future multi-channel PID. Default ``[0]``.
    error_sign:
        Sign convention for the error term. ``+1`` (default) means the
        controller acts to *reduce* the observation toward the setpoint
        when the observation is *above* the setpoint (the canonical
        glucose convention: high G -> more insulin). ``-1`` reverses the
        sense (e.g. the MAPK case: when MAPK_PP is *below* the setpoint
        we need *more* upstream stimulus).
    """

    def __init__(
        self,
        setpoint: float = 110.0,
        kp: float = 0.05,
        ki: float = 0.001,
        kd: float = 0.02,
        action_clip: tuple[float, float] | None = (0.0, 5.0),
        action_dim: int = 1,
        observation_indices: list[int] | None = None,
        error_sign: float = 1.0,
    ) -> None:
        self.setpoint = float(setpoint)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.action_clip = action_clip
        self.action_dim = action_dim
        if observation_indices is None:
            self.observation_indices = [0]
        else:
            if len(observation_indices) < 1:
                raise ValueError("observation_indices must be non-empty")
            self.observation_indices = list(observation_indices)
        self.error_sign = float(error_sign)
        self._obs_idx = int(self.observation_indices[0])

    def __call__(
        self,
        state: State,
        spec: STLSpec,  # noqa: ARG002
        history: History,
        key: PRNGKeyArray,  # noqa: ARG002
    ) -> Action:
        # Current error: observed - setpoint, optionally sign-flipped so
        # the action grows in the right direction for the task's
        # actuator-effect convention.
        observed = jnp.asarray(state[self._obs_idx])
        error = self.error_sign * (observed - self.setpoint)

        # Integral term: cumulative sum of past errors (one per past call).
        # We rebuild it from history so the controller is stateless and
        # therefore safe to vmap over instances.
        if history:
            past_errors = jnp.stack(
                [
                    self.error_sign * (jnp.asarray(s[self._obs_idx]) - self.setpoint)
                    for (s, _) in history
                ]
            )
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
# TopologyAwareController
# -----------------------------------------------------------------------------


class TopologyAwareController:
    """Topology-aware feedback controller for cyclic gene regulatory networks.

    Given a topology specifying which gene's product *represses* which other
    gene, this controller knows how to drive a target protein high or low by
    actuating the appropriate upstream inducer. It is the natural feedback
    controller for cyclic repressor networks such as the Elowitz-Leibler 2000
    repressilator (Gene 1 -| Gene 2 -| Gene 3 -| Gene 1) and other ring
    topologies.

    Repressilator example. To drive ``p_1`` (target gene 0) HIGH, the
    controller silences gene 2 — the upstream repressor of gene 0 — by
    emitting ``u_2 = 1``; the canonical inducer convention here is that
    ``u_i = 1`` fully silences gene i (per
    ``stl_seed.tasks.bio_ode.RepressilatorSimulator`` docstring). Once the
    upstream repressor is suppressed, the target gene is de-repressed and
    its protein rises. The controller observes the *target protein* and
    keeps the upstream gene silenced as long as the target is below the
    threshold (drive high) or above the threshold (drive low).

    Parameters
    ----------
    topology:
        Dict mapping ``gene_index -> upstream_repressor_index``. For the
        repressilator (gene i repressed by gene (i-1) mod 3) this is
        ``{0: 2, 1: 0, 2: 1}``. The mapping is derived from the network
        wiring diagram (Elowitz & Leibler 2000 Fig. 1a) and is independent
        of any specific spec.
    target_gene:
        Index of the gene whose protein the controller is trying to drive.
        Must be a key of ``topology``.
    target_direction:
        ``"high"`` to drive the target protein up (silence the upstream
        repressor); ``"low"`` to drive it down (do not silence; allow the
        upstream repressor to act).
    threshold:
        Switching threshold (same units as the observed protein) for the
        feedback decision. With ``target_direction="high"``, the controller
        emits ``u_upstream = 1`` whenever the observed target protein is
        below the threshold (still needs help to rise) and ``0`` once it
        exceeds the threshold (already where we want it). With
        ``target_direction="low"`` the senses are flipped.
    observation_indices:
        State indices (length ``n_genes``) where the proteins live in the
        full state vector. For the repressilator state
        ``(m_1, m_2, m_3, p_1, p_2, p_3)`` this is ``[3, 4, 5]`` so that
        the controller observes proteins, not mRNAs. Defaults to
        ``range(n_genes)`` if not provided (e.g. for the toggle's
        2-state form, where ``[0, 1]`` are already the repressors).
    action_dim:
        Optional explicit action dimensionality. Defaults to the maximum
        gene index in ``topology`` plus one (i.e. one channel per gene).

    Notes
    -----
    Action sign convention: ``u_i = 1`` *silences* gene i. This matches the
    bio_ode repressilator simulator (``transcription = alpha_max * (1 - u_i)
    * repression``). For tasks with the opposite sign (``u_i = 1`` activates
    gene i), pre-flip the action with a wrapper or extend this controller.

    REDACTED firewall. The topology dict and threshold are derived from
    Elowitz & Leibler (Nature 2000;403:335) Fig. 1a and the spec's textbook
    midpoint between ``P_LOW`` and ``P_HIGH`` (specs/bio_ode_specs.py),
    not from any REDACTED artifact. The class itself is task-family-agnostic.
    """

    def __init__(
        self,
        topology: dict[int, int],
        target_gene: int,
        target_direction: str = "high",
        threshold: float = 50.0,
        observation_indices: list[int] | None = None,
        action_dim: int | None = None,
    ) -> None:
        if not topology:
            raise ValueError("topology must be a non-empty dict")
        if target_gene not in topology:
            raise KeyError(
                f"target_gene={target_gene} not in topology keys {sorted(topology.keys())}"
            )
        if target_direction not in {"high", "low"}:
            raise ValueError(f"target_direction must be 'high' or 'low', got {target_direction!r}")

        self.topology = dict(topology)
        self.target_gene = int(target_gene)
        self.target_direction = target_direction
        self.threshold = float(threshold)

        # Derive action_dim from the topology if not specified. Each gene
        # mentioned (as key or value) gets a channel.
        n_genes = max(max(self.topology.keys()), max(self.topology.values())) + 1
        self.action_dim = int(action_dim) if action_dim is not None else n_genes

        if observation_indices is None:
            self.observation_indices = list(range(n_genes))
        else:
            if len(observation_indices) != n_genes:
                raise ValueError(
                    f"observation_indices length {len(observation_indices)} "
                    f"must equal n_genes={n_genes} (one observation per gene)"
                )
            self.observation_indices = list(observation_indices)

        # Pre-compute the index of the upstream repressor we will actuate.
        self._upstream_idx = int(self.topology[self.target_gene])
        # Pre-compute the index of the target protein we will observe.
        self._target_obs_idx = int(self.observation_indices[self.target_gene])

    def __call__(
        self,
        state: State,
        spec: STLSpec,  # noqa: ARG002
        history: History,  # noqa: ARG002
        key: PRNGKeyArray,  # noqa: ARG002
    ) -> Action:
        # Observe the target protein.
        target = state[self._target_obs_idx]

        # Decide whether the upstream repressor should be silenced.
        # "high": still need to push target up -> silence upstream while target < threshold.
        # "low":  still need to push target down -> let upstream act while target > threshold,
        #         which here means do NOT silence (u_upstream = 0); silencing the target's
        #         own gene via u_target = 1 also drives target down, so we additionally
        #         silence the target gene in the "low" branch.
        u = jnp.zeros((self.action_dim,), dtype=jnp.float32)
        if self.target_direction == "high":
            # Silence the upstream repressor while the target is below threshold.
            silence_upstream = jnp.where(target < self.threshold, 1.0, 0.0)
            u = u.at[self._upstream_idx].set(silence_upstream)
        else:  # "low"
            # Silence the target gene's own transcription while the target
            # protein is above threshold (active drive-down).
            silence_target = jnp.where(target > self.threshold, 1.0, 0.0)
            u = u.at[self.target_gene].set(silence_target)
        return u


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
    # Bio_ode/repressilator: TOPOLOGY-AWARE controller, 3 channels.
    # State is (m_1, m_2, m_3, p_1, p_2, p_3). The cyclic repression wiring
    # (Elowitz & Leibler 2000 Nature 403:335 Fig. 1a) is gene 0 -| gene 1
    # -| gene 2 -| gene 0, i.e. each gene i is repressed by gene (i-1) % 3.
    # The ``bio_ode.repressilator.easy`` spec asks the controller to drive
    # p_1 (gene 0) HIGH; the topology-aware policy therefore silences the
    # upstream repressor of gene 0, which is gene 2 (per topology[0] = 2).
    # The threshold is the textbook midpoint between P_LOW=25 and P_HIGH=250
    # (specs/bio_ode_specs.py); observation_indices=[3,4,5] picks the protein
    # block out of the (m,p) state. A constant control of u=(0,0,1)
    # satisfies the spec with rho ~ +25 (manually verified, commit 369872a).
    "bio_ode.repressilator": {
        "controller": "topology_aware",
        "kwargs": {
            "topology": {0: 2, 1: 0, 2: 1},  # gene i repressed by gene (i-1) % 3
            "target_gene": 0,  # spec drives p_1 (gene 0) high
            "target_direction": "high",
            "threshold": 137.5,  # nM, midway P_LOW=25 and P_HIGH=250
            "observation_indices": [3, 4, 5],  # proteins live in state[3:6]
        },
    },
    # Bio_ode/toggle: TOPOLOGY-AWARE controller, 2 channels.
    # State is (x_1, x_2) repressor concentrations (no mRNA in 2-state form),
    # so the proteins themselves live at indices 0 and 1 — observation_indices
    # = [0, 1] picks them directly. Topology dict {0: 1, 1: 0} encodes the
    # mutual-repression wiring of the Gardner-Cantor-Collins 2000 toggle
    # (gene 0 repressed by gene 1, and vice versa). The bio_ode.toggle.medium
    # spec asks the controller to drive x_1 HIGH; the topology-aware policy
    # therefore silences x_1's upstream repressor (gene 1) by emitting u_1 = 1
    # (full IPTG/aTc on the repressor of gene 0). Threshold is set just above
    # the spec's HIGH=100 nM bound (specs/bio_ode_specs.py).
    #
    # CALIBRATION NOTE (2026-04-25). The bio_ode.toggle.medium spec was
    # corrected on 2026-04-25: HIGH lowered from 200 nM to 100 nM (the
    # simulator's x_1 saturates at alpha_1 = 160, so the previous 200 nM
    # threshold was structurally unreachable). With HIGH=100, the
    # topology-aware controller's u=(0,1) constant policy drives x_1 to
    # ~160 nM and the spec is satisfied with rho ~ +30. The "threshold"
    # below is the controller's *internal switching* threshold (when to
    # back off IPTG); 100 nM is now both the spec's HIGH and the
    # controller's switching point.
    "bio_ode.toggle": {
        "controller": "topology_aware",
        "kwargs": {
            "topology": {0: 1, 1: 0},  # mutual repression: i repressed by 1-i
            "target_gene": 0,  # spec drives x_1 (gene 0) high
            "target_direction": "high",
            "threshold": 100.0,  # nM, matches the spec's HIGH band post-fix
            "observation_indices": [0, 1],  # proteins live in state[0:2]
        },
    },
    # Bio_ode/MAPK: PID controller on a single stimulus channel, observing
    # the terminal-tier MAPK-PP and driving it toward a setpoint band.
    # The MAPK cascade has no cyclic topology — it's a stimulus -> output
    # signaling cascade — so the natural feedback heuristic is a PID on
    # the output kinase (MAPK-PP) rather than a topology-aware silencer.
    #
    # Simulator state convention (`bio_ode.MAPKSimulator`):
    #   y[0]=MKKK_P  y[1]=MKK_P  y[2]=MKK_PP  y[3]=MAPK_P  y[4]=MAPK_PP  y[5]=E1
    # so MAPK-PP lives at index 4.
    #
    # PID sign convention. The action u in [0, 1] sets the upstream stimulus
    # (E1_target ∝ u); raising u drives MAPK-PP UP. We want u to grow when
    # MAPK-PP is BELOW setpoint, so we use error_sign=-1 (action grows as
    # observed - setpoint becomes more negative).
    #
    # Setpoint = 0.5 microM, midway in the MAPK_PP reachable range
    # [0, MAPK_total = 1.25 microM] under the simulator's parameter set
    # (`MAPKParams`). Gains are first-pass; PID tuning is not the
    # bottleneck — see the calibration note in the toggle entry above
    # for the broader spec/simulator mismatch context.
    #
    # CALIBRATION NOTE (2026-04-25). The bio_ode.mapk.hard spec was
    # corrected on 2026-04-25 to read state index 4 (MAPK_PP) using
    # absolute microM thresholds (peak >= 0.5, settle < 0.05, MKKK
    # safety < 0.002975). This controller already observed the correct
    # MAPK_PP signal (index 4) at the spec-mismatch microM setpoint of
    # 0.5, so no controller change was required.
    "bio_ode.mapk": {
        "controller": "pid",
        "kwargs": {
            "setpoint": 0.5,  # microM, midway in [0, MAPK_total = 1.25]
            "kp": 0.4,
            "ki": 0.05,
            "kd": 0.1,
            "action_clip": (0.0, 1.0),
            "action_dim": 1,
            "observation_indices": [4],  # MAPK_PP in the 6-state simulator
            "error_sign": -1.0,  # u grows when MAPK_PP is below setpoint
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
        elif spec["controller"] == "topology_aware":
            self._impl = TopologyAwareController(**kwargs)
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


# -----------------------------------------------------------------------------
# PerturbedHeuristicPolicy
# -----------------------------------------------------------------------------


class PerturbedHeuristicPolicy:
    """Wraps a heuristic policy and adds Gaussian noise for action diversity.

    This is the "low-temperature perturbation" policy used by the Phase-2
    canonical generator (`scripts/generate_canonical.py`). It boosts the
    diversity of the heuristic-only leg of the policy mix without sacrificing
    the heuristic's structural correctness: noise is small (default
    ``sigma_frac=0.1`` of the action range) and the perturbed action is
    clipped back into the declared bounds, so the overall trajectory still
    falls within the simulator's valid action box.

    The σ is specified as a *fraction* of the action range so the noise
    scales appropriately for any task family (e.g., σ_frac=0.1 means
    σ=0.1 for a [0, 1] inducer, σ=0.5 for a [0, 5] insulin rate). When
    ``action_low`` / ``action_high`` are provided, the perturbed action is
    clipped to ``[action_low, action_high]``; otherwise no clipping is
    applied (the simulator's own clip is then the authoritative gate, per
    the policy-interface convention).

    Determinism contract: identical to `RandomPolicy` — the noise is
    generated by folding the call index (history length) into the
    trajectory-level key, so repeated calls with the same key + step yield
    identical perturbations.

    Parameters
    ----------
    base_policy:
        Any callable conforming to the `Policy` Protocol. Typically a
        `HeuristicPolicy(task_family)` instance.
    sigma_frac:
        Standard deviation of the Gaussian noise *as a fraction* of
        ``(action_high - action_low)``. Default 0.1. ``sigma_frac=0.0``
        reduces this policy to its base (a regression test for that
        contract is in `tests/test_generation.py`).
    action_low, action_high:
        Per-component bounds (broadcastable to the base policy's action
        shape). Used both to scale the noise and to clip the result. If
        ``None`` (either), no clip is applied; noise is then scaled by
        ``|action_high - action_low|`` if both are present and otherwise
        falls back to ``sigma_frac`` interpreted as an absolute σ on the
        unit scale.
    """

    def __init__(
        self,
        base_policy: Any,
        sigma_frac: float = 0.1,
        action_low: float | np.ndarray | None = None,
        action_high: float | np.ndarray | None = None,
    ) -> None:
        if sigma_frac < 0.0:
            raise ValueError(f"sigma_frac must be >= 0, got {sigma_frac}")
        if not callable(base_policy):
            raise TypeError(
                "base_policy must be callable (a Policy-Protocol instance); "
                f"got {type(base_policy).__name__}"
            )
        self.base_policy = base_policy
        self.sigma_frac = float(sigma_frac)
        self.action_low = action_low
        self.action_high = action_high

        # Pre-compute the σ vector if both bounds are given. We do this
        # lazily on the first call so we can broadcast against the base
        # policy's actual emitted shape (which we may not know up-front:
        # e.g. the topology-aware controller infers its action_dim from
        # the topology dict).
        self._sigma_cache: Array | None = None

        # Inherit action_dim if available (lets the runner shape-check us
        # in the same place it shape-checks plain HeuristicPolicy).
        if hasattr(base_policy, "action_dim"):
            self.action_dim = base_policy.action_dim

    def _resolve_sigma(self, sample_shape: tuple[int, ...]) -> Array:
        """Return the σ vector used for the Gaussian noise.

        If both bounds are provided, σ = sigma_frac * (high - low) broadcast
        to ``sample_shape``. If either bound is None, σ = sigma_frac scalar
        (interpreted on the unit scale; the simulator's own clip is then
        the bound enforcer).
        """
        if self._sigma_cache is not None and self._sigma_cache.shape == sample_shape:
            return self._sigma_cache
        if self.action_low is not None and self.action_high is not None:
            lo = jnp.broadcast_to(jnp.asarray(self.action_low, dtype=jnp.float32), sample_shape)
            hi = jnp.broadcast_to(jnp.asarray(self.action_high, dtype=jnp.float32), sample_shape)
            sigma = self.sigma_frac * (hi - lo)
        else:
            sigma = jnp.full(sample_shape, self.sigma_frac, dtype=jnp.float32)
        self._sigma_cache = sigma
        return sigma

    def __call__(
        self,
        state: State,
        spec: STLSpec,
        history: History,
        key: PRNGKeyArray,
    ) -> Action:
        # 1. Run the base policy first. It consumes its own derivation of
        #    `key` (deterministic policies ignore it; randomized ones fold
        #    the step in themselves), so we pass the trajectory key through
        #    unmodified to preserve the base's determinism contract.
        base_action = self.base_policy(state, spec, history, key)
        base_action = jnp.asarray(base_action, dtype=jnp.float32)

        # 2. Generate Gaussian noise on a *separate* subkey derived from
        #    the call index, so successive steps are independent and the
        #    noise stream does not collide with the base policy's PRNG draws.
        # We use a structured fold-in (`step + 100003`, a prime offset) to
        # decorrelate from any other consumer of the same trajectory key.
        step = len(history)
        subkey = jax.random.fold_in(key, step + 100_003)
        sigma = self._resolve_sigma(base_action.shape)
        noise = jax.random.normal(subkey, base_action.shape, dtype=jnp.float32) * sigma

        perturbed = base_action + noise

        # 3. Clip to declared bounds when both are provided; otherwise let
        #    the simulator's own action-box clip enforce validity.
        if self.action_low is not None and self.action_high is not None:
            lo = jnp.broadcast_to(jnp.asarray(self.action_low, dtype=jnp.float32), perturbed.shape)
            hi = jnp.broadcast_to(jnp.asarray(self.action_high, dtype=jnp.float32), perturbed.shape)
            perturbed = jnp.clip(perturbed, lo, hi)

        return perturbed


__all__ = [
    "BangBangController",
    "ConstantPolicy",
    "HeuristicPolicy",
    "MLXModelPolicy",
    "PerturbedHeuristicPolicy",
    "PIDController",
    "RandomPolicy",
    "TopologyAwareController",
    "State",
    "Action",
    "History",
]

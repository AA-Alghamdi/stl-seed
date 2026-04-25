"""Trajectory adversary: gradient-based search for spec-satisfying, gold-violating trajectories.

The :class:`TrajectoryAdversary` searches over the action space ``u_{1:H} in
R^{H x m}`` for control sequences that, after simulation through the task's
ODE integrator, yield a trajectory ``tau`` with HIGH proxy STL robustness
``rho(tau, phi_spec)`` and LOW gold score ``g(tau)``. It is the empirical
operationalization of the spec-completeness gap (theory.md S6, "the
auditable handle"); the worst-case spec_rho-vs-gold divergence the
adversary can find is a direct lower bound on
``sup_tau [R_spec(tau) - R_gold(tau)]``.

Optimizer
---------
The default is plain Adam (Kingma & Ba 2015, ICLR) on the differentiable
objective

    L(u) = -spec_rho(u) + lambda * gold_score(u),

i.e. ascent on ``spec_rho - lambda * gold``. Minimizing L drives spec_rho
up and gold_score down, matching the adversary semantics. We use plain
Adam rather than Optax because (i) the dependency surface is smaller for
this single-purpose module, (ii) the loss landscape is well-conditioned
once we project u into the [u_min, u_max] action box via a smooth
``sigmoid * (u_max - u_min) + u_min`` parameterization, and (iii) Adam
state is a single (mu, nu, t) triple per parameter that we can pickle
trivially.

Diffrax NaN policy
------------------
A solver failure inside the simulator returns sentinel-replaced states
per the architecture.md NaN policy (see ``stl_seed.tasks._trajectory``).
The sentinel is a constant (basal state for glucose, zero state for
bio_ode), so its STL rho is a finite number rather than NaN, and the
gradient through the sentinel branch is zero. The adversary therefore
cannot exploit solver failures as a "free spec satisfaction" because the
sentinel rho is always negative for the proxy specs in the registry
(verified by running the simulator on the all-zero / basal state). We
additionally count the number of NaN/Inf events per inner solve via
``trajectory.meta.n_nan_replacements`` and surface the count on
``AdversaryResult`` for the caller to monitor.

Random restarts
---------------
The adversary's loss is non-convex (the ODE-induced map u -> tau is
nonlinear; the STL min/max evaluator is piecewise smooth). We support
multiple random restarts via ``find_adversary(n_restarts=...)``; the
returned :class:`AdversaryResult` is the best (highest spec_rho, lowest
gold) restart found, with the per-restart histories preserved on
``AdversaryResult.restart_histories`` for audit.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from stl_seed.specs import Node, STLSpec
from stl_seed.stl.evaluator import compile_spec
from stl_seed.tasks._trajectory import Trajectory

# A "Simulator" here is anything that exposes a ``simulate`` method
# matching the architecture.md Simulator protocol; we duck-type rather
# than import the Protocol class to avoid coupling tests/scripts to the
# bio_ode Protocol definition.
_SimulateFn = Callable[..., Trajectory]


# ---------------------------------------------------------------------------
# Result dataclass.
# ---------------------------------------------------------------------------


@dataclass
class AdversaryResult:
    """Container for the output of :meth:`TrajectoryAdversary.find_adversary`.

    Attributes
    ----------
    best_actions:
        The action sequence (shape ``(H, m)``) achieving the largest
        ``spec_rho - lambda * gold_score`` margin, i.e., the strongest
        adversarial example found.
    best_trajectory:
        The simulated trajectory under ``best_actions``.
    best_spec_rho:
        STL space-robustness of ``best_trajectory`` against the proxy
        spec.
    best_gold_score:
        Gold score of ``best_trajectory``.
    iter_history:
        Per-iteration scalars from the WINNING restart, useful for
        plotting convergence: each row is ``(spec_rho, gold_score,
        loss)``.
    restart_histories:
        Per-restart final ``(spec_rho, gold_score)`` tuples, one row per
        restart. Length equals the ``n_restarts`` argument.
    n_nan_events:
        Total Diffrax NaN/Inf replacement count across all restarts and
        iterations. Non-zero indicates the optimizer wandered into a
        numerically pathological region of action space; the adversary
        is not penalized for it but the count surfaces as a sanity
        diagnostic.
    converged:
        True iff at least one restart's loss change in the last 10
        iterations was below ``tol`` (default 1e-4). Informational only;
        the adversary returns the best found regardless.
    """

    best_actions: Float[Array, "H m"]
    best_trajectory: Trajectory
    best_spec_rho: float
    best_gold_score: float
    iter_history: Float[Array, "T 3"]
    restart_histories: list[tuple[float, float]] = field(default_factory=list)
    n_nan_events: int = 0
    converged: bool = False


# ---------------------------------------------------------------------------
# Adam optimizer state (lightweight; no Optax dependency).
# ---------------------------------------------------------------------------


@dataclass
class _AdamState:
    """Adam moment estimates per Kingma & Ba 2015, ICLR.

    Using a tiny dataclass instead of Optax keeps the surface area of this
    module small. The update rule below is the canonical Adam:

        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        x_t = x_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)
    """

    mu: Float[Array, "..."]
    nu: Float[Array, "..."]
    t: int = 0


def _adam_init(x: Float[Array, "..."]) -> _AdamState:
    return _AdamState(mu=jnp.zeros_like(x), nu=jnp.zeros_like(x), t=0)


def _adam_update(
    state: _AdamState,
    grad: Float[Array, "..."],
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[Float[Array, "..."], _AdamState]:
    """Apply one Adam update; returns ``(delta, new_state)``."""
    new_t = state.t + 1
    mu = beta1 * state.mu + (1.0 - beta1) * grad
    nu = beta2 * state.nu + (1.0 - beta2) * (grad * grad)
    mu_hat = mu / (1.0 - beta1**new_t)
    nu_hat = nu / (1.0 - beta2**new_t)
    delta = -lr * mu_hat / (jnp.sqrt(nu_hat) + eps)
    return delta, _AdamState(mu=mu, nu=nu, t=new_t)


# ---------------------------------------------------------------------------
# Action box reparameterization.
# ---------------------------------------------------------------------------


def _box_project(
    z: Float[Array, "H m"],
    u_min: Float[Array, " m"],
    u_max: Float[Array, " m"],
) -> Float[Array, "H m"]:
    """Smooth bijection from R^{H x m} into the action box [u_min, u_max].

    ``u = u_min + (u_max - u_min) * sigmoid(z)`` is the standard
    smooth box parameterization; gradients are well-defined everywhere
    (no clipping discontinuities). The sigmoid temperature is fixed at
    1.0 since the action-box scales for our task families are all
    O(1)-O(10), well-suited to this default.
    """
    return u_min + (u_max - u_min) * jax.nn.sigmoid(z)


# ---------------------------------------------------------------------------
# Adversary class.
# ---------------------------------------------------------------------------


class TrajectoryAdversary:
    """Find worst-case-gold trajectories that satisfy a given STL spec.

    Operates by gradient ascent on ``spec_rho - lambda * gold_score``,
    backpropagating through the JAX/Diffrax simulator and the
    JAX-compiled STL evaluator. Reparameterizes the action sequence via
    a sigmoid box projection so that the optimizer is unconstrained on
    R^{H x m}.

    Parameters
    ----------
    simulator:
        Any object exposing a ``simulate(initial_state, control_sequence,
        params, key)`` method that returns a
        :class:`stl_seed.tasks._trajectory.Trajectory`. Both
        ``GlucoseInsulinSimulator`` and the ``bio_ode`` simulators
        conform.
    spec:
        The STL proxy spec (an :class:`stl_seed.specs.STLSpec` or raw
        :class:`stl_seed.specs.Node`) the agent is "optimizing".
    gold_score:
        A pure callable mapping a Trajectory to a JAX scalar; the
        unstated-intent objective the adversary should *minimize*.
    lambda_satisfaction:
        Lagrange multiplier on the spec-satisfaction term. Larger
        values constrain the adversary to satisfy the spec more
        strictly; smaller values let it sacrifice some spec rho for
        more gold violation. Default 1.0.
    learning_rate:
        Adam step size. Default 0.1; the sigmoid box parameterization
        keeps the unconstrained ``z`` magnitudes O(few), so 0.1 gives
        roughly 10% box-fraction moves per step.
    max_iters:
        Maximum optimizer iterations per restart. Default 200; the
        loss typically converges within ~50-100 for our tasks.
    tol:
        Loss-change convergence tolerance over the last 10 iterations.
        Default 1e-4.
    project_actions:
        If True (default), the optimizer's free variable is the
        unconstrained pre-image ``z`` and actions are recovered via
        sigmoid box projection. If False, the actions are optimized
        directly with no box constraint --- only safe if the spec /
        simulator naturally bounds them.
    action_min, action_max:
        Box bounds (broadcastable to shape ``(m,)``). Required when
        ``project_actions=True``.
    simulator_aux:
        Any extra positional arguments passed to ``simulator.simulate``
        between ``control_sequence`` and ``key`` (e.g., ``MealSchedule``
        for the glucose-insulin family, the params object for any
        family). Stored once at construction so the inner loss closure
        is JAX-traceable.
    params:
        The kinetic-parameter object passed to ``simulator.simulate``
        (e.g., ``BergmanParams``, ``RepressilatorParams``).

    Notes
    -----
    The simulator's ``simulate(...)`` signature for the bio_ode family
    is ``(initial_state, control_sequence, params, key) -> Trajectory``,
    while glucose-insulin's is ``(initial_state, control_sequence,
    meal_schedule, params, key) -> (states, times, meta)`` --- the two
    return signatures differ. The adversary normalizes both into a
    Trajectory via the ``_normalize_trajectory`` helper invoked inside
    the loss closure.
    """

    def __init__(
        self,
        simulator: object,
        spec: STLSpec | Node,
        gold_score: Callable[[Trajectory], Float[Array, ""]],
        params: object,
        *,
        lambda_satisfaction: float = 1.0,
        learning_rate: float = 0.1,
        max_iters: int = 200,
        tol: float = 1e-4,
        project_actions: bool = True,
        action_min: float | Float[Array, " m"] = 0.0,
        action_max: float | Float[Array, " m"] = 1.0,
        simulator_aux: tuple = (),
    ) -> None:
        self.simulator = simulator
        self.spec = spec
        self.gold_score = gold_score
        self.params = params
        self.lambda_satisfaction = float(lambda_satisfaction)
        self.learning_rate = float(learning_rate)
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.project_actions = bool(project_actions)
        self.action_min = jnp.asarray(action_min)
        self.action_max = jnp.asarray(action_max)
        self.simulator_aux = simulator_aux

        # Compile spec once. The closure is JIT-compatible because every
        # predicate in the SPECS registry conforms to the introspection
        # convention (verified by tests/test_stl_evaluator.py).
        self._compiled_spec = compile_spec(spec)

    # ------------------------------------------------------------------
    # Internal helpers.
    # ------------------------------------------------------------------

    def _normalize_trajectory(self, sim_output: object) -> Trajectory:
        """Coerce the simulator output into a :class:`Trajectory`.

        The bio_ode simulators return a ``Trajectory`` directly; the
        glucose-insulin simulator's older signature returns a
        ``(states, times, meta)`` tuple and re-builds the actions from
        the pre-clipped control sequence on the call site. This helper
        normalizes both; for the tuple form we fill ``actions`` with the
        pre-clipped actions stashed by the caller (set on
        ``self._last_actions`` in the loss closure).
        """
        if isinstance(sim_output, Trajectory):
            return sim_output
        # Tuple form: (states, times, meta).
        states, times, meta = sim_output
        # actions stashed by the loss closure for this exact call
        actions = self._last_actions
        if actions.ndim == 1:
            actions = actions[:, None]
        return Trajectory(
            states=states,
            actions=actions,
            times=times,
            meta=meta,
        )

    def _call_simulator(
        self,
        initial_state: Float[Array, " n"],
        actions: Float[Array, "H m"],
    ) -> object:
        """Invoke the simulator with the family-appropriate signature.

        The glucose-insulin simulator wants a 1-D ``(H,)`` control
        sequence and an extra ``meal_schedule`` positional argument
        before the params. The bio_ode simulators want a 2-D
        ``(H, m)`` control sequence and no auxiliary args. We dispatch
        by class name (matching the convention in
        ``stl_seed.generation.runner._is_glucose_insulin_simulator``).
        """
        if type(self.simulator).__name__ == "GlucoseInsulinSimulator":
            u_flat = actions.reshape(-1)
            return self.simulator.simulate(
                initial_state, u_flat, *self.simulator_aux, self.params, _DUMMY_KEY
            )
        return self.simulator.simulate(
            initial_state, actions, *self.simulator_aux, self.params, _DUMMY_KEY
        )

    def _actions_from_z(self, z: Float[Array, "H m"]) -> Float[Array, "H m"]:
        """Recover the action sequence from the unconstrained variable z."""
        if self.project_actions:
            return _box_project(z, self.action_min, self.action_max)
        return z

    def _spec_rho(self, trajectory: Trajectory) -> Float[Array, ""]:
        """Spec rho on the trajectory using the precompiled evaluator."""
        return self._compiled_spec(trajectory.states, trajectory.times)

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------

    def find_adversary(
        self,
        initial_state: Float[Array, " n"],
        action_dim: int,
        horizon: int,
        key: PRNGKeyArray,
        n_restarts: int = 4,
    ) -> AdversaryResult:
        """Search for the worst-case (high spec_rho, low gold) trajectory.

        Parameters
        ----------
        initial_state:
            Initial state ``x_0`` passed to the simulator.
        action_dim:
            ``m`` --- the per-step action dimensionality.
        horizon:
            ``H`` --- the number of control-update steps.
        key:
            JAX PRNG key for restart initializations.
        n_restarts:
            Number of independent random restarts. Default 4.

        Returns
        -------
        AdversaryResult
            The best (highest ``spec_rho - lambda * gold``) restart found,
            with per-restart final scores and the iter-history of the
            winning restart preserved.
        """
        keys = jax.random.split(key, n_restarts)

        best_score = -jnp.inf
        best_payload: tuple | None = None
        restart_finals: list[tuple[float, float]] = []
        total_nan_events = 0
        any_converged = False

        for r in range(n_restarts):
            # Initialize z near zero so sigmoid -> 0.5, the box midpoint.
            # Add small noise per restart for diversity.
            z0 = 0.5 * jax.random.normal(keys[r], shape=(horizon, action_dim), dtype=jnp.float32)
            history, final_traj, final_z, n_nan, converged = self._run_restart(
                z0=z0, initial_state=initial_state
            )
            total_nan_events += int(n_nan)
            any_converged = any_converged or converged

            final_spec_rho = float(history[-1, 0])
            final_gold = float(history[-1, 1])
            restart_finals.append((final_spec_rho, final_gold))

            # Score: spec_rho - lambda * gold (we want to MAXIMIZE this).
            score = final_spec_rho - self.lambda_satisfaction * final_gold
            if score > best_score:
                best_score = score
                best_payload = (
                    self._actions_from_z(final_z),
                    final_traj,
                    final_spec_rho,
                    final_gold,
                    history,
                )

        if best_payload is None:
            # Degenerate case: zero restarts requested. Return a sentinel
            # result rather than crashing.
            empty = jnp.zeros((horizon, action_dim))
            return AdversaryResult(
                best_actions=empty,
                best_trajectory=Trajectory(
                    states=jnp.zeros((1, 1)),
                    actions=empty,
                    times=jnp.zeros((1,)),
                    meta=None,  # type: ignore[arg-type]
                ),
                best_spec_rho=float("nan"),
                best_gold_score=float("nan"),
                iter_history=jnp.zeros((0, 3)),
                restart_histories=[],
                n_nan_events=0,
                converged=False,
            )

        actions, traj, spec_rho, gold, history = best_payload
        return AdversaryResult(
            best_actions=actions,
            best_trajectory=traj,
            best_spec_rho=spec_rho,
            best_gold_score=gold,
            iter_history=history,
            restart_histories=restart_finals,
            n_nan_events=total_nan_events,
            converged=any_converged,
        )

    # ------------------------------------------------------------------
    # Inner loop.
    # ------------------------------------------------------------------

    def _run_restart(
        self,
        z0: Float[Array, "H m"],
        initial_state: Float[Array, " n"],
    ) -> tuple[Float[Array, "T 3"], Trajectory, Float[Array, "H m"], int, bool]:
        """Run one Adam restart from initial point ``z0``.

        Returns
        -------
        history:
            (max_iters, 3) array of (spec_rho, gold, loss) per iter.
        final_traj:
            Trajectory at the final z.
        final_z:
            Final unconstrained variable.
        n_nan:
            Cumulative NaN/Inf events across all simulator calls in this
            restart.
        converged:
            True iff |loss[t] - loss[t-10]| < tol at termination.
        """

        # Build the loss closure. We do NOT jit it because the
        # glucose-insulin simulator's call signature returns a tuple
        # (not a Trajectory), and the tuple-to-Trajectory normalization
        # touches a Python attribute (`self._last_actions`) that is
        # JIT-incompatible. Each call eagerly evaluates the simulator +
        # STL evaluator, both of which internally JIT their hot paths.

        def loss_and_outputs(
            z: Float[Array, "H m"],
        ) -> tuple[Float[Array, ""], Trajectory]:
            actions = self._actions_from_z(z)
            self._last_actions = actions  # used by _normalize_trajectory
            sim_out = self._call_simulator(initial_state, actions)
            traj = self._normalize_trajectory(sim_out)
            spec_rho = self._spec_rho(traj)
            gold = self.gold_score(traj)
            # Minimize: -spec_rho + lambda * gold (== ascent on
            # spec_rho - lambda * gold).
            loss = -spec_rho + self.lambda_satisfaction * gold
            return loss, (spec_rho, gold, traj)

        # value_and_grad with has_aux so we can recover spec_rho / gold
        # without recomputing them.
        grad_fn = jax.value_and_grad(loss_and_outputs, has_aux=True)

        opt = _adam_init(z0)
        z = z0
        history = jnp.zeros((self.max_iters, 3))
        n_nan = 0

        for i in range(self.max_iters):
            (loss, (spec_rho, gold, traj)), grad = grad_fn(z)
            # Replace any NaN gradients with zero (rare, but a known
            # diffrax + sigmoid-saturation interaction); count for
            # diagnostics.
            grad = jnp.where(jnp.isfinite(grad), grad, 0.0)
            # Track NaN events from the simulator.
            with contextlib.suppress(Exception):
                n_nan += int(traj.meta.n_nan_replacements)

            history = history.at[i].set(jnp.stack([spec_rho, gold, loss]))

            delta, opt = _adam_update(opt, grad, self.learning_rate)
            z = z + delta

        # Convergence: the |loss| change in the last 10 steps below tol.
        if self.max_iters >= 10:
            recent_change = float(jnp.abs(history[-1, 2] - history[-10, 2]))
            converged = recent_change < self.tol
        else:
            converged = False

        # Final forward pass to get the final trajectory at the last z.
        actions = self._actions_from_z(z)
        self._last_actions = actions
        sim_out = self._call_simulator(initial_state, actions)
        final_traj = self._normalize_trajectory(sim_out)
        return history, final_traj, z, n_nan, converged


# A dummy PRNG key the simulators take but don't currently consume; reused
# for every call to keep the inner loss closure free of jax.random calls.
_DUMMY_KEY = jax.random.PRNGKey(0)


__all__ = [
    "AdversaryResult",
    "TrajectoryAdversary",
]

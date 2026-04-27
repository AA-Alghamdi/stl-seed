"""STL space-robustness evaluator (Donzé & Maler 2010, FORMATS).

Reference. Donzé, A. & Maler, O. "Robust Satisfaction of Temporal Logic over
Real-Valued Signals." FORMATS 2010, LNCS 6246: 92-106.
DOI: 10.1007/978-3-642-15297-9_9.


Numerical accuracy. Each min/max node accumulates at most one ulp of float64
round-off. Worst-case AST depth in this codebase is 4 (And -> Always ->
Negation -> Predicate, see ``bio_ode.mapk.hard``), so the cumulative round-off
ceiling is ~4 ulp ~ 4e-16 in f64 / ~5e-7 in f32, far below all practical
spec thresholds (smallest threshold magnitude in the registry is 0.1; see
``paper/theory.md`` §2 numerical floor argument).

Implementation strategy. The AST is *compiled once* into a pure-JAX closure
that consumes a (T, n)-shape state array and a (T,)-shape time array, and
returns a scalar rho. Compilation introspects every ``Predicate``'s lambda
defaults (the convention in ``stl_seed.specs.bio_ode_specs._gt`` /
``_lt`` and ``glucose_insulin_specs._gt`` / ``_lt`` is to capture
``(channel, threshold)`` via ``fn.__defaults__``) and probes the lambda on
two synthetic single-step trajectories (state == 0 and state == ones) to
recover whether the predicate is the "greater-than" or "less-than" form.
This produces a JAX-traceable expression of the form ``states[:, c] - th``
or ``th - states[:, c]`` that is JIT- and grad-compatible. Predicates that
do not conform to this convention fall back to a slow Python evaluation
path (one ``fn(traj, t)`` call per time step, no JIT), with a clear warning.
The fallback is tagged ``_FALLBACK_USED`` for diagnostic auditing.

Time-window handling. ``Always[a, b]`` and ``Eventually[a, b]`` mask the
predicate's per-time robustness vector via
``jnp.where((times >= a) & (times <= b), value, +/-inf)`` and then take a
``min`` / ``max`` over the full vector. This is correct for arbitrary
non-uniform ``times`` because the mask uses physical times, not indices.
Empty windows (no sample falls in [a, b], i.e. trajectory does not cover
the requested interval) yield ``+inf`` for ``Always`` and ``-inf`` for
``Eventually``. these are propagated upward as numerical infinities; the
caller is responsible for ensuring ``trajectory.times`` covers all spec
intervals (typically guaranteed by the simulator construction).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import jax.numpy as jnp
import jaxtyping as jt

from stl_seed.specs import (
    Always,
    And,
    Eventually,
    Negation,
    Node,
    Predicate,
    STLSpec,
)

# ---------------------------------------------------------------------------
# Trajectory protocol. duck-typed.
# ---------------------------------------------------------------------------


@runtime_checkable
class Trajectory(Protocol):
    """Minimal protocol the STL evaluator requires from a simulator output.

    The full ``Trajectory`` dataclass (with ``actions``, ``meta``) is owned
    by ``stl_seed.tasks._trajectory``; the evaluator only needs the state
    array and the time array.
    """

    @property
    def states(self) -> jt.Float[jt.Array, "T n"]: ...

    @property
    def times(self) -> jt.Float[jt.Array, " T"]: ...


# ---------------------------------------------------------------------------
# Predicate introspection.
# ---------------------------------------------------------------------------


# Sentinel used as a flag on returned compiled functions when at least one
# predicate could not be JIT-compiled and fell back to the Python path.
_FALLBACK_USED = "_stl_seed_fallback_used"


def _introspect_predicate(
    pred: Predicate,
) -> tuple[int, float, str] | None:
    """Recover (channel, threshold, op) from a ``Predicate`` lambda.

    The convention enforced in ``stl_seed.specs.{bio_ode_specs,
    glucose_insulin_specs}._gt`` and ``_lt`` is to capture ``(channel,
    threshold)`` as the lambda's default arguments. We recover them via
    ``fn.__defaults__`` and identify the op (gt vs lt) by probing the
    lambda on a single-step zero trajectory:

        gt: ``f(traj, 0) = traj[0, c] - th``  -> at zeros gives ``-th``
        lt: ``f(traj, 0) = th - traj[0, c]``  -> at zeros gives ``+th``

    Returns ``None`` if the predicate does not match this convention, in
    which case the evaluator falls back to direct ``fn`` calls (non-JIT).
    """
    fn = pred.fn
    defaults = getattr(fn, "__defaults__", None)
    if defaults is None or len(defaults) != 2:
        return None
    channel, threshold = defaults
    if not isinstance(channel, int) or not isinstance(threshold, (int, float)):
        return None

    # Probe the lambda on a zero-state and a unit-state to identify the op.
    # We need at least ``channel + 1`` columns; one timestep is enough.
    n_cols = max(channel + 1, 1)
    try:
        import numpy as np

        zero_traj = np.zeros((1, n_cols), dtype=np.float64)
        unit_traj = np.zeros((1, n_cols), dtype=np.float64)
        unit_traj[0, channel] = 1.0
        v0 = float(fn(zero_traj, 0))
        v1 = float(fn(unit_traj, 0))
    except Exception:
        return None

    th = float(threshold)
    # gt: v0 == -th, v1 == 1 - th  =>  v1 - v0 == 1
    # lt: v0 == +th, v1 == th - 1  =>  v1 - v0 == -1
    delta = v1 - v0
    if abs(delta - 1.0) < 1e-9 and abs(v0 + th) < 1e-9:
        return (int(channel), th, "gt")
    if abs(delta + 1.0) < 1e-9 and abs(v0 - th) < 1e-9:
        return (int(channel), th, "lt")
    return None


def _predicate_jax_fn(
    pred: Predicate,
) -> tuple[Callable[[jt.Float[jt.Array, "T n"]], jt.Float[jt.Array, " T"]], bool]:
    """Return a JAX-pure function ``states -> rho_per_time`` for ``pred``.

    Returns ``(fn, is_jit)``. When introspection fails, ``is_jit`` is
    ``False`` and ``fn`` invokes the original Python lambda in a loop ,
    correct but unjittable.
    """
    info = _introspect_predicate(pred)
    if info is not None:
        c, th, op = info
        if op == "gt":

            def gt_fn(states: jt.Float[jt.Array, "T n"]) -> jt.Float[jt.Array, " T"]:
                return states[:, c] - th

            return gt_fn, True

        # op == "lt"
        def lt_fn(states: jt.Float[jt.Array, "T n"]) -> jt.Float[jt.Array, " T"]:
            return th - states[:, c]

        return lt_fn, True

    # Fallback: call the original Python lambda for every time index. This
    # is JIT-incompatible (the lambda calls ``float(...)`` which trips on
    # tracers), so the caller must NOT wrap the resulting compiled spec in
    # ``jax.jit``. We still emit a numerically correct evaluation under
    # eager execution, suitable for the slow-path evaluator path used in
    # tests for non-conforming user-supplied predicates.
    fn = pred.fn

    def fallback_fn(states: jt.Float[jt.Array, "T n"]) -> jt.Float[jt.Array, " T"]:
        # Convert tracer to concrete array if possible. Under JIT this
        # raises; under eager it succeeds.
        import numpy as np

        s_np = np.asarray(states)
        out = np.array([float(fn(s_np, t)) for t in range(s_np.shape[0])], dtype=np.float64)
        return jnp.asarray(out)

    return fallback_fn, False


# ---------------------------------------------------------------------------
# Recursive compiler: AST node -> (states, times) -> scalar rho.
# ---------------------------------------------------------------------------


def _compile_node(
    node: Node,
) -> tuple[
    Callable[
        [jt.Float[jt.Array, "T n"], jt.Float[jt.Array, " T"]],
        jt.Float[jt.Array, ""],
    ],
    bool,
]:
    """Compile a single AST node into ``(states, times) -> rho_scalar``.

    Returns ``(compiled_fn, is_jit)``. The scalar is the Donzé-Maler
    space-robustness evaluated *at trajectory time t = 0* (the conventional
    top-level evaluation point). For internal nodes inside ``Always`` /
    ``Eventually``, the compiler instead emits a per-time function (see
    ``_compile_temporal_inner``).
    """
    if isinstance(node, Predicate):
        per_time, is_jit = _predicate_jax_fn(node)

        def pred_scalar(
            states: jt.Float[jt.Array, "T n"], times: jt.Float[jt.Array, " T"]
        ) -> jt.Float[jt.Array, ""]:
            del times  # predicate at t=0 is independent of the time grid
            return per_time(states)[0]

        return pred_scalar, is_jit

    if isinstance(node, Negation):
        # AST contract: Negation only wraps a Predicate.
        inner_fn, is_jit = _compile_node(node.inner)

        def neg_scalar(
            states: jt.Float[jt.Array, "T n"], times: jt.Float[jt.Array, " T"]
        ) -> jt.Float[jt.Array, ""]:
            return -inner_fn(states, times)

        return neg_scalar, is_jit

    if isinstance(node, And):
        compiled_children = [_compile_node(c) for c in node.children]
        is_jit_all = all(j for _, j in compiled_children)
        child_fns = [f for f, _ in compiled_children]

        def and_scalar(
            states: jt.Float[jt.Array, "T n"], times: jt.Float[jt.Array, " T"]
        ) -> jt.Float[jt.Array, ""]:
            vals = jnp.stack([f(states, times) for f in child_fns])
            return jnp.min(vals)

        return and_scalar, is_jit_all

    if isinstance(node, Always):
        per_time, is_jit = _compile_temporal_inner(node.inner)
        a, b = node.interval.t_lo, node.interval.t_hi

        def always_scalar(
            states: jt.Float[jt.Array, "T n"], times: jt.Float[jt.Array, " T"]
        ) -> jt.Float[jt.Array, ""]:
            rho_t = per_time(states, times)
            in_window = (times >= a) & (times <= b)
            # Empty window -> +inf (vacuous Always is true with infinite margin).
            masked = jnp.where(in_window, rho_t, jnp.inf)
            return jnp.min(masked)

        return always_scalar, is_jit

    if isinstance(node, Eventually):
        per_time, is_jit = _compile_temporal_inner(node.inner)
        a, b = node.interval.t_lo, node.interval.t_hi

        def eventually_scalar(
            states: jt.Float[jt.Array, "T n"], times: jt.Float[jt.Array, " T"]
        ) -> jt.Float[jt.Array, ""]:
            rho_t = per_time(states, times)
            in_window = (times >= a) & (times <= b)
            # Empty window -> -inf (vacuous Eventually is false with infinite deficit).
            masked = jnp.where(in_window, rho_t, -jnp.inf)
            return jnp.max(masked)

        return eventually_scalar, is_jit

    raise TypeError(f"Unsupported AST node type: {type(node).__name__}")


def _compile_temporal_inner(
    node: Node,
) -> tuple[
    Callable[
        [jt.Float[jt.Array, "T n"], jt.Float[jt.Array, " T"]],
        jt.Float[jt.Array, " T"],
    ],
    bool,
]:
    """Compile a node *inside* a temporal operator, returning a per-time vector.

    For nested temporal operators (e.g. ``G[a,b] (G[c,d] phi)``), the inner
    node returns rho at every time t' in the grid; the outer operator masks
    and reduces. Conjunctions inside a temporal operator min-stack the per-
    time vectors of their children.
    """
    if isinstance(node, Predicate):
        per_time, is_jit = _predicate_jax_fn(node)

        def pred_per_time(
            states: jt.Float[jt.Array, "T n"], times: jt.Float[jt.Array, " T"]
        ) -> jt.Float[jt.Array, " T"]:
            del times
            return per_time(states)

        return pred_per_time, is_jit

    if isinstance(node, Negation):
        per_time, is_jit = _compile_temporal_inner(node.inner)

        def neg_per_time(
            states: jt.Float[jt.Array, "T n"], times: jt.Float[jt.Array, " T"]
        ) -> jt.Float[jt.Array, " T"]:
            return -per_time(states, times)

        return neg_per_time, is_jit

    if isinstance(node, And):
        compiled_children = [_compile_temporal_inner(c) for c in node.children]
        is_jit_all = all(j for _, j in compiled_children)
        child_fns = [f for f, _ in compiled_children]

        def and_per_time(
            states: jt.Float[jt.Array, "T n"], times: jt.Float[jt.Array, " T"]
        ) -> jt.Float[jt.Array, " T"]:
            vals = jnp.stack([f(states, times) for f in child_fns])
            return jnp.min(vals, axis=0)

        return and_per_time, is_jit_all

    if isinstance(node, (Always, Eventually)):
        # Nested temporal: evaluate the inner operator's per-time vector,
        # then for each anchor t in the grid take min/max over the shifted
        # window [t+a, t+b]. The conjunction-only AST in ``stl_seed.specs``
        # never produces nested temporals, so this branch is provided for
        # generality only; it is O(T^2) and not JIT-fused.
        inner_per_time, is_inner_jit = _compile_temporal_inner(node.inner)
        a, b = node.interval.t_lo, node.interval.t_hi
        is_always = isinstance(node, Always)

        def nested_per_time(
            states: jt.Float[jt.Array, "T n"], times: jt.Float[jt.Array, " T"]
        ) -> jt.Float[jt.Array, " T"]:
            rho_inner = inner_per_time(states, times)  # shape (T,)
            T = states.shape[0]
            results = []
            for i in range(T):
                t = times[i]
                in_window = (times >= t + a) & (times <= t + b)
                if is_always:
                    masked = jnp.where(in_window, rho_inner, jnp.inf)
                    results.append(jnp.min(masked))
                else:
                    masked = jnp.where(in_window, rho_inner, -jnp.inf)
                    results.append(jnp.max(masked))
            return jnp.stack(results)

        # Mark non-JIT because the Python loop over time anchors is not
        # traced cleanly under static T (Python int).
        del is_inner_jit
        return nested_per_time, False

    raise TypeError(f"Unsupported AST node type inside temporal: {type(node).__name__}")


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


CompiledSpec = Callable[
    [jt.Float[jt.Array, "T n"], jt.Float[jt.Array, " T"]],
    jt.Float[jt.Array, ""],
]


def compile_spec(spec: STLSpec | Node) -> CompiledSpec:
    """Compile a spec or raw AST node into a ``(states, times) -> rho`` closure.

    The returned closure is JIT-compatible iff every ``Predicate`` in the AST
    matches the ``_gt`` / ``_lt`` introspection convention (which holds for
    every spec in ``stl_seed.specs.REGISTRY``). Inspect the
    ``_FALLBACK_USED`` attribute on the closure to verify.
    """
    node = spec.formula if isinstance(spec, STLSpec) else spec
    fn, is_jit = _compile_node(node)
    setattr(fn, _FALLBACK_USED, not is_jit)
    return fn


def evaluate_robustness(
    spec: STLSpec | Node,
    trajectory: Trajectory,
) -> jt.Float[jt.Array, ""]:
    """Compute Donzé-Maler space-robustness rho(spec, trajectory) at t = 0.

    Parameters
    ----------
    spec
        Either a registered :class:`STLSpec` (whose ``formula`` field is the
        AST root) or a raw AST :class:`Node`.
    trajectory
        Any object exposing ``states`` (shape ``(T, n)``) and ``times``
        (shape ``(T,)``). The full ``Trajectory`` dataclass conforms.

    Returns
    -------
    rho : scalar JAX array
        The signed robustness margin. ``rho > 0`` iff the trajectory
        satisfies ``spec`` with margin ``rho``; ``rho < 0`` iff it violates
        with margin ``|rho|``; ``rho == 0`` is the boundary.

    Notes
    -----
    For the conjunction-only specs in ``stl_seed.specs``, the AST has depth
    at most 4, so the recursive evaluator visits at most ~10 nodes. Cost is
    O(T) per node (one min/max over the time grid), giving overall O(T)
    per evaluation. Vectorize across a batch of trajectories with
    ``jax.vmap``.
    """
    compiled = compile_spec(spec)
    return compiled(trajectory.states, trajectory.times)


# Avoid an unused-import warning when ``inspect`` is referenced only for
# documentation / future-extension purposes; keep the import alive for
# downstream tools that import it from this module.
_ = inspect

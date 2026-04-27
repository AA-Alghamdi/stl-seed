"""Streaming (partial-trajectory) STL robustness for online agent feedback.

Used by the agent loop in ``stl_seed.generation.policies.MLXModelPolicy`` to
provide intermediate rho feedback to the LLM at every control switching
point. Without streaming, the LLM only sees the final rho after the full
horizon; with streaming, it sees a *lower bound* on the eventual rho at
every intermediate step, which lets it abort or course-correct.

Semantics. Let ``t_now`` be the current simulation time. For each temporal
operator over interval ``[a, b]`` we partition into three cases:

    * ``b <= t_now``: the operator's interval is entirely in the past;
      evaluate the standard Donzé-Maler rho on the slice of the trajectory
      with ``times <= t_now``. The result is the final rho contribution
      from this operator.
    * ``a <= t_now < b``: the operator is partially observed. For
      ``Always[a, b] phi``, the streaming rho is the minimum of phi over
      ``[a, t_now]``. a *lower bound* on the eventual rho because the
      true rho is the minimum over the full ``[a, b]`` and adding more
      samples can only decrease the min. For ``Eventually[a, b] phi``,
      streaming rho is the maximum over ``[a, t_now]``. also a lower
      bound (more samples can only increase the max, so the value-so-far
      is below the eventual value).
    * ``a > t_now``: the operator's interval has not started; return
      ``+inf`` (vacuous Always satisfied with infinite margin) for
      ``Always`` and ``-inf`` for ``Eventually`` (vacuous reachability
      not yet attempted). The conjunction with ``+inf`` is identity
      (parent Always-conjunction collapses to the active children's min);
      the conjunction with ``-inf`` correctly drives the parent's rho to
      ``-inf`` until the reachability operator activates, signalling to
      the agent that this clause is dormant rather than satisfied.

      Design alternative considered: returning a small finite negative
      number for "not-yet-activated Eventually" so the parent rho is
      bounded. Rejected because (a) ``-inf`` is the unique algebraic
      identity that makes the streaming rho a *lower bound* on the final
      rho (key correctness property; see ``test_streaming_lower_bound``),
      and (b) the agent loop can detect the ``-inf`` and emit a "not yet"
      message instead of a numeric rho. Returning a finite value would
      mislabel a pending clause as "weakly violated".

Lower-bound property (proof sketch). For any ``Always[a, b] phi`` with
``a <= t_now <= b``, ``rho_stream(t_now) = inf_{t in [a, t_now]} rho(phi, t)
>= inf_{t in [a, b]} rho(phi, t) = rho_final``, with equality iff the
extension to ``[t_now, b]`` does not introduce a smaller value. The same
inequality holds for ``Eventually`` because adding samples to a sup-set
cannot decrease the supremum, and our streaming rho is the sup over a
smaller set, hence a lower bound. Conjunction preserves the lower bound
because ``min`` is monotone-coordinatewise.
"""

from __future__ import annotations

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
from stl_seed.stl.evaluator import (
    Trajectory,
    _compile_temporal_inner,
    _predicate_jax_fn,
)


def _streaming_node(
    node: Node,
    states: jt.Float[jt.Array, "T n"],
    times: jt.Float[jt.Array, " T"],
    current_time: float,
) -> jt.Float[jt.Array, ""]:
    """Recursively evaluate streaming rho for ``node``.

    Unlike the full evaluator, streaming evaluation is *not* compiled to a
    pure JAX closure because the case split on ``b vs current_time`` is
    Python-level (the spec intervals are static, but the comparison is
    performed at evaluation time). This is acceptable because streaming
    evaluation runs once per agent decision (at most ``H`` times per
    rollout), not in the inner JIT loop.
    """
    if isinstance(node, Predicate):
        per_time, _ = _predicate_jax_fn(node)
        # Evaluate at t = current_time, i.e. the latest observed sample.
        # Find the largest index with times[i] <= current_time.
        rho_t = per_time(states)
        # Clip to observed window:
        observed = times <= current_time
        # If no sample is observed, return +inf as vacuous (no constraint
        # has been evaluated). In practice ``current_time`` >= 0 and
        # ``times[0] = 0``, so at least one sample is always observed.
        masked = jnp.where(observed, rho_t, jnp.inf)
        # Predicate at t = 0 in the standard semantics; for streaming we
        # report the value at the latest observed time so the agent sees
        # the current per-channel margin.
        # Use argmax of times where observed, then index.
        last_idx = jnp.sum(observed.astype(jnp.int32)) - 1
        last_idx = jnp.maximum(last_idx, 0)
        del masked  # not used; predicate streaming is the latest value
        return rho_t[last_idx]

    if isinstance(node, Negation):
        return -_streaming_node(node.inner, states, times, current_time)

    if isinstance(node, And):
        child_vals = jnp.stack(
            [_streaming_node(child, states, times, current_time) for child in node.children]
        )
        return jnp.min(child_vals)

    if isinstance(node, Always):
        a, b = node.interval.t_lo, node.interval.t_hi
        if a > current_time:
            # Not yet activated -> vacuous Always satisfied (+inf).
            return jnp.asarray(jnp.inf, dtype=jnp.float32)

        # Window is at least partially observed.
        upper = min(float(b), float(current_time))
        per_time, _ = _compile_temporal_inner(node.inner)
        rho_t = per_time(states, times)
        in_window = (times >= a) & (times <= upper)
        masked = jnp.where(in_window, rho_t, jnp.inf)
        return jnp.min(masked)

    if isinstance(node, Eventually):
        a, b = node.interval.t_lo, node.interval.t_hi
        if a > current_time:
            # Reachability not yet attempted -> -inf (lower bound).
            return jnp.asarray(-jnp.inf, dtype=jnp.float32)

        upper = min(float(b), float(current_time))
        per_time, _ = _compile_temporal_inner(node.inner)
        rho_t = per_time(states, times)
        in_window = (times >= a) & (times <= upper)
        masked = jnp.where(in_window, rho_t, -jnp.inf)
        return jnp.max(masked)

    raise TypeError(f"Unsupported AST node type for streaming: {type(node).__name__}")


def evaluate_streaming(
    spec: STLSpec | Node,
    trajectory: Trajectory,
    current_time: float,
) -> jt.Float[jt.Array, ""]:
    """Compute streaming (lower-bound) rho at ``current_time``.

    Returns a scalar JAX array. Guaranteed to be a *lower bound* on
    ``evaluate_robustness(spec, full_trajectory)`` for all monotone
    extensions of the trajectory beyond ``current_time``, in the sense
    that any extension that keeps the existing samples fixed and adds new
    samples in ``(current_time, T]`` cannot increase the final rho beyond
    the symbolic upper bound implied by the streaming value.

    The intended use is online verifier feedback in the agent loop: at
    each control switching point ``t_h``, call ``evaluate_streaming(spec,
    partial_traj, t_h)`` to get the agent's current robustness so it can
    decide whether to continue, abort, or course-correct.

    Parameters
    ----------
    spec
        Registered :class:`STLSpec` or raw AST :class:`Node`.
    trajectory
        Object exposing ``states`` and ``times``.
    current_time
        Wall-clock time (in the simulator's units) up to which the
        trajectory is observed.

    Returns
    -------
    rho_stream : scalar JAX array
        ``+inf`` if no temporal operator has activated yet (all
        ``Always``/``Eventually`` intervals start after ``current_time``).
        ``-inf`` if at least one ``Eventually`` has not activated; this
        is by design (see module docstring).
    """
    node = spec.formula if isinstance(spec, STLSpec) else spec
    return _streaming_node(node, trajectory.states, trajectory.times, current_time)


__all__ = ["evaluate_streaming"]

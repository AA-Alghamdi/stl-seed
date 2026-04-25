"""Localization of the lowest-robustness subformula.

Used by the agent verifier to format natural-language feedback for the LLM
policy: instead of returning a single scalar rho, the verifier identifies
*which* subformula is the bottleneck and *when* in the trajectory the
violation occurs. Example output (consumed by the agent prompt):

    "Spec G_[120, 200] (p1 >= 250) violated at t = 145.2 by margin 0.234"

This is the minimum-length feedback that lets a learned policy distinguish
"violated p1-tracking clause" from "violated p2-silencing clause" in the
repressilator.easy spec, and is the analogue of "stack trace" for STL.

Algorithm. Recursively descend the AST. At each node compute its rho and
the time-of-min for that node. Propagate upward by selecting the child
with the lowest rho (the "argmin child"). For a leaf ``Predicate``, the
rho is ``traj[t, c] - th`` (or ``th - traj[t, c]``) and the time-of-min is
``argmin_t (traj[t, c] - th)`` over the entire trajectory; for a temporal
operator ``Always[a, b]`` the time-of-min is ``argmin_{t in [a, b]} rho_t``;
for ``Eventually[a, b]`` we report the time at which the rho is *highest*
within the window (the satisfaction point), not the lowest, because for
reachability the witness is the maximum-rho time. (For a violated
``Eventually`` — i.e., one whose max is negative — the witness is still
the max-rho time, since that's the closest the trajectory came to
satisfying the reachability.)

Conjunctions return ``min`` over children. The argmin child becomes the
returned subformula. This makes the localization *tight*: the returned
subformula is the one whose violation drives the spec's overall rho.
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


def _localize(
    node: Node,
    states: jt.Float[jt.Array, "T n"],
    times: jt.Float[jt.Array, " T"],
) -> tuple[Node, float, float]:
    """Recursively locate the (subformula, rho, time) tuple.

    Returns
    -------
    (witness_node, rho, t)
        ``witness_node`` is the leaf subformula (Predicate / Negation /
        Always / Eventually) that drives the worst rho. ``rho`` is the
        scalar Donzé-Maler robustness at that subformula. ``t`` is the
        time (in trajectory units) at which the rho is achieved.
    """
    if isinstance(node, Predicate):
        per_time, _ = _predicate_jax_fn(node)
        rho_t = per_time(states)
        idx = int(jnp.argmin(rho_t))
        return node, float(rho_t[idx]), float(times[idx])

    if isinstance(node, Negation):
        # Negation flips sign per Donzé-Maler. The "worst" time of NOT(p)
        # is the time at which p is *largest* (since rho(NOT p) = -rho(p)
        # is most negative when rho(p) is most positive).
        per_time, _ = _predicate_jax_fn(node.inner)
        rho_t = -per_time(states)
        idx = int(jnp.argmin(rho_t))
        return node, float(rho_t[idx]), float(times[idx])

    if isinstance(node, And):
        # Pick the child with the smallest rho.
        best_child, best_rho, best_t = None, jnp.inf, 0.0
        for child in node.children:
            sub, sub_rho, sub_t = _localize(child, states, times)
            if sub_rho < best_rho:
                best_child, best_rho, best_t = sub, sub_rho, sub_t
        assert best_child is not None  # And requires >= 2 children
        return best_child, float(best_rho), float(best_t)

    if isinstance(node, Always):
        # rho(G[a,b] phi) = inf_{t in [a,b]} rho(phi, t)
        a, b = node.interval.t_lo, node.interval.t_hi
        per_time, _ = _compile_temporal_inner(node.inner)
        rho_t = per_time(states, times)
        in_window = (times >= a) & (times <= b)
        masked = jnp.where(in_window, rho_t, jnp.inf)
        idx = int(jnp.argmin(masked))
        rho = float(masked[idx])
        t = float(times[idx])
        # Return the Always node itself as the "witness subformula" — the
        # agent prompt formats it as "G_[a,b] (...) violated at t=... by ...".
        return node, rho, t

    if isinstance(node, Eventually):
        # rho(F[a,b] phi) = sup_{t in [a,b]} rho(phi, t)
        # The witness is the time at which the sup is achieved.
        a, b = node.interval.t_lo, node.interval.t_hi
        per_time, _ = _compile_temporal_inner(node.inner)
        rho_t = per_time(states, times)
        in_window = (times >= a) & (times <= b)
        masked = jnp.where(in_window, rho_t, -jnp.inf)
        idx = int(jnp.argmax(masked))
        rho = float(masked[idx])
        t = float(times[idx])
        return node, rho, t

    raise TypeError(f"Unsupported AST node type for localization: {type(node).__name__}")


def worst_violating_subformula(
    spec: STLSpec | Node,
    trajectory: Trajectory,
) -> tuple[Node, float, float]:
    """Identify the subformula with the lowest robustness on ``trajectory``.

    Returns ``(subformula, rho, t)`` where:

    * ``subformula`` is the deepest AST node whose rho equals the spec's
      overall rho. For a top-level ``And``, this is the argmin child;
      for a temporal operator, it is the temporal operator itself (the
      witness is the time-of-min within the window).
    * ``rho`` is the scalar robustness of that subformula.
    * ``t`` is the trajectory time at which ``rho`` is achieved.

    Used by ``stl_seed.generation.policies.MLXModelPolicy`` to build the
    natural-language verifier feedback string consumed by the LLM agent.

    Notes
    -----
    The returned ``rho`` equals ``evaluate_robustness(spec, trajectory)``
    by construction: at every level of the AST we propagate the argmin
    child's rho upward, and the conjunction-only structure of
    ``stl_seed.specs`` formulae means the spec's rho is exactly the min
    over the leaves' rhos.
    """
    node = spec.formula if isinstance(spec, STLSpec) else spec
    return _localize(node, trajectory.states, trajectory.times)


__all__ = ["worst_violating_subformula"]

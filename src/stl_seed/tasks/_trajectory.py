"""Shared `Trajectory` and `TrajectoryMeta` types for `stl-seed` task families.

This module is the canonical home of the locked `Trajectory` interface from
`paper/architecture.md` (subphase 1.3). Both the glucose-insulin family and
the bio_ode family (Repressilator, Toggle, MAPK) construct values of these
types as the output of `Simulator.simulate(...)`. Downstream stages (STL
evaluator, filter, training, eval) consume the same canonical pytree.

Design notes
------------

* `Trajectory` and `TrajectoryMeta` are `equinox.Module` subclasses so that
  they are first-class JAX pytrees: vmap-friendly, jit-friendly, and
  serializable via `equinox.tree_serialise_leaves` if desired later.
* All numerical fields are `jaxtyping`-annotated `Float[Array, "..."]`. Even
  scalar diagnostics (`n_nan_replacements`, `final_solver_result`,
  `used_stiff_fallback`) are 0-d JAX arrays rather than plain Python ints,
  because mixing Python scalars into an `eqx.Module` pytree breaks `jit`
  retracing semantics (the trace would specialize on every distinct integer
  value).
* The `meta` field is itself a pytree, so the whole `Trajectory` is one
  pytree leaf-set; `jax.tree.map(...)` walks both states/actions/times AND
  meta together.

This file does NOT import from `REDACTED`, `REDACTED`, `REDACTED`,
`REDACTED`, or `REDACTED*`. The `Trajectory` interface is
documented in `paper/architecture.md` and was authored independently of any
REDACTED artifact.
"""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Float


class TrajectoryMeta(eqx.Module):
    """Diagnostic metadata returned alongside the trajectory.

    All fields are 0-d JAX arrays (not Python scalars) so that the dataclass
    is a valid pytree and `simulate(...)` is jit-compatible. To get plain
    Python values for logging, do `int(meta.n_nan_replacements)` etc. AFTER
    the jitted call returns.

    Fields
    ------
    n_nan_replacements:
        Count of NaN/Inf state values that were replaced by sentinel values
        during the integration (per the architecture.md NaN policy).
    final_solver_result:
        Diffrax `RESULTS` integer code at the end of integration. Zero
        indicates clean success; non-zero codes are documented in
        `diffrax._solution.RESULTS` (e.g., `max_steps_reached`).
    used_stiff_fallback:
        0/1 flag: 1 if the simulator switched to a stiff solver (Kvaerno5)
        for this run, 0 otherwise. Currently always 0; the stiff-fallback
        path is exposed via constructor argument and is not yet auto-
        selected at runtime.
    """

    n_nan_replacements: Float[Array, ""]
    final_solver_result: Float[Array, ""]
    used_stiff_fallback: Float[Array, ""]


class Trajectory(eqx.Module):
    """Canonical trajectory pytree for the full stl-seed pipeline.

    A `Trajectory` is the sole output type of every `Simulator.simulate(...)`
    call (architecture.md §"Simulator interface"). It is consumed unmodified
    by the STL evaluator, the filter, the training tokenizer, and the eval
    harness. Treat the field shapes as load-bearing — many downstream
    operations vmap or scan over `states[:, channel]` and assume the second
    axis is the state-dim axis.

    Fields
    ------
    states:
        Per-save-time state vector, shape `(T, n)` where `T` is
        `n_save_points` and `n` is the simulator's `state_dim`.
    actions:
        Piecewise-constant control schedule used during the integration,
        shape `(H, m)` where `H` is the simulator's `horizon` (number of
        control points) and `m` is `action_dim`. Stored alongside the
        trajectory so the training tokenizer has the agent's emitted
        action sequence in canonical form without re-deriving it from
        the simulator state.
    times:
        Save-time grid, shape `(T,)`, in the simulator's time units (minutes
        for both glucose-insulin and bio_ode).
    meta:
        Diagnostic metadata (`TrajectoryMeta`), as a sub-pytree.
    """

    states: Float[Array, "T n"]
    actions: Float[Array, "H m"]
    times: Float[Array, " T"]
    meta: TrajectoryMeta


__all__ = [
    "Trajectory",
    "TrajectoryMeta",
]

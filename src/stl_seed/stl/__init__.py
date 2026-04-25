"""JAX-native STL robustness evaluation for ``stl-seed``.

Implements standard space-robustness semantics due to Donzé & Maler
(FORMATS 2010, "Robust Satisfaction of Temporal Logic over Real-Valued
Signals", DOI: 10.1007/978-3-642-15297-9_9) on the AST defined in
``stl_seed.specs``:

    rho(p >= c, tau, t)        = tau(t) - c
    rho(NOT p, tau, t)         = -rho(p, tau, t)
    rho(phi AND psi, tau, t)   = min(rho(phi, tau, t), rho(psi, tau, t))
    rho(G_[a,b] phi, tau, t)   = inf_{t' in [t+a, t+b]} rho(phi, tau, t')
    rho(F_[a,b] phi, tau, t)   = sup_{t' in [t+a, t+b]} rho(phi, tau, t')

REDACTED firewall (per ``CLAUDE.md``): this module does NOT import REDACTED
``pystl``. The semantics above are coded from scratch on the conjunction-only
``stl_seed.specs`` AST (see ``paper/REDACTED.md`` Part C).

Public API:

    evaluate_robustness(spec, trajectory) -> jt.Float[jt.Array, ""]
        Full-trajectory rho. JIT-compatible.

    evaluate_streaming(spec, trajectory, current_time) -> jt.Float[jt.Array, ""]
        Partial-trajectory rho for online use during the agent loop.

    worst_violating_subformula(spec, trajectory) -> tuple[STLNode, float, float]
        Localization of the lowest-robustness subformula. Returns
        (subformula, min_rho, time_of_min). Used by the agent verifier to
        format natural-language feedback.

    Trajectory: a minimal ``Protocol`` describing the simulator output the
        evaluator consumes. The full ``Trajectory`` dataclass is owned by
        ``stl_seed.tasks._trajectory`` (added by Subphase 1.3 agent A8); we
        accept any duck-typed object exposing ``states`` and ``times``.
"""

from __future__ import annotations

from stl_seed.stl.evaluator import (
    Trajectory,
    compile_spec,
    evaluate_robustness,
)
from stl_seed.stl.streaming import evaluate_streaming
from stl_seed.stl.worst_subformula import worst_violating_subformula

__all__ = [
    "Trajectory",
    "compile_spec",
    "evaluate_robustness",
    "evaluate_streaming",
    "worst_violating_subformula",
]

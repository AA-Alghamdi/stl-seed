"""Backend-agnostic STL specification AST and registries.

Phase 1 / subphase 1.2 deliverable. These specifications describe the *form*
of STL formulae that ``stl-seed`` will use to score control trajectories. A
concrete robustness evaluator is wired in subphase 1.3 (it can target either
``stljax`` for autograd, or a NumPy reference implementation).

REDACTED firewall posture (see ``paper/REDACTED.md`` Part C):

* Every formula in this package is composed exclusively from ``Always``,
  ``Eventually``, conjunction, and predicate-level negation. No top-level
  disjunction. No implication. No ``Until``.
* Every numerical threshold is sourced from biological / clinical literature
  and cited inline at the point of use. No threshold is transcribed from the
  REDACTED paper, ``REDACTED.py``, or any ``REDACTED*.py`` artifact.
* The signal scales used here (protein concentrations in nM, glucose in
  mg/dL, normalised phosphorylation fraction in [0, 1], time in minutes) do
  not coincide with the REDACTED dimensionless ``x_i ∈ [0, 1.5]`` on ``t ∈
  [0, 25]`` regime, so even accidental numerical overlap is structurally
  excluded.

Public API:

>>> from stl_seed.specs import REGISTRY
>>> spec = REGISTRY["bio_ode.repressilator.easy"]
>>> spec.signal_dim, spec.horizon_minutes
(3, 200.0)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# AST nodes (backend-agnostic).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Predicate:
    """Atomic predicate ``f(signal[t], t) >= 0``.

    ``fn`` consumes the full trajectory array of shape ``(T, signal_dim)``
    plus a discrete time index, and returns a scalar real value. By STL
    convention, the predicate is *true* when ``fn(...) >= 0``; the signed
    value is the predicate-level robustness.
    """

    name: str
    fn: Callable[[np.ndarray, int], float]


@dataclass(frozen=True)
class Negation:
    """Predicate-level negation only (per firewall §C.1).

    The wrapped node MUST be a :class:`Predicate`. Negation outside an
    atomic predicate (e.g. ``Negation(Always(...))``) is forbidden because it
    can introduce disjunctive structure under De Morgan, which would violate
    the conjunction-only requirement.
    """

    inner: Predicate

    def __post_init__(self) -> None:
        if not isinstance(self.inner, Predicate):
            raise TypeError(
                "Negation may only wrap a Predicate (firewall §C.1)."
            )


@dataclass(frozen=True)
class Interval:
    """Closed bounded time interval ``[t_lo, t_hi]`` in *physical* units.

    Units are minutes for every spec in this package.
    """

    t_lo: float
    t_hi: float

    def __post_init__(self) -> None:
        if not (self.t_lo <= self.t_hi):
            raise ValueError(f"Interval requires t_lo <= t_hi, got {self}")


@dataclass(frozen=True)
class Always:
    """``G_[t_lo, t_hi] inner`` — inner must hold throughout the interval."""

    inner: Node
    interval: Interval


@dataclass(frozen=True)
class Eventually:
    """``F_[t_lo, t_hi] inner`` — inner must hold at some point in the interval."""

    inner: Node
    interval: Interval


@dataclass(frozen=True)
class And:
    """N-ary conjunction. The only Boolean combinator allowed in this package."""

    children: tuple[Node, ...]

    def __post_init__(self) -> None:
        if len(self.children) < 2:
            raise ValueError("And requires at least two children")


# Type alias for any node.
Node = Predicate | Negation | Always | Eventually | And


# ---------------------------------------------------------------------------
# Spec wrapper.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class STLSpec:
    """A textbook-derived STL specification with provenance metadata.

    Attributes
    ----------
    name : str
        Registry key, e.g. ``"bio_ode.repressilator.easy"``.
    formula : Node
        Backend-agnostic AST.
    signal_dim : int
        Number of channels of the trajectory signal.
    horizon_minutes : float
        Total simulated horizon ``T`` in minutes.
    description : str
        Human-readable English statement of the spec.
    citations : tuple[str, ...]
        Inline citations for every numerical threshold.
    formula_text : str
        Standard STL syntax for the formula, for inclusion in the paper and
        the firewall audit.
    """

    name: str
    formula: Node
    signal_dim: int
    horizon_minutes: float
    description: str
    citations: tuple[str, ...]
    formula_text: str
    metadata: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry — populated by the per-family modules.
# ---------------------------------------------------------------------------


REGISTRY: dict[str, STLSpec] = {}


def register(spec: STLSpec) -> STLSpec:
    """Register a spec under its ``name``. Raises on duplicate keys."""

    if spec.name in REGISTRY:
        raise KeyError(f"Spec {spec.name!r} already registered")
    REGISTRY[spec.name] = spec
    return spec


# Import the per-family modules to populate the registry. Done at the bottom
# so that AST symbols above are available to the spec modules.
from stl_seed.specs import bio_ode_specs as bio_ode_specs  # noqa: E402,F401
from stl_seed.specs import glucose_insulin_specs as glucose_insulin_specs  # noqa: E402,F401

__all__ = [
    "Predicate",
    "Negation",
    "Interval",
    "Always",
    "Eventually",
    "And",
    "Node",
    "STLSpec",
    "REGISTRY",
    "register",
    "bio_ode_specs",
    "glucose_insulin_specs",
]

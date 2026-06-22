"""Toy coding-agent simulator (minimum viable cell).

Bridges ``stl-seed`` from biomolecular ODEs to a coding-agent task per
``paper/coding_task_design.md``. This is a *spike*, not the full design.

Major scope cuts vs. the design doc
-----------------------------------

1. **Dataset.** ``paper/coding_task_design.md`` calls for HumanEval-mutated
   (~50 problems) with `ast`-based mutation generation. This module instead
   ships ``TINY_TASKS``, a hand-written set of 5 buggy Python functions with
   unit-test suites of size 3. No HumanEval dependency.

2. **Vocabulary.** The design specifies ``V_op x V_loc`` factorization with
   ``|V| = 390``. This module uses a flat discrete vocabulary of size
   ``K = 5``: ``{do_nothing, fix_typo, add_check, add_return, fix_off_by_one}``.

3. **State / measurement vector.** The design specifies six channels
   (test_pass_rate, lint, type, ast_parse, new_imports, patch_size). This
   module uses one channel. ``test_pass_rate ∈ [0, 1]``.

4. **Simulator backend.** The design recommends a sandboxed subprocess on
   each candidate ``c_h``. This module ships a *direct-evaluation* backend:
   each task carries a hand-coded ``apply_edit`` function that maps
   ``(current_code_state, edit) -> next_code_state``, plus a
   ``score_test_pass_rate(code_state) -> float`` callable. No subprocess,
   no `subprocess.run`, no temp files. This is the hardest of the cuts: the
   simulator is *not* exercising real Python, only a hand-written semantic
   model of it. We document this so that the prototype's outputs cannot be
   confused with a HumanEval result.

5. **STL specs.** Only the easy spec ``coding.fix.easy`` is wired (see
   ``stl_seed.specs.coding_specs``).

What still works honestly
-------------------------

* The simulator returns a duck-typed object with ``states`` and ``times``
  fields, so ``stl_seed.stl.evaluator.compile_spec`` can score it without
  modification. The verifier-fidelity term ``R_spec - R_verifier`` collapses
  to the same Donze-Maler floor as in the bio-ODE family. that part of the
  design's claim ports correctly even with the hand-coded simulator.
* The ``do_nothing`` action is genuinely a no-op (matches the design's
  ``null_op`` slot), so a "wait" policy is in the action space.
* Failures are sentinel-handled: any ``apply_edit`` exception or
  ``score_test_pass_rate`` exception is replaced by ``test_pass_rate = 0``
  for that step, mirroring the bio-ODE NaN guard.

Author: Abdullah AlGhamdi. Date: 2026-04-26.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Action vocabulary (K = 5 flat).
# ---------------------------------------------------------------------------

ACTIONS: tuple[str, ...] = (
    "do_nothing",
    "fix_typo",
    "add_check",
    "add_return",
    "fix_off_by_one",
)
K: int = len(ACTIONS)


# ---------------------------------------------------------------------------
# Lightweight code-state model.
#
# Each `task` carries a *flag dict* describing what bugs are present in the
# current candidate. Edits flip flags. The test_pass_rate is a deterministic
# function of the flag dict. This is a *direct-evaluation* model rather than
# a real subprocess sandbox; see the docstring's "scope cuts" item 4.
# ---------------------------------------------------------------------------

CodeState = dict[str, bool]


@dataclass(frozen=True)
class CodingTask:
    """One toy buggy-function task.

    Attributes
    ----------
    name:
        Identifier, e.g. ``"toy.add_two.has_typo"``.
    initial_state:
        The starting flag dict (representing the initial buggy code).
    apply_edit:
        ``(state, action_name) -> next_state``. Pure; raises on malformed
        action names (caller's bug, surfaced).
    score:
        ``state -> test_pass_rate ∈ [0, 1]``. Deterministic. Pure.
    needed_actions:
        The minimal set of edits a perfect agent would emit (used in tests
        and to confirm the satisfier exists in the vocabulary).
    """

    name: str
    initial_state: CodeState
    apply_edit: Callable[[CodeState, str], CodeState]
    score: Callable[[CodeState], float]
    needed_actions: tuple[str, ...] = field(default=())


# ---------------------------------------------------------------------------
# Five toy tasks. Each has 1-3 active bugs that map directly onto entries in
# ACTIONS. Some tasks have multiple bugs so the agent must compose edits.
# ---------------------------------------------------------------------------


def _make_simple_task(name: str, bug_flag: str, fix_action: str) -> CodingTask:
    """Single-bug task: one flag, one matching fix action."""

    def apply(state: CodeState, action: str) -> CodeState:
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}")
        new = dict(state)
        if action == fix_action and new.get(bug_flag, False):
            new[bug_flag] = False
        # do_nothing or wrong action: no change
        return new

    def score(state: CodeState) -> float:
        # Single bug: pass rate is 1.0 if fixed, 0.0 if still buggy.
        # We use 3 tests so the rate has 4-level resolution {0, 1/3, 2/3, 1}.
        # For a single-bug task: 0/3 if buggy, 3/3 if fixed.
        return 0.0 if state.get(bug_flag, False) else 1.0

    return CodingTask(
        name=name,
        initial_state={bug_flag: True},
        apply_edit=apply,
        score=score,
        needed_actions=(fix_action,),
    )


def _make_double_bug_task() -> CodingTask:
    """Two-bug task: needs ``fix_typo`` and ``add_check`` to fully pass."""

    def apply(state: CodeState, action: str) -> CodeState:
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}")
        new = dict(state)
        if action == "fix_typo" and new.get("has_typo", False):
            new["has_typo"] = False
        elif action == "add_check" and new.get("missing_null_check", False):
            new["missing_null_check"] = False
        return new

    def score(state: CodeState) -> float:
        # 3 tests: typo gates test 0, null-check gates tests 1+2.
        passing = 0
        if not state.get("has_typo", False):
            passing += 1
        if not state.get("missing_null_check", False):
            passing += 2
        return passing / 3.0

    return CodingTask(
        name="toy.parse_and_normalize.two_bugs",
        initial_state={"has_typo": True, "missing_null_check": True},
        apply_edit=apply,
        score=score,
        needed_actions=("fix_typo", "add_check"),
    )


TINY_TASKS: tuple[CodingTask, ...] = (
    _make_simple_task("toy.add_two.has_typo", "has_typo", "fix_typo"),
    _make_simple_task("toy.range_loop.off_by_one", "off_by_one", "fix_off_by_one"),
    _make_simple_task("toy.return_missing.no_return", "no_return", "add_return"),
    _make_simple_task("toy.handle_none.missing_check", "missing_null_check", "add_check"),
    _make_double_bug_task(),
)


# ---------------------------------------------------------------------------
# Trajectory + simulator.
# ---------------------------------------------------------------------------


@dataclass
class CodingTrajectory:
    """Duck-typed trajectory consumed by ``stl_seed.stl.evaluator``.

    Only ``states`` (shape ``(T, 1)``) and ``times`` (shape ``(T,)``) are
    required by the STL evaluator. We carry ``actions`` and a failure
    counter as bookkeeping; nothing else uses them inside this prototype.
    """

    states: np.ndarray
    times: np.ndarray
    actions: tuple[str, ...]
    n_apply_failures: int


def simulate(task: CodingTask, edit_sequence: tuple[str, ...]) -> CodingTrajectory:
    """Apply ``edit_sequence`` step-by-step to ``task.initial_state``.

     On any exception in ``apply_edit`` or ``score``, the offending step's
     ``test_pass_rate`` is sentinelled to ``0`` and ``n_apply_failures`` is
     incremented. The current code state is *not* advanced on a failed apply
    . it is held over, mirroring the bio-ODE NaN-guard convention of "do not
     let one corrupt step contaminate downstream samples".

     Returns a trajectory with `T = len(edit_sequence) + 1` rows, including
     the initial-state measurement at t = 0.
    """

    H = len(edit_sequence)
    states = np.zeros((H + 1, 1), dtype=np.float64)
    times = np.arange(H + 1, dtype=np.float64)
    n_failures = 0

    current = dict(task.initial_state)
    try:
        states[0, 0] = float(task.score(current))
    except Exception:
        states[0, 0] = 0.0
        n_failures += 1

    for h, action in enumerate(edit_sequence, start=1):
        try:
            current = task.apply_edit(current, action)
        except Exception:
            n_failures += 1
            # current held over; emit sentinel measurement
            states[h, 0] = 0.0
            continue
        try:
            states[h, 0] = float(task.score(current))
        except Exception:
            states[h, 0] = 0.0
            n_failures += 1

    return CodingTrajectory(
        states=states,
        times=times,
        actions=tuple(edit_sequence),
        n_apply_failures=n_failures,
    )


# ---------------------------------------------------------------------------
# Self-test.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Sanity: every task's needed_actions sequence must drive test_pass_rate to 1.0.
    for task in TINY_TASKS:
        traj = simulate(task, task.needed_actions)
        final = float(traj.states[-1, 0])
        assert final == 1.0, f"{task.name}: expected 1.0, got {final}"
        # And do_nothing-only must leave it at the initial buggy rate.
        traj_idle = simulate(task, ("do_nothing",) * len(task.needed_actions))
        assert float(traj_idle.states[-1, 0]) < 1.0, (
            f"{task.name}: do_nothing should not pass tests"
        )
    print(f"OK: {len(TINY_TASKS)} tasks pass needed_actions self-test.")

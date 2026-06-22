"""Demo: standard sampler vs. beam-search warmstart on the toy coding-agent cell.

Spike for ``paper/coding_task_design.md``. Runs the two samplers against
``stl_seed.tasks.coding_toy.TINY_TASKS`` under the easy STL spec
``coding.fix.easy``, prints a final-rho table.

The two samplers are implemented in-script (not via the JAX-typed
``stl_seed.inference`` infrastructure) because the toy simulator is plain
NumPy / Python, not a JAX simulator pytree, and porting the toy through the
canonical sampler harness was out of scope for this 12-minute spike. The
algorithms are however functionally equivalent to:

* ``StandardSampler`` (``stl_seed.inference.baselines``): vanilla LLM
  sampling at temperature 1.0, no verifier feedback. We use a uniform-prior
  proxy LLM (no Qwen, no MLX) as in the design's flat-prior caveat.
* ``BeamSearchWarmstartSampler`` (``stl_seed.inference.beam_search_warmstart``):
  beam expansion across the discrete vocabulary at each step, scoring
  partial trajectories by terminal STL rho. No gradient-refinement phase
  (would require a differentiable simulator, which we do not have).

Run from repo root:

    .venv/bin/python scripts/coding_toy_demo.py

Author: Abdullah AlGhamdi. Date: 2026-04-26.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo-root import shim so the script runs without `pip install -e .`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np

import stl_seed.specs.coding_specs  # noqa: F401  (registers the spec)
from stl_seed.specs import REGISTRY
from stl_seed.stl.evaluator import evaluate_robustness
from stl_seed.tasks.coding_toy import (
    ACTIONS,
    TINY_TASKS,
    CodingTask,
    CodingTrajectory,
    simulate,
)

# ---------------------------------------------------------------------------
# Samplers (in-script, plain Python, no JAX).
# ---------------------------------------------------------------------------


def standard_sampler(
    task: CodingTask,
    horizon: int,
    rng: np.random.Generator,
) -> tuple[CodingTrajectory, tuple[str, ...]]:
    """Vanilla random sampling at temperature 1.0 over a uniform-prior LLM.

    Each step draws an action uniformly at random. No verifier feedback.
    """

    edits = tuple(rng.choice(ACTIONS) for _ in range(horizon))
    return simulate(task, edits), edits


def beam_search_warmstart(
    task: CodingTask,
    horizon: int,
    spec,
    beam_width: int = 4,
) -> tuple[CodingTrajectory, tuple[str, ...]]:
    """Discrete beam search over the action vocabulary, scored by terminal rho.

    At each step every active beam is expanded by every action in
    ``ACTIONS``, simulated through the *full* horizon (with the rest of
    the steps padded by ``do_nothing``), scored by STL rho, and the top
    ``beam_width`` partial sequences are kept. After ``H`` steps the best
    full sequence is returned.

    This is faithful to the discrete-search half of
    ``stl_seed.inference.beam_search_warmstart`` (no gradient-refinement
    phase since the toy simulator is non-differentiable, exactly the
    structural-distinction finding the design doc names in §6).
    """

    # Beam state: list of (partial_actions, rho_estimate).
    beams: list[tuple[tuple[str, ...], float]] = [((), -np.inf)]

    for _step in range(horizon):
        candidates: list[tuple[tuple[str, ...], float]] = []
        for partial, _ in beams:
            for a in ACTIONS:
                new_partial = partial + (a,)
                # Pad with do_nothing for scoring, then evaluate full rho.
                padded = new_partial + ("do_nothing",) * (horizon - len(new_partial))
                traj = simulate(task, padded)
                rho = float(evaluate_robustness(spec, traj))
                candidates.append((new_partial, rho))
        # Keep top-k by rho.
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]

    best_actions, _ = beams[0]
    return simulate(task, best_actions), best_actions


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> None:
    spec = REGISTRY["coding.fix.easy"]
    horizon = int(spec.horizon_minutes)  # we reuse 'minutes' as step-count

    n_seeds = 5
    print(f"Spec: {spec.name}  formula: {spec.formula_text}")
    print(f"Horizon (steps): {horizon}, vocab K = {len(ACTIONS)}, n_seeds = {n_seeds}")
    print()
    print(f"{'task':<42} {'sampler':<22} {'rho_mean':>10} {'rho_max':>10} {'fail':>6}")
    print("-" * 95)

    for task in TINY_TASKS:
        # Standard sampler: average over seeds (it is stochastic).
        std_rhos: list[float] = []
        std_fail = 0
        for s in range(n_seeds):
            rng = np.random.default_rng(seed=1234 + s)
            traj, _ = standard_sampler(task, horizon, rng)
            rho = float(evaluate_robustness(spec, traj))
            std_rhos.append(rho)
            if rho < 0:
                std_fail += 1

        # Beam search: deterministic (no LLM stochasticity), one run.
        beam_traj, beam_actions = beam_search_warmstart(task, horizon, spec, beam_width=4)
        beam_rho = float(evaluate_robustness(spec, beam_traj))

        print(
            f"{task.name:<42} {'standard(uniform)':<22} "
            f"{np.mean(std_rhos):>10.4f} {np.max(std_rhos):>10.4f} "
            f"{std_fail:>3}/{n_seeds}"
        )
        print(
            f"{task.name:<42} {'beam_search(B=4)':<22} "
            f"{beam_rho:>10.4f} {beam_rho:>10.4f} "
            f"{'0' if beam_rho >= 0 else '1':>3}/1"
        )
        print(f"  beam best actions: {beam_actions}")
        print()


if __name__ == "__main__":
    main()

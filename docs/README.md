# `stl-seed` documentation

`stl-seed` tests whether SERA-style soft verification (Shen et al., arXiv:2601.20789) extends to scientific control: a small open-weights LLM emits a piecewise-constant control schedule `u_{1:H}` for an ODE-driven biomolecular or physiological system, the trajectory is scored by a Donzé-Maler STL robustness margin `rho`, and the SFT loss is filtered or weighted by `rho` so gradient updates favour schedules that satisfy the formal spec.

This page is the documentation root. The narrative starts in the repo [`README.md`](../README.md); the formal claims live in [`paper/`](../paper/).

## Install

```bash
uv sync --extra mlx --extra dev      # Apple Silicon
uv sync --extra cuda --extra dev     # Linux + CUDA
uv sync --extra dev                  # CPU only (examples 01/02 work)
```

## Run

```bash
uv run python examples/01_basic_simulation.py    # ODE + STL, ~5 s
uv run python examples/02_stl_filtering.py       # generate + filter, ~30 s
uv run python examples/03_mlx_training_minimal.py  # MLX QLoRA, ~1-2 min
```

## Concepts

| Object            | Contract                                                 | Source                                |
| ----------------- | -------------------------------------------------------- | ------------------------------------- |
| `Simulator`       | `simulate(x0, u, params, key) -> Trajectory`             | `src/stl_seed/tasks/`                 |
| `STLSpec`         | conjunction-only AST + `formula_text` + cited thresholds | `src/stl_seed/specs/`                 |
| `Policy`          | `(state, spec, history, key) -> action`                  | `src/stl_seed/generation/policies.py` |
| `FilterCondition` | `filter(trajectories, rhos) -> (kept, weights)`          | `src/stl_seed/filter/conditions.py`   |

Four task families ship today: glucose-insulin (Bergman 1979 + Dalla Man 2007), repressilator (Elowitz & Leibler 2000), toggle (Gardner-Cantor-Collins 2000), MAPK (reduced Huang-Ferrell 1996 / Markevich 2004). Three filter conditions: `HardFilter`, `QuantileFilter`, `ContinuousWeightedFilter`. all raise `FilterError` rather than silently degrade.

The full per-module API surface is documented in the source docstrings; `tests/test_stl_evaluator.py` is the most pedagogical entry point for the STL evaluator (it builds tiny ASTs by hand and asserts Donzé-Maler equalities one node at a time).

## Reading `rho`

`rho > 0`: trajectory satisfies `spec` with margin `rho` (in the spec's natural units. mg/dL for glucose specs, monomers/cell for repressilator). `rho < 0`: violation with margin `|rho|`. `rho == 0`: boundary. The Donzé-Maler space-robustness is the smallest predicate-level signed margin over the spec's temporal intervals.

## Where to read next

- Formal claim, hypotheses, statistical model: [`paper/theory.md`](../paper/theory.md)
- Cross-task validation: [`paper/cross_task_validation.md`](../paper/cross_task_validation.md)
- Inference method (gradient-guided STL decoding): [`paper/inference_method.md`](../paper/inference_method.md)
- Empirical results: [`paper/unified_comparison_results.md`](../paper/unified_comparison_results.md)
- Compute-cost Pareto: [`paper/compute_cost_results.md`](../paper/compute_cost_results.md)

## Gotchas

- `GlucoseInsulinSimulator.simulate(...)` returns `(states, times, meta)`, not a `Trajectory` pytree. the runner adapter wraps it. The bio_ode simulators DO return `Trajectory` directly. Asymmetry is intentional and documented.
- Glucose-insulin actions are `(H,)` insulin rates in U/h; bio_ode actions are `(H, m)` dimensionless inducer fractions in `[0, 1]`. Simulators clip on entry.
- `MLXModelPolicy` is Apple Silicon only. raises `RuntimeError` at construction time on other hosts. On Linux/CUDA use `BNBBackend`.

# Coding-Agent Toy Cell — Spike Results

Author: Abdullah AlGhamdi. Date: 2026-04-26. Status: 12-minute spike.

This document reports the outcome of a *minimum viable* implementation of the coding-agent task cell sketched in `paper/coding_task_design.md`. The goal was to demonstrate, end-to-end and in under twelve minutes, that the existing `stl-seed` STL evaluator and a (greatly reduced) sampler grid can score trajectories from a coding-agent simulator at all. A full HumanEval- mutated implementation is deferred per the design doc's 2-3 day estimate.

Every line below is honest about what shipped and what did not.

## 1. What was implemented

Three files, all `shipped and works`:

- `src/stl_seed/tasks/coding_toy.py` — toy simulator with five hand-coded buggy-function tasks. Returns a duck-typed trajectory (`states` shape `(T, 1)`, `times` shape `(T,)`) compatible with the existing STL evaluator. Sentinel-on-exception policy mirrors the bio-ODE NaN guard. Self-test passes (`python src/stl_seed/tasks/coding_toy.py` → "OK: 5 tasks pass needed_actions self-test").
- `src/stl_seed/specs/coding_specs.py` — registers `coding.fix.easy` with formula `F_[0, 6] (test_pass_rate > 0.5)`. The spec uses the same `Eventually + Predicate` form as `bio_ode` specs and respects the firewall §C.1 conjunction-only convention.
- `scripts/coding_toy_demo.py` — runs `standard` (uniform-random) and `beam_search_warmstart` (`B=4`, no gradient refinement) against `TINY_TASKS`, prints per-task rho.

The script exits cleanly. The integration with `stl_seed.stl.evaluator.evaluate_robustness` is the real path; the rho numbers below come out of the same Donze-Maler evaluator that scores the bio-ODE trajectories.

## 2. What was skipped vs. the design

| Design element                                              | Status              |
| ----------------------------------------------------------- | ------------------- |
| HumanEval-mutated dataset (~50 problems)                    | stubbed for future  |
| `subprocess.run` build-and-measure backend                  | stubbed for future  |
| `V_op × V_loc` factored vocabulary (\`                      | V                   |
| Six-channel measurement vector (test, lint, type, ast, ...) | shipped and partial |
| Three STL specs (easy / medium / hard)                      | shipped and partial |
| `BestOfN`, `ContinuousBoN`, `RolloutTree`, `HorizonFolded`  | stubbed for future  |
| Integration with the JAX-typed `stl_seed.inference` API     | stubbed for future  |
| Mutation-catalog generator script                           | stubbed for future  |
| Tests (`tests/test_coding_*.py`)                            | stubbed for future  |
| Parquet harness (`run_coding_unified_comparison.py`)        | stubbed for future  |

The `shipped and partial` rows are: only **one** of six measurement channels (test_pass_rate) is implemented, and only **one** of three specs (easy) is wired. Medium and hard, plus the lint / type / ast / imports / patch-size channels, are explicit follow-up items.

The largest scope cut, called out in the simulator's docstring, is that the toy tasks have **no real Python source**. Each task is a flag dict ("has_typo", "missing_null_check", ...) plus a hand-coded `apply_edit` function. We did not run a subprocess; we did not parse Python; we did not shell out to ruff or mypy. This is `shipped and partial`: the STL/sampler half is real, the simulator half is a hand-written stand-in for what HumanEval+subprocess would do.

## 3. Results on the toy dataset

Spec: `coding.fix.easy` formula: `F_[0, 6] (test_pass_rate > 0.5)` Horizon: 6 steps. Vocab: K = 5. Standard sampler: 5 seeds. Beam: B = 4.

| Task                               | Sampler            | rho (mean) | rho (max) | Failures |
| ---------------------------------- | ------------------ | ---------: | --------: | -------: |
| `toy.add_two.has_typo`             | standard (uniform) |    +0.3000 |   +0.5000 |      1/5 |
| `toy.add_two.has_typo`             | beam_search (B=4)  |    +0.5000 |   +0.5000 |      0/1 |
| `toy.range_loop.off_by_one`        | standard (uniform) |    +0.3000 |   +0.5000 |      1/5 |
| `toy.range_loop.off_by_one`        | beam_search (B=4)  |    +0.5000 |   +0.5000 |      0/1 |
| `toy.return_missing.no_return`     | standard (uniform) |    +0.1000 |   +0.5000 |      2/5 |
| `toy.return_missing.no_return`     | beam_search (B=4)  |    +0.5000 |   +0.5000 |      0/1 |
| `toy.handle_none.missing_check`    | standard (uniform) |    +0.1000 |   +0.5000 |      2/5 |
| `toy.handle_none.missing_check`    | beam_search (B=4)  |    +0.5000 |   +0.5000 |      0/1 |
| `toy.parse_and_normalize.two_bugs` | standard (uniform) |    +0.1667 |   +0.5000 |      2/5 |
| `toy.parse_and_normalize.two_bugs` | beam_search (B=4)  |    +0.5000 |   +0.5000 |      0/1 |

Beam-search's recovered fix sequence on the two-bug task is `(add_check, fix_typo, do_nothing, ..., do_nothing)` — both required edits, then idle out the rest of the horizon. The trailing `do_nothing` calls confirm the design's `null_op` slot is doing its job in the vocabulary.

Standard-sampler `rho < 0` happens at the 1-5 to 2-5 rate per task. The gap is structural: with `K = 5` and a single-fix task the chance of emitting the unique fix action at *some* step in `H = 6` is `1 - (4/5)^6 ≈ 0.738`. Empirically we see ~0.6-0.8 success rates per task, consistent with the analytic floor.

## 4. Comparison to bio_ode behavior

The methodology gap the design names — *standard* (no verifier feedback) versus *beam-search-warmstart* (discrete search over a vocabulary that provably contains the satisfier) — **reproduces in the toy coding cell**.

| Axis                     | bio_ode.repressilator.easy                  | coding.fix.easy (toy)                                 |
| ------------------------ | ------------------------------------------- | ----------------------------------------------------- |
| Vocabulary contains fix? | Yes (`u = (0,0,1)` discretization)          | Yes (each `needed_action` ∈ ACTIONS by construction)  |
| Standard-sampler rho     | Sub-saturating (per `unified_comparison_*`) | Mean 0.10-0.30, fails 1-2 of 5 seeds                  |
| Beam-search rho          | Saturating (vocabulary-enumeration win)     | Saturating at +0.5 on every task                      |
| Reason beam wins         | Cliff-shaped rho, narrow attractor          | Discrete one-hot fix; flat-prior LLM has no attractor |

The bio_ode story is "continuous samplers (gradient, CMA-ES, hybrid) saturate on cliff-geometry rho with a single satisfying corner; switch to vocabulary-enumeration beam search". The toy coding story is the same algebraic shape ported to a `K = 5` discrete vocabulary: standard random sampling's success probability is `1 - (1 - p_fix)^H` and is bounded below 1, while beam-search at any `B ≥ 1` is exhaustive in width-1 and can never miss. This is the same structural-search-vs-continuous-search distinction my prior memory entry names.

What the toy cell does **not** reproduce is the failure mode that needed beam-search in the first place on bio_ode — non-convex rho landscapes with multi-step coordination requirements (`G_[120,200]` clauses where all downstream actions must cooperate). The easy spec is a single `Eventually` over a single channel; it has no analog of repressilator's sustained-high requirement. To reproduce the *interesting* part of the bio_ode finding the cell would need at minimum the medium spec from the design doc, with its `G_[H/2, H] (test_pass_rate >= 0.8)` sustained-pass clause. That is `stubbed for future`.

## 5. What would need to extend this to HumanEval

In rough order of effort:

1. **Real simulator backend.** Replace `apply_edit` (flag-flip) and `score` (deterministic function) with a sandboxed `subprocess.run` call against a pytest-instrumented HumanEval test runner. ~150 LOC of subprocess plumbing per the design doc §2; main risk is wall-time budget (the design's 30 s/eval cap means H = 12 trajectories cost up to 6 min each, so the full 50-problem × 6-sampler grid is ~6 wall-CPU-h).
1. **Mutation-catalog generator.** `scripts/generate_he_mutations.py`, ~50 LOC of `ast.NodeTransformer`. Deterministic given a seed list.
1. **`V_op × V_loc` vocabulary.** ~150 LOC of `ast` rewrite logic; the factored representation requires only a small change to the sampler to enumerate the cartesian product, the rest of the pipeline is shape-preserving (`|V| = 390` is just a different number of bins).
1. **Multi-channel measurement vector.** Add `ruff`/`pyflakes`/`pyright` subprocess calls and AST-parse / import-set diffs. The STL evaluator is already channel-indexed, so this is a simulator-side change only.
1. **Medium and hard specs.** Direct port from the design doc's §5. Both compose `Always` and `Eventually` plus channel-2/3/4 predicates; no new evaluator nodes needed.
1. **Wire to the canonical `stl_seed.inference` API.** The toy uses plain Python samplers because the JAX sampler protocol expects a JAX simulator. Either (a) port the toy to a JAX-typed `Simulator` subclass, or (b) add a `PythonSimulatorAdapter` wrapper. Option (b) is lower-friction and keeps the rest of the harness untouched.
1. **Tests.** ≥6 tests per the design doc §10 implementation checklist.
1. **Parquet harness + bar chart.** Drop into the unified-comparison template, exactly as `paper/unified_comparison_results.md`.

The one item the spike *cannot* simply scale up to is the gradient-sampler exclusion. As the design doc §6 names: there is no `∂rho/∂e_h` for any discrete edit, and we declined to manufacture a learned surrogate. That exclusion is structural, not a scope cut, and should be reported in the final paper as a positive finding (gradient guidance fundamentally requires a differentiable simulator; coding-agent simulators are not).

## 6. Files of record

- `/Users/abdullahalghamdi/stl-seed/src/stl_seed/tasks/coding_toy.py`
- `/Users/abdullahalghamdi/stl-seed/src/stl_seed/specs/coding_specs.py`
- `/Users/abdullahalghamdi/stl-seed/scripts/coding_toy_demo.py`
- `/Users/abdullahalghamdi/stl-seed/paper/coding_toy_results.md` (this file)

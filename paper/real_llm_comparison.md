# Real-LLM hard-spec comparison: does the methodology survive a real prior?

Pre-registered, falsification-shaped head-to-head against `Qwen3-0.6B-bf16` (via MLX). The unified-comparison harness (`paper/unified_comparison_results.md`) runs against a **flat-prior LLM** (uniform logits over the action vocabulary) to isolate what each sampler does with the verifier signal alone. The risk that audit raised is straightforward: any "+128× lift" headline measured against a uniform proxy is at risk of evaporating once a real language-model prior is wired in. This script measures whether the methodology contribution (beam-search warmstart) actually beats the real-LLM baseline on tasks where it should matter.

Reproduce with `uv run python scripts/real_llm_hard_specs.py`. Results in `runs/real_llm_hard_specs/{results.parquet, results.jsonl, verdict.json}`.

## Pre-registered outcome rule

- `METHODOLOGY MATTERS` iff beam-search reaches ρ > 0 on ≥ 2 of 4 tasks where `StandardSampler + Qwen3-0.6B` reaches ρ ≤ 0 on a majority of seeds.
- `METHODOLOGY DOES NOT MATTER` if `StandardSampler + Qwen3-0.6B` already reaches ρ > 0 on a majority of seeds across all 4 tasks.
- `METHODOLOGY MAYBE MATTERS` for any in-between outcome.

The rule is in the script docstring (`scripts/real_llm_hard_specs.py:14-44`) and was pre-registered before the runs were executed.

## Result

**VERDICT: METHODOLOGY MATTERS.** Beam-search rescues all four tasks where `Qwen3-0.6B + StandardSampler` fails on a majority of seeds (4/4 ≥ 2/4 threshold).

| Task                              | Standard sat / n | Standard ρ̄ | Beam sat / n |  Beam ρ̄ | Methodology gap |
| --------------------------------- | ---------------: | ---------: | -----------: | ------: | --------------: |
| `bio_ode.repressilator.easy`      |            0 / 3 |   −247.582 |        3 / 3 | +25.000 |        +272.582 |
| `bio_ode.toggle.medium`           |            0 / 3 |    −99.960 |        3 / 3 | +29.992 |        +129.952 |
| `bio_ode.mapk.hard`               |            0 / 3 |     −0.500 |        3 / 3 |  +0.002 |          +0.502 |
| `cardiac.suppress_after_two.hard` |            0 / 3 |     −1.434 |        3 / 3 |  +0.850 |          +2.284 |

Three fixed seeds per (task, sampler) cell (`{3000, 3001, 3002}`). 24 runs total. Wall-clock `≈ 5 min` on M5 Pro for the whole sweep; the repressilator beam-search cell dominates at `≈ 80 s` per seed (K = 125 vocabulary forces ~125 LLM-score forward passes per control step, × 8-beam × 10 horizon).

## Why this is the test that matters

Without this script, the unified-comparison "+128× lift" is against `jnp.zeros(K)`. uniform logits, not a real LLM. With this script, the comparison is against the same Qwen3-0.6B base both samplers consume, under matched temperature `0.5` and matched seeds. The gap is entirely attributable to the sampler's use of the verifier signal: standard sampling reads the LLM logits and picks; beam-search warmstart enumerates the action vocabulary, scores each candidate under a model-predictive constant-extrapolation lookahead, and seeds gradient refinement from the top-B. Same LLM, same temperature, same seeds, same horizon. different inference recipe.

## Honest caveats

- **Zero per-seed variance under standard sampling.** `Qwen3-0.6B` at temperature 0.5 picks identical action sequences across the three seeds on every task here, producing zero-variance ρ. This is the LLM-prior collapsing to a single mode (low-entropy logits dominate the temperature-rescaled softmax); it is not a sampling bug. Beam-search has the same property because the beam selection is deterministic given the LLM's scores. A larger LLM (1.7B, 4B) and / or a higher temperature would change this; the seed loop here exists to expose variance if it exists, not to assume it.
- **Vocabulary contains the answer (beam-search).** As called out in `paper/cross_task_validation.md` line 85 and the README, the dense `k_per_dim = 5` lattice for repressilator and toggle is engineered so the satisfying corner is in the vocabulary by construction. Beam-search plus constant-extrapolation lookahead therefore reduces to argmax over constant policies once the satisfying corner has score `> 0`. This is a structural property of the inference recipe, not a free win; the contribution is the structural-search-vs-continuous-search distinction. The same critique applies symmetrically to the rollout-tree heuristic in cross_task_validation.md.
- **MAPK and cardiac gaps are small in absolute units.** ρ̄ = +0.002 on MAPK and +0.850 on cardiac just clear the satisfaction threshold. These are sat-frac wins, not magnitude wins, and the satisfaction- boundary phenomenon should not be over-read as a saturation result the way `bio_ode.repressilator.easy` (+25.000) and `bio_ode.toggle.medium` (+29.992) can be.
- **Three seeds is small.** This is a falsification test, not a population estimate. The unified-comparison harness uses N = 4 seeds for the same reason; both are tight on power, large enough to distinguish a methodology-rescue (sat-frac jumps from 0/3 to 3/3) from a no-rescue, not large enough to estimate the population-mean ρ precisely.
- **Qwen3-0.6B is the smallest of the planned model ladder.** The 1.7B and 4B sizes were not run here because the cost-per-cell at K = 125 beam-search inflates with model size and the falsification rule does not require them. If the canonical SFT sweep finishes them, the follow-up question is whether standard sampling at 4B post-SFT closes the methodology gap on the easier tasks (MAPK, cardiac).

## Files

- Pre-registered driver: `scripts/real_llm_hard_specs.py`
- Per-cell results: `runs/real_llm_hard_specs/results.parquet`, `runs/real_llm_hard_specs/results.jsonl`
- Verdict (machine-readable): `runs/real_llm_hard_specs/verdict.json`
- Raw run log: `runs/real_llm_hard_specs/run.log`

# stl-seed

[![CI](https://github.com/AA-Alghamdi/stl-seed/actions/workflows/ci.yml/badge.svg)](https://github.com/AA-Alghamdi/stl-seed/actions/workflows/ci.yml)
[![Lint](https://github.com/AA-Alghamdi/stl-seed/actions/workflows/lint.yml/badge.svg)](https://github.com/AA-Alghamdi/stl-seed/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/AA-Alghamdi/stl-seed/branch/main/graph/badge.svg)](https://codecov.io/gh/AA-Alghamdi/stl-seed)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Differentiable STL robustness as inference-time guidance for small open-weights LLM agents on biomolecular control.

## Headline

Gradient-guided STL decoding lifts mean ρ on `glucose_insulin.tir.easy` from +0.16 (standard sampling) to +19.91 (saturating the spec) at matched compute. The hybrid sampler hits the +20.0 ceiling on every seed. On `bio_ode.repressilator.easy` the same method floors at ρ ≈ −250: gradient guidance does not transfer when the satisfying region requires multi-step coordinated planning.

![Unified sampler comparison](paper/figures/unified_comparison.png)

N=8 seeds, 95% bootstrap CIs. Per-cell numbers in [`paper/unified_comparison_results.md`](paper/unified_comparison_results.md). Reproduce with `uv run python scripts/run_unified_comparison.py`.

## What and why

stl-seed tests whether SERA's soft-verification recipe ([Shen et al. 2026](https://arxiv.org/abs/2601.20789)) extends to scientific control when the soft signal is a *formal STL specification* rather than an engineered patch-overlap proxy. Two contributions follow.

**A new inference method.** Differentiable STL gradient-guided decoding. At each generation step, the LLM's logits are biased by ∇ρ propagated through the simulator and Donzé–Maler evaluator stack. One ODE solve + one backward pass per step, against N forward solves for best-of-N. Implementation in [`src/stl_seed/inference/gradient_guided.py`](src/stl_seed/inference/gradient_guided.py).

**An empirical decomposition of the verifier gap.** Write `R_gold − R_proxy = (R_gold − R_spec) + (R_spec − R_verifier)`. For formal STL the second term is bounded by float64 epsilon. The trajectory adversary in [`src/stl_seed/analysis/`](src/stl_seed/analysis/) measures the first term: on glucose-insulin it finds a satisfying trajectory with gold score 2.27 below the random satisfying baseline. Against a learned process reward model ([Setlur et al. 2024](https://arxiv.org/abs/2410.08146)), STL AUC is 1.000 vs PAV 0.000 on repressilator (PAV is anti-informative on this task) and 1.000 vs 0.819 on glucose-insulin. PAV never matches STL at any tested train size between 100 and 2000 trajectories.

## Install

```bash
uv sync --extra mlx     # Apple Silicon (development)
uv sync --extra cuda    # CUDA / RunPod (canonical sweep)
```

## Demo

```bash
$ stl-seed sample --task glucose_insulin --sampler gradient_guided --guidance-weight 2
task=glucose_insulin spec=glucose_insulin.tir.easy sampler=gradient_guided
final_rho = 19.9100
steps_changed_by_guidance = 12 / 12
```

Available samplers: `standard`, `bon`, `bon_continuous`, `gradient_guided`, `hybrid`.

## What doesn't work

Gradient guidance fails on tasks where 1-step lookahead is uninformative. `bio_ode.repressilator.easy` requires sustained silencing of gene-3 across 10 control steps; the local gradient does not point toward this attractor. A sweep over default-action initializations × λ ∈ {0, 5, 50} confirms the failure is structural, not tuning. Hybrid recovers part of the gap on glucose, none on repressilator. Full diagnosis in [`paper/cross_task_validation.md`](paper/cross_task_validation.md).

The spec auto-tuner in [`src/stl_seed/specs/calibration.py`](src/stl_seed/specs/calibration.py) finds threshold values 10–100× more discriminative than the textbook hand-set choices ([`paper/REDACTED.md`](paper/REDACTED.md)).

## Tests

426 passed, 6 platform-skipped, 2 expected-fails (cross-task transfer + STL discretization edge case). 91% line coverage on `src/stl_seed/`. Property-based tests in [`tests/test_stl_properties.py`](tests/test_stl_properties.py) verify 13 algebraic invariants of the Donzé–Maler robustness semantics ([FORMATS 2010](https://doi.org/10.1007/978-3-642-15297-9_9)): negation antisymmetry, conjunction-as-min, De Morgan dualities, interval-shrinkage monotonicity, predicate scaling, others.

## Status

Phase 1 (theory, library, local pilot) shipped 2026-04-24. Phase 2 is the canonical 18-cell SFT sweep on RunPod: 3 model sizes × 3 filter conditions × 2 task families, projected $5–15 of $25 cap. The mock-backend dry-run validates the full pipeline locally and caught five bugs that would have failed the real run. Single command when ready:

```bash
python scripts/run_canonical_sweep.py --confirm
```

## Acknowledgments

Murat Arcak for STL formal-methods grounding. Hanna REDACTED et al. for the STL-on-biomolecular-ODE precedent ([REDACTED 2025](https://arxiv.org/abs/2412.15227)). Karen Leung et al. for [STLCG++](https://arxiv.org/abs/2501.04194). REDACTED et al. for SERA. Setlur et al. for PAVs.

```bibtex
@misc{alghamdi2026stlseed,
  author = {Abdullah AlGhamdi},
  title  = {stl-seed: Soft-verified SFT for scientific control with formal STL verifiers},
  year   = {2026},
  url    = {https://github.com/AA-Alghamdi/stl-seed},
}
```

Apache 2.0.

# stl-seed results: navigation dashboard

A 30-second index over the artifact's paper drafts, theory notes, and code. For the full project pitch, see [`STORY.md`](../STORY.md). For install + demo, see [`README.md`](../README.md).

## At a glance (60 seconds)

**Headline.** Real `Qwen3-0.6B-bf16` plus standard sampling fails 4 of 4 hard biomolecular control specs; structural-search inference-time methodology (beam-search warmstart) rescues all 4. The methodology gap survives roughly 3x quantization compression and roughly 3x size scaling. The first SERA-saturation transition appears on `bio_ode.toggle.medium` at 1.7B.

**The money plot.** [`paper/figures/money_plot.png`](figures/money_plot.png) (caption: [`paper/figures/money_plot.md`](figures/money_plot.md)). Companion: [`paper/figures/money_plot_inset.png`](figures/money_plot_inset.png).

**Numerical headline (5 models x 4 hard tasks, sat-fraction standard vs beam):**

| Model             | repressilator | toggle.medium | mapk.hard  | cardiac.hard | Verdict             |
| ----------------- | ------------- | ------------- | ---------- | ------------ | ------------------- |
| `qwen3-0.6b`      | 0/3 vs 3/3    | 0/3 vs 3/3    | 0/3 vs 3/3 | 0/3 vs 3/3   | METHODOLOGY MATTERS |
| `qwen3-0.6b-8bit` | 0/3 vs 3/3    | 0/3 vs 3/3    | 0/3 vs 3/3 | 0/3 vs 3/3   | METHODOLOGY MATTERS |
| `qwen3-0.6b-4bit` | 0/3 vs 3/3    | 1/3 vs 3/3    | 0/3 vs 3/3 | 0/3 vs 3/3   | METHODOLOGY MATTERS |
| `qwen3-1.7b`      | 0/3 vs 3/3    | 3/3 vs 3/3 S  | 0/3 vs 3/3 | 0/3 vs 3/3   | METHODOLOGY MATTERS |
| `qwen3-1.7b-4bit` | 0/3 vs 3/3    | 3/3 vs 3/3 S  | 0/3 vs 3/3 | 0/3 vs 3/3   | METHODOLOGY MATTERS |

S = SERA-saturation cell. Full per-cell parquet: `runs/quant_size_sweep/`. Hierarchical Bayes posterior on the population gap: `delta_global = +82.82` rho-units, 95% CrI `[+35.22, +129.44]`.

## What's inside (organized by reading time)

**5-minute path.** [`STORY.md`](../STORY.md) -> [`paper/real_llm_comparison.md`](real_llm_comparison.md) -> [`paper/quant_size_results.md`](quant_size_results.md). End state: you understand the verdict and what's behind it.

**15-minute path.** Add [`paper/landscape_theorem.md`](landscape_theorem.md) -> [`paper/scaling_laws.md`](scaling_laws.md) -> [`paper/pav_v2.md`](pav_v2.md). End state: you understand why the asymmetry exists and what the strengthened baseline says.

**30-minute path.** Add [`paper/hierarchical_bayes.md`](hierarchical_bayes.md) -> [`paper/information_theory.md`](information_theory.md) -> [`paper/memory_pareto.md`](memory_pareto.md) -> [`paper/stlcgpp_comparison.md`](stlcgpp_comparison.md). End state: you have the Bayesian uncertainty, the bits-of-MI argument, the QLoRA-lineage memory framing, and the formal STLCG++ delta.

**Workshop submission path.** [`paper/fmai_2026_draft.md`](fmai_2026_draft.md) + [`paper/venue_targets.md`](venue_targets.md) + [`paper/cross_task_validation.md`](cross_task_validation.md). End state: you can review the FMAI 2026 submission as written.

## Documents (one-line summary each)

Reading-time estimates assume 250 wpm.

- [`paper/abstract.md`](abstract.md) (4.9 KB, 3 min). Phase-1 result, falsification verdict, Phase-2 status, honest failure mode.
- [`paper/real_llm_comparison.md`](real_llm_comparison.md) (5.9 KB, 4 min). Pre-registered Qwen3-0.6B head-to-head; verdict METHODOLOGY MATTERS on 4/4 hard tasks.
- [`paper/quant_size_results.md`](quant_size_results.md) (9.7 KB, 6 min). 5 models x 4 tasks x 2 samplers x 3 seeds = 120 cells; gap survives quantization and size.
- [`paper/landscape_theorem.md`](landscape_theorem.md) (37 KB, 20 min). Smooth-margin radius vs cliff index regime characterization for gradient guidance.
- [`paper/scaling_laws.md`](scaling_laws.md) (20 KB, 12 min). Inference-time compute scaling: power law on smooth tasks, step function on narrow attractors.
- [`paper/pav_v2.md`](pav_v2.md) (9.7 KB, 6 min). PAV strengthened with model selection + Setlur on-policy MC labels; STL-PAV gap tightens to -0.038.
- [`paper/hierarchical_bayes.md`](hierarchical_bayes.md) (7.9 KB, 5 min). NumPyro NUTS over 120 cells; population gap +82.82 rho-units, P(>0) = 0.9995.
- [`paper/information_theory.md`](information_theory.md) (14 KB, 9 min). I(STL; success) vs I(PAV; success) in bits; STL noise floor 16 orders of magnitude below PAV.
- [`paper/memory_pareto.md`](memory_pareto.md) (11 KB, 6 min). Memory-quality Pareto in QLoRA / bitsandbytes axes; 1.7B-4bit at 3.27 GB strictly dominates 1.7B-bf16.
- [`paper/stlcgpp_comparison.md`](stlcgpp_comparison.md) (17 KB, 11 min). Formal comparison to STLCG (Leung 2020) and STLCG++ (Hashemi 2025); two novel contributions.
- [`paper/coding_task_design.md`](coding_task_design.md) (34 KB, 22 min). Design doc for the SERA-native coding-agent task cell.
- [`paper/venue_targets.md`](venue_targets.md) (24 KB, 15 min). 18-row submission table with deadlines verified 2026-04-26.
- [`paper/fmai_2026_draft.md`](fmai_2026_draft.md) (26 KB, 14 min). 4-page ICML 2026 FMAI workshop submission draft.
- [`paper/cross_task_validation.md`](cross_task_validation.md) (19 KB, 11 min). Negative result and structural diagnosis for gradient guidance on repressilator.
- [`paper/inference_method.md`](inference_method.md) (11 KB, 6 min). Gradient-guided sampling algorithm + matched-compute caveat.
- [`paper/theory.md`](theory.md) (25 KB, 14 min). Theoretical foundation: STL formal soft verifier, soft-filtered SFT formalism, Goodhart decomposition.
- [`paper/unified_comparison_results.md`](unified_comparison_results.md) (10.8 KB, 5 min). 9 samplers x 5 task families x 4 seeds = 180 cells (flat-prior LLM).
- [`paper/compute_cost_results.md`](compute_cost_results.md) (19.5 KB, 12 min). Per-task 2x3 compute-cost Pareto frontier across 9 samplers.
- [`paper/pav_comparison.md`](pav_comparison.md) (4.0 KB, 2 min). Original (v1) PAV comparison; superseded by `pav_v2.md`.
- [`paper/adversary_findings.md`](adversary_findings.md) (3.5 KB, 2 min). Trajectory adversary lower bound on the spec-completeness gap (-2.27 rho-units).
- [`paper/power_analysis_empirical.md`](power_analysis_empirical.md) (7.7 KB, 4 min). Pilot ICC = 0.9979; pooled SE 0.0098 vs TOST threshold 0.0171.
- [`STORY.md`](../STORY.md) (13 KB, 7 min). Pitch document tying everything together.
- [`RELEASE_v0.1.0.md`](../RELEASE_v0.1.0.md) (2.9 KB, 1 min). PyPI v0.1.0 release notes.
- [`docs/blog/stl_seed_intro.md`](../docs/blog/stl_seed_intro.md) (10 KB, 6 min). Long-form blog post on the verifier-fidelity argument.

## Code (one-line summary each)

**Drivers (in `scripts/`).**

- [`scripts/real_llm_hard_specs.py`](../scripts/real_llm_hard_specs.py). Day-1 falsification harness on real Qwen3-0.6B; pre-registered outcome rule.
- [`scripts/quant_size_sweep.py`](../scripts/quant_size_sweep.py). 5 models x 4 tasks x 2 samplers x 3 seeds = 120-cell quantization x size sweep.
- [`scripts/run_unified_comparison.py`](../scripts/run_unified_comparison.py). 9 samplers x 5 task families x 4 seeds = 180-cell flat-prior unified comparison.
- [`scripts/benchmark_compute_cost.py`](../scripts/benchmark_compute_cost.py). Wall-clock + simulator-call cost frontier across all samplers.
- [`scripts/scaling_laws_analysis.py`](../scripts/scaling_laws_analysis.py). Power-law fits over the cost-benchmark parquet.
- [`scripts/hierarchical_bayes.py`](../scripts/hierarchical_bayes.py). NumPyro NUTS over the 120-cell methodology gap.
- [`scripts/_info_theory_compute.py`](../scripts/_info_theory_compute.py). MI in bits between verifier score and success indicator.
- [`scripts/run_pav_comparison_v2.py`](../scripts/run_pav_comparison_v2.py). Strengthened PAV with model selection + on-policy MC labels.
- [`scripts/run_pav_comparison.py`](../scripts/run_pav_comparison.py). Original (v1) PAV comparison driver.
- [`scripts/run_canonical_sweep.py`](../scripts/run_canonical_sweep.py). Phase-2 SFT sweep (queued on RunPod).
- [`scripts/run_canonical_eval.py`](../scripts/run_canonical_eval.py). Eval harness for the canonical sweep.
- [`scripts/generate_canonical.py`](../scripts/generate_canonical.py). Builds the canonical generation corpus.
- [`scripts/auto_tune_specs.py`](../scripts/auto_tune_specs.py). Spec calibration finder.
- [`scripts/run_adversary.py`](../scripts/run_adversary.py). Trajectory adversary for the spec-completeness lower bound.
- [`scripts/canonical_analysis.py`](../scripts/canonical_analysis.py). Aggregation over the canonical corpus.
- [`scripts/power_analysis_real.py`](../scripts/power_analysis_real.py). Empirical ICC + pooled SE for the registered TOST.
- [`scripts/smoke_test_mlx.py`](../scripts/smoke_test_mlx.py). Apple Silicon QLoRA smoke (loss 1.484 -> 0.466 in 15 s).
- [`scripts/smoke_test_bnb.py`](../scripts/smoke_test_bnb.py). bitsandbytes / TRL smoke for the canonical-sweep backend.
- [`scripts/validate_phase2_pipeline.py`](../scripts/validate_phase2_pipeline.py). End-to-end mock-backend dry-run for Phase-2.
- [`scripts/sweep_monitor.py`](../scripts/sweep_monitor.py). Live monitor over `runs/*/sweep.jsonl`.
- [`scripts/release_prep.py`](../scripts/release_prep.py). Build + twine-check helper for PyPI releases.
- [`scripts/filter_pilot.py`](../scripts/filter_pilot.py), [`scripts/generate_pilot.py`](../scripts/generate_pilot.py). Pilot generation + soft-filter dry runs.

**`src/stl_seed/` subpackages.**

- [`stl/`](../src/stl_seed/stl/). Donze-Maler 2010 evaluator + streaming + worst-subformula diagnostic.
- [`specs/`](../src/stl_seed/specs/). Spec registry (bio_ode, glucose_insulin, cardiac) + auto-calibration.
- [`tasks/`](../src/stl_seed/tasks/). ODE simulators: glucose-insulin (Bergman), bio_ode (repressilator/toggle/MAPK), cardiac AP (FitzHugh-Nagumo).
- [`inference/`](../src/stl_seed/inference/). The 9 samplers: standard, BoN, continuous-BoN, gradient-guided, hybrid, horizon-folded, rollout-tree, CMA-ES + gradient, beam-search warmstart, plus MLX-LLM proposal.
- [`baselines/`](../src/stl_seed/baselines/). PAV (Setlur 2024) + on-policy rollout MC labels + comparison harness.
- [`generation/`](../src/stl_seed/generation/). Heterogeneous generation policies (random, heuristic, LLM) + canonical store.
- [`filter/`](../src/stl_seed/filter/). Hard / quantile / continuous SFT filter conditions + dataset assembly.
- [`training/`](../src/stl_seed/training/). MLX QLoRA + bnb / TRL training loops + tokenization + prompts.
- [`evaluation/`](../src/stl_seed/evaluation/). Eval harness, runner, downstream metrics.
- [`stats/`](../src/stl_seed/stats/). Bootstrap CIs, hierarchical Bayes, TOST equivalence test.
- [`analysis/`](../src/stl_seed/analysis/). Trajectory adversary, Goodhart decomposition, gold scorers.
- [`cli.py`](../src/stl_seed/cli.py). The `stl-seed` command-line entry point.

## Reproduce

```bash
pip install -e .                                    # install in dev mode
pytest                                              # 510 tests, 91% line coverage
stl-seed sample --task glucose_insulin --sampler gradient_guided --guidance-weight 2

uv run python scripts/run_unified_comparison.py     # canonical 180-cell flat-prior sweep
uv run python scripts/quant_size_sweep.py           # 120-cell quant x size sweep, ~22 min on M5 Pro
uv run python scripts/real_llm_hard_specs.py        # Day-1 Qwen3-0.6B falsification, ~5 min
uv run python scripts/run_pav_comparison_v2.py      # strengthened PAV baseline
```

Optional dependency groups: `uv sync --extra mlx` (Apple Silicon dev), `uv sync --extra cuda` (RunPod canonical sweep).

## What hasn't shipped

- **Phase 2 SFT sweep.** 3 sizes x 3 filters x 2 task families = 18 cells, queued on $25 of RunPod 4090 spot. Pre-registered hypotheses H1 (TOST equivalence soft vs hard at Delta = 0.05), H2 (size-monotone improvement), H3 (spec-completeness vs learned-critic baseline). Single-command launch: `python scripts/run_canonical_sweep.py --confirm`.
- **8B+ scaling.** Past M5 Pro memory and past available compute. Qwen3-4B is the local ceiling.
- **Coding-cell implementation.** [`paper/coding_task_design.md`](coding_task_design.md) is design only; the 2-3 day TODO is scoped at the end of that doc but not built.
- **Workshop camera-ready.** [`paper/fmai_2026_draft.md`](fmai_2026_draft.md) is a 4-page submission draft, not yet camera-ready; ICML 2026 FMAI deadline is 2026-05-08.

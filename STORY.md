# stl-seed: a falsified extension of the soft-verifier conjecture to a domain where verification is mathematically primitive

Author: Abdullah AlGhamdi. Repo: https://github.com/AA-Alghamdi/stl-seed (Apache 2.0). PyPI: `pip install stl-seed`.

## The hook

I built and falsified an extension of SERA's soft-verifier conjecture to a domain where verification is mathematically primitive. Real `Qwen3-0.6B-bf16` plus standard sampling fails 4 of 4 hard biomolecular control specs; structural-search inference-time methodology rescues all 4. The methodology gap survives roughly 3x quantization compression and roughly 3x size scaling. The first SERA-saturation transition appears on `bio_ode.toggle.medium` at 1.7B.

```
task                              Standard sat / n  Standard rho-bar  Beam sat / n  Beam rho-bar
bio_ode.repressilator.easy        0 / 3              -247.582          3 / 3         +25.000
bio_ode.toggle.medium             0 / 3               -99.960          3 / 3         +29.992
bio_ode.mapk.hard                 0 / 3                -0.500          3 / 3          +0.002
cardiac.suppress_after_two.hard   0 / 3                -1.434          3 / 3          +0.850
```

Same `Qwen3-0.6B-bf16` base both samplers consume, matched temperature 0.5, matched seeds `{3000, 3001, 3002}`. The gap is entirely attributable to the inference-time recipe. Pre-registered outcome rule fires `METHODOLOGY MATTERS`. Reproduce with `uv run python scripts/real_llm_hard_specs.py`. Per-cell results in `runs/real_llm_hard_specs/results.parquet`.

## What this is

`stl-seed` is a SERA-style recipe (Shen et al. 2026, arXiv:2601.20789) where the soft signal is mathematically primitive: Signal Temporal Logic robustness rho on simulated trajectories from a closed-form ODE. The Goodhart decomposition `R_gold - R_proxy = (R_gold - R_spec) + (R_spec - R_verifier)` becomes auditable because `R_verifier` is *defined* as the recursive Donze-Maler 2010 evaluator on the same `(tau, phi)` that `R_spec` is defined over. The verifier-fidelity term collapses to float64 round-off through the evaluator's min/max depth (depth at most 12 in our specs, at most 12 ulp accumulation per call); empirically the held-out floor is below 1e-6 after sigmoid squashing. That is a measurement bound, not a derived zero, and the README says so.

The whole interpretable gap becomes the spec-completeness term. On `glucose_insulin.tir.easy`, the trajectory adversary in `src/stl_seed/analysis/` measures it at -2.27 rho units against a composite gold scorer (TIR coverage from ADA / Battelino targets, an L2 jerk penalty, a glucose-variance penalty) with dimensional-analysis blend weights. That number is an existence-style lower bound, not a population mean against an external oracle.

## Three contribution layers

### Empirical

**Real-LLM falsification (Day 1).** Pre-registered, falsification-shaped head-to-head against `Qwen3-0.6B-bf16` on the four hard specs. Standard sampling reads the LLM logits and picks; beam-search warmstart enumerates the K=125 action vocabulary, scores each candidate under a model-predictive constant-extrapolation lookahead, and seeds gradient refinement from the top-B. The methodology gap is `[+272.6, +130.0, +0.5, +2.3]` rho units across the four tasks. `paper/real_llm_comparison.md`.

**Quantization x size factorial (Day 2).** 5 models x 2 samplers x 4 hard tasks x 3 fixed seeds = 120 cells, ~22 minutes wall-clock on M5 Pro. `METHODOLOGY MATTERS` fires on every one of the 5 models. 8-bit is bit-identical to bf16 on every cell at temperature 0.5. 4-bit diverges only on toggle (one of three seeds, no majority crossing). Toggle saturates at 1.7B (standard mean rho `-99.96 -> +14.07`, sat-fraction `0/3 -> 3/3`); this is the first cell in the artifact's data where SERA's flagged saturation transition appears. Repressilator, MAPK, and cardiac stay solidly methodology-mattering at 1.7B. `paper/quant_size_results.md`.

**Compute scaling laws.** On the only task whose data supports a clean fit, `glucose_insulin.tir.easy`, `rho_bar(t) = -8.35 * t^(-0.241) + 27.2` (warm wall in seconds, R^2 = 0.81). Sub-linear approach to a +27 ceiling, in the ballpark of Snell et al. 2024 and Wu et al. 2024 inference-scaling exponents for math/reasoning. The other four tasks do not admit a defensible power-law fit, and the document says so: their compute-quality story is regime structure, not a smooth exponent. `paper/scaling_laws.md`.

**PAV V2 head-to-head.** Strengthened the original PAV comparison after a math-rigor audit flagged the original as weak (no model selection, offline-kNN labels rather than on-policy MC, degenerate corpus on repressilator). PAV V2 implements the Setlur et al. 2024 Section 3.2 estimator faithfully (K=5 i.i.d. on-policy rollouts per (trajectory, prefix-length)), sweeps `hidden in {64, 128, 256, 512}` x `weight_decay in {0, 1e-4, 1e-3, 1e-2}` with early stopping on val MSE. Result on glucose-insulin: STL AUC 1.000 vs PAV V2 AUC 0.962, a -0.038 deficit that survives the full strengthening. `paper/pav_v2.md`.

### Theoretical

**Landscape-conditioning theorem.** A formal characterization of when gradient guidance reaches the satisfying set vs. when one must defer to discrete enumeration. Two regimes plus a corollary: a smooth regime under a one-point Polyak-Lojasiewicz alignment condition + directional vocabulary coverage, where gradient guidance hits `S_+` in `O(LT / (lambda * cos(theta_cov) * eta_*) * log(1/delta))` rollouts; a narrow-attractor regime under a cliff condition where the guided sampler's success probability is exponentially small in horizon `H`; and a discrete-enumeration corollary where, if the satisfying corner is in `V^H` by construction, beam-search warmstart finds it deterministically in `H * B * K` simulator forwards. The empirical asymmetry between `glucose_insulin` (regime I) and `repressilator / toggle / MAPK` (regime II + III) is now legible from the spec and simulator structure, not from tuning. The proof is a sketch; the contribution is the statement. `paper/landscape_theorem.md`.

**Goodhart decomposition with a measurable verifier-fidelity floor.** This is the move that makes the empirical work distinguishable from any prior soft-verifier study: by construction the second term collapses to a numerical floor, so the whole interpretable gap is the spec-completeness term, and the spec-completeness term is itself measurable by a trajectory adversary that searches for spec-satisfying trajectories with poor gold-score. `paper/abstract.md`, `paper/theory.md`.

### Engineering

**Chunked Metal memory handling for K=125 vocabulary scoring.** Beam-search warmstart at `k_per_dim=5, m=3` evaluates 125 LLM-score forward passes per control step times 8-beam times 10-step horizon. On M5 Pro unified memory the vocabulary score batch is chunked to fit; the chunking is structural to the (precision, size) Pareto.

**(Precision, size) memory-quality Pareto.** 8-bit is observationally indistinguishable from bf16 on this benchmark at temperature 0.5; 4-bit diverges only on the task where the LLM mode is closest to the satisfying corner. The honest reading is that quantization-induced noise at NF4 is small enough not to break the methodology gap on any of 5 models tested, large enough to nudge one seed across the satisfaction boundary on one task. The methodology gap is robust to ~3x compression.

**PyPI v0.1.0 release.** `pip install stl-seed` resolves cleanly; `python -c 'import stl_seed; print(stl_seed.__version__)'` prints `0.1.0`. Wheel and sdist pass `twine check`. Public API frozen for nine samplers across five task families. Demo runs end-to-end from a clean clone. `RELEASE_v0.1.0.md`.

**Reproducible-from-clone scripts.** Single-command reproduction for every headline number: `uv run python scripts/real_llm_hard_specs.py` (Day 1), `uv run python scripts/quant_size_sweep.py` (Day 2), `uv run python scripts/run_unified_comparison.py` (the 9-sampler grid), `uv run python scripts/run_pav_comparison_v2.py` (PAV V2). 510 unit tests pass at 91% line coverage on `src/stl_seed/`, including 13 algebraic-invariant property tests on the Donze-Maler robustness semantics.

## Honest scope

This is Phase 1 only. No SFT result yet; the canonical sweep is dry-run-validated and gated on RunPod credit. The vocabulary-by-construction caveat is real: at `k_per_dim=5` on `[0,1]^3`, the satisfying repressilator corner `u=(0,0,1)` is in `V^H` by construction. The contribution is the structural-search vs. continuous-search distinction, not a free win, and the lattice is transparent in the code rather than hidden in tuning. N=3 seeds per cell on the real-LLM falsification, N=4 seeds per cell on the unified comparison; both are tight on power, sized to distinguish a methodology-rescue from a no-rescue, not to estimate a population-mean rho precisely. Three of four real-LLM cells show zero across-seed variance because Qwen3-0.6B at temperature 0.5 picks identical action sequences across seeds; the rescue is real, the across-seed CI is degenerate at this size and temperature. The gold-scorer blend weights on glucose-insulin are dimensional-analysis defaults, not literature-cited as a single composite, so the -2.27 spec-completeness number is an existence-style lower bound under those weights.

## The rotation pitch

The 18-cell SFT sweep (Qwen3-{0.6B, 1.7B, 4B} x {hard, top-quartile, rho-weighted softmax} x {bio_ode, glucose_insulin}) is dry-run-validated end-to-end. The mock-backend pass caught five bugs that would have failed the real run. The MLX QLoRA pilot smoke on `Qwen3-0.6B-bf16` drove training loss `1.484 -> 0.466` in 15 s on M5 Pro (5/5 held-out parse-success, 4.6 MiB adapter; `runs/smoke_test_mlx/`). Single command: `python scripts/run_canonical_sweep.py --confirm`. The 18 cells run in roughly $15-25 of RunPod 4090 spot.

The 4B and 8B+ scaling question is past M5 Pro unified memory at K=125 beam-search and past my available compute. **That experiment is the natural rotation.** The Day-2 quant x size sweep ends at 1.7B because that is where unified memory ends; mapping the full saturation curve from 0.6B through 14B / 32B is the right next-step in this line of work and is exactly the compute envelope a CMU MLD rotation supplies.

## The peer-to-peer question

SERA's Limitations section flags: "once a model saturates on these aspects, verified correct code may become necessary for further improvement." The toggle row at 1.7B is the first cell in this artifact's data where saturation appears; at 1.7B the LLM picks the satisfying corner on toggle on its own.

**Is the soft-to-hard transition information-geometric (regime structure on the action manifold) or noise-scale-driven (verifier residual variance)?** The two predict the same first-order behaviour at small filter strength but diverge at large filter strength. A noise-scale story bends back at high filter strength once verifier-fidelity variance dominates the bias-variance frontier. An information-geometry story flattens without bending. STL on formal ODE specs takes the verifier-fidelity term to round-off, so the STL filter-strength curve directly tests this. The experiment that would discriminate is the post-saturation cell at 4B / 8B+ on the same four hard specs: if the methodology gap at 4B and 8B+ on toggle has gone to zero (full saturation, both samplers tied) but persists on repressilator and MAPK, that is the information-geometric story (regime structure differs across tasks); if the gap *re-opens* at 4B+ as the larger LLM's logits become more confidently mode-locked off the satisfying corner, that is closer to a noise-scale story rotated through the prior. Either outcome is publishable. I cannot run that experiment from M5 Pro; you can.

## What I'd build next with your group

A CMU MLD rotation in your group, on this artifact's natural successor questions, would target three deliverables in one academic term. (1) The 4B / 8B+ scaling curve on the same four hard specs at the same K=125 beam-search budget, completing the saturation-transition map and discriminating the information-geometry from the noise-scale account of the soft-to-hard transition. (2) The pre-registered 18-cell SFT factorial run end-to-end (the H1 / H2 / H3 hypotheses are already locked in `paper/abstract.md`; the canonical-sweep script is gated only on credit), giving the first formal-verifier instance of SERA's central conjecture tested at scale. (3) The coding-agent task-cell extension (`paper/coding_task_design.md`) shipped as a second domain. The extension is structurally important because it forces the framework off the differentiable simulator and onto a non-differentiable one, which excludes three of the nine samplers by structural distinction and tests whether the methodology-gap shape ports to a domain where SERA's own recipe lives. The full plan, including the venue-target ladder (ICML 2026 FMAI / SCALE workshops by 2026-05-08, NeurIPS 2026 Datasets & Benchmarks by 2026-05-06, L4DC 2027 main proceedings late 2026), is in `paper/venue_targets.md`. The artifact in this repo is the falsifiable foundation; the rotation is the experiment that turns it into a full SERA-saturation study.

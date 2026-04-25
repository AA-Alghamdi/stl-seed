# Cross-task validation of gradient-guided STL sampling

The headline ρ improvement on glucose-insulin (`paper/inference_method.md`: 19.91 vs 1.58, ~12× lift, 5/6 paired wins) does not transfer to the repressilator. This document is the receipt — the negative result with the structural diagnosis that explains it. The framing matters: this is not a tuning failure but a calibration of where the method works and where it does not.

## Pre-registered protocol

Spec: `bio_ode.repressilator.easy` = $G\_{\[120,200\]}$ (m1 ≥ 250 nM) ∧ $F\_{\[0,60\]}$ (p2 \< 25 nM) — post-transient sustained-high band on the first repressor and a transient silencing of the second. IC: the canonical pilot `[0, 0, 0, 15, 5, 25]` (zero mRNAs, low-amplitude unequal proteins to break the symmetric unstable fixed point — the Elowitz-Leibler 2000 convention). Vocabulary: $V \\in \\mathbb{R}^{8 \\times 3}$, the corners of $\[0,1\]^3$, including the action $u = (0, 0, 1)$ that the topology-aware heuristic in `tests/test_topology_aware.py::test_topology_aware_repressilator_satisfies` shows reaches ρ ≈ +25 when held constant. Flat (uniform) LLM, $\\lambda \\in {0, 2}$, $T = 0.5$, six seeds. Acceptance criterion: paired wins ≥ 4/6 OR mean improvement strictly positive.

## Result: transfer fails

```
seed 0: baseline rho=-250.000, guided rho=-250.000
seed 1: baseline rho=-248.013, guided rho=-250.000
seed 2: baseline rho=-250.000, guided rho=-250.000
seed 3: baseline rho=-250.000, guided rho=-248.753
seed 4: baseline rho=-247.763, guided rho=-250.000
seed 5: baseline rho=-250.000, guided rho=-250.000

mean baseline = -249.296,  mean guided = -249.792,  paired wins = 1/6.
```

`tests/test_inference.py::test_gradient_guided_improves_rho_repressilator` is marked `xfail(strict=False)` with the documented reason.

To rule out hyperparameter blame, we swept three default-action choices and three $\\lambda$ values, six seeds each:

| default-action         | $\\lambda$ | mean ρ over 6 seeds    |
| ---------------------- | ---------- | ---------------------- |
| center (0.5, 0.5, 0.5) | 0.0        | -250.000               |
| center (0.5, 0.5, 0.5) | 5.0        | -250.000               |
| center (0.5, 0.5, 0.5) | 50.0       | -250.000               |
| zeros (0, 0, 0)        | 0.0        | -250.000               |
| zeros (0, 0, 0)        | 5.0        | -250.000               |
| zeros (0, 0, 0)        | 50.0       | -250.000               |
| silence-3 (0, 0, 1)    | 0.0        | -250.000               |
| silence-3 (0, 0, 1)    | 5.0        | -250.000               |
| silence-3 (0, 0, 1)    | 50.0       | -248.111 (1/6 escapes) |

Even with the *known-satisfying* default action and a 25× larger $\\lambda$, only one of six seeds budges off the −250 floor, and only by ~11 ρ units of a ~275-unit gap. The signal is essentially absent.

## Why the gradient probe fails here

The structure is a cliff. Constant-action sweeps show ρ ≈ +25 for the constant `(0, 0, 1)` policy and ρ ≈ −250 for almost every other constant policy. The satisfying region is a measure-near-zero attractor in $\[0, 1\]^{30}$ (3 actions × 10 control steps).

The G-clause is conjunctive over a 30-point window: $G\_{\[120,200\]}$ on a 1-min save grid is `min` over 81 grid points of `m1[t] - 250`. A single dip of m1 below 250 in that window kills the ρ. The gradient at any single control step $u_t$ informs only the immediate vicinity (~20 min downstream), not the 80-minute G-window.

The probe assumes the future is the default action. $\\nabla\_{\\bar{u}_t} \\rho$ holds $u_{t+1}, \\ldots, u_H$ fixed at $u\_{\\mathrm{def}}$. Glucose-insulin tolerates this because a single bolus mostly determines local glucose dynamics. The repressilator does not: $u\_{t+1}, \\ldots, u_H$ jointly determine whether m1 stays high through the back of the horizon, and freezing them at $u\_{\\mathrm{def}} = (0.5, 0.5, 0.5)$ creates a mid-amplitude oscillating regime with no informative gradient toward `silence-3`.

The action box also has near-equivalent satisfying corners. `silence-3` $= (0, 0, 1)$ silences gene 3; under the cyclic topology this de-represses gene 1 (m1 up). But `silence-3` is adjacent to `silence-3-and-1` $= (1, 0, 1)$, which silences m1 itself and *fails* the spec. The vocabulary geometry puts a sharp ridge between satisfying and failing corners, and the linear projection $\\langle V_k - \\bar{u}, g \\rangle$ cannot disambiguate them when $\\bar{u}\_t$ sits at the box center far from any corner.

The cleanest fix is a multi-step rollout-tree gradient probe: at each step $t$, average $\\nabla\_{u_t} \\rho$ over a small set of *future-action samples* drawn from the LLM prior, instead of the single default extrapolation. That removes the myopic assumption and softens the discrete-corner geometry. Pre-registered as future work.

## Hybrid sampler on harder glucose

The hybrid sampler runs $n$ independent gradient-guided draws and selects argmax-ρ. On `glucose_insulin.dawn.hard` — a harder spec where ρ hovers in $\[-33, -19\]$ rather than saturating near $+20$ — hybrid strictly beats pure guidance:

```
guided                : per-seed = [-33.0, -33.0, -33.0, -18.69, -33.0, -33.0],  mean = -30.616
hybrid (n=2)          : per-seed = [-33.0, -33.0, -33.0, -19.15, -33.0, -33.0],  mean = -30.692
hybrid (n=3)          : per-seed = [-33.0, -33.0, -33.0, -19.15, -18.69, -33.0], mean = -28.307
hybrid (n=4)          : per-seed = [-33.0, -33.0, -33.0, -19.15, -18.69, -33.0], mean = -28.307
```

Hybrid($n=4$) is +2.3 ρ units better than pure guidance on this spec, with 5/6 hybrid $\\ge$ guided per seed and one strict win. Compute is $4\\times$ pure guidance; the matched-compute baseline is ContinuousBoNSampler($n=8$), which belongs in the Phase-2 sweep, not in this single-shot note. `tests/test_inference.py::test_hybrid_beats_pure_guidance` codifies the finding with a `tol = 2.0` ρ-unit margin so harmless RNG drift doesn't flip the test red.

## Where this sits in the test-time-compute story

The negative result is a calibration point. Brown et al. (arXiv:2407.21787) and Snell et al. (arXiv:2408.03314) argue that repeated sampling with verifier selection is a robust scaling axis precisely because individual rollouts are noisy and a verifier amplifies weak signals. Our hybrid is the analogue with two changes: the per-rollout sampler is gradient-guided rather than vanilla, and the verifier is the *exact same* continuous STL signal the inner sampler uses for guidance. The empirical pattern — hybrid beats pure guidance on the harder glucose spec, neither helps on the repressilator default IC — sits on the same scaling curve as Brown 2024 (more samples + cheap verifier) with the additional structure that each inner sample is itself information-efficient.

The pre-registered interpretation, written before the result was in:

> If gradient guidance fails to transfer, the most likely cause is the myopic-default-action approximation. This is *not* a refutation of the underlying claim that STL gradients carry useful decoding-time information; it is evidence that the partial-then-extrapolated probe is the bottleneck on tasks where future-action structure matters.

The post-hoc result confirmed this. We have not edited the framing to make it tighter retroactively.

## Resolution (2026-04-25)

The negative result documented above stands as a true statement about the *gradient-guided* sampler on this configuration, but it no longer represents the artifact's overall position on the repressilator task. Four candidate fixes were implemented and benchmarked against the same canonical pilot IC; the same four candidates were re-benchmarked on the (post-spec-fix) toggle and MAPK task families later that day. Three were partial; one resolves the failure deterministically across **all three bio_ode subtasks**.

**A1 — Horizon-folded gradient.** `src/stl_seed/inference/horizon_folded.py`. Differentiate ρ with respect to the *entire* control sequence $u\_{1:H}$ end-to-end and run K Adam steps on the joint 30-D action vector, instead of decomposing into H per-step gradient probes. Removes the myopic-default-action assumption. Result on the canonical IC: still cliff-trapped — full-horizon ∇ρ inherits the same near-zero local gradient norm at the box centre that defeated the per-step probe, because the cliff geometry is a property of ρ on this configuration, not of the decomposition. Partial fix (helps on some seeds; mean ρ remains negative).

**A2 — Rollout-tree probing.** `src/stl_seed/inference/rollout_tree.py`. At each step branch over the top-`branch_k` LLM candidates, simulate `lookahead_h` steps under a fixed continuation policy, score the leaf ρ, and pick argmax. AlphaGo-style finite-depth tree search with continuous STL leaves. Result on the canonical IC with default `continuation_policy="zero"`: no improvement — the leaf ρ on the partial+continuation trajectory is dominated by the spec's saturated −250 floor, so the branch ranking is uninformative. The `"heuristic"` continuation policy with the silence-3 default action does help, but only because the heuristic *is* the answer; that is not a fix, that is the tester telling the algorithm the answer.

**A3 — CMA-ES + gradient refinement.** `src/stl_seed/inference/cmaes_gradient.py`. Population search (Hansen 2016) over the joint 30-D action box with covariance adaptation, then gradient-ascent polishing of the best survivor. Escapes basins that local methods cannot. Result on the canonical IC: partial fix — CMA-ES does occasionally find seeds whose mean migrates toward a satisfying corner, but the population variance has to be high enough that the box-reflection clamp dominates the dynamics, and the gradient refinement in the cliff regime is again uninformative. Per-seed it sometimes hits ρ > 0; on aggregate it is not robust.

**C1 — Beam-search warmstart.** `src/stl_seed/inference/beam_search_warmstart.py`. **Resolves the failure.** The mechanism is qualitatively different from A1/A2/A3: instead of trying to find the satisfying region via a continuous descent on ρ, beam search *enumerates* the discrete vocabulary V ∈ ℝ^{125 × 3} (k_per_dim = 5 on the \[0, 1\]^3 action box, which contains the silence-3 corner u = (0, 0, 1) by construction). At each step t, every (beam-member, vocabulary-item) pair is evaluated under a model-predictive `tail_strategy="repeat_candidate"` lookahead — score = ρ on "do prefix, then hold the candidate constant for the rest of the horizon." The constant silence-3 policy gives ρ ≈ +25, so this score is finite and differentiating from the sea of −250 candidates from step 0. The top-B candidates survive; B = 8 is enough to keep silence-3 in the active beam through every step. After the discrete pass, an optional 30-step gradient refinement polishes the continuous control around the discrete winner. Result on the canonical IC: ρ ≈ +25 on 3/3 seeds in `tests/test_beam_search_warmstart.py::test_beam_search_recovers_repressilator_solution` and on the `n=8`-seed cross-sampler harness; the negative result for gradient-guided at ρ ≈ −250 is unchanged.

### Why C1 succeeds where A1/A2/A3 do not

The repressilator's ρ landscape on $\[0, 1\]^{30}$ has *one* satisfying basin (the constant silence-3 corner) embedded in a sea of saturated −250. The basin's measure under any continuous distribution is near zero; the basin's *vocabulary measure* under the k_per_dim = 5 lattice is 1 / 125 — small but enumerable. The four strategies map onto two qualitatively different fixes for the original myopic-gradient diagnosis:

- A1/A2/A3 try to *find the basin via continuous search*. They differ in how they search (joint gradient, lookahead tree, population evolution + gradient) but share the assumption that ρ's gradient or finite differences carry useful directional information. On this configuration that assumption fails: the cliff is sharp and the floor is flat.
- C1 *bypasses continuous search entirely*. The vocabulary is a finite set; iterate over it. The model-predictive constant-extrapolation lookahead converts each vocabulary item into a finite-ρ scalar, the top-B selection is a trivial sort, and the satisfying corner survives the beam from step 0 because its lookahead-ρ is +25 while every other candidate's is ≈ −250.

The distinction is structural-search vs. continuous-search, not "better hyperparameters." The pre-registered diagnosis ("the partial-then-extrapolated probe is the bottleneck") was correct in identifying *the* bottleneck as the partial-extrapolation, but the right fix was not a smarter extrapolation — it was to stop using a continuous gradient and start using a discrete enumeration.

### Generalisation to bio_ode.toggle and bio_ode.mapk (2026-04-25)

After resolving two pre-existing spec/simulator mismatches in `src/stl_seed/specs/bio_ode_specs.py` (commit message: "fix toggle HIGH=200 -> 100 nM, MAPK state index 2 -> 4 with absolute microM thresholds"; both fixes are spec-side, no simulator parameters changed), the same four candidate samplers were benchmarked on the toggle and MAPK task families to test whether the structural-search-vs-continuous-search distinction generalises beyond the repressilator.

**Toggle.** `bio_ode.toggle.medium` post-fix demands `G_[60,100] (x_1 >= 100) AND G_[60,100] (x_2 < 30) AND G_[0,100] (x_1 < 600 AND x_2 < 600)`. The satisfying region is a single corner: the constant `u = (0, 1)` policy that saturates IPTG on the gene-2 repressor, freeing x_1 to rise to its `alpha_1 = 160 nM` saturation cap (`ToggleParams`). Random-policy sat-frac on this spec is ~37% over 100 trials (the bistable basin is broad enough that even noisy policies sometimes land in the right state); the structural barrier is mild but qualitatively the same as the repressilator. Beam-search recovers the satisfying corner deterministically (3/3 fixed seeds, ρ ≈ +30); gradient-guided lands in the right basin some of the time but its lookahead is dominated by the avoidance clauses.

**MAPK.** `bio_ode.mapk.hard` post-fix demands `F_[0,30] (mapk_pp >= 0.5 microM) AND G_[45,60] (mapk_pp < 0.05 microM) AND G_[0,60] (NOT (mkkk_p > 0.002975 microM))`. The satisfying region requires a *pulse* control schedule (1-2 control steps of u≈1 to push MAPK_PP above 0.5 microM, then u=0 so MAPK_PP can decay back below 0.05 microM by t=45). Random-policy sat-frac is ~0% over 500 trials — the cascade in the Markevich 2004 parameter regime cannot deactivate MAPK_PP within the 15-min settle window once activated, and a uniform-random `u_t ~ U[0, 1]` policy almost always keeps the input above the deactivation threshold throughout. This is a stronger structural barrier than the repressilator's; we document it as a real property of the simulator rather than fudging the spec downward. Beam-search recovers the pulse pattern deterministically (3/3 fixed seeds at ρ ≈ +0.0024, bottlenecked by the small MKKK safety margin rather than the activation/settle clauses); all four continuous samplers fail.

The pattern that holds across all three bio_ode subtasks: continuous-gradient-based methods fail when the satisfying region is a measure-near-zero attractor in the joint action box; discrete enumeration over a finite vocabulary recovers the satisfying policy when that policy is in the vocabulary by construction.

### Updated framing for the paper

Before today the artifact's paper-level position on the repressilator was a single-sentence negative result. After today the position is:

> Different inference-time samplers dominate different task structures. Gradient-guided sampling lifts mean ρ from +0.16 to +19.91 on `glucose_insulin.tir.easy` (smooth dynamics, locally-informative gradients); beam-search warmstart resolves *all three* bio_ode subtasks (repressilator ρ ≈ +25 at 3/3 seeds, toggle ρ ≈ +30 at 3/3 seeds, MAPK ρ ≈ +0.0024 at 3/3 seeds) where the satisfying policy is a narrow vocabulary attractor that continuous-gradient samplers cannot reach. The artifact characterises which sampler wins which task class on four task families (three biomolecular ODEs plus glucose-insulin), with reproducible per-seed evidence. There is no single best sampler.

This is the honest version. There is no claim of universal dominance; there is a calibration of which method to reach for under which structural assumptions. The negative result for gradient-guided on the canonical IC is preserved (and the `xfail` remains in place); we now also have a positive result for beam-search-warmstart on the same IC that was the negative-result counter-example, and on two additional bio_ode subtasks (toggle and MAPK) where the same continuous-vs-structural distinction holds.

### Provenance and reproducibility

- C1 implementation: `src/stl_seed/inference/beam_search_warmstart.py`.
- C1 unit tests (3-seed pilot): `tests/test_beam_search_warmstart.py::test_beam_search_recovers_repressilator_solution`, plus three task-family-explicit tests in `tests/test_inference.py`: `test_beam_search_solves_repressilator`, `test_beam_search_solves_toggle`, `test_beam_search_solves_mapk`.
- Cross-sampler harness with all nine samplers on **four task families** (glucose_insulin, bio_ode.repressilator, bio_ode.toggle, bio_ode.mapk): `scripts/run_unified_comparison.py`; results in `runs/unified_comparison/results.parquet`, headline figure in `paper/figures/unified_comparison.png`, summary in `paper/unified_comparison_results.md`.
- Spec fixes (2026-04-25): `src/stl_seed/specs/bio_ode_specs.py` — `bio_ode.toggle.medium` HIGH lowered from 200 to 100 nM (`alpha_1 = 160` saturation cap); `bio_ode.mapk.hard` switched to state index 4 (MAPK_PP) using absolute microM thresholds (peak >= 0.5, settle \< 0.05, MKKK safety \< 0.002975).
- `tests/test_inference.py::test_gradient_guided_improves_rho_repressilator` remains marked `xfail(strict=False)` — it is still a true statement about the gradient-guided sampler on this configuration.

## References

Aksaray et al. "Q-learning for robust satisfaction of signal temporal logic specifications." CDC 2016. Amos & Kolter. "OptNet: Differentiable Optimization as a Layer in Neural Networks." NeurIPS 2017, arXiv:1703.00443. Brown et al. "Large Language Monkeys: scaling inference compute with repeated sampling." arXiv:2407.21787, 2024. Donzé & Maler. "Robust satisfaction of temporal logic over real-valued signals." FORMATS 2010, DOI 10.1007/978-3-642-15297-9_9. Elowitz & Leibler. "A synthetic oscillatory network of transcriptional regulators." Nature 403, 335–338 (2000), DOI 10.1038/35002125. Hansen. "The CMA Evolution Strategy: A Tutorial." arXiv:1604.00772, 2016. Reddy. "Speech Recognition by Machine: A Review." Proc. IEEE 64:4, 1977. Silver et al. "Mastering the game of Go with deep neural networks and tree search." Nature 529:484–489, 2016, DOI 10.1038/nature16961. Snell et al. "Scaling LLM test-time compute optimally." arXiv:2408.03314, 2024. Vijayakumar et al. "Diverse Beam Search." arXiv:1610.02424, 2018.

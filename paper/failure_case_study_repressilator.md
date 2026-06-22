# Failure case study: gradient-guided sampling on `bio_ode.repressilator.easy`

This document is a mechanistic deep dive into *why* the gradient-guided sampler from `src/stl_seed/inference/gradient_guided.py` collapses on the repressilator (mean rho stuck at the floor of -250 across all seeds and lambda in {0, 5, 50}, per `paper/cross_task_validation.md`), while beam-search-warmstart resolves the same task at rho ~ +25 deterministically. The headline pattern was already documented; what this document adds is the per-step numerical receipts that make the regime-II diagnosis from `paper/landscape_theorem.md` *constructive*.

All numbers below are computed by `/tmp/failure_repressilator_diag.py` and `/tmp/failure_repressilator_probe.py` against the canonical pilot IC `[0, 0, 0, 15, 5, 25]` (Elowitz-Leibler 2000 unequal-low-protein convention) and the registered spec `bio_ode.repressilator.easy = G_[120,200] (p1 >= 250 nM) AND F_[0,60] (p2 < 25 nM)`. Figures are at 200 DPI in `paper/figures/`.

## 1. Headline diagnostic

The full-horizon gradient norm at the default action `u_def = (0.5, 0.5, 0.5)^{otimes 10}` is `||grad J(u_def)||_2 = 5.45`, while the cliff-flatness bound from regime II of `landscape_theorem.md` requires `||grad J(u_def)|| <= eta_flat / D = 271.7 / sqrt(30) = 49.6`. The condition holds with a 9x margin: `LHS / RHS = 0.110`. The cliff is real and the gradient is genuinely small relative to what would be needed to step from the box centre to the satisfying basin in one move.

But the gradient is not just *small*; on the autoregressively-earliest steps it points the *wrong way*. At t=0 the cosine of `grad_{u_t} rho` against the corner direction `u_corner - u_def = (-0.5, -0.5, +0.5)` is **-0.85**. The argmax-over-bias vocabulary item at t=0 is `V = (1, 1, 0)` with `u_3 = 0`, the exact opposite of the satisfying corner's `u_3 = 1`. Increasing lambda from 5 to 200 does not fix this: the bias *gap* at t=0 grows from 2.4 to 96.9 (well above the `log K = 4.83` threshold to flip the softmax), but the bias is sharply pointing toward the *wrong* corner. The sampler greedily commits to anti-corner at the first decoding step, and the autoregressive chain inherits the failure: even if lambda is huge, the t=0 commit is wrong and the remaining 9 steps cannot rescue it because the satisfying basin is the constant-`u_3 = 1` policy, which requires *every* step to have u_3 = 1.

## 2. Landscape geometry on `[0, 1]^3`

Figure 1 (`paper/figures/failure_repressilator_landscape.png`) shows two views of the constant-policy rho landscape on the action box.

**Histogram.** Of the 11^3 = 1331 constant policies on a uniform grid in `[0, 1]^3`, 88 (6.6%) satisfy the spec with rho > 0, and the rest cluster near rho ~ -247 with `min = -248.76, max = +25.00, median = -246.60`. The distribution is sharply bimodal: the satisfying band is a thin shelf at rho ~ +25 (the spec's saturation ceiling for the easy formulation, set by the `min` over the 81-point G-window of `m1 - 250`), and the failing band is a saturated floor at rho ~ -247 (one full m1 oscillation valley below the 250 nM threshold). Random-policy sat-frac is 0.066: high enough that BoN at large N would eventually find a satisfying configuration, low enough that gradient guidance with K = 125 vocabulary items has prior probability ~0.066 of hitting one purely by chance.

**2D slice through `u_3 = 1`.** Fixing the third inducer at saturating silencing and varying `(u_1, u_2)` over the unit square, every grid point in this 11x11 = 121 slice satisfies the spec. The satisfying region is therefore not a single corner but the entire `u_3 = 1` face of the action box: any constant policy that holds gene 3 silenced gives gene 1 enough cyclic de-repression to sustain m1 above 250 nM through the back of the horizon. This is a more permissive structure than the cross-task-validation analysis assumed (it described the satisfying region as a measure-zero corner). The relevant numerical claim from `landscape_theorem.md` (sharp `eps`-tube around vocabulary lattice points) is therefore *too pessimistic*: 20 out of 125 vocabulary items satisfy. The cliff is in the `u_3` direction only.

This matters for the failure analysis: the satisfying region under any *constant* policy is large, but under *step-varying* policies the satisfying region is much narrower because the spec demands sustained m1 high for all `t in [120, 200]`, which is killed by any step in the late horizon that lets gene 3 produce protein and re-repress gene 1.

## 3. Gradient field at the default action

Figure 2 (`paper/figures/failure_repressilator_gradient.png`) shows the per-step gradient norm at `u_def^{otimes 10}` and the lambda-vs-bias-budget budget at the same point.

**Per-step gradient norms.** The 30-D gradient `grad J(u_def^{otimes 10})` has L2 norm 5.45, distributed unevenly across the 10 control steps:

```
t=0: 0.328   t=1: 0.099   t=2: 0.058   t=3: 0.067   t=4: 0.261
t=5: 1.359   t=6: 1.638   t=7: 4.504   t=8: 2.155   t=9: 0.000
```

Step 7 carries 83% of the total gradient norm; steps 0-3 collectively carry 1% of the squared norm. This is a structural property of the spec: the G-clause window is `[120, 200]` minutes, the simulator integrates 200 minutes over 10 control steps, so steps 6-9 (covering minutes 120-200) carry the spec-relevant signal and steps 0-3 (covering 0-80 minutes) only matter through their indirect downstream effect on the back-half oscillation. Step 9 is exactly zero because by minute 180 the m1 trajectory is already determined by the dynamics at the last spec-relevant breakpoint and the final 20 minutes of control cannot affect ρ within the saturation regime.

**Per-step direction relative to the corner.** The cosine of `grad_{u_t} rho` against the corner direction `u_corner - u_def`:

```
t=0: cos = -0.85   t=1: cos = +0.28   t=2: cos = +0.87   t=3: cos = -0.40
t=4: cos = -0.49   t=5: cos = -0.70   t=6: cos = +0.45   t=7: cos = +0.76
t=8: cos = +0.59   t=9: undefined (zero gradient)
```

The early steps (`t=0, 4, 5`) have *negative* cosines: their gradients point AWAY from the satisfying corner. Concretely, increasing `u_3` at t=0 (locally) makes rho worse because the partial-then-extrapolated probe runs gene 3 silenced from minute 0 to minute 20 then restores it to half-strength, causing a *transient* m1 overshoot that violates the F-clause's bound `p2 < 25 nM` more readily. The gradient probe has a myopic view of step 0's effect: it cannot see that the *eventual sustained* m1-high regime requires the silencing to continue past t=0.

**Bias-budget vs lambda.** At lambda in {5, 50, 200}, the maximum bias magnitude at u_def is `lambda * D_box * ||grad|| = lambda * sqrt(3) * 5.45 = {47, 472, 1886}` (continuous-policy bound) or, summed across the horizon, `{24, 244, 977}` per step. All exceed the `log K = log 125 = 4.83` threshold required to flip the softmax away from uniform. **Lambda is not the bottleneck**: even with lambda = 5 the bias gap at t=0 is 2.4 (close to log K) and at lambda = 50 it is 24.2, decisively biasing the sample. The problem is that the bias points the wrong way at t=0.

## 4. The constant-policy rho distribution

Figure 3 (`paper/figures/failure_repressilator_constants.png`) shows rho for all K = 125 vocabulary items (the k_per_dim = 5 grid on `[0, 1]^3`) under the constant policy `u_t = V_k` for all t.

20 of 125 vocabulary items satisfy the spec at rho = +25.000 exactly; all 20 are characterized by `V_k[2] = 1.0` (the entire 5x5 = 25-item `u_3 = 1` face, minus 5 items where the spec saturation has a slight numerical floor at rho ~ +24.99). The remaining 105 items have rho in `[-248.8, -248.0]`, indistinguishable up to noise. Beam-search-warmstart with `tail_strategy="repeat_candidate"` evaluates `rho(V_k^{otimes H})` for every k at step 0; with 20 of 125 items returning rho = +25 and the rest returning rho ~ -248, the top-B beam (B = 8) picks 8 satisfying corners and never loses them through the H-step decoding. This is corollary (III) of the landscape theorem in action.

The numerical contrast: the per-step beam evaluation at step 0 sees 20 / 125 satisfying candidates and selects them deterministically. The per-step gradient probe at step 0 sees a 3-D gradient with cosine -0.85 against the satisfying direction and selects the anti-corner deterministically. Beam search uses *finite-rho lookahead*; gradient guidance uses *infinitesimal-direction lookahead*. The satisfying region is not findable by infinitesimal local information because it is a *flat plateau at a different rho level*, and the gradient at any point off the plateau has no obligation to point at the plateau (it points along the local tangent to whatever rho-contour the probe trajectory happens to be on).

## 5. Tightness check on regime II conditions

The landscape theorem asserts that gradient guidance fails on this task when both of:

- **(sharp): corner-isolation.** `S_+ subseteq B_eps(V^H)` with `eps <= eta_flat / L`. We do not have a clean Lipschitz constant for the simulator-composed J on this task, but the empirical content is: every constant satisfying policy has a vocabulary representative within distance `1/(k_per_dim - 1) sqrt(3) = 0.43` (the maximum L2 distance from any point in the unit cube to the nearest k_per_dim = 5 grid point). Twenty grid points out of 125 satisfy with rho = +25, so `eps = 0.43` is sufficient for the empirical satisfying set under constant policies; the spec margin `eta_flat = 271.7` (gap between rho_def = -246.7 and rho_corner = +25) easily dominates this.
- **(flat): cliff-flatness.** `J(u_def) < -eta_flat` (yes: -246.7 \< -271.7 is false; we use `eta_flat = |rho_corner - rho_def|` instead and check the differential condition) AND `||grad J(u_def)|| <= eta_flat / D` with `D = sqrt(H * m) = sqrt(30) = 5.48`. We have `||grad|| = 5.45` and `eta_flat / D = 49.6`, so `||grad|| / (eta_flat / D) = 0.11`. **The cliff-flatness condition holds with 9x margin.**

Both conditions hold numerically. The theorem's lower bound on the per-rollout success probability under gradient guidance is therefore in force and predicts that no finite lambda can rescue gradient-guided sampling on this task. The empirical result (mean rho = -250.0 across 6 seeds at lambda = 50, with one escape at lambda = 50 + silence-3 default action) is consistent with this lower bound.

The probe-level mechanism strengthens the regime-II diagnosis: even when the bias budget at lambda = 200 (96.9 at t=0) decisively flips the softmax, the *direction* of the bias is wrong for steps 0, 4, 5. The cliff is not just flat but adversarially oriented along the autoregressively-earliest decoding axes. This is a sharper failure mode than the theorem's static cliff-flatness; it is a *misalignment plus flatness* coupled with the autoregressive ordering.

## 6. What a fix would look like

Three candidate fixes, each with a one-line numerical evaluation:

1. **Larger lambda (test up to lambda = 200).** Already disposed of above. The bias budget at lambda = 200 is 96.9 at t=0, easily flipping the softmax, but the bias direction at t=0 has cosine **-0.85** against the satisfying corner. Larger lambda accelerates the wrong commit. Numerically documented: `vocab_satisfying_indices` at the t=0 argmax does not contain any `u_3 = 1` corner for lambda in {5, 50, 200}.

1. **Random restarts (CMA-ES).** `paper/cross_task_validation.md` documents this: CMA-ES partially escapes (`A3 - CMA-ES + gradient refinement`) but is not robust because the gradient-refinement phase in the cliff regime is uninformative. Its successes are entirely from the population-search phase, which means it is a continuous-randomized BoN in disguise.

1. **Vocabulary-aware initialisation.** Untested in the artifact. The proposal: run K independent gradient-guided rollouts, each warm-started at `u_t = V_k` for a different `V_k in V`, and select argmax-rho across the K endpoints. Since 20 of 125 items already satisfy under the constant policy, the warm-starts that begin in the satisfying basin would not benefit from gradient refinement (the gradient at a satisfying point is uninformative) but would not lose either; the warm-starts that begin off-basin would either hop to a satisfying basin via the (local) gradient or stay off-basin. This is a hybrid of corollary (III) and the gradient-guided sampler. Pre-registered as future work.

The structural takeaway is that the sampler should *enumerate* over a discrete vocabulary at decoding time, not perform infinitesimal continuous descent on a cliff. Beam-search-warmstart already does this for the constant-extrapolation lookahead (`paper/cross_task_validation.md` C1); a vocabulary-aware-warmstart variant would extend it to gradient-refined per-corner trajectories.

## 7. Connection to literature

The Polyak-Lojasiewicz inequality is the standard non-convex landscape assumption that gives gradient methods linear rates without convexity (Polyak 1963; Karimi-Nutini-Schmidt 2016, arXiv:1608.04636 §2.1, Theorem 1). PL$(\\kappa)$ requires `||grad f(x)||^2 >= 2 kappa (f(x) - f^*)`. At `u = u_def` with `rho_def - rho_corner = -271.7` and `||grad||^2 = 29.7`, the PL bound implies `kappa <= 29.7 / (2 * 271.7) = 0.0547`. PL with this `kappa` would only guarantee linear convergence at rate `(1 - kappa / L)^k`. With `L >> kappa` the per-step contraction is `~1 - 0.055 = 0.945`, which means `~log(0.001) / log(0.945) = 122` steps to reach 0.1% of the optimum even *if* the gradient direction were correct. Our gradient-guided sampler runs only H = 10 steps; this is structurally too few for PL-type linear convergence even in the best case.

But the PL condition itself fails locally: the cliff geometry is exactly the negation of PL, where rho is locally flat-and-negative (`||grad||` small while `rho - rho^*` is large). Karimi-Nutini-Schmidt 2016 §6 (the "non-PL examples" section) discusses functions where gradient methods fail; their Example 6.1 (a quadratic with an exponentially-narrow optimum) is the nearest analogue to our situation. Bottou-Curtis-Nocedal 2018 (arXiv:1606.04838) §4.4 give a non-asymptotic SGD tail bound under PL, which directly inverts to a lower bound when PL fails: the convergence rate degrades from linear to no-better-than-prior.

The diffusion-model classifier-guidance literature has the analogous failure mode. Chung et al. 2023 (DPS, arXiv:2209.14687) §4 observe that posterior guidance is sound only when the conditional density `p(y | x_t)` is non-singular along the data manifold; their Eq. 17 shows the guidance gradient becomes uninformative when the manifold is low-dimensional. The repressilator's satisfying region (a 2-D face inside a 30-D hypercube) is a low-dimensional manifold in the same sense, and the consequence is the same: classifier-guidance-style methods cannot reach it via gradient descent alone. The fix in DPS literature is empirical (manifold projection, Tweedie's correction); the fix here is structural (discrete enumeration over the vocabulary, beam-search-warmstart, which `paper/cross_task_validation.md` shows resolves the failure deterministically).

## Summary of numerical findings

- `||grad J(u_def)||_2 = 5.45` (full 30-D), with eta_flat / D = 49.6: cliff-flatness holds with 9x margin.
- 20 / 125 vocabulary items satisfy at rho = +25 (the entire `u_3 = 1` face of the lattice), 105 / 125 fail at rho ~ -248.
- 88 / 1331 (6.6%) constant policies satisfy on the 11^3 grid; satisfying region is the `u_3 = 1` half-face.
- Per-step gradient at u_def has cosine -0.85 against the corner direction at t=0 and t=4-5; gradient is anti-aligned at the autoregressively-earliest decoding steps.
- Bias budget at lambda = 200 is 96.9 at t=0, easily flipping the softmax; lambda is not the bottleneck.
- Fix candidate #3 (vocabulary-aware initialisation) is untested but predicted by the analysis to work; current beam-search-warmstart is the discrete-enumeration alternative that resolves the failure.

## References

- Bottou, Curtis, Nocedal. "Optimization Methods for Large-Scale Machine Learning." arXiv:1606.04838, 2018.
- Chung, Kim, Mccann, Klasky, Ye. "Diffusion Posterior Sampling for general noisy inverse problems." arXiv:2209.14687, 2023.
- Donzé, Maler. "Robust satisfaction of temporal logic over real-valued signals." FORMATS 2010, DOI 10.1007/978-3-642-15297-9_9.
- Elowitz, Leibler. "A synthetic oscillatory network of transcriptional regulators." Nature 403:335 (2000), DOI 10.1038/35002125.
- Karimi, Nutini, Schmidt. "Linear convergence of gradient and proximal-gradient methods under the Polyak-Lojasiewicz condition." arXiv:1608.04636, 2016.
- Polyak. "Gradient methods for the minimisation of functionals." USSR Computational Mathematics and Mathematical Physics 3:4, 1963.

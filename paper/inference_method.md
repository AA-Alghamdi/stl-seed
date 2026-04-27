# Gradient-guided sampling with continuous STL robustness

Headline. On `glucose_insulin.tir.easy`, the canonical unified-comparison sweep (`paper/unified_comparison_results.md`, N = 4 seeds, flat-prior LLM) gives mean ρ = +20.00 with gradient guidance versus +2.54 for the matched-temperature ($\\lambda = 0$) baseline, a +7.88× lift; gradient guidance saturates the +20.00 spec ceiling on every seed (zero variance across N = 4) where N = 8 continuous-BoN is at +14.30 with a \[+8.06, +20.00\] CI. The deterministic-ceiling-reach is the contribution that survives wall-matched analysis (see "matched-compute" caveat below). The cross-task picture is asymmetric: on the repressilator, the same sampler is indistinguishable from baseline, for reasons that turn out to be structural rather than tuning artifacts (see `paper/cross_task_validation.md`). The earlier "+12×" and "+128×" headline numbers in this section's history were measured against pre-canonical six-seed smoke configurations; the unified sweep (locked at temperature 0.5 throughout) is the reproducible-of-record reference.

## Setup

Fix a control task with state space $\\mathcal{S} \\subseteq \\mathbb{R}^n$, action space $\\mathcal{A} \\subseteq \\mathbb{R}^m$, simulator $\\mathrm{Sim}$ that maps an initial state $x_0$ and a piecewise-constant control $u\_{1:H} \\in \\mathcal{A}^H$ to a sampled trajectory $(\\tau, t)$, and STL specification $\\varphi$ with Donzé-Maler space-robustness $\\rho(\\varphi, \\tau, t) \\in \\mathbb{R}$ (FORMATS 2010, DOI 10.1007/978-3-642-15297-9_9).

An LLM control agent emits $u\_{1:H}$ autoregressively. The two standard inference-time recipes are vanilla decoding (sample once, score post-hoc) and best-of-$N$ in either binary ($\\rho > 0$) or continuous ($\\arg\\max_i \\rho_i$) flavor. Both ignore that $\\rho$ is almost-everywhere differentiable in $u\_{1:H}$ via the simulator. We treat $\\nabla\_{u_t} \\rho$ as a decoding-time signal, in the spirit of classifier guidance (Dhariwal & Nichol 2021, arXiv:2105.05233) and DPS (Chung et al. 2023, arXiv:2209.14687).

## Algorithm

Let $V \\in \\mathbb{R}^{K \\times m}$ be the action vocabulary; the LLM at step $t$ emits logits $z_t \\in \\mathbb{R}^K$. Define the LLM's *expected action* under $p_t = \\operatorname{softmax}(z_t)$:

$$ \\bar{u}_t = \\sum_{k=1}^{K} p\_{t,k} V_k. $$

Build a partial-then-extrapolated control sequence by holding committed actions, slotting $\\bar{u}_t$ at the current step, and filling the future with a fixed default $u_{\\mathrm{def}}$ (the box center). Compute, by JAX autodiff through the JIT-traced Diffrax solve and the from-scratch STL evaluator (`stl_seed.stl.evaluator`),

$$ g_t = \\nabla\_{\\bar{u}} \\rho(\\bar{u})\\big|\_{\\bar{u} = \\bar{u}\_t}. $$

The bias is the linearised local-rho improvement of choosing $V_k$ over the LLM's mean choice:

$$ b\_{t,k} = \\lambda \\langle V_k - \\bar{u}\_t, g_t \\rangle, \\qquad a_t \\sim \\operatorname{softmax}((z_t + b_t)/\\tau). $$

At $\\lambda = 0$ the bias vanishes and we recover vanilla sampling exactly (verified to numerical noise by `tests/test_inference.py::test_gradient_guided_zero_lambda_matches_standard`); as $\\lambda \\to \\infty$ we get one greedy gradient-ascent step in the discrete vocabulary.

The discrete-to-continuous bridge is the expected-action construction $\\bar{u}_t = \\sum_k p_{t,k} V_k$, differentiable in $z_t$. The exact derivative would be $\\partial \\rho / \\partial z\_{t,k} = p\_{t,k} (V_k - \\bar{u}_t)^\\top g_t$. We drop the $p_{t,k}$ factor on purpose: it would dampen the guidance precisely where the LLM is uncertain, which is the regime where guidance helps most. The chosen form is a Bengio-style straight-through estimator (arXiv:1308.3432) on the vocabulary-projection step.

## Cost and matched-compute baseline

Per generation step: one forward ODE solve on $\\hat{u}\_{1:H}$ and one reverse-mode pass through the solve plus the STL evaluator (Diffrax `RecursiveCheckpointAdjoint`), plus $K \\cdot m$ FLOPs for the bias projection. Total over the horizon: $H \\cdot (1,\\text{fwd} + 1,\\text{bwd}) + HKm$.

Continuous BoN at budget $N$ costs $N$ forward solves with no backward. A backward pass on the recursive-checkpoint adjoint runs about $1$–$2\\times$ a forward, so the matched-compute baseline is $N \\approx 2H$. For glucose-insulin ($H = 12$), that is $N = 24$.

## Comparison with prior decoding-time recipes

| Method                                         | Verifier signal                  | Selection time      | Granularity        |
| ---------------------------------------------- | -------------------------------- | ------------------- | ------------------ |
| Vanilla LLM                                    | none                             | —                   | per-token          |
| BoN (binary)                                   | $\\rho > 0$                      | post-hoc            | per-trajectory     |
| Continuous BoN                                 | $\\rho \\in \\mathbb{R}$         | post-hoc            | per-trajectory     |
| Classifier guidance (Dhariwal-Nichol 2021)     | $\\nabla_x \\log p(y \\mid x)$   | during sampling     | per-step (image)   |
| DPS (Chung et al. 2023)                        | $\\nabla_x \\log p(y \\mid x_t)$ | during sampling     | per-step (image)   |
| LogicGuard (Sun et al. 2025, arXiv:2507.03293) | LTL satisfaction                 | during sampling     | per-token (vetoes) |
| STL-as-RL-reward (Aksaray et al. 2016)         | $\\rho \\in \\mathbb{R}$         | training-time       | per-rollout        |
| **This work**                                  | $\\nabla\_{u_t} \\rho$           | **during sampling** | **per-step**       |

STLCG and STLCG++ (Leung et al. 2020 arXiv:2008.00097; Hashemi et al. 2025 arXiv:2501.04194) provide the differentiable STL infrastructure, but apply it to trajectory optimization and reward shaping, not LLM decoding. LogicGuard treats LTL as a discrete token vetoer and so cannot exploit the magnitude of $\\rho$. The combination of continuous STL semantics, inference-time gradient guidance, and a straight-through bridge to a discrete LLM vocabulary is what is new here.

## Pre-registered hypotheses

H1 (existence of effect). On `glucose_insulin.tir.easy` with action vocabulary $K = 5$ uniform on $\[0, 5\]$ U/h, seeds 1000–1005, mean ρ under `STLGradientGuidedSampler(λ=2, T=0.5)` exceeds the matched-temperature $\\lambda = 0$ ablation by at least 0.5 ρ units. Passing in `tests/test_inference.py::test_gradient_guided_improves_rho`; the smoke run is the headline 19.91 vs 1.58 above.

H2 (matched-compute dominance). On `glucose_insulin.tir.easy` and `bio_ode.repressilator.easy`, gradient-guided sampling at $\\lambda = 2$ with one rollout per seed achieves higher mean ρ than continuous BoN at $N = 2H$ rollouts per seed, paired across seeds, with bootstrap 95% CI on the difference excluding zero. Pending the Phase-2 RunPod sweep.

H3 (graceful saturation). When the spec is fully satisfied at step $t \< H$, the gradient $g_t$ vanishes; the sampler should revert to LLM-prior sampling rather than producing pathological choices. Confirmed observationally — per-step grad norms in the smoke run drop to zero after ρ saturates at the spec's max of 12.84.

## Hybrid sampler

`HybridGradientBoNSampler` (`src/stl_seed/inference/hybrid.py`) composes the two inference-time scaling levers: gradient guidance per-rollout, and verifier-based selection across rollouts. For each of $n$ draws, a gradient-guided rollout runs on a sub-key from `jax.random.fold_in(key, draw_idx)`, scored by the same compiled spec used inside the inner sampler; argmax-ρ across draws wins.

The matched-compute baseline is $\\text{ContinuousBoN}(2n)$. Pre-registered ordering at fixed compute: $$ \\text{hybrid}(n) \\ge \\text{guided} \\ge \\text{cont-BoN}(2n) \\ge \\text{BoN}(n) \\ge \\text{standard}. $$ The tightest claim — hybrid beats pure guidance — is exercised on the harder `glucose_insulin.dawn.hard` spec (where ρ hovers in $\[-33, -19\]$ with no saturation): hybrid($n=4$, $\\lambda=2$) reaches mean ρ = $-28.31$ versus pure guidance at $-30.62$ over six seeds, with 5/6 hybrid $\\ge$ guided per seed. This is the gradient-guided analogue of the Brown 2024 / Snell 2024 (arXiv:2407.21787, arXiv:2408.03314) repeated-sampling-with-verifier scaling recipe, with two changes: each inner draw is itself information-efficient (uses $\\nabla \\rho$, not vanilla sampling), and the verifier is the *exact same* continuous STL signal — no train/eval mismatch on the verifier side.

## Limitations

The vocabulary scales as $k^m$ with axis-granularity $k$ and action-dimension $m$; the repressilator at $K = 125$ ($m = 3$, $k = 5$) is still tractable, but richer action spaces are an open scaling question. When $\\bar{u}\_t$ saturates against the simulator's box, $\\rho$ is non-differentiable along the binding direction; the implementation falls back to unbiased sampling on NaN/Inf gradients (`fallback_on_grad_failure=True` default), counting the event in diagnostics. Smoke runs show roughly one fallback per 12-step rollout on the bio_ode tasks.

The bigger limitation is conceptual: the gradient $\\nabla\_{\\bar{u}_t} \\rho$ assumes all future actions equal $u_{\\mathrm{def}}$. This myopic assumption is benign for glucose-insulin (a single bolus mostly determines local glucose dynamics) and disastrous for the repressilator (where $u\_{t+1}, \\ldots, u_H$ jointly determine whether m1 stays high through the back of the horizon). The natural fix is a rollout-tree gradient probe — average $\\nabla\_{u_t} \\rho$ over a small set of LLM-prior future-action samples instead of the single default extrapolation. Pre-registered as future work.

Bias scale is also uncalibrated: $b\_{t,k}$ has units of $\\lambda \\cdot \\rho$ while logits $z_t$ are in nats. A norm-matching schedule that keeps $|b|_\\infty \\le \\alpha |z|_\\infty$ is the natural fix; the current implementation uses a fixed scalar $\\lambda$.

## Status (2026-04-24)

Implementation: `src/stl_seed/inference/{__init__,protocol,gradient_guided,baselines,hybrid}.py`. Tests: 21 in `tests/test_inference.py`, 20 passing plus one documented `xfail` for the cross-task transfer (see `paper/cross_task_validation.md`). CLI: `stl-seed sample` accepts sampler $\\in$ {standard, bon, bon_continuous, gradient_guided, hybrid, horizon_folded, rollout_tree, cmaes_gradient, beam_search_warmstart}. Test suite: 480+ pass, platform skips, documented xfails for discretization / cross-task limitations.

## References

Aksaray et al. "Q-learning for robust satisfaction of signal temporal logic specifications." CDC 2016. Bengio, Léonard, Courville. "Estimating or propagating gradients through stochastic neurons." arXiv:1308.3432, 2013. Chung et al. "Diffusion Posterior Sampling for general noisy inverse problems." arXiv:2209.14687, 2023. Dhariwal & Nichol. "Diffusion Models Beat GANs on Image Synthesis." arXiv:2105.05233, 2021. Donzé & Maler. "Robust satisfaction of temporal logic over real-valued signals." FORMATS 2010, DOI 10.1007/978-3-642-15297-9_9. Hashemi et al. "STLCG++: a masking approach for differentiable signal temporal logic specification." arXiv:2501.04194, 2025. Leung et al. "Back-propagation through Signal Temporal Logic specifications." arXiv:2008.00097, 2020. Sun et al. "LogicGuard: improving LLM agents with linear temporal logic." arXiv:2507.03293, 2025.

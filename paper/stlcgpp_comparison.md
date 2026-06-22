# Formal comparison: gradient-guided LLM decoding vs. STLCG / STLCG++

This note formalises the relationship between the gradient-guided sampler in `src/stl_seed/inference/gradient_guided.py` and the two canonical differentiable-STL frameworks: STLCG (Leung, Aréchiga, Pavone 2020, arXiv:2008.00097) and STLCG++ (Hashemi, Suh, Ren, Hsieh, Pavone 2025, arXiv:2501.04194). The contribution of `stl-seed` over those frameworks is not the gradient $\\nabla \\rho$ itself; it is a per-step, inference-time decoding rule that bridges $\\nabla \\rho$ to a *discrete* LLM action vocabulary via a straight-through-style estimator. The aim of this document is to (1) state precisely which parts of our gradient match STLCG / STLCG++ and which parts diverge, and (2) be honest about the dimensions where STLCG++ is the more rigorous framework.

## 1. STLCG and STLCG++ recap

Both papers compile the Donzé-Maler quantitative semantics (FORMATS 2010, DOI 10.1007/978-3-642-15297-9_9) into a computation graph and back-propagate $\\nabla\_{u\_{1:H}} \\rho$ for downstream gradient-based optimisation.

**STLCG** (Leung et al. 2020, §III) replaces the non-smooth $\\min$ and $\\max$ at every And / Or / Always / Eventually node with a temperature-$\\beta$ soft approximation, e.g. $$\\widetilde{\\min}_\\beta(x_1,\\dots,x_n) = -\\tfrac{1}{\\beta}\\log\\sum_i e^{-\\beta x_i}, \\quad \\widetilde{\\max}_\\beta(x) = \\tfrac{1}{\\beta}\\log\\sum_i e^{\\beta x_i},$$ recovering the exact $\\min/\\max$ as $\\beta \\to \\infty$. The resulting $\\widetilde{\\rho}_\\beta$ is everywhere $C^\\infty$ and the smoothing bias is $O(\\log n / \\beta)$. Their experiments parameterise $u_{1:H}$ either as a free vector (trajectory optimisation) or as a neural-network policy; no LLM or discrete vocabulary appears.

**STLCG++** (Hashemi et al. 2025, §IV) drops the temperature smoothing and instead introduces a *masking* scheme: temporal operators unroll the time axis into a 2-D matrix and apply $\\min / \\max$ over the unmasked cells in a single fused kernel. At a non-differentiable cell the framework picks one branch via the standard $\\min / \\max$ argument selector (Eq. 3 and the masked operations $\\mathcal{M} \\odot\_- \\mathcal{S}$ and $\\mathcal{M} \\odot\_+ \\mathcal{S}$ in §IV-B), and §IV-D shows that the $\\mathrm{logsumexp}$ family is recursively consistent in the smoothing limit. The output is a Clarke-subgradient-correct selection on the measure-zero non-differentiable set. Their experiments are again trajectory optimisation with continuous controls (single-integrator dynamics, gradient descent on $u$); no LLM or discrete vocabulary.

In both frameworks, the object that is differentiated is $J(u\_{1:H}) = \\rho \\circ \\mathrm{Sim}(x_0, u\_{1:H})$, with $u\_{1:H}$ a *continuous* decision variable.

## 2. Our gradient construction

`src/stl_seed/stl/evaluator.py` implements the same Donzé-Maler semantics but with *pure* JAX `jnp.min` / `jnp.max` at every And / Always / Eventually node (lines 239, 254, 269, 322; And and the temporal masks all reduce via `jnp.min` or `jnp.max` over the in-window mask). No softmin / softmax smoothing, and no STLCG++-style branch-selection masking beyond what `jax.grad` produces by default for `jnp.min` / `jnp.max`. Concretely JAX returns the `argmin` / `argmax` index's gradient exactly as STLCG++ would; on the measure-zero tie set it selects an arbitrary-but-consistent index.

Our sampler (`src/stl_seed/inference/gradient_guided.py`, `_compute_bias` at line 505) uses this gradient as follows. At decoding step $t$, given LLM logits $z_t \\in \\mathbb{R}^K$ over an action vocabulary $V \\in \\mathbb{R}^{K\\times m}$:

1. Form the LLM's *expected action* (line 525-526): $\\bar{u}_t = \\sum_{k=1}^K p\_{t,k} V_k$, where $p_t = \\mathrm{softmax}(z_t)$.
1. Build the partial-then-extrapolated control (lines 532-540): $\\hat{u}_t = (u_1,\\dots,u_{t-1}, \\bar{u}_t, u_{\\mathrm{def}}, \\dots, u\_{\\mathrm{def}})$.
1. Compute $\\rho$ and its gradient through the JIT-traced Diffrax solve plus the STL evaluator (line 545): $g_t = \\nabla\_{\\bar{u}\_t}, J(\\hat{u}\_t)$.
1. Project onto vocabulary deviations (lines 568-569): $b\_{t,k} = \\lambda \\langle V_k - \\bar{u}\_t,, g_t \\rangle$, then $a_t \\sim \\mathrm{softmax}((z_t + b_t)/T)$.

The simulator in step 3 is a vanilla Diffrax solve; the STL evaluator is the same one shared by `BestOfNSampler`; the gradient is taken by `jax.value_and_grad(rho_from_control, argnums=1)` (line 418).

## 3. Equivalence in the smooth interior

**Proposition 1 (gradient equivalence on the differentiable cell).** Let $u^\\star \\in U^H$ be a control at which every $\\min$ / $\\max$ node in the evaluator has a unique active argument (no ties), and assume $\\mathrm{Sim}(x_0,\\cdot)$ is differentiable at $u^\\star$ (which holds whenever the Diffrax adaptive-step solver does not change schedule across an infinitesimal perturbation of $u^\\star$). Then $$\\nabla\_{u^\\star}, J^{\\mathrm{stl\\text{-}seed}} ;=; \\lim\_{\\beta\\to\\infty} \\nabla\_{u^\\star}, J^{\\mathrm{STLCG}}_\\beta ;=; \\nabla_{u^\\star}, J^{\\mathrm{STLCG!+!+}}.$$

*Proof sketch.* Off the tie set, $\\jnp{min}$ and $\\jnp{max}$ agree pointwise with $\\widetilde{\\min}_\\infty$ and $\\widetilde{\\max}_\\infty$ and with STLCG++'s mask-then-reduce; their gradients agree by elementary calculus (the active-argument selector is constant in a neighbourhood, so the chain rule reduces to differentiating the active branch). The remaining $\\rho \\circ \\mathrm{Sim}$ composition is the same Donzé-Maler robustness in all three frameworks. $\\square$

The proposition says the headline novelty of `stl-seed` is *not* in the gradient computation. On the open dense set where ties do not occur, our $\\nabla \\rho$ is bit-identical to what STLCG++ computes and to the $\\beta\\to\\infty$ limit of what STLCG computes.

## 4. The discrete-vocabulary STE bridge (Proposition 2)

STLCG and STLCG++ assume $u\_{1:H}$ is a continuous decision variable. In LLM decoding the control at step $t$ is a *categorical* random variable $a_t \\in {1,\\dots,K}$, materialised as $u_t = V\_{a_t}$. There is no gradient through the categorical sampling step, so STLCG / STLCG++ as stated do not apply.

The expected-action construction $\\bar{u}_t = \\sum_k p_{t,k} V_k$ embeds the categorical distribution into $U \\subset \\mathbb{R}^m$ via its mean. Differentiating $\\rho(\\hat{u}_t)$ in $z_t$ via the chain rule gives $$\\frac{\\partial \\rho}{\\partial z_{t,k}} ;=; p\_{t,k},(V_k - \\bar{u}_t)^\\top g_t.$$ Our bias $b_{t,k} = \\lambda(V_k - \\bar{u}_t)^\\top g_t$ drops the $p_{t,k}$ factor.

**Proposition 2 (STE bridge).** Let $\\pi\_\\lambda(k\\mid z_t,g_t) = \\mathrm{softmax}(z_t + \\lambda(V - \\bar{u}\_t)g_t)\_k$ be the bias-augmented sampling distribution.

(i) At $\\lambda = 0$, $\\pi_0 = \\mathrm{softmax}(z_t) = p_t$ exactly (no bias is added), so vanilla LLM decoding is recovered (verified numerically in `tests/test_inference.py::test_gradient_guided_zero_lambda_matches_standard`).

(ii) The bias $b\_{t,k}$ is the first-order Taylor expansion of $J(\\hat{u}\_t \\text{ with } u_t = V_k) - J(\\hat{u}\_t \\text{ with } u_t = \\bar{u}\_t)$ about $u_t = \\bar{u}\_t$: $$J(\\hat{u}\_t \\mid u_t = V_k) - J(\\hat{u}\_t \\mid u_t = \\bar{u}\_t) ;=; (V_k - \\bar{u}\_t)^\\top g_t ;+; O\\bigl(|V_k-\\bar{u}\_t|^2\\bigr).$$ The bias is therefore the linearised one-step improvement in $\\rho$ from moving to vocabulary item $V_k$.

(iii) Dropping the $p\_{t,k}$ factor is a Bengio-Léonard-Courville straight-through estimator (arXiv:1308.3432) on the projection step. The unbiased chain-rule gradient $p\_{t,k}(V_k - \\bar{u}\_t)^\\top g_t$ damps guidance precisely on low-probability tokens, which is the regime in which guidance is most useful; STE removes that damping.

*Proof sketch.* (i) is direct from $b_t = \\lambda \\cdot 0 = 0$ at $\\lambda = 0$. (ii) is Taylor's theorem applied to $J$ in the $u_t$ slot of the partial control, with remainder bounded by $\\frac{1}{2}|V_k - \\bar{u}_t|^2 \\cdot \\sup_{\\xi}|\\nabla^2_u J(\\xi)|$ (uniform over the box by A1-A2 of `paper/landscape_theorem.md`). (iii) follows from the standard STE definition: replacing the discrete-sampling Jacobian by an identity on the relaxed mean. $\\square$

This is the first novelty axis. STLCG / STLCG++ have no discrete-decoding step in their pipeline, so neither framework states or needs an STE bridge.

## 5. Per-step inference-time MPC extension (Proposition 3)

STLCG (§V) and STLCG++ (§V) optimise $u\_{1:H}$ by gradient descent on the *entire* sequence at once, treating the trajectory as a fixed-horizon batch. Our setting differs in two ways: (a) control is committed *one step at a time*, autoregressively, with the LLM proposing logits $z_t$ given the history; (b) the gradient is computed at decoding step $t$ on a partial-then-extrapolated control with future actions $u\_{t+1},\\dots,u_H = u\_{\\mathrm{def}}$.

**Proposition 3 (model-predictive extension).** The gradient-guided sampler is a model-predictive control variant of the STLCG / STLCG++ gradient: at each step $t$ it computes $$g_t = \\nabla\_{\\bar{u}_t}, J\\bigl(u_{1:t-1},, \\bar{u}_t,, u_{\\mathrm{def}}^{\\otimes(H-t)}\\bigr),$$ which is the STLCG / STLCG++ gradient of a *constant-tail extrapolated trajectory*. As $t \\to H$ the extrapolation horizon vanishes and the gradient converges to the trajectory-optimisation gradient on the fully-committed prefix.

*Proof sketch.* By construction the inner $J$ is the same composition $\\rho \\circ \\mathrm{Sim}$ that STLCG / STLCG++ differentiate. The substitution $u\_{t+1:H} = u\_{\\mathrm{def}}$ is a first-order MPC short-horizon approximation; in the language of receding-horizon control (Mayne, Rawlings, Diehl 2017, *Model Predictive Control*) it is the default-action terminal heuristic. The convergence statement is a direct consequence of the chain rule: at $t = H$ the extrapolation tail is empty and $g_H = \\nabla\_{u_H} J(u\_{1:H})$ exactly. $\\square$

This is the second novelty axis. STLCG / STLCG++ as written cannot run per-step; they require the full $u\_{1:H}$ to be available before any gradient can be taken. Our pipeline takes the gradient of a one-shot extrapolated rollout *at every decoding step*, which is what makes the gradient usable as an inference-time logit bias.

## 6. The $\\lambda \\to \\infty$ equivalence

**Proposition 4 (greedy limit).** Suppose the LLM prior is uniform ($z_t \\equiv 0$) and the sampling temperature is fixed to $T = 1$. Then $$\\lim\_{\\lambda \\to \\infty}; \\pi\_\\lambda(k \\mid z_t,g_t) ;=; \\delta\_{k = k^\\star_t}, \\qquad k^\\star_t = \\arg\\max_k \\langle V_k - \\bar{u}\_t,, g_t\\rangle = \\arg\\max_k \\langle V_k,, g_t\\rangle.$$ That is, gradient-guided sampling at infinite gain reduces to the per-step argmax over $V_k$ on the linearised STLCG / STLCG++ objective at the partial-then-extrapolated control.

*Proof sketch.* Under a uniform prior, $b\_{t,k}$ becomes the only term in the biased logit. For any pair $k \\ne k'$, $\\Pr\[a_t = k\]/\\Pr\[a_t = k'\] = \\exp(\\lambda(b\_{t,k} - b\_{t,k'}))$. As $\\lambda \\to \\infty$ the ratio diverges to $+\\infty$ if $b\_{t,k} > b\_{t,k'}$ and to $0$ otherwise, so the softmax converges to the indicator of $\\arg\\max_k b\_{t,k}$. The $\\bar{u}\_t$ term is $k$-independent and drops out of the argmax, leaving $\\arg\\max_k \\langle V_k, g_t\\rangle$. $\\square$

In words: the $\\lambda \\to \\infty$ limit *is* a one-step gradient ascent on the STLCG / STLCG++ objective, projected onto the vocabulary. The two novelties (STE bridge, MPC extension) reduce, in this corner of parameter space, to the per-step linear-objective discretisation of the underlying continuous-control gradient that STLCG / STLCG++ compute.

## 7. Honesty: STLCG++ is the more rigorous boundary handler

The evaluator in `src/stl_seed/stl/evaluator.py` uses pure $\\jnp{min}$ / $\\jnp{max}$ with default JAX autodiff. On the measure-zero non-differentiable set this returns *some* element of the Clarke subdifferential (the gradient of whichever branch JAX's tie-breaking selects). It is consistent in the sense that two evaluator calls with the same inputs return the same subgradient, but it is not necessarily the *best* subgradient in the Clarke sense; STLCG++'s explicit masking formulation (§IV-B, Eq. 3) makes the branch selection auditable and provably one-sided correct.

Of the three approaches:

1. **STLCG soft-$\\beta$** is everywhere $C^\\infty$; the cost is an $O(\\log n / \\beta)$ smoothing bias that propagates into the gradient. For STL satisfaction (sign of $\\rho$) the bias is benign; for tight margin tracking it is not.
1. **STLCG++ masking** is exact $\\min / \\max$ with explicit Clarke subgradient selection; no smoothing bias, well-defined behaviour at ties.
1. **`stl-seed` pure JAX** is exact $\\min / \\max$ with implicit JAX tie-breaking; correct on the differentiable cell, *probably* correct on the tie set, but the documentation does not formalise the choice.

By that ranking, STLCG++ is the most rigorous; we are second; STLCG is third (smoothing bias). This is a future-work improvement direction, not a published novelty for `stl-seed`.

## 8. Future work: STLCG++ as a drop-in

The evaluator interface is intentionally narrow: `compile_spec(spec) -> Callable[[states, times], scalar_rho]` (`src/stl_seed/stl/evaluator.py`, line 372). Any implementation conforming to that signature is interchangeable with the current pure-JAX one. STLCG++ ships a PyTorch + JAX implementation (https://uw-ctrl.github.io/stlcg/); a wrapper that converts our `Node` AST into STLCG++'s formula objects, evaluates on `(states, times)`, and returns a scalar would slot into `STLGradientGuidedSampler` with no other changes. The straight-through bridge (§4) and the MPC extension (§5) are agnostic to which evaluator provides $\\nabla \\rho$.

Concretely the migration path is:

- Add `stl_seed.stl.evaluator_stlcgpp.compile_spec_stlcgpp` returning a `(states, times) -> scalar_rho` closure backed by STLCG++.
- Route via a `evaluator: Literal["donze_maler", "stlcgpp"]` argument on `STLGradientGuidedSampler.__init__`.
- Re-run `tests/test_gradient_guided_zero_lambda_matches_standard` to confirm the $\\lambda = 0$ collapse still holds (it is evaluator-agnostic by construction).

The equivalence theorems in §3-§6 carry over verbatim: Proposition 1 says the gradient values agree on the differentiable cell, so swapping the evaluator does not change the smooth-regime behaviour. The improvement is on the tie set, which is empirically rare for the bio_ode and glucose-insulin specs (no NaN / Inf gradient events fired by `fallback_on_grad_failure` in the unified-comparison sweep, beyond the documented one-per-rollout boundary saturation in `paper/inference_method.md`).

## 9. Summary of novelty boundaries

| Component                               |     STLCG     | STLCG++ | `stl-seed` |
| --------------------------------------- | :-----------: | :-----: | :--------: |
| Donzé-Maler $\\rho$ semantics           |      yes      |   yes   |    yes     |
| Backprop $\\nabla_u \\rho$ through ODE  |      yes      |   yes   |    yes     |
| $\\min$/$\\max$ smoothing               | soft-$\\beta$ | masking |  pure JAX  |
| Continuous trajectory optimisation      |      yes      |   yes   |    n/a     |
| Discrete vocabulary $V \\subset U$      |      no       |   no    |    yes     |
| Categorical sampling $a_t$ + STE bridge |      no       |   no    |    yes     |
| Per-step inference-time MPC application |      no       |   no    |    yes     |
| LLM logit-bias formulation              |      no       |   no    |    yes     |

The two rows that are *only* checked in the rightmost column are the two novelty axes. Everything above that line is shared infrastructure.

## References

- Bengio, Léonard, Courville. "Estimating or propagating gradients through stochastic neurons." arXiv:1308.3432, 2013.
- Donzé, Maler. "Robust satisfaction of temporal logic over real-valued signals." FORMATS 2010, DOI 10.1007/978-3-642-15297-9_9.
- Hashemi, Suh, Ren, Hsieh, Pavone. "STLCG++: a masking approach for differentiable signal temporal logic specifications." arXiv:2501.04194, 2025. (§IV-B masking, Eq. 3 quantitative semantics, §IV-D logsumexp consistency.)
- Leung, Aréchiga, Pavone. "Back-propagation through Signal Temporal Logic specifications." arXiv:2008.00097, 2020. (§III soft-$\\beta$ approximation, §V trajectory-optimisation experiments.)
- Mayne, Rawlings, Diehl. *Model Predictive Control: Theory, Computation, and Design.* Nob Hill, 2017.

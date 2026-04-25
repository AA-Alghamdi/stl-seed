# Gradient-guided sampling with continuous STL robustness

*Technical specification for the inference-time decoding contribution of `stl-seed`. Written 2026-04-24.*

## 1. Problem and notation

Fix a control task with state space $\mathcal{S} \subseteq \mathbb{R}^n$, action space $\mathcal{A} \subseteq \mathbb{R}^m$, simulator $\mathrm{Sim} : \mathcal{S} \times \mathcal{A}^H \to (\mathcal{S}^T \times [0, T_{\max}]^T)$ that maps an initial state $x_0$ and a piecewise-constant control schedule $u_{1:H} \in \mathcal{A}^H$ to a sampled trajectory $(\tau, t)$ over $T$ save points, and STL specification $\varphi$ with Donzé-Maler space-robustness $\rho(\varphi, \tau, t) \in \mathbb{R}$ (Donzé & Maler 2010, FORMATS, DOI 10.1007/978-3-642-15297-9_9).

An LLM control agent emits $u_{1:H}$ autoregressively, conditioning on $x_0$ and the action history. Standard inference-time recipes are:

1. **Vanilla decoding.** Sample one $u_{1:H}$ from the LLM's autoregressive distribution; report $\rho$ post-hoc.
2. **Best-of-$N$ (BoN).** Generate $N$ candidates; return the first with $\rho > 0$ (binary), or the argmax of $\rho$ (continuous).

Both ignore that $\rho$ is a continuous, almost-everywhere differentiable function of the trajectory, hence (by chain rule through the simulator) a function of $u_{1:H}$. We propose using $\nabla_{u_t} \rho$ at decoding time, in the spirit of classifier guidance for diffusion models (Dhariwal & Nichol 2021, arXiv:2105.05233) and Diffusion Posterior Sampling (Chung et al. 2023, arXiv:2209.14687).

## 2. Algorithm: STL-robustness gradient-guided sampling

Let $V \in \mathbb{R}^{K \times m}$ be a finite action vocabulary; the LLM at step $t$ emits logits $z_t \in \mathbb{R}^K$ over $V$. Let $\bar{u}$ denote the LLM's *preferred mean action* under the softmax distribution $p_t = \operatorname{softmax}(z_t)$:

$$
\bar{u}_t \;=\; \sum_{k=1}^{K} p_{t,k} V_k \;\in\; \mathcal{A}.
$$

Define the *partial-then-extrapolated* control sequence

$$
\hat{u}_{1:H}(\bar{u}_t) \;=\; (u_1^{\star}, \ldots, u_{t-1}^{\star}, \bar{u}_t, u_{\mathrm{def}}, \ldots, u_{\mathrm{def}}),
$$

where $u_h^{\star}$ for $h<t$ are the actions already committed in this rollout and $u_{\mathrm{def}}$ is a fixed default action (the L^∞-most-neutral choice, e.g. the center of the action box).

Let $\rho(\bar{u}_t) = \rho(\varphi, \mathrm{Sim}(x_0, \hat{u}_{1:H}(\bar{u}_t)))$ denote the resulting full-horizon robustness as a function of $\bar{u}_t$. We compute, by JAX autodiff through the JIT-traced simulator and the from-scratch STL evaluator (`stl_seed.stl.evaluator`),

$$
g_t \;=\; \nabla_{\bar{u}} \rho(\bar{u})\big|_{\bar{u} = \bar{u}_t} \;\in\; \mathbb{R}^m.
$$

We then form a logit bias by linearising the improvement in $\rho$ when moving from $\bar{u}_t$ to vocabulary item $V_k$:

$$
b_{t,k} \;=\; \lambda \; \langle V_k - \bar{u}_t,\; g_t \rangle.
$$

Finally we sample

$$
a_t \;\sim\; \operatorname{softmax}\bigl((z_t + b_t)/\tau\bigr), \qquad u_t^{\star} \;=\; V_{a_t},
$$

with sampling temperature $\tau \ge 0$ ($\tau = 0$ collapses to greedy argmax).

**Properties.**

* $\lambda = 0 \Rightarrow b_t = 0 \Rightarrow$ recovers vanilla LLM sampling exactly (modulo numerical tracer-vs-eager noise from the JAX autodiff path; verified by `tests/test_inference.py::test_gradient_guided_zero_lambda_matches_standard`).
* $\lambda \to \infty \Rightarrow$ selects the vocabulary item maximising $\langle V_k, g_t \rangle$, i.e. one greedy gradient-ascent step in the discrete vocabulary.
* $b_{t,k} \approx \lambda \cdot (\rho(V_k) - \rho(\bar{u}_t))$ to first order, so the bias is the linearised local-rho improvement of choosing $V_k$ over the LLM's mean choice. This matches Tweedie / score-matching intuition: we shift mass toward higher-$\rho$ regions in proportion to the score $g_t$.

### 2.1 Discrete-vs-continuous bridge (straight-through estimator)

The LLM emits a categorical distribution; the simulator and STL evaluator consume continuous actions. The bridge is the *expected-action* construction $\bar{u}_t = \sum_k p_{t,k} V_k$, which is differentiable in $z_t$. The gradient flows as

$$
\frac{\partial \rho}{\partial z_{t,k}} \;=\; p_{t,k}\,(V_k - \bar{u}_t)^{\top} g_t.
$$

We use $b_{t,k} = \lambda \langle V_k - \bar{u}_t, g_t \rangle$ rather than this exact derivative because the additional $p_{t,k}$ factor would re-weight the bias by the LLM's *current* probability, dampening the guidance precisely where the LLM is uncertain (the regime where guidance helps most). The chosen form is equivalent to a Bengio-style straight-through estimator (Bengio, Léonard & Courville 2013, arXiv:1308.3432) applied to the *vocabulary-projection step*.

## 3. Connection to prior work

| Method | Verifier signal | Selection time | Granularity |
|---|---|---|---|
| Vanilla LLM | none | — | per-token |
| BoN (binary) | $\rho > 0$ | post-hoc | per-trajectory |
| Continuous BoN | $\rho \in \mathbb{R}$ | post-hoc | per-trajectory |
| Classifier guidance (Dhariwal & Nichol 2021) | $\nabla_x \log p(y \mid x)$ | during sampling | per-step (image) |
| DPS (Chung et al. 2023) | $\nabla_x \log p(y \mid x_t)$ | during sampling | per-step (image) |
| LTLCrit / LogicGuard (Sun et al. 2025, arXiv:2507.03293) | LTL satisfaction | during sampling | per-token (vetoes) |
| STL-as-RL-reward (Aksaray et al. 2016) | $\rho \in \mathbb{R}$ | training-time | per-rollout |
| **This work** | $\nabla_{u_t} \rho \in \mathbb{R}^m$ | **during sampling** | **per-step** |

The technical novelty is the combination of (i) STL's continuous semantics, (ii) inference-time gradient guidance, and (iii) the straight-through bridge to a discrete LLM vocabulary. STLCG / STLCG++ (Leung et al. 2020 arXiv:2008.00097; Hashemi et al. 2025 arXiv:2501.04194) provide the differentiable STL infrastructure but were applied to trajectory optimisation and reward shaping, not LLM decoding. LogicGuard uses LTL as a *discrete* token vetoer and so cannot exploit the magnitude of $\rho$.

## 4. Computational cost

Per generation step $t$:

* **Forward**: 1 ODE solve on $\hat{u}_{1:H}$ (Diffrax JIT-traced).
* **Backward**: 1 reverse-mode pass through the ODE solve and the STL evaluator (Diffrax `RecursiveCheckpointAdjoint`).
* **Bias projection**: $K \cdot m$ FLOPs.

Total over the horizon: $H \cdot (1\,\text{fwd} + 1\,\text{bwd}) + H K m$ FLOPs.

Compare with continuous BoN at sample budget $N$: $N$ forward solves and $N$ rho evaluations, no backward. Setting $N \approx 2H$ matches the gradient-guided cost approximately (a backward pass costs ~1-2x a forward pass on Diffrax adjoint). The pre-registered fair-compute comparison is

* Gradient-guided sampler with $\lambda > 0$ and one rollout, vs.
* Continuous BoN with $N = 2H$ rollouts.

For the glucose-insulin task ($H = 12$), this means $N = 24$ for matched compute.

## 5. Pre-registered hypotheses

**H1 (existence of effect).** Fix task family `glucose_insulin`, spec `glucose_insulin.tir.easy`, action vocabulary $K=5$ uniform on $[0, 5]$ U/h, seed sweep $\{1000, \ldots, 1005\}$. Then mean $\rho$ under `STLGradientGuidedSampler(guidance_weight = 2.0, sampling_temperature = 0.5)` exceeds mean $\rho$ under `STLGradientGuidedSampler(guidance_weight = 0.0, sampling_temperature = 0.5)` (the matched-temperature lambda = 0 ablation) by at least 0.5 rho units. *Status: passing in `tests/test_inference.py::test_gradient_guided_improves_rho`. Smoke run produced mean rho 19.91 (guided) vs 1.58 (baseline) — a ~12x improvement and 5/6 paired wins.*

**H2 (matched-compute dominance).** On `glucose_insulin.tir.easy` and `bio_ode.repressilator.easy`, gradient-guided sampling at $\lambda = 2$ with one rollout per seed achieves higher *mean* $\rho$ than continuous BoN at $N = 2H$ rollouts per seed, paired across seeds, with bootstrap 95% CI on the difference excluding zero. *Pending Phase-2 sweep on RunPod.*

**H3 (saturation graceful degradation).** When the spec is fully satisfied at step $t < H$, the gradient $g_t$ vanishes (rho is locally insensitive to $u_t$); the sampler should then revert to LLM-prior sampling rather than producing pathological choices. *Verified observationally: per-step grad norms in the smoke run drop to zero after rho saturates at the max of 12.84.*

## 5.5 Hybrid sampler: gradient-guided + Best-of-$N$ selection

`src/stl_seed/inference/hybrid.py` implements `HybridGradientBoNSampler`, which composes the two inference-time scaling levers of (i) gradient guidance per-rollout and (ii) verifier-based selection across rollouts. For each of $n$ draws, the sampler runs `STLGradientGuidedSampler` on a sub-key derived via `jax.random.fold_in(key, draw_idx)` and scores the resulting trajectory by the same compiled spec used inside the inner sampler; it then returns the argmax-$\rho$ trajectory across the $n$ draws.

**Compute cost.** $n \cdot H \cdot (1\,\text{fwd} + 1\,\text{bwd})$ — one inner gradient-guided rollout per draw. The matched-compute baseline is $\text{ContinuousBoN}(n_{\text{bon}} = 2n)$, since a backward pass is approximately $1\text{–}2\times$ a forward pass on the Diffrax recursive-checkpoint adjoint.

**Pre-registered hypothesis.** At fixed compute budget,
$$
\text{mean }\rho_{\text{hybrid}(n)} \;\ge\; \text{mean }\rho_{\text{guided}} \;\ge\; \text{mean }\rho_{\text{cont-BoN}(2n)} \;\ge\; \text{mean }\rho_{\text{BoN}(n)} \;\ge\; \text{mean }\rho_{\text{standard}}.
$$
The tightest claim — hybrid $\ge$ pure guidance — is verified by `tests/test_inference.py::test_hybrid_beats_pure_guidance` on the harder `glucose_insulin.dawn.hard` spec (where rho hovers in $[-33, -19]$ with no saturation): hybrid($n=4$, $\lambda = 2$) achieves mean $\rho = -28.31$ versus pure guidance at $-30.62$ over six seeds — a $+2.3$ rho-unit improvement with 5/6 hybrid $\ge$ guided per seed.

**Connection to test-time-compute scaling.** The hybrid sampler is the gradient-guided analogue of the Snell 2024 / Brown 2024 ("Large Language Monkeys", arXiv:2407.21787) repeated-sampling-with-verifier scaling recipe. Two changes from the literature recipe: each inner draw is itself information-efficient (uses $\nabla \rho$ as a per-step decoding signal, not vanilla LLM sampling), and the verifier is the *exact same* continuous STL signal — so there is no train-eval mismatch on the verifier side. The cross-task empirical analysis in `paper/cross_task_validation.md` shows the hybrid composes cleanly with gradient guidance on tasks where guidance has signal, and degenerates gracefully (to ContinuousBoN with a constant per-draw tax) on tasks where guidance does not — which is the right behaviour for a scaling lever.

## 6. Implementation status (2026-04-24)

* `src/stl_seed/inference/__init__.py`, `protocol.py`, `gradient_guided.py`, `baselines.py`, `hybrid.py`.
* `tests/test_inference.py` — 21 tests, 20 passing + 1 documented `xfail` for cross-task transfer (see `paper/cross_task_validation.md`).
* `stl-seed sample` CLI command supports sampler $\in$ {standard, bon, bon_continuous, gradient_guided, hybrid}.
* REDACTED firewall: clean (`scripts/REDACTED.sh` returns OK).
* Test-suite regression: 0 — all 421 tracked tests pass (6 skipped on Apple Silicon for CUDA-only paths, 2 xfailed for documented discretization / cross-task limitations).

## 7. Limitations and failure modes

* **Vocabulary granularity.** $K$ scales as $k^m$ with axis-granularity $k$ and action-dimension $m$. The repressilator ($m = 3$) at $k = 5$ gives $K = 125$, still tractable; richer action vocabularies are an open scaling question.
* **Gradient pathologies at the action-box boundary.** When $\bar{u}_t$ saturates against the simulator's clip, $\rho$ is non-differentiable in $\bar{u}_t$ along the binding direction. The implementation falls back to unbiased sampling on NaN/Inf gradients (`fallback_on_grad_failure = True` default), recording the event in diagnostics. Smoke runs show ~1 fallback per 12-step rollout on the bio_ode tasks.
* **Default-action extrapolation bias.** The gradient $\nabla_{\bar{u}} \rho$ is computed assuming all future actions equal $u_{\mathrm{def}}$. This is a known myopic-planning bias; a corrected version would average over future-action distributions (a roll-out tree), at cost of an extra factor in compute. Pre-registered as future work.
* **Bias scale calibration.** The bias $b_{t,k}$ has units of $\lambda \cdot \rho$, while logits $z_t$ have units of nats. A bias-norm-matching variant ($\lambda$ scheduled to keep $\|b\|_{\infty} \le \alpha \|z\|_{\infty}$) is the natural calibration; we adopted a fixed scalar $\lambda$ for simplicity in the current implementation.

## 8. References

* Aksaray et al. "Q-learning for robust satisfaction of signal temporal logic specifications." CDC 2016.
* Bengio, Léonard & Courville. "Estimating or propagating gradients through stochastic neurons for conditional computation." arXiv:1308.3432, 2013.
* Chung et al. "Diffusion Posterior Sampling for general noisy inverse problems." arXiv:2209.14687, 2023.
* Dhariwal & Nichol. "Diffusion Models Beat GANs on Image Synthesis." arXiv:2105.05233, 2021.
* Donzé & Maler. "Robust satisfaction of temporal logic over real-valued signals." FORMATS 2010, DOI 10.1007/978-3-642-15297-9_9.
* Hashemi et al. "STLCG++: a masking approach for differentiable signal temporal logic specification." arXiv:2501.04194, 2025.
* Leung et al. "Back-propagation through Signal Temporal Logic specifications: infrastructure for control synthesis and machine learning." arXiv:2008.00097, 2020.
* Sun et al. "LogicGuard: improving LLM agents with linear temporal logic." arXiv:2507.03293, 2025.

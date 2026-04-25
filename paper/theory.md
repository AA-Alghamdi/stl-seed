# Theoretical Foundation: stl-seed

*A formal-verification instantiation of soft-filtered SFT for scientific control with LLM agents.*

Author: Abdullah AlGhamdi
Date: 2026-04-24
Target venue: workshop submission concurrent with CMU MS-AIE matriculation (Aug 2026); primary audience REDACTED' group (MLD).

> **Note on task family naming (added 2026-04-24, post-implementation):**
> This document was drafted before the task families were fully implemented and uses placeholder names `{gene-toggle, predator-prey/lv}`. The shipped artifact uses **`{bio_ode, glucose_insulin}`** with bio_ode covering 3 subdomains (repressilator, toggle switch, MAPK cascade). All hypotheses, statistical models, and power analyses transfer directly — only the family labels differ. See [`paper/architecture.md`](architecture.md) and [`src/stl_seed/tasks/`](../src/stl_seed/tasks/) for the canonical task definitions, and [`paper/REDACTED.md`](REDACTED.md) for the actual STL specs. The empirical pilot ([`paper/power_analysis_empirical.md`](power_analysis_empirical.md)) confirmed the design remains adequately powered (MDE ≈ 0.024 vs claimed ≥ 0.08).

---

## 1. Problem statement

We study LLM-driven *scientific control*: an autoregressive language-model policy emits a discrete-time control sequence that drives a continuous-time dynamical system toward a behavior specified by a temporal-logic formula. Concretely, fix a state space X = ℝ^n and an action space U = ℝ^m. A controlled trajectory is a continuous map τ : [0, T] → X obtained by integrating an ODE

dx/dt = f(x(t), u(t); θ),  x(0) = x_0,

where θ ∈ Θ ⊂ ℝ^d collects (kinetic, mechanical, electrical) parameters that are *fixed* and supplied by literature priors, and u : [0, T] → U is a piecewise-constant control with H switching points u_{1:H} = (u_1, …, u_H). Concretely, u(t) = u_{⌈t·H/T⌉}. The per-step state is s_t := τ(t·T/H) ∈ ℝ^n. Existence and uniqueness of τ given (θ, u_{1:H}, x_0) follow from local Lipschitz continuity of f in x (Picard-Lindelöf; verified for the Hill-type, Lotka-Volterra-type, and gene-toggle dynamics used here), and τ ∈ C([0, T], ℝ^n) when f is continuous and u is bounded measurable.

The behavioral specification is an STL formula φ over signal predicates μ_i(x) > 0, built from the Boolean connectives and the temporal operators 𝒰 (until), 𝒢_I (always over interval I ⊆ [0, T]), 𝒻_I (eventually). The STL syntax and Boolean semantics are due to Maler and Nickovic [arXiv:cs/0408019; DOI:10.1007/978-3-540-30206-3_12]. The quantitative *robustness* semantics ρ(τ, φ) ∈ ℝ — positive iff τ ⊨ φ, with magnitude measuring the signed margin to the nearest violation — is due to Donzé and Maler [DOI:10.1007/978-3-642-15297-9_9, FORMATS 2010] and is recursively defined on the formula tree using min/max over the predicate margins. We use the standard space-robustness ρ throughout (not the time-robustness variant of Donzé et al.).

The agent is a frozen pretrained LLM g_M : Σ* → Σ* of size M ∈ {Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B} [Qwen3 technical report, arXiv:2505.09388 placeholder]. A decoder D maps a sampled token stream to u_{1:H}. The composition π_M(u_{1:H} | x_0, φ) := g_M ∘ D induces a stochastic policy. For each (x_0, φ) we draw N samples (the per-instance budget), simulate τ_i = Φ(θ_fixed, u_{1:H}^{(i)}; x_0), and score ρ_i = ρ(τ_i, φ). The verifier-density v ∈ {hard, quantile, continuous} controls how those scores are reduced into a training signal (Section 2). The eval objective is success rate at budget N, defined as Pr_{u ~ π_M}[ρ(τ, φ_eval) > 0] under best-of-N decoding.

The setting differs from standard RLHF in three ways. (i) The reward ρ(τ, φ) is *formal*: it is computed by a deterministic algorithm on the simulated trajectory, not learned from preferences. (ii) The reward is *non-Markovian* in the action sequence — STL's temporal operators look across the whole horizon. (iii) The policy is constrained to emit u in physical units that the simulator accepts, so the action space is *grounded*. These three properties make stl-seed a clean testbed for soft-filtered SFT recipes designed for verifiable rewards [SERA, Shen et al. 2026, arXiv:2601.20789].

## 2. Soft-filtered SFT formalism

We abstract SERA's recipe [arXiv:2601.20789] in a form that makes the choice of filter density v explicit. Let π_ref denote a reference *generation* distribution constructed as a heterogeneous mixture: π_ref = (1/3)·π_random + (1/3)·π_heuristic + (1/3)·π_LLM, where π_random samples u_{1:H} uniformly from the bounded action box, π_heuristic uses three hand-coded controllers (bang-bang, PID with literature gains, sinusoidal probing) selected uniformly, and π_LLM is the smallest model (Qwen3-0.6B) at temperature 1.0 with no in-context examples. The mixture is deliberate: each leg covers different regions of the trajectory manifold, and the heuristic leg ensures non-trivial coverage even when the small LLM produces degenerate outputs.

**Stage 1 — Sample.** Draw N_gen trajectories {(x_0^{(i)}, φ^{(i)}, u_{1:H}^{(i)}, τ_i)}_{i=1}^{N_gen} where (x_0^{(i)}, φ^{(i)}) is sampled from a problem distribution 𝒟_train and u_{1:H}^{(i)} ~ π_ref(· | x_0^{(i)}, φ^{(i)}).

**Stage 2 — Score.** Compute ρ_i = ρ(τ_i, φ^{(i)}) ∈ ℝ via the recursive Donzé-Maler evaluator [DOI:10.1007/978-3-642-15297-9_9]. The evaluator is exact in float64 modulo the floating-point epsilon of the predicate evaluations (≈ 1e-15 per min/max node; depth ≤ 12 in our specs gives total error ≤ 12 · 1e-15 ≈ 1.2e-14, far below all practical thresholds).

**Stage 3 — Filter.** Construct a training set D_v as a function of v:

- **hard:** D_hard = {(x_0^{(i)}, φ^{(i)}, u_{1:H}^{(i)}) : ρ_i > 0}. This is the SERA-baseline / RFT recipe [Yuan et al. 2023, arXiv:2308.01825] and discards every "almost-correct" trajectory. The empty-D_hard edge case (which is non-trivial when φ is hard) is handled by retaining the top-1 trajectory with a flag to monitor.
- **quantile:** D_quant = top ⌈0.25 · N_gen⌉ trajectories of {τ_i} ranked by ρ_i. This is uniform-weight SFT on the empirical top quartile.
- **continuous:** D_cont uses *all* N_gen trajectories with importance weights w_i = exp(ρ_i / β) / Σ_j exp(ρ_j / β). The temperature β is set per-batch to β = std({ρ_i}_{i=1}^{N_gen}); this scaling is unitless because it normalizes by the empirical spread (analogous to the standardization step in process-advantage value reweighting [Setlur et al., PAV, arXiv:2410.08146]; PAV motivates the std-rescaling because raw-scale advantages couple the softmax sharpness to the reward unit, which is brittle when ρ has units of concentration·time).

**Stage 4 — SFT.** For each v ∈ {hard, quantile, continuous} we minimize the weighted negative log-likelihood

L_v(θ) = - Σ_{i ∈ D_v} w_i · log p_θ( decode^{-1}(u_{1:H}^{(i)}) | encode(x_0^{(i)}, φ^{(i)}) ),

with w_i ≡ 1 for v ∈ {hard, quantile} and w_i = softmax(ρ_i / β)_i for v = continuous. The decode^{-1} map serializes u_{1:H}^{(i)} into the same tokenization scheme used at sampling. The encoder produces a system prompt containing (x_0, φ) in a fixed JSON schema; we use neither chain-of-thought distillation nor reasoning traces, since our concern is the soft-vs-hard *filter* axis and adding rationale tokens would confound it. Optimization is one-epoch AdamW (lr 1e-5, cosine decay, warmup 3%, batch 32 sequences, bf16) on the eight LoRA-decomposed projection matrices [Hu et al., arXiv:2106.09685] of each Qwen3 attention block, rank 16. LoRA is chosen over full-parameter SFT to keep the artifact reproducible on a single A100; full-parameter is documented as a future extension.

The continuous-weighted condition can be derived as the M-step of a one-iteration EM with reward-as-evidence [Dayan-Hinton 1997 reward-as-likelihood; rederived in Peng et al. AWR, arXiv:1910.00177]: treating ρ as a log-importance ratio and applying KL-regularized policy improvement, the optimal target distribution is π*(u | x, φ) ∝ π_ref(u | x, φ) · exp(ρ / β), and projecting back to the parametric family by KL minimization yields exactly the importance-weighted MLE above. This places "continuous" on the same theoretical footing as hard-filtered RFT — both are coordinate descents in policy space — with hard recovered as the β → 0+ limit (one-hot weight on the argmax-ρ trajectory per problem).

## 3. Pre-registered hypotheses

All hypotheses are registered before the canonical sweep. Falsification criteria are stated numerically.

**H1 (headline equivalence).** Across the 3×3×2 grid of {filter v} × {model size M} × {task family F ∈ {gene-toggle, predator-prey}}, the eval success rate satisfies

|p̂_{quant, M, F} − p̂_{hard, M, F}| ≤ 0.05  AND  |p̂_{cont, M, F} − p̂_{hard, M, F}| ≤ 0.05,

simultaneously for all 18 cells, evaluated on the same held-out instance set. This is an *equivalence* claim, so the formal test is two one-sided tests (TOST [Schuirmann 1987, DOI:10.1007/BF01068419; Lakens 2017, DOI:10.1177/1948550617697177]) at α = 0.05 with Δ = 0.05. **H1_null:** at least one cell has |Δp̂| > 0.05 with TOST p > 0.05, i.e. equivalence fails. Rejecting H1 — finding a cell where soft significantly *beats* or *underperforms* hard — is itself publishable, especially in the latter direction since SERA's central claim was equivalence in coding [arXiv:2601.20789].

**H2 (size-monotone improvement).** For at least one task family F* ∈ {gene-toggle, predator-prey} and at least one v* ∈ {quantile, continuous}, the posterior-mean success rate is strictly monotone in M:

E[p_{v*, Qwen3-0.6B, F*}] < E[p_{v*, Qwen3-1.7B, F*}] < E[p_{v*, Qwen3-4B, F*}],

with each strict inequality holding with posterior probability ≥ 0.9 under the model of Section 4. **H2_null:** the monotonicity fails for both task families and both soft-filter conditions — a flat or non-monotone scaling curve, which would suggest the verifier signal is saturated below the smallest model's capacity ceiling.

**H3 (Goodhart decomposition).** Define R_proxy(τ) := σ(ρ(τ, φ_spec) / κ) ∈ (0, 1) for a fixed scale κ (the squashing avoids unbounded contribution from extreme ρ; κ is the median |ρ| from π_ref samples). Define R_verifier(τ) as the *evaluated* version of R_proxy from the same algorithm — i.e., the recursive Donzé-Maler evaluator on identical inputs, run a second time. Then R_proxy − R_verifier = 0 exactly in symbolic semantics; numerically we predict |R_proxy − R_verifier| ≤ 1e-6 per evaluation for all τ in the held-out set, dominated by float64 accumulation from the recursive min/max. This is the *verifier-fidelity* term. The *spec-completeness* term R_gold − R_spec is measured by introducing a stricter held-out spec φ_gold (constructed by tightening every numerical threshold in φ_spec by 10% and adding two unobserved-in-training conjuncts; see §6). **H3_null:** verifier-fidelity term exceeds 1e-6 (would indicate an evaluator bug), or the spec-completeness term cannot be empirically separated from learned-critic baseline noise (would deflate the central novelty claim).

The three hypotheses are jointly registered in `paper/preregistration.md` (separate file) with frozen task-family definitions, frozen φ_spec/φ_gold, and the random-seed schedule for held-out splits.

## 4. Statistical analysis plan

We model trial-level outcomes hierarchically in NumPyro [arXiv:1912.11554]. Index trials by (m, v, f, i, s, N), where m ∈ {Q06, Q17, Q40} indexes model size, v ∈ {hard, quant, cont}, f ∈ {toggle, lv} indexes task family, i ∈ {1, …, n_f} indexes instances within family (n_toggle = n_lv = 25), s ∈ {1, …, 5} indexes seed, and N ∈ {1, 2, 4, 8, 16, 32, 64, 128} indexes the best-of-N budget. Y_{m,v,f,i,s,N} ∈ {0, 1} is the success indicator: 1 iff at least one of the N samples drawn under (m, v) for instance (f, i) on seed s achieves ρ(τ, φ_eval) > 0.

The likelihood is Bernoulli with success probability following a saturating power-law in N:

p_{m,v,f,i}(N) = A_{m,v,f,i} · (1 − N^{-b_{m,v,f,i}}),    N ≥ 1,

where A ∈ (0, 1) is the asymptotic ceiling (probability that *some* successful u_{1:H} exists in the support of π_{m,v} for instance (f, i)) and b > 0 is the rate at which best-of-N approaches that ceiling. This functional form is a one-parameter generalization of the BoN coverage curve in [Brown et al. 2024, arXiv:2407.21787] and is tractable for posterior inference because logit A and log b are unconstrained.

**Hierarchical structure.** Let

logit A_{m,v,f,i} = μ_A + α^A_m + φ^A_f + δ^A_v · 𝟙{v ≠ hard} + γ^A_{mf} + ε^A_{mvfi},
log b_{m,v,f,i}  = μ_b + α^b_m + φ^b_f + δ^b_v · 𝟙{v ≠ hard} + γ^b_{mf} + ε^b_{mvfi}.

The δ^· effects encode the soft-vs-hard contrast for v ∈ {quant, cont} and are the parameters of interest for H1. We use sum-to-zero parametrizations for α, φ, γ to avoid the standard intercept-confounding identifiability trap.

**Priors.**
- μ_A ~ Normal(0, 1²); μ_b ~ Normal(0, 1²).
- δ^A_v, δ^b_v ~ Normal(0, 1²) for v ∈ {quant, cont} (weakly informative; spans roughly ±25 percentage points on the probability scale at the median).
- α^·_m, φ^·_f, γ^·_{mf} ~ Normal(0, τ²_·) with τ_· ~ HalfNormal(0, 0.5²) (random-effects variance; HalfNormal upper-tails at ≈ 1).
- ε^·_{mvfi} ~ Normal(0, σ²_·) with σ_· ~ HalfNormal(0, 0.3²) (instance-by-condition idiosyncratic variation; smaller because instance variation is partially absorbed by φ^·_f).

The HalfNormal tail choices follow Gelman's rule of thumb [Gelman 2006, DOI:10.1214/06-BA117A] for variance components in hierarchical logistic models: scale the prior so the implied 95th percentile equals roughly the largest plausible variance rather than the smallest.

**Reporting.** The primary inferential summary is the joint posterior P(δ^A_v > 0 AND δ^b_v > 0) for each v ∈ {quant, cont}. For H1 (equivalence), we report TOST decisions on the marginal posterior of the implied probability gap (p_{v,m,f}(N=128) − p_{hard,m,f}(N=128)) at Δ = 0.05. Per-cell posterior means with 89% credible intervals (per Kruschke [DOI:10.1037/a0029146]) are exploratory.

**MCMC config.** NUTS, 4 chains, 2000 warmup + 2000 draws each, target_accept = 0.9, max_tree_depth = 10. Convergence: R̂ < 1.01 and bulk ESS > 400 for every primary parameter (μ_·, δ^·_v) and every τ_·, σ_·; for the leaf ε's we relax to ESS > 100. Posterior predictive checks via held-out seed 6 (drawn after fitting on seeds 1-5 only) at the eight BoN budgets, with a rank-histogram diagnostic.

## 5. Power analysis

The locked design has 3 sizes × 3 filters × 2 task families × 25 instances/family × 5 seeds × 8 BoN budgets = 36,000 trials. Crucially, the BoN budgets reuse samples — N_max = 128 per (m, v, f, i, s) cell, and we compute success at N ∈ {1, 2, …, 128} via a single read of the ρ-sorted stack. This means the 8 budget points are *not* independent within a cell; their correlation is structurally near-1. We therefore conduct power analysis at the level of the (m, v, f, i, s) cell, with N treated as a curve covariate.

**Effective sample size.** With n_seeds = 5 trials per (m, v, f, i) configuration and a within-task ICC of ρ_ICC ≈ 0.4 (estimated from a pilot of 30 trajectories per cell, observing ICC = 0.39 ± 0.06), the design effect is 1 + (5 − 1) · 0.4 = 2.6, so n_eff per (m, v, f) configuration is 25 · 5 / 2.6 ≈ 48. Across the 18 (m, v, f) cells, total n_eff ≈ 864.

**Fisher information.** At the prior-median (A = 0.6, b = 0.25), a single observation Y_N ~ Bernoulli(p(N)) has Fisher information I_N(A, b) = (∂p/∂(A, b))ᵀ (∂p/∂(A, b)) / [p(1 − p)]. Working through:
∂p/∂A = 1 − N^{-b};
∂p/∂b = A · N^{-b} · ln N.
At N = 128, ln N ≈ 4.85, N^{-b} = 128^{-0.25} ≈ 0.30. So ∂p/∂A ≈ 0.70, ∂p/∂b ≈ 0.60 · 0.30 · 4.85 ≈ 0.87. p ≈ 0.42, p(1 − p) ≈ 0.244. The diagonal Fisher entries at N = 128 are I_AA ≈ 0.49 / 0.244 ≈ 2.0 and I_bb ≈ 0.76 / 0.244 ≈ 3.1 per observation, with cross-term I_Ab ≈ 0.70 · 0.87 / 0.244 ≈ 2.5.

Aggregating across the 8 BoN budgets per cell with the structural correlation r = 0.7 (estimated from the same pilot), the effective Fisher per cell is [I_AA, I_Ab; I_Ab, I_bb] · 8 / (1 + 7 · 0.7) = [I_AA, I_Ab; I_Ab, I_bb] · 1.38. With cell-level pooled n_eff ≈ 48 (from above), the Fisher matrix per (m, v, f) configuration is approximately 48 · 1.38 · I = 66.2 · I.

Inverting and reading off SE(δ^A) — the contrast between soft and hard within a (m, f) cell, paired across seed-instance pairs — gives SE(δ^A_logit) ≈ √(2 / (66.2 · 2.0)) ≈ 0.123 on the logit scale. On the probability scale at p = 0.42, this is roughly 0.123 · p(1 − p) ≈ 0.030.

**MDE.** For a one-sided test at α = 0.05 and power 0.8, MDE ≈ 2.49 · SE = 0.075 on the probability scale. For TOST at Δ = 0.05 and equivalence power 0.8, the requirement is SE ≤ Δ / (z_{1−α} + z_{1−β/2}) ≈ 0.05 / (1.645 + 1.282) ≈ 0.017. Our SE ≈ 0.030 *exceeds* this threshold, so the TOST equivalence test is underpowered at the per-cell level.

**Borrowing strength.** The hierarchical model partially pools across the 18 (m, v, f) cells via the random-effects variance prior. With the sum-to-zero parametrization and τ_· ~ HalfNormal(0, 0.5²), the effective n for the global δ^A contrast (averaging across m, f) is approximately 18 × per-cell n_eff = 864, reducing SE on the *global* δ^A by √18 ≈ 4.2, to ≈ 0.007. This is well below the 0.017 TOST threshold, so the *global* equivalence test is appropriately powered. *Per-cell* TOST is exploratory; the registered primary endpoint is the global contrast. This distinction is registered in the analysis plan to prevent post-hoc inflation.

**Equivalence-vs-difference framing.** H1 is an equivalence hypothesis: "soft is *as good as* hard within Δ = 0.05." TOST [DOI:10.1007/BF01068419] is the appropriate test, not the standard two-sided. The standard two-sided test is reported alongside as a *secondary* descriptive statistic, but does not drive the H1 decision. Confusing equivalence with non-significance is the modal error in soft-RFT comparison papers and we explicitly avoid it; Lakens [DOI:10.1177/1948550617697177] has an accessible treatment.

## 6. Goodhart decomposition theorem

Let R_gold(τ), R_spec(τ), R_proxy(τ), R_verifier(τ) ∈ ℝ denote, respectively: the (latent, oracle) reward for the underlying behavior of interest; the reward induced by the *spec* φ_spec we wrote down; the reward used at training time as the soft signal; and the reward computed by the algorithm we actually run. The identity

R_gold(τ) − R_proxy(τ) = [R_gold(τ) − R_spec(τ)] + [R_spec(τ) − R_verifier(τ)] + [R_verifier(τ) − R_proxy(τ)]

is a tautology. We collapse the last term to zero by defining R_proxy ≡ R_verifier (the proxy is whatever the algorithm computes), giving

**R_gold(τ) − R_proxy(τ) = [R_gold(τ) − R_spec(τ)]_spec-completeness + [R_spec(τ) − R_verifier(τ)]_verifier-fidelity.**

For STL with formal robustness ρ as the verifier, the second term is identically zero in symbolic semantics: R_spec(τ) is *defined* as ρ(τ, φ_spec) and R_verifier(τ) is what the Donzé-Maler evaluator [DOI:10.1007/978-3-642-15297-9_9] returns for the same (τ, φ_spec). The two are equal by construction modulo float64 round-off; the recursive min/max evaluator over a depth-12 STL formula accumulates at most 12 ulps ≈ 1.2 · 10^{-14} per evaluation, which we bound empirically at ≤ 1 · 10^{-6} after the σ-squashing in Section 3.

The substantive content is what this *exposes*. In RLHF and learned reward modeling, R_verifier is a learned approximation of R_spec (which is itself an approximation of R_gold), and the two error terms are *entangled* in the verifier's training residual. Gao, Schulman, and Hilton [arXiv:2210.10760] establish empirically that learned RMs exhibit a Goodhart-style overoptimization curve in best-of-N where the proxy reward continues to climb while the gold reward turns over; their decomposition cannot separate which fraction of the gap is verifier noise vs. spec misspecification because the only handle on R_spec is the learned RM itself.

STL ρ collapses the verifier-fidelity term to a numerical floor and turns the entire R_gold − R_proxy gap into the *spec-completeness* term. This is the auditable handle: a researcher inspecting an STL spec can in principle reason about whether a behavior φ_gold ⊃ φ_spec is missed, because both are written in the same logic. We operationalize this by constructing φ_gold from φ_spec via two augmentations: (a) tightening every numerical threshold by 10% (e.g., a 𝒢_{[0,T]} (x_1 > 0.5) becomes 𝒢_{[0,T]} (x_1 > 0.55)), and (b) adding two conjuncts that were *withheld* during training (a no-overshoot constraint 𝒢_{[T/2, T]} (x_1 < 1.1 · setpoint) and a control-effort cap 𝒢_{[0, T]} (|u| < u_max · 0.8)). On the held-out trajectory set, we report (R_gold − R_spec) and (R_spec − R_verifier) separately; the prediction is that the latter is at the 1e-6 floor while the former grows monotonically with model capacity (the better the policy fits φ_spec, the more headroom there is for φ_gold to differ).

The comparison baseline is a *learned-critic* proxy: a small reward model fine-tuned to regress ρ-from-rollouts, evaluated on the same held-out set. The prediction is that the learned critic exhibits *both* a non-zero verifier-fidelity term (regression error) and a spec-completeness term, and that the two cannot be disentangled from observation alone. STL exposes the spec-completeness term in isolation, which is the central theoretical contribution of the artifact.

## 7. Failure mode taxonomy

Five failure modes ranked by ex-ante probability with mitigations.

**FM1 — Trivial replication (~30%).** SERA's "soft works as well as hard" replicates without surprise because the gap was zero in this domain to begin with. Symptom: every (m, v, f) cell shows |Δp̂| < 0.01. The publishable claim collapses because we replicated SERA on a domain where the result was uninformative. *Mitigation:* the task family construction is calibrated against a pilot (n = 30 trajectories per cell) such that hard-filter success at N = 128 lies in [0.3, 0.7]. Cells where π_ref already saturates near 1.0 are excluded from the canonical sweep at the design stage, before any model training.

**FM2 — STL filtering doesn't beat random filtering (~25%).** Symptom: a control arm where D_v is constructed by random subsampling matches D_quant within MDE on every cell. Means ρ does not capture useful task-quality information for these specs (for example, because every accepted trajectory is so unstructured that the ρ score is dominated by an early-time predicate that any random control satisfies). *Mitigation:* before any SFT, validate that ρ_i correlates (Spearman r > 0.3) with a held-out *gold* score on the trajectory store. If r < 0.3 on a given task family, drop the family from the canonical sweep and document. This preflight is in `experiments/preflight_rho_correlation.py`.

**FM3 — Models too small (~20%).** Symptom: π_v eval success at floor (≤ 5%) for all v across all model sizes. The Qwen3-{0.6, 1.7, 4}B family is below the capacity threshold for credible scientific control on these tasks. *Mitigation:* difficulty calibration is on the trajectory-generation side, not just the eval side: the per-instance horizon H, the number of state dimensions exposed in the prompt, and the predicate density of φ are all swept in the pilot until a heuristic (PID with literature gains) hits ~50% success. If even Qwen3-4B with hard filtering hits floor on the resulting tasks, the artifact pivots to a "negative result on small-model scientific control with current prompt schemas" framing.

**FM4 — Backend numerical divergence (~15%).** Symptom: the canonical RunPod (bf16, A100, bnb 4-bit base) sweep produces materially different numbers than the local MLX (mps, fp16) pilot. Has bitten me before. Manifestations: (i) ρ_i differing in sign across backends due to ODE integrator choice; (ii) LoRA training dynamics differing because the bnb-quantized base introduces noise that MLX's fp16 base does not. *Mitigation:* a "preflight" step on RunPod replicates a tiny slice (1 model × 1 filter × 1 task family × 2 instances × 2 seeds × 4 BoN budgets) of the MLX pilot and checks that per-instance success rates match within 5 percentage points before launching the full canonical sweep. ODE integration uses Diffrax `Tsit5` with `rtol=1e-6, atol=1e-9` on both backends to remove integrator confounds. NaN/Inf events on Diffrax solves are replaced with zeros and counted in a separate field of the trial record (per the project rule in `~/CLAUDE.md`).

**FM5 — Compute overrun (~10%).** Symptom: the canonical sweep doesn't finish by day 14. *Mitigation:* hard checkpoint at day 8. If compute is on the critical path, the fallback is a 2 × 2 × 2 sub-design (Qwen3-{0.6B, 4B} × {hard, cont} × {toggle, lv}) that still touches all three hypotheses with reduced power, and the missing cells are annotated as "not run" rather than imputed. Imputation across model sizes is explicitly forbidden in the analysis plan.

## 8. REDACTED firewall

The author's prior work on the REDACTED paper (REDACTED et al.; physics-informed STL parameter synthesis) shares simulator infrastructure with this artifact but is mathematically a different optimization problem. The firewall is stated formally below.

**REDACTED problem.**
max_{θ ∈ Θ_phys} ρ( τ(θ; x_0), φ ),  where τ solves dx/dt = f(x, u_fixed; θ).

The optimization variable is the physical parameter vector θ ∈ ℝ^d, the control u is fixed (or absent — the system is autonomous), x_0 is given, and the loss landscape is shaped by the physics-prior penalties C1-C11 documented in `REDACTED.py` and the augmented-Lagrangian / CEGAR machinery in `REDACTED.py`.

**stl-seed problem.**
max_{u_{1:H} ∈ U^H} ρ( τ(θ_fixed, u_{1:H}; x_0), φ ),  where θ_fixed is drawn from BRENDA [DOI:10.1093/nar/gky1048] *fresh* for each task family and *not* taken from any REDACTED tuned solution.

The optimization variable is the control sequence u_{1:H} ∈ ℝ^{H · m}, the parameters θ are fixed and literature-sourced, and the loss is the unmodified ρ — there is no Augmented Lagrangian, no CEGAR loop, no residual-NN correction term, no C1-C11 physics-filter penalties, no conjunction-vs-implication spec-form ablation. The optimization is performed *implicitly* by a learned policy π_θ_LLM via best-of-N decoding rather than by a numerical solver.

**Firewall checklist.** No REDACTED artifact appears in stl-seed:
- (i) No file in `stl-seed/` imports from `~/REDACTED.py`, `~/REDACTED.py`, `~/REDACTED.py`. (Verified by grep at `paper/firewall_grep.txt`.)
- (ii) No θ value in `stl-seed/configs/` matches any REDACTED-tuned θ to within 5 significant figures. All θ values are pulled from BRENDA / SABIO-RK / KEGG with citation strings recorded in the config.
- (iii) The STL spec library (`stl-seed/specs/`) is independently authored: gene-toggle and predator-prey specs are written from textbook descriptions [Strogatz 2014 nonlinear dynamics, Gardner et al. 2000 toggle-switch, DOI:10.1038/35002131] without reference to REDACTED's repressilator/Hill specs.
- (iv) Reproducibility scripts (`REDACTED.py` etc.) are not invoked anywhere in the stl-seed pipeline.

The shared infrastructure (Diffrax integrator, Donzé-Maler ρ evaluator) is acknowledged in both papers as a software dependency, not a methodological overlap. The two papers are submittable to disjoint venues (REDACTED vs. NeurIPS workshop) without overlap-of-contribution concerns.

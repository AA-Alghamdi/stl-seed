# Theoretical Foundation

A formal-verification instantiation of soft-filtered SFT for scientific control with LLM agents.

Author: Abdullah AlGhamdi. Date: 2026-04-24. Target venue: workshop submission concurrent with CMU MS-AIE matriculation (Aug 2026); primary audience REDACTED' group (MLD).

## Problem statement

We study LLM-driven *scientific control*: an autoregressive language-model policy emits a discrete-time control sequence that drives a continuous-time dynamical system toward a behavior specified by a temporal-logic formula. Fix a state space $X = \\mathbb{R}^n$ and an action space $U = \\mathbb{R}^m$. A controlled trajectory is a continuous map $\\tau : \[0, T\] \\to X$ obtained by integrating

$$ \\dot{x} = f(x(t), u(t); \\theta), \\quad x(0) = x_0, $$

where $\\theta \\in \\Theta \\subset \\mathbb{R}^d$ collects (kinetic, mechanical, electrical) parameters that are *fixed* and supplied by literature priors, and $u : \[0, T\] \\to U$ is a piecewise-constant control with $H$ switching points $u\_{1:H}$, so $u(t) = u\_{\\lceil t \\cdot H/T \\rceil}$. The per-step state is $s_t := \\tau(t \\cdot T/H) \\in \\mathbb{R}^n$. Existence and uniqueness of $\\tau$ given $(\\theta, u\_{1:H}, x_0)$ follow from local Lipschitz continuity of $f$ in $x$ (Picard-Lindelöf; verified for the Hill-type, Bergman-minimal-model, and bio_ode (repressilator) dynamics used here), and $\\tau \\in C(\[0, T\], \\mathbb{R}^n)$ when $f$ is continuous and $u$ is bounded measurable.

The behavioral specification is an STL formula $\\varphi$ over signal predicates $\\mu_i(x) > 0$, built from Boolean connectives and the temporal operators $\\mathcal{U}$ (until), $\\mathcal{G}\_I$ (always over $I \\subseteq \[0, T\]$), $\\mathcal{F}\_I$ (eventually). The STL syntax and Boolean semantics are due to Maler and Nickovic \[arXiv:cs/0408019; DOI 10.1007/978-3-540-30206-3_12\]. The quantitative *space-robustness* $\\rho(\\tau, \\varphi) \\in \\mathbb{R}$ — positive iff $\\tau \\models \\varphi$, with magnitude measuring the signed margin to the nearest violation — is due to Donzé and Maler \[DOI 10.1007/978-3-642-15297-9_9, FORMATS 2010\] and is recursively defined on the formula tree using min/max over the predicate margins. We use the standard space-robustness throughout (not the time-robustness variant).

The agent is a frozen pretrained LLM $g_M : \\Sigma^\* \\to \\Sigma^\*$ of size $M \\in {$Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B$}$ \[Qwen3 technical report, arXiv:2505.09388\]. A decoder $D$ maps a sampled token stream to $u\_{1:H}$. The composition $\\pi_M(u\_{1:H} \\mid x_0, \\varphi) := g_M \\circ D$ is a stochastic policy. For each $(x_0, \\varphi)$ we draw $N$ samples (the per-instance budget), simulate $\\tau_i = \\Phi(\\theta\_\\text{fixed}, u\_{1:H}^{(i)}; x_0)$, and score $\\rho_i = \\rho(\\tau_i, \\varphi)$. The verifier-density $v \\in {$hard, quantile, continuous$}$ controls how those scores reduce into a training signal (next section). The eval objective is success rate at budget $N$: $\\Pr\_{u \\sim \\pi_M}\[\\rho(\\tau, \\varphi\_\\text{eval}) > 0\]$ under best-of-$N$ decoding.

The setting differs from standard RLHF in three ways. The reward $\\rho(\\tau, \\varphi)$ is *formal* — computed by a deterministic algorithm on the simulated trajectory, not learned from preferences. It is *non-Markovian* in the action sequence — STL's temporal operators look across the whole horizon. And the policy is constrained to emit $u$ in physical units that the simulator accepts, so the action space is *grounded*. These three properties make stl-seed a clean testbed for soft-filtered SFT recipes designed for verifiable rewards \[SERA, Shen et al. 2026, arXiv:2601.20789\].

## Soft-filtered SFT formalism

We abstract SERA's recipe in a form that makes the choice of filter density $v$ explicit. Let $\\pi\_\\text{ref}$ be a reference *generation* distribution constructed as a heterogeneous mixture: $\\pi\_\\text{ref} = (1/3)\\pi\_\\text{random} + (1/3)\\pi\_\\text{heuristic} + (1/3)\\pi\_\\text{LLM}$, where $\\pi\_\\text{random}$ samples $u\_{1:H}$ uniformly from the bounded action box, $\\pi\_\\text{heuristic}$ uses three hand-coded controllers (bang-bang, PID with literature gains, sinusoidal probing) selected uniformly, and $\\pi\_\\text{LLM}$ is the smallest model (Qwen3-0.6B) at temperature 1.0 with no in-context examples. The mixture is deliberate: each leg covers different regions of the trajectory manifold, and the heuristic leg ensures non-trivial coverage even when the small LLM produces degenerate outputs.

Stage 1, sample. Draw $N\_\\text{gen}$ trajectories ${(x_0^{(i)}, \\varphi^{(i)}, u\_{1:H}^{(i)}, \\tau_i)}$ with $(x_0^{(i)}, \\varphi^{(i)})$ from a problem distribution $\\mathcal{D}_\\text{train}$ and $u_{1:H}^{(i)} \\sim \\pi\_\\text{ref}$.

Stage 2, score. Compute $\\rho_i = \\rho(\\tau_i, \\varphi^{(i)})$ via the recursive Donzé-Maler evaluator. Exact in float64 modulo the floating-point epsilon of the predicate evaluations (~$10^{-15}$ per min/max node; depth ≤ 12 in our specs gives total error $\\le 12 \\cdot 10^{-15} \\approx 1.2 \\cdot 10^{-14}$, far below all practical thresholds).

Stage 3, filter. Construct a training set $D_v$ as a function of $v$. **Hard:** $D\_\\text{hard} = {(x_0^{(i)}, \\varphi^{(i)}, u\_{1:H}^{(i)}) : \\rho_i > 0}$ — the SERA-baseline / RFT recipe \[Yuan et al. 2023, arXiv:2308.01825\], which discards every "almost-correct" trajectory. The empty-$D\_\\text{hard}$ edge case (non-trivial when $\\varphi$ is hard) is handled by retaining the top-1 trajectory with a flag to monitor. **Quantile:** $D\_\\text{quant}$ = top $\\lceil 0.25 N\_\\text{gen} \\rceil$ trajectories of ${\\tau_i}$ ranked by $\\rho_i$, uniform-weight SFT on the empirical top quartile. **Continuous:** $D\_\\text{cont}$ uses *all* $N\_\\text{gen}$ trajectories with importance weights $w_i = \\exp(\\rho_i / \\beta) / \\sum_j \\exp(\\rho_j / \\beta)$. Temperature $\\beta$ is set per-batch to $\\beta = \\text{std}({\\rho_i})$; this rescaling is unitless because it normalizes by the empirical spread (analogous to PAV \[Setlur et al., arXiv:2410.08146\], which motivates the std-rescaling because raw-scale advantages couple the softmax sharpness to the reward unit, brittle when $\\rho$ has units of concentration·time).

Stage 4, SFT. For each $v$ minimize the weighted negative log-likelihood

$$ L_v(\\theta) = - \\sum\_{i \\in D_v} w_i \\cdot \\log p\_\\theta\\bigl( \\text{decode}^{-1}(u\_{1:H}^{(i)}) \\mid \\text{encode}(x_0^{(i)}, \\varphi^{(i)}) \\bigr), $$

with $w_i \\equiv 1$ for $v \\in {$hard, quantile$}$ and $w_i = \\text{softmax}(\\rho_i/\\beta)\_i$ for continuous. The encoder produces a system prompt containing $(x_0, \\varphi)$ in a fixed JSON schema; we use neither chain-of-thought distillation nor reasoning traces, since our concern is the soft-vs-hard *filter* axis and adding rationale tokens would confound it. Optimization is one-epoch AdamW (lr 1e-5, cosine decay, warmup 3%, batch 32 sequences, bf16) on the eight LoRA-decomposed projection matrices \[Hu et al., arXiv:2106.09685\] of each Qwen3 attention block, rank 16. LoRA over full-parameter SFT to keep the artifact reproducible on a single A100.

The continuous-weighted condition can be derived as the M-step of a one-iteration EM with reward-as-evidence \[Dayan-Hinton 1997; rederived in Peng et al. AWR, arXiv:1910.00177\]: treating $\\rho$ as a log-importance ratio and applying KL-regularized policy improvement, the optimal target is $\\pi^\*(u \\mid x, \\varphi) \\propto \\pi\_\\text{ref}(u \\mid x, \\varphi) \\cdot \\exp(\\rho/\\beta)$, and projecting back to the parametric family by KL minimization yields exactly the importance-weighted MLE above. This puts "continuous" on the same theoretical footing as hard-filtered RFT — both are coordinate descents in policy space — with hard recovered as the $\\beta \\to 0^+$ limit (one-hot weight on the argmax-$\\rho$ trajectory per problem).

## Pre-registered hypotheses

All registered before the canonical sweep; falsification criteria are numerical.

H1 (headline equivalence). Across the $3 \\times 3 \\times 2$ grid of ${$filter $v} \\times {$model size $M} \\times {$task family $F \\in {$bio_ode (repressilator), glucose_insulin$}}$,

$$ |\\hat p\_{\\text{quant}, M, F} - \\hat p\_{\\text{hard}, M, F}| \\le 0.05 ;\\text{ AND }; |\\hat p\_{\\text{cont}, M, F} - \\hat p\_{\\text{hard}, M, F}| \\le 0.05, $$

simultaneously for all 18 cells, on the same held-out instance set. This is an *equivalence* claim, so the test is two one-sided tests (TOST \[Schuirmann 1987, DOI 10.1007/BF01068419; Lakens 2017, DOI 10.1177/1948550617697177\]) at $\\alpha = 0.05$ with $\\Delta = 0.05$. H1_null: at least one cell has $|\\Delta \\hat p| > 0.05$ with TOST $p > 0.05$. Rejecting H1 — finding a cell where soft significantly *beats* or *underperforms* hard — is itself publishable, especially in the latter direction, since SERA's central claim was equivalence in coding.

H2 (size-monotone improvement). For at least one $F^\* \\in {$bio_ode (repressilator), glucose_insulin$}$ and at least one $v^\* \\in {$quantile, continuous$}$,

$$ \\mathbb{E}\[p\_{v^*, \\text{0.6B}, F^*}\] \< \\mathbb{E}\[p\_{v^*, \\text{1.7B}, F^*}\] \< \\mathbb{E}\[p\_{v^*, \\text{4B}, F^*}\], $$

with each strict inequality at posterior probability $\\ge 0.9$. H2_null: monotonicity fails for both task families and both soft-filter conditions (a flat or non-monotone scaling curve, suggesting the verifier signal is saturated below the smallest model's capacity ceiling).

H3 (Goodhart decomposition). Define $R\_\\text{proxy}(\\tau) := \\sigma(\\rho(\\tau, \\varphi\_\\text{spec})/\\kappa) \\in (0, 1)$ for a fixed scale $\\kappa$ (the squashing avoids unbounded contribution from extreme $\\rho$; $\\kappa$ is the median $|\\rho|$ from $\\pi\_\\text{ref}$ samples). Define $R\_\\text{verifier}(\\tau)$ as the *evaluated* version of $R\_\\text{proxy}$ from the same algorithm — the recursive Donzé-Maler evaluator on identical inputs, run a second time. Then $R\_\\text{proxy} - R\_\\text{verifier} = 0$ exactly in symbolic semantics; numerically we predict $|R\_\\text{proxy} - R\_\\text{verifier}| \\le 10^{-6}$ per evaluation for all $\\tau$ in the held-out set, dominated by float64 accumulation. The *spec-completeness* term $R\_\\text{gold} - R\_\\text{spec}$ is measured by introducing a stricter held-out spec $\\varphi\_\\text{gold}$ (tightening every numerical threshold by 10% and adding two unobserved-in-training conjuncts; see §6). H3_null: verifier-fidelity term exceeds $10^{-6}$ (would indicate an evaluator bug), or the spec-completeness term cannot be empirically separated from learned-critic baseline noise.

The three are jointly registered in `paper/preregistration.md` with frozen task-family definitions, frozen $\\varphi\_\\text{spec}/\\varphi\_\\text{gold}$, and the random-seed schedule for held-out splits.

## Statistical analysis plan

We model trial-level outcomes hierarchically in NumPyro \[arXiv:1912.11554\]. Index trials by $(m, v, f, i, s, N)$, where $m \\in {$Q06, Q17, Q40$}$, $v \\in {$hard, quant, cont$}$, $f \\in {$bio_ode, gluc$}$, $i \\in {1, \\ldots, 25}$ instances within family, $s \\in {1, \\ldots, 5}$ seeds, $N \\in {1, 2, 4, 8, 16, 32, 64, 128}$ BoN budgets. $Y\_{m,v,f,i,s,N} \\in {0, 1}$ is the success indicator: 1 iff at least one of the $N$ samples drawn under $(m, v)$ for instance $(f, i)$ on seed $s$ achieves $\\rho(\\tau, \\varphi\_\\text{eval}) > 0$.

Likelihood is Bernoulli with success probability following a saturating power-law in $N$:

$$ p\_{m,v,f,i}(N) = A\_{m,v,f,i} \\cdot (1 - N^{-b\_{m,v,f,i}}), \\quad N \\ge 1, $$

where $A \\in (0, 1)$ is the asymptotic ceiling (probability that *some* successful $u\_{1:H}$ exists in the support of $\\pi\_{m,v}$ for instance $(f, i)$) and $b > 0$ is the rate at which BoN approaches that ceiling. This functional form is a one-parameter generalization of the BoN coverage curve in Brown et al. 2024 \[arXiv:2407.21787\] and is tractable for posterior inference because logit $A$ and log $b$ are unconstrained.

Hierarchical structure:

$$ \\begin{aligned} \\text{logit } A\_{m,v,f,i} &= \\mu_A + \\alpha^A_m + \\varphi^A_f + \\delta^A_v \\cdot \\mathbb{1}{v \\neq \\text{hard}} + \\gamma^A\_{mf} + \\varepsilon^A\_{mvfi}, \\ \\log b\_{m,v,f,i} &= \\mu_b + \\alpha^b_m + \\varphi^b_f + \\delta^b_v \\cdot \\mathbb{1}{v \\neq \\text{hard}} + \\gamma^b\_{mf} + \\varepsilon^b\_{mvfi}. \\end{aligned} $$

The $\\delta^\\cdot$ effects encode the soft-vs-hard contrast for $v \\in {$quant, cont$}$ and are the parameters of interest for H1. We use sum-to-zero parametrizations for $\\alpha, \\varphi, \\gamma$ to avoid intercept-confounding identifiability issues.

Priors. $\\mu_A, \\mu_b \\sim \\mathcal{N}(0, 1^2)$; $\\delta^A_v, \\delta^b_v \\sim \\mathcal{N}(0, 1^2)$ for $v \\in {$quant, cont$}$ (weakly informative; spans roughly $\\pm 25$ percentage points on the probability scale at the median); $\\alpha^\\cdot_m, \\varphi^\\cdot_f, \\gamma^\\cdot\_{mf} \\sim \\mathcal{N}(0, \\tau^2\_\\cdot)$ with $\\tau\_\\cdot \\sim \\text{HalfNormal}(0, 0.5^2)$; $\\varepsilon^\\cdot\_{mvfi} \\sim \\mathcal{N}(0, \\sigma^2\_\\cdot)$ with $\\sigma\_\\cdot \\sim \\text{HalfNormal}(0, 0.3^2)$. The HalfNormal tail choices follow Gelman's rule of thumb \[Gelman 2006, DOI 10.1214/06-BA117A\] for variance components in hierarchical logistic models.

Reporting. Primary inferential summary is the joint posterior $P(\\delta^A_v > 0 \\text{ AND } \\delta^b_v > 0)$ for each $v$. For H1 (equivalence), TOST decisions on the marginal posterior of the implied probability gap $(p\_{v,m,f}(N=128) - p\_{\\text{hard},m,f}(N=128))$ at $\\Delta = 0.05$. Per-cell posterior means with 89% credible intervals (per Kruschke \[DOI 10.1037/a0029146\]) are exploratory. MCMC: NUTS, 4 chains, 2000 warmup + 2000 draws each, target_accept = 0.9, max_tree_depth = 10. Convergence: $\\hat R \< 1.01$ and bulk ESS > 400 for every primary parameter and every $\\tau\_\\cdot, \\sigma\_\\cdot$; for the leaf $\\varepsilon$'s we relax to ESS > 100. Posterior predictive checks via held-out seed 6 (drawn after fitting on seeds 1-5 only) at the eight BoN budgets, with a rank-histogram diagnostic.

## Power analysis

The locked design has $3 \\times 3 \\times 2 \\times 25 \\times 5 \\times 8 = 36{,}000$ trials. Crucially, the BoN budgets reuse samples — $N\_\\text{max} = 128$ per cell, and we compute success at each $N$ via a single read of the $\\rho$-sorted stack. Their correlation is structurally near-1, so power analysis runs at the cell level with $N$ as a curve covariate.

With $n\_\\text{seeds} = 5$ trials per $(m, v, f, i)$ configuration and a within-task ICC of $\\rho\_\\text{ICC} \\approx 0.4$ (estimated from a pilot of 30 trajectories per cell, observing ICC $= 0.39 \\pm 0.06$), the design effect is $1 + (5 - 1) \\cdot 0.4 = 2.6$, so $n\_\\text{eff}$ per $(m, v, f)$ configuration is $25 \\cdot 5 / 2.6 \\approx 48$. Across the 18 cells, total $n\_\\text{eff} \\approx 864$.

Fisher information at the prior-median $(A = 0.6, b = 0.25)$: a single observation $Y_N \\sim \\text{Bernoulli}(p(N))$ has $I_N(A, b) = (\\partial p / \\partial(A,b))^\\top (\\partial p / \\partial(A,b)) / \[p(1 - p)\]$. Working through: $\\partial p / \\partial A = 1 - N^{-b}$; $\\partial p / \\partial b = A \\cdot N^{-b} \\cdot \\ln N$. At $N = 128$, $\\ln N \\approx 4.85$, $N^{-b} = 128^{-0.25} \\approx 0.30$. So $\\partial p / \\partial A \\approx 0.70$, $\\partial p / \\partial b \\approx 0.60 \\cdot 0.30 \\cdot 4.85 \\approx 0.87$. $p \\approx 0.42$, $p(1-p) \\approx 0.244$. Diagonal Fisher entries at $N = 128$: $I\_{AA} \\approx 2.0$, $I\_{bb} \\approx 3.1$ per observation, cross-term $I\_{Ab} \\approx 2.5$.

Aggregating across the 8 BoN budgets per cell with structural correlation $r = 0.7$ (estimated from the same pilot), the effective Fisher per cell is the per-observation matrix scaled by $8 / (1 + 7 \\cdot 0.7) = 1.38$. With cell-level pooled $n\_\\text{eff} \\approx 48$, the Fisher matrix per $(m, v, f)$ is approximately $48 \\cdot 1.38 \\cdot I = 66.2 \\cdot I$. Inverting gives $\\text{SE}(\\delta^A\_\\text{logit}) \\approx \\sqrt{2/(66.2 \\cdot 2.0)} \\approx 0.123$ on the logit scale; on the probability scale at $p = 0.42$, roughly $0.030$.

MDE for a one-sided test at $\\alpha = 0.05$ and power 0.8: $\\approx 2.49 \\cdot \\text{SE} = 0.075$. For TOST at $\\Delta = 0.05$ and equivalence power 0.8: $\\text{SE} \\le \\Delta / (z\_{1-\\alpha} + z\_{1-\\beta/2}) \\approx 0.05 / 2.927 \\approx 0.017$. Our SE $\\approx 0.030$ *exceeds* this — TOST is underpowered at the per-cell level.

The hierarchical model partially pools across cells. With sum-to-zero parametrization and $\\tau\_\\cdot \\sim \\text{HalfNormal}(0, 0.5^2)$, the effective $n$ for the global $\\delta^A$ contrast (averaging across $m, f$) is approximately $18 \\times$ per-cell $n\_\\text{eff} = 864$, reducing SE on the *global* $\\delta^A$ by $\\sqrt{18} \\approx 4.2$, to $\\approx 0.007$. Well below the 0.017 TOST threshold. The *global* equivalence test is appropriately powered; *per-cell* TOST is exploratory. Registered to prevent post-hoc inflation.

H1 is an equivalence hypothesis: "soft is *as good as* hard within $\\Delta = 0.05$." TOST is the appropriate test, not the standard two-sided. Confusing equivalence with non-significance is the modal error in soft-RFT comparison papers; Lakens \[DOI 10.1177/1948550617697177\] has an accessible treatment.

## Goodhart decomposition theorem

Let $R\_\\text{gold}(\\tau), R\_\\text{spec}(\\tau), R\_\\text{proxy}(\\tau), R\_\\text{verifier}(\\tau) \\in \\mathbb{R}$ denote, respectively, the latent oracle reward; the reward induced by the *spec* $\\varphi\_\\text{spec}$ we wrote down; the reward used at training time as the soft signal; and the reward computed by the algorithm we run. The identity

$$ R\_\\text{gold}(\\tau) - R\_\\text{proxy}(\\tau) = \[R\_\\text{gold} - R\_\\text{spec}\] + \[R\_\\text{spec} - R\_\\text{verifier}\] + \[R\_\\text{verifier} - R\_\\text{proxy}\] $$

is a tautology. We collapse the last term by defining $R\_\\text{proxy} \\equiv R\_\\text{verifier}$ (the proxy is whatever the algorithm computes), giving

$$ R\_\\text{gold} - R\_\\text{proxy} = \\underbrace{\[R\_\\text{gold} - R\_\\text{spec}\]}_\\text{spec-completeness} + \\underbrace{\[R_\\text{spec} - R\_\\text{verifier}\]}\_\\text{verifier-fidelity}. $$

For STL with formal robustness $\\rho$ as the verifier, the second term is identically zero in symbolic semantics: $R\_\\text{spec}(\\tau)$ is *defined* as $\\rho(\\tau, \\varphi\_\\text{spec})$ and $R\_\\text{verifier}(\\tau)$ is what the Donzé-Maler evaluator returns for the same input. Equal by construction modulo float64 round-off; the recursive min/max evaluator over a depth-12 STL formula accumulates at most 12 ulps $\\approx 1.2 \\cdot 10^{-14}$ per evaluation, which we bound empirically at $\\le 10^{-6}$ after the $\\sigma$-squashing.

The substantive content is what this *exposes*. In RLHF and learned reward modeling, $R\_\\text{verifier}$ is a learned approximation of $R\_\\text{spec}$ (which is itself an approximation of $R\_\\text{gold}$), and the two error terms are *entangled* in the verifier's training residual. Gao, Schulman & Hilton \[arXiv:2210.10760\] establish empirically that learned RMs exhibit a Goodhart-style overoptimization curve in best-of-$N$ where the proxy reward continues to climb while the gold reward turns over; their decomposition cannot separate which fraction of the gap is verifier noise vs spec misspecification because the only handle on $R\_\\text{spec}$ is the learned RM itself.

STL $\\rho$ collapses verifier-fidelity to a numerical floor and turns the entire gap into the spec-completeness term. This is the auditable handle: a researcher inspecting an STL spec can in principle reason about whether a behavior $\\varphi\_\\text{gold} \\supset \\varphi\_\\text{spec}$ is missed, because both are written in the same logic. We operationalize this by constructing $\\varphi\_\\text{gold}$ from $\\varphi\_\\text{spec}$ via two augmentations: (a) tightening every numerical threshold by 10% (e.g., $\\mathcal{G}_{\[0,T\]}(x_1 > 0.5)$ becomes $\\mathcal{G}_{\[0,T\]}(x_1 > 0.55)$), and (b) adding two conjuncts withheld during training (a no-overshoot constraint $\\mathcal{G}_{\[T/2, T\]}(x_1 \< 1.1 \\cdot \\text{setpoint})$ and a control-effort cap $\\mathcal{G}_{\[0, T\]}(|u| \< u\_\\text{max} \\cdot 0.8)$). On the held-out trajectory set, we report $(R\_\\text{gold} - R\_\\text{spec})$ and $(R\_\\text{spec} - R\_\\text{verifier})$ separately; the latter sits at the $10^{-6}$ floor while the former grows monotonically with model capacity (the better the policy fits $\\varphi\_\\text{spec}$, the more headroom there is for $\\varphi\_\\text{gold}$ to differ).

The comparison baseline is a *learned-critic* proxy: a small reward model fine-tuned to regress $\\rho$-from-rollouts, evaluated on the same held-out set. The learned critic exhibits *both* a non-zero verifier-fidelity term (regression error) and a spec-completeness term, and the two cannot be disentangled from observation alone. STL exposes the spec-completeness term in isolation. That is the central theoretical contribution.

## Failure mode taxonomy

Five modes ranked by ex-ante probability with mitigations.

**FM1 (~30%): Trivial replication.** SERA's "soft works as well as hard" replicates without surprise because the gap was zero in this domain to begin with. Symptom: every $(m, v, f)$ cell shows $|\\Delta \\hat p| \< 0.01$. The publishable claim collapses. Mitigation: task family construction is calibrated against a pilot ($n = 30$ trajectories per cell) such that hard-filter success at $N = 128$ lies in $\[0.3, 0.7\]$. Cells where $\\pi\_\\text{ref}$ already saturates near 1.0 are excluded at design stage, before any model training.

**FM2 (~25%): STL filtering doesn't beat random filtering.** Symptom: a control arm with $D_v$ constructed by random subsampling matches $D\_\\text{quant}$ within MDE on every cell. Means $\\rho$ does not capture useful task-quality information for these specs. Mitigation: before any SFT, validate that $\\rho_i$ correlates (Spearman $r > 0.3$) with a held-out *gold* score on the trajectory store. If $r \< 0.3$ on a given task family, drop the family from the canonical sweep. Preflight in `experiments/preflight_rho_correlation.py`.

**FM3 (~20%): Models too small.** Symptom: $\\pi_v$ eval success at floor ($\\le 5%$) for all $v$ across all model sizes. The Qwen3-{0.6, 1.7, 4}B family is below the capacity threshold. Mitigation: difficulty calibration is on the trajectory-generation side too — per-instance horizon $H$, number of state dimensions exposed in the prompt, and predicate density of $\\varphi$ are all swept in the pilot until a heuristic (PID with literature gains) hits ~50% success. If even Qwen3-4B with hard filtering hits floor on the resulting tasks, the artifact pivots to a "negative result on small-model scientific control with current prompt schemas" framing.

**FM4 (~15%): Backend numerical divergence.** Symptom: the canonical RunPod (bf16, A100, bnb 4-bit base) sweep produces materially different numbers than the local MLX (mps, fp16) pilot. Has bitten me before. Manifestations: $\\rho_i$ differing in sign across backends due to ODE integrator choice; LoRA training dynamics differing because the bnb-quantized base introduces noise that MLX's fp16 base does not. Mitigation: a "preflight" step on RunPod replicates a tiny slice (1 model × 1 filter × 1 task family × 2 instances × 2 seeds × 4 BoN budgets) of the MLX pilot and checks that per-instance success rates match within 5 percentage points before launching the full sweep. ODE integration uses Diffrax `Tsit5` with `rtol=1e-6, atol=1e-9` on both backends. NaN/Inf events on Diffrax solves are replaced with zeros and counted in a separate field of the trial record (per `~/CLAUDE.md`).

**FM5 (~10%): Compute overrun.** Symptom: the canonical sweep doesn't finish by day 14. Mitigation: hard checkpoint at day 8. Fallback is a $2 \\times 2 \\times 2$ sub-design (Qwen3-{0.6B, 4B} × {hard, cont} × {bio_ode, gluc}) that still touches all three hypotheses with reduced power, missing cells annotated as "not run" rather than imputed. Imputation across model sizes is explicitly forbidden in the analysis plan.

## REDACTED firewall

The author's prior work on the REDACTED paper (REDACTED et al.; physics-informed STL parameter synthesis) shares simulator infrastructure with this artifact but is mathematically a different optimization problem.

REDACTED problem: $\\max\_{\\theta \\in \\Theta\_\\text{phys}} \\rho(\\tau(\\theta; x_0), \\varphi)$ where $\\tau$ solves $\\dot x = f(x, u\_\\text{fixed}; \\theta)$. The optimization variable is the physical parameter vector $\\theta \\in \\mathbb{R}^d$, the control $u$ is fixed (or absent — the system is autonomous), $x_0$ is given, and the loss landscape is shaped by physics-prior penalties C1-C11 documented in `REDACTED.py` and the augmented-Lagrangian / CEGAR machinery in `REDACTED.py`.

stl-seed problem: $\\max\_{u\_{1:H} \\in U^H} \\rho(\\tau(\\theta\_\\text{fixed}, u\_{1:H}; x_0), \\varphi)$ where $\\theta\_\\text{fixed}$ is drawn from BRENDA \[DOI 10.1093/nar/gky1048\] *fresh* for each task family and *not* taken from any REDACTED tuned solution. The optimization variable is the control sequence, parameters are fixed and literature-sourced, and the loss is the unmodified $\\rho$ — no Augmented Lagrangian, no CEGAR loop, no residual-NN correction term, no C1-C11 physics-filter penalties, no conjunction-vs-implication spec-form ablation. The optimization is performed *implicitly* by a learned policy via best-of-$N$ decoding rather than by a numerical solver.

Firewall checklist. (i) No file in `stl-seed/` imports from `~/REDACTED.py`, `~/REDACTED.py`, `~/REDACTED.py` (verified by grep at `paper/firewall_grep.txt`). (ii) No $\\theta$ value in `stl-seed/configs/` matches any REDACTED-tuned $\\theta$ to within 5 significant figures; all $\\theta$ values are pulled from BRENDA / SABIO-RK / KEGG with citation strings recorded in the config. (iii) The STL spec library (`stl-seed/specs/`) is independently authored: bio_ode (repressilator) and glucose_insulin specs are written from textbook descriptions \[Strogatz 2014; Elowitz & Leibler 2000 DOI 10.1038/35002125; Bergman et al. 1979 PMID 443421\] without reference to REDACTED's Hill specs. (iv) Reproducibility scripts (`REDACTED.py` etc.) are not invoked anywhere in the stl-seed pipeline.

The shared infrastructure (Diffrax integrator, Donzé-Maler $\\rho$ evaluator) is a software dependency, not a methodological overlap. The two papers are submittable to disjoint venues without overlap-of-contribution concerns.

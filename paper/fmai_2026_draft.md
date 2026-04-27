# When Soft Beats Hard, and When It Doesn't: STL Robustness as a Formal Soft Verifier for Inference-Time LLM Control

## Abstract

We test whether the SERA recipe (soft-verifier-guided inference for small open-weights LLMs) extends to scientific control when the soft signal is a *formal* Signal Temporal Logic (STL) robustness $\\rho(\\tau, \\varphi)$ on a closed-form ODE simulator, rather than a learned or hand-engineered proxy. The headline finding is that **methodology choice dominates verifier choice**: on a pre-registered, falsification-shaped head-to-head against `Qwen3-0.6B` over four narrow-attractor biomolecular and cardiac specs, standard sampling fails on a strict majority of seeds across all four tasks ($\\bar\\rho \\in {-247.6, -100.0, -0.5, -1.4}$), and a discrete-enumeration sampler (beam-search warmstart) rescues all four ($\\bar\\rho \\in {+25.0, +30.0, +0.002, +0.85}$, 12/12 satisfaction). The pre-registered verdict fires `METHODOLOGY MATTERS`, ruling out the flat-prior strawman. A subsequent quantization $\\times$ size factorial (5 Qwen3 variants $\\times$ 4 hard tasks $\\times$ 3 seeds = 120 cells) shows the methodology gap survives $\\sim 3\\times$ quantization compression and $\\sim 3\\times$ size scaling; the toggle task saturates at 1.7B, the first appearance of SERA's saturation transition in our data. We give a landscape theorem characterizing when gradient guidance reaches the satisfying set (smooth regime, Polyak-Lojasiewicz alignment with directional vocabulary coverage) versus when it cannot (narrow-attractor regime, exponential-in-$H$ decay) and an inference-time scaling-law fit showing power-law shapes on smooth tasks and step-function shapes on narrow-attractor tasks. Because the verifier is the recursive Donze-Maler evaluator on the same trajectories the spec is defined over, the verifier-fidelity term in the Goodhart decomposition collapses to float64 round-off ($\\le 12$ ulp on our specs); the only auditable gap is spec-completeness, which we lower-bound at $-2.27$ $\\rho$-units against a hand-tuned composite gold scorer.

## 1. Introduction

Soft-verifier-guided inference for small LLM agents (SERA; Shen et al. 2026) has emerged as a dominant recipe for instructing small open-weights LLMs on verifiable tasks. The soft signal is almost always *constructed*: a learned reward model, a judge LLM, or an engineered proxy such as patch-overlap on a code edit. The Goodhart decomposition

$$R\_\\text{gold} - R\_\\text{proxy} = (R\_\\text{gold} - R\_\\text{spec}) + (R\_\\text{spec} - R\_\\text{verifier})$$

remains unauditable in this regime because the verifier-fidelity error term $(R\_\\text{spec} - R\_\\text{verifier})$ is entangled with the verifier's training residual, and the spec-completeness term $(R\_\\text{gold} - R\_\\text{spec})$ is rarely measured at all. From a failure-modes perspective this is unsatisfying: when soft-verified inference goes wrong, we cannot tell whether the spec was incomplete or the verifier was unfaithful.

We propose using **formal STL robustness on closed-form ODE simulators** as the soft signal. STL robustness $\\rho(\\tau, \\varphi)$ (Donze and Maler 2010; Fainekos and Pappas 2009) is a recursively-defined real-valued functional of the trajectory and the formula. When the simulator is a closed-form ODE and the formula is fixed, $\\rho$ is computed by recursive `min`/`max` on signal differences; the verifier-fidelity term collapses to float64 round-off through the evaluator's depth (depth $\\le 12$ on our specs, $\\le 12$ ulp accumulation per call). The whole interpretable gap becomes the spec-completeness term, which is lower-bounded by a trajectory adversary and thus auditable.

**Contributions.**

1. **Verifier-fidelity-free soft verification** via formal STL robustness on closed-form ODE simulators (Section 3). The Goodhart decomposition's verifier-fidelity term is empirically $\\le 10^{-6}$ after $\\sigma$-squashing on held-out trajectories; the auditable spec-completeness gap is $-2.27$ $\\rho$-units on `glucose_insulin.tir.easy` against a hand-tuned composite gold scorer.

1. **Pre-registered, falsification-shaped real-LLM head-to-head** (Section 4.1) showing `METHODOLOGY MATTERS` on `Qwen3-0.6B` (4/4 hard tasks rescued by discrete enumeration). The pre-registered outcome rule rules out the flat-prior strawman that "soft verification helps when the LLM has no opinion."

1. **Quantization $\\times$ size factorial** (Section 4.2) showing the methodology gap survives $\\sim 3\\times$ quantization compression and $\\sim 3\\times$ size scaling. The toggle task saturates at 1.7B (standard $\\bar\\rho: -99.96 \\to +14.07$), the first appearance of the SERA-saturation transition in our data.

1. **Landscape theorem** (Section 3.2) giving a formal characterization of when gradient guidance succeeds (regime I, smooth) versus when it cannot (regime II, narrow attractor with exponential-in-$H$ decay).

1. **Task-structure-dependent inference scaling laws** (Section 4.3): smooth dynamics yield power laws ($b \\approx -0.24$, $R^2 = 0.81$ on glucose-insulin); narrow attractors yield step functions, with no smooth scaling exponent defensible.

The framing is squarely on FMAI's "verified trade-offs" pillar: STL robustness is the verifier, the 9-sampler comparison is the trade-off, and the methodology-matters-but-only-for-some-task-structures finding is the failure-mode taxonomy.

## 2. Setup

**Tasks.** Five biomolecular ODE control families spanning two physical time-scales: (i) `glucose_insulin.tir.easy` (Bergman minimal model; smooth dynamics, time-in-range spec); (ii) `bio_ode.repressilator.easy` (Elowitz-Leibler synthetic oscillator; silence-3 corner); (iii) `bio_ode.toggle.medium` (genetic toggle switch); (iv) `bio_ode.mapk.hard` (MAPK cascade with pulse pattern); (v) `cardiac.suppress_after_two.hard` (FitzHugh-Nagumo cardiac action potential; millisecond time-scale). Specs are written in STL with predicates over the simulator state. Full spec definitions and predicate forms are in the artifact.

**Action vocabulary.** Each task has a continuous action box $U \\subseteq \\mathbb{R}^m$. We discretize via a uniform per-axis grid with $k\_\\text{per-dim} = 5$, giving $|V| = K = k\_\\text{per-dim}^m \\in {25, 125}$ depending on $m$. The horizon $H$ is task-specific (8-12 control steps).

**LLM proposer.** `Qwen3-0.6B-bf16` (and four other Qwen3 variants for the quantization $\\times$ size factorial) via MLX. The prompt encodes the task; the LLM emits logits over $V$ at each control step. We sample at temperature $T = 0.5$ across all reported runs.

**STL robustness.** $\\rho : \\mathbb{R}^{T \\times n} \\to \\mathbb{R}$ is computed by the recursive Donze-Maler quantitative semantics (Donze and Maler 2010), implemented as JAX-traced `min`/`max` reductions on differentiable signals. STLCG and STLCG++ (Leung 2020; Hashemi et al. 2025) supply the masking semantics that make autodiff well-defined on the measure-zero non-differentiable set.

**Samplers.** We compare: `standard` (LLM-only sampling), `bon` and `bon_continuous` (best-of-$N$), `gradient_guided` (logit bias by $\\nabla\_{\\bar u_t} \\rho$ propagated through Diffrax + Donze-Maler), `hybrid`, `horizon_folded` (full-horizon gradient over flattened control), `rollout_tree` (one-step lookahead with constant tail), `cmaes_gradient` (CMA-ES + gradient refinement), and `beam_search_warmstart` (beam search over $V$ with model-predictive constant-extrapolation lookahead).

**Pre-registration.** The outcome rule for the falsification head-to-head (Section 4.1) is committed in `scripts/real_llm_hard_specs.py:14-44` before the runs were executed: `METHODOLOGY MATTERS` iff beam-search reaches $\\rho > 0$ on $\\ge 2$ of 4 tasks where standard sampling reaches $\\rho \\le 0$ on a majority of seeds.

## 3. Method

### 3.1 Verifier-fidelity-free soft verification

The Goodhart decomposition $R\_\\text{gold} - R\_\\text{proxy} = (R\_\\text{gold} - R\_\\text{spec}) + (R\_\\text{spec} - R\_\\text{verifier})$ is a tautology, but its second term is interesting only when the verifier is a learned approximation of $R\_\\text{spec}$. When the verifier *is* the recursive Donze-Maler evaluator on the same $(\\tau, \\varphi)$ that $R\_\\text{spec}$ is defined over, the second term collapses to a numerical floor.

**Claim (verifier-fidelity floor).** *For float64 evaluation of $\\rho$ on a depth-$d$ formula tree with $d \\le 12$, the cumulative round-off bound is $\\le 12 \\cdot 2^{-52} \\approx 2.7 \\cdot 10^{-15}$ per evaluation. Empirically, on our specs after $\\sigma$-squashing, the held-out floor is $\\le 10^{-6}$.*

This is an empirical bound, not a proof of zero (soft $\\rho$-weighted SFT composes $\\sigma$ around $\\rho$ which can amplify condition number near saturation), but it puts the verifier-fidelity term orders of magnitude below any Goodhart phenomenon a learned RM faces.

The auditable gap is therefore $R\_\\text{gold} - R\_\\text{spec}$. We measure this with a trajectory adversary that searches for a satisfying trajectory whose gold score is minimal. On `glucose_insulin.tir.easy`, against a composite gold scorer with literature-cited components (TIR coverage from ADA / Battelino targets; an $L^2$ jerk penalty; a glucose-variance penalty; blend weights set by dimensional-analysis defaults), the adversary finds a satisfying trajectory with gap $-2.27$ $\\rho$-units. The honest framing is that this is an existence-style lower bound under those weights, not a population-mean against an external oracle. Against a learned process-reward baseline (Setlur et al. 2024; PAV with on-policy MC labels and 4-cell hyperparameter selection), STL achieves AUC 1.000 versus PAV-v2 0.962 on glucose-insulin (a $-0.038$ AUC residual after model selection).

### 3.2 Landscape theorem

Let $J(u) = \\rho(\\text{Sim}(x_0, u))$ be the composition of the simulator and the STL evaluator, $J : U^H \\to \\mathbb{R}$. Define the satisfying set $S\_+ = {u : J(u) > 0}$. We separate two structural regimes.

**Theorem (informal).** *Under standing assumptions A1-A5 (Lipschitz simulator and STL, a.e. differentiability of $J$, vocabulary covers $U$, float64 floor):*

**(I) Smooth regime.** *If there exists $\\eta\_\\star > 0$ such that $S\_+^{,\\eta\_\\star}$ is non-empty and convex and a Polyak-Lojasiewicz-type alignment $\\langle \\nabla J(u), u^\\star - u\\rangle \\ge \\kappa(J(u^\\star) - J(u))$ holds for some $u^\\star \\in S\_+^{,\\eta\_\\star}$, and the vocabulary geometry has directional coverage $\\cos\\theta\_\\text{cov} > 0$, then for sufficiently large $\\lambda$ gradient-guided sampling reaches $S\_+$ in*

$$\\mathbb{E}\[\\text{rollouts to first hit}\] = O\\left(\\frac{LT}{\\lambda \\cos\\theta\_\\text{cov}, \\eta\_\\star} \\log\\frac{1}{\\delta}\\right).$$

**(II) Narrow-attractor regime.** *If the satisfying set is contained in an $\\varepsilon$-tube around the discrete vocabulary lattice (cliff condition: $\\varepsilon \\le \\eta\_\\flat / L$, $J(\\bar u_0) \< -\\eta\_\\flat$, $|\\nabla J(\\bar u_0)| \\le \\eta\_\\flat / D$), then for any finite $\\lambda, T, N$,*

$$\\Pr\_{u \\sim \\pi^\\text{guided}}\[u \\in S\_+\] \\le \\beta \\cdot (\\Pr\_\\text{prior}\[u \\in S\_+\] + e^{-cN}).$$

*When $|S\_+|/|U^H| \\le K^{-H}$ (the satisfying set is a single vocabulary corner), the right-hand side is $O(\\beta K^{-H} + e^{-cN})$: exponentially small in the horizon.*

**(III) Discrete-enumeration corollary.** *Under (II), if a constant-vocabulary policy $u^\\star = V\_{k^\\star}^{\\otimes H}$ has $J(u^\\star) > 0$, then beam-search warmstart with constant-extrapolation lookahead and beam width $B \\ge 1$ finds $u^\\star$ in deterministic compute $H \\cdot B \\cdot K$ simulator forwards.*

The full statement and proof sketch are in the supplementary `landscape_theorem.md`. Two honest caveats: the constant $\\lambda^\\star$ in (I) is not tight, and the vocabulary-design assumption in (III) is engineering ("if the user picks $V$ to cover a satisfying corner") not a structural theorem.

The empirical correspondence is legible. Glucose-insulin meets (I) and saturates the spec ceiling at every seed under gradient guidance ($+20.00$, zero variance). Repressilator, toggle, and MAPK meet (II): gradient guidance produces $\\bar\\rho \\in \[-250, -190\]$ across all $\\lambda$ tested. Beam-search warmstart meets (III): the satisfying corner lives in $V^H$ with $k\_\\text{per-dim} = 5$, and discrete enumeration finds it deterministically.

## 4. Results

### 4.1 Real-LLM falsification: `METHODOLOGY MATTERS`

The unified comparison harness uses a flat-prior LLM (uniform logits over $V$) to isolate what each sampler does with the verifier signal alone. The risk is that any "+128$\\times$ lift" headline measured against a uniform proxy evaporates once a real LLM prior is wired in. We run a pre-registered head-to-head against `Qwen3-0.6B-bf16` over the four narrow-attractor specs; 3 fixed seeds per cell; matched temperature 0.5; matched seeds; matched horizon.

| Task                              | Standard sat / n | Standard $\\bar\\rho$ | Beam sat / n | Beam $\\bar\\rho$ |        Gap |
| --------------------------------- | ---------------: | --------------------: | -----------: | ----------------: | ---------: |
| `bio_ode.repressilator.easy`      |              0/3 |            $-247.582$ |          3/3 |         $+25.000$ | $+272.582$ |
| `bio_ode.toggle.medium`           |              0/3 |             $-99.960$ |          3/3 |         $+29.992$ | $+129.952$ |
| `bio_ode.mapk.hard`               |              0/3 |              $-0.500$ |          3/3 |          $+0.002$ |   $+0.502$ |
| `cardiac.suppress_after_two.hard` |              0/3 |              $-1.434$ |          3/3 |          $+0.850$ |   $+2.284$ |

The pre-registered verdict fires `METHODOLOGY MATTERS` (4/4 $\\ge$ 2/4 threshold). The gap is entirely attributable to the sampler's use of the verifier signal: standard sampling reads the LLM logits and picks; beam-search warmstart enumerates $V$, scores each candidate under a model-predictive constant-extrapolation lookahead, and seeds gradient refinement from the top-$B$. Wall-clock $\\approx$ 5 min on M5 Pro for the whole sweep.

Two honest caveats. First, `Qwen3-0.6B` at $T = 0.5$ produces zero per-seed variance on every task here: the LLM-prior collapses to a single mode because low-entropy logits dominate the temperature-rescaled softmax. This is mode-collapse, not a sampling bug. Second, the MAPK and cardiac gaps are small in absolute units ($+0.002$, $+0.850$); they are sat-fraction wins at the satisfaction boundary, not magnitude wins.

### 4.2 Quantization $\\times$ size factorial

We replicate the falsification across 5 Qwen3 variants (0.6B / 1.7B $\\times$ bf16 / 4-bit / 8-bit) over the same 4 hard tasks at 3 seeds, 120 cells total, $\\sim$22 min wall on M5 Pro.

| Model             | Sat (standard) | Sat (beam-search) | Verdict             |
| ----------------- | -------------: | ----------------: | ------------------- |
| `qwen3-0.6b-bf16` |           0/12 |             12/12 | METHODOLOGY MATTERS |
| `qwen3-0.6b-8bit` |           0/12 |             12/12 | METHODOLOGY MATTERS |
| `qwen3-0.6b-4bit` |           1/12 |             12/12 | METHODOLOGY MATTERS |
| `qwen3-1.7b-bf16` |           3/12 |             12/12 | METHODOLOGY MATTERS |
| `qwen3-1.7b-4bit` |           3/12 |             12/12 | METHODOLOGY MATTERS |

The verdict is unanimous. Three structural facts emerge. (i) 8-bit quantization is bit-identical to bf16 on every cell at $T = 0.5$; 4-bit diverges only on toggle (1/3 seeds, no majority crossing). (ii) Scaling 0.6B $\\to$ 1.7B saturates toggle at the LLM level (standard $\\bar\\rho: -99.96 \\to +14.07$, 3/3 sat). This is the first appearance of the SERA-saturation transition in our data: at 1.7B the LLM picks the satisfying corner on toggle on its own, no verifier needed. (iii) Repressilator, MAPK, and cardiac stay solidly in the methodology-mattering regime at 1.7B; standard fails on every seed at every model size.

The headline strengthens: the methodology gap is robust to $\\sim 3\\times$ quantization compression and $\\sim 3\\times$ size scaling, and the only task that breaks the gap is the one where SERA's saturation transition appears for free. Mapping the full saturation curve from 0.6B $\\to$ 8B+ is past available compute; 4B is at M5 Pro's unified-memory edge for a $K = 125$ beam-search cell.

### 4.3 Inference-time compute scaling laws

We fit the saturating power law $\\bar\\rho(t) = a t^b + c$ across 9 samplers per task, fitting per-task on warm wall-clock seconds.

| Task                    | Class            |      $b$ | $R^2$ | Qualitative shape                                      |
| ----------------------- | ---------------- | -------: | ----: | ------------------------------------------------------ |
| `glucose_insulin`       | smooth           | $-0.241$ |  0.81 | smooth saturation toward $+20.75$ ceiling              |
| `bio_ode.mapk`          | smooth-tight     | $-0.927$ |  0.10 | binary: 3 succeed, 6 stall at $-1$                     |
| `cardiac_ap`            | step             | $-3.000$ |  0.76 | step at $t \\sim 0.03$s; $b$ at boundary               |
| `bio_ode.toggle`        | narrow-attractor | $-0.375$ |  0.05 | bimodal: 3 cross, 6 stall at $-100$                    |
| `bio_ode.repressilator` | narrow-attractor | $-0.000$ |  0.23 | step: 8/9 stall at $-220$, beam jumps to $+25$ at 100s |

Only `glucose_insulin` admits a defensible power-law fit ($b = -0.241$, sub-linear, consistent with Snell et al. 2024's reasoning-task regime). On the four narrow-attractor and tight-ceiling tasks, the legible shape is regime structure (which samplers succeed at all), not a scaling exponent. The headline: **inference-time compute scaling laws for STL-verified scientific control are task-structure-dependent.** Smooth dynamics yield power laws favoring continuous compute investment; narrow attractors yield step functions, with vocabulary design determining the location of the vertical edge.

This diverges from the LLM-reasoning scaling story precisely where Setlur et al. 2024 (PAV) and Beirami et al. 2024 (BoN-KL) implicitly assume: a verifier whose gradient or sampling-conditioned posterior is locally informative. STL-rho on `bio_ode.repressilator` is *correct* (it returns $-250$ to non-satisfying controls and $+25$ to the satisfying corner); what fails is the *gradient* at non-satisfying points, which does not point toward the silence-3 corner.

## 5. Discussion

The empirical and theoretical pictures align. Under the smooth-regime conditions of the landscape theorem (PL alignment, directional coverage, dense $S\_+$), gradient-guided sampling reaches the spec ceiling deterministically; the inference-time scaling exponent is sub-linear and consistent with the LLM-reasoning literature. Under the narrow-attractor conditions (cliff geometry, $\\varepsilon$-tube around vocabulary), gradient guidance is exponentially small in $H$, and discrete enumeration over a vocabulary that contains a satisfying corner finds it in $H \\cdot B \\cdot K$ simulator forwards. The task-structure-dependent scaling exponents are predicted by which regime the spec/simulator pair falls into, and the SERA-saturation transition appearing on toggle at 1.7B is a third axis: at sufficient model scale, the LLM's internal mode-selection itself crosses the satisfaction boundary, removing the need for the verifier on that task.

For the FMAI failure-modes audience, the headline is: **the methodology-vs-LLM-prior trade-off has a sharp structural signature.** Reporting "methodology matters" or "doesn't matter" without conditioning on task structure is the failure mode. We propose two operational diagnostics: (i) measure the smooth-margin radius $r\_{\\eta\_\\star}$ and the cliff index $\\chi(\\delta)$ of the simulator-spec composition before choosing a sampler; (ii) check whether the LLM's mode-selection at the relevant scale already crosses the satisfaction boundary (the toggle-at-1.7B pattern).

The verifier-fidelity-free reframing has implications beyond bio-control. Wherever a closed-form simulator and a temporal-logic spec are available (cardiac modeling, robot trajectory verification, chemical reaction network control, glucose management, certain coding-agent settings), the same Goodhart decomposition applies, and the same "methodology vs prior" trade-off can be measured under audit conditions. The artifact's coding-agent task-cell design (HumanEval-mutated, factored patch vocabulary $|V| = 390$, six-channel state, three STL specs of varying difficulty) is a near-term test of how far the framework generalizes to non-differentiable simulators where only the discrete-enumeration samplers apply. Failure to extend would itself be informative about which structural property (differentiability of the simulator? continuity of the spec?) is load-bearing.

## 6. Limitations

**Phase 1 only.** This is an inference-time-only result. No SFT sweep is reported. The pre-registered Phase 2 (training-time SERA-mimic on Qwen3-{0.6B, 1.7B, 4B} $\\times$ 3 filter conditions $\\times$ 2 task families = 18 cells, hypotheses H1-H3 on TOST equivalence of soft and hard at $\\Delta = 0.05$, size-monotone improvement, spec-completeness against learned-critic baseline) is queued, gated on $\\sim$$25 of RunPod 4090 spot. The MLX QLoRA pilot smoke validated the pipeline (training loss $1.484 \\to 0.466$ in 15 s on M5 Pro; 5/5 held-out parse-success; 4.6 MiB adapter); the canonical sweep is one command away.

**Vocabulary by construction.** Beam-search warmstart's rescue rests on the satisfying corner being in the $k\_\\text{per-dim} = 5$ lattice. For repressilator, $u = (0, 0, 1) \\in V$; for toggle, $u = (0, 1) \\in V$. This is engineering, not a free win. The contribution is the structural-search-vs-continuous-search distinction; the vocabulary-tuning choice is transparent in code and theorem.

**$N = 3$ seeds.** This is a falsification test, not a population estimate. Per-seed variance is zero on most cells because Qwen3 at $T = 0.5$ deterministically picks the same vocabulary entry across the three seeds; CIs are tight by construction. Higher temperature or larger $N$ would loosen the CIs and may change the one-of-three crossings on `0.6b-4bit + toggle`.

**Hardware: M5 Pro local.** Wall-clock numbers are not portable to CUDA. The qualitative cross-over structure is hardware-independent (it depends on which sampler can find the satisfying region at all); the quantitative cross-over budgets are M5-Pro-specific.

**Spec-completeness gap.** The $-2.27$ $\\rho$-unit lower bound on `glucose_insulin.tir.easy` uses dimensional-analysis blend weights set by the author for the composite gold scorer, not literature-cited as a single composite. The number is an existence-style lower bound under those weights, not a population-mean against an external oracle. The directional claim (verifier-fidelity is float-64-bounded; spec-completeness is the auditable gap) is unaffected, but the magnitude depends on the gold scorer's blend.

**Proof-sketch theorem.** The landscape theorem is a characterization, not a tight rate bound. The constant $\\lambda^\\star$ in regime (I) is not tight; the vocabulary-design assumption in (III) is engineering not structural. Reviewers should read regime (I)'s bound asymptotically, not as a usable hyperparameter recommendation.

**Single-LLM family.** All real-LLM results use Qwen3. Whether the methodology gap shape replicates on Llama-3, Gemma-3, or Phi-4 at comparable sizes is open; we expect it to, but have not measured.

**No SFT result here.** The headline (METHODOLOGY MATTERS, robust to quantization and size scaling, with one task crossing into saturation at 1.7B) is an inference-time finding. The training-time question (does SFT close the gap, and does soft training match hard training in the SERA sense) is the queued Phase 2 contribution and is not in this paper.

## References

Aksaray, D., Jones, A., Kong, Z., Schwager, M., and Belta, C. (2016). Q-Learning for Robust Satisfaction of Signal Temporal Logic Specifications. *CDC 2016*.

Beirami, A., Agarwal, A., Berant, J., D'Amour, A., Eisenstein, J., Nagpal, C., and Suresh, A. T. (2024). Theoretical guarantees on the best-of-n alignment policy. *arXiv:2401.01879*.

Donze, A. and Maler, O. (2010). Robust satisfaction of temporal logic over real-valued signals. *FORMATS 2010*. DOI:10.1007/978-3-642-15297-9_9.

Fainekos, G. E. and Pappas, G. J. (2009). Robustness of temporal logic specifications for continuous-time signals. *Theoretical Computer Science* 410(42):4262-4291.

Hashemi, N., Suh, J., Ren, A., Hsieh, J., and Pavone, M. (2025). STLCG++: a masking approach for differentiable signal temporal logic. *arXiv:2501.04194*.

Karimi, H., Nutini, J., and Schmidt, M. (2016). Linear convergence of gradient and proximal-gradient methods under the Polyak-Lojasiewicz condition. *arXiv:1608.04636*.

Krasowski, H. et al. (2024). STL-guided parameter inference for biomolecular ODEs. (Krasowski STL-control line; cited as prior work establishing STL as a control signal in biomolecular settings.)

Leung, K., Arechiga, N., and Pavone, M. (2020). Back-propagation through Signal Temporal Logic specifications. *arXiv:2008.00097*.

Setlur, A., Nagpal, C., Fisch, A., Geng, X., Eisenstein, J., Agarwal, R., Agarwal, A., Berant, J., and Kumar, A. (2024). Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning. *arXiv:2410.08146*.

Shen, S., Tormoen, J., Shah, P., Farhadi, A., and Dettmers, T. (2026). SERA: Soft-verified inference for small open-weights LLMs. *arXiv:2601.20789*.

Snell, C., Lee, J., Xu, K., and Kumar, A. (2024). Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Model Parameters. *arXiv:2408.03314*.

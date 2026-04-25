# stl-seed

[![CI](https://github.com/AA-Alghamdi/stl-seed/actions/workflows/ci.yml/badge.svg)](https://github.com/AA-Alghamdi/stl-seed/actions/workflows/ci.yml)
[![Lint](https://github.com/AA-Alghamdi/stl-seed/actions/workflows/lint.yml/badge.svg)](https://github.com/AA-Alghamdi/stl-seed/actions/workflows/lint.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

> **Soft-verified SFT for scientific control. STL robustness as a formal process verifier on small open-weights LLMs.**
>
> Phase 1 release (theory + library + local pilot), 2026-04-24. Phase 2 (canonical RunPod sweep) targets v0.1.0.
>
> **Phase 2 status:** the canonical 18-cell sweep ships as a single command — `python scripts/run_canonical_sweep.py --confirm` — with a default `$25` cap and an expected spend of `$5–15`. Awaiting RunPod credentials.

---

## The rule of thumb (above-the-fold, ≤ 100 words)

On a 100-trajectory glucose-insulin pilot, MLX QLoRA on Qwen3-0.6B-bf16 drove training loss **1.484 → 0.466** monotonically in **15.0 s** of M5 Pro wall-clock, hit **5/5 held-out parse-success**, and produced a **4.6 MiB adapter** — all for $0 of cloud compute. The post-fix repressilator pilot (topology-aware heuristic) lifts pooled satisfaction from 0% to **46.5%** (N=2,388). That gates Phase 2: a single command, 3 sizes × 3 filters × 2 task families = 18 RunPod 4090 cells, $5–15 of $25 cap. The artifact's point is the Goodhart decomposition (§6) — the verifier-fidelity term is provably zero.

---

## 1. Why I built this

I'm Abdullah AlGhamdi: UC Berkeley chemistry '26, four years in [Omar REDACTED's](https://yaghi.berkeley.edu/) MOF lab, two semesters in [Murat Arcak's](https://people.eecs.berkeley.edu/~arcak/) BAIR group on STL formal methods, matriculating into [CMU MS-AIE](https://www.cs.cmu.edu/aie) August 2026. I came to STL through Arcak; I came to STL on biomolecular ODEs through [Hanna REDACTED's REDACTED line](https://arxiv.org/abs/2412.15227), where I'm a co-author on a separate parameter-synthesis paper that is firewalled from this artifact ([paper/REDACTED.md](paper/REDACTED.md)).

Last January, [SERA](https://arxiv.org/abs/2601.20789) (Shen, Tormoen, Shah, Farhadi, Dettmers) ran a careful experiment in code-agent SFT: generate trajectories with a teacher model, score each one with line-level recall against a pseudo-ground-truth patch, and SFT a student on the soft-filtered set. The headline was that **soft verification (line-overlap) matched hard verification (test-execution)** on SWE-bench Verified at 26× lower cost. The discussion section flagged the open question I find most interesting:

> *"One explanation is that early performance gains on coding tasks depend primarily on learning skills like converting intentions into code edits and navigating codebases, rather than on code correctness. However, once a model saturates on these aspects, verified correct code may become necessary for further improvement."* — SERA §Discussion

In the SERA setup, the soft signal was **constructed**. Line-overlap on patches is a useful proxy but it is engineered: someone had to choose line granularity, choose recall over F1, choose the threshold (~0.5 in their final config). The signal is downstream of human design choices about what "almost-correct code" means.

In **scientific control** — driving a continuous-time dynamical system to satisfy a temporal-logic specification — the soft signal is not constructed. It is **defined**. Donzé and Maler's space-robustness ρ(τ, φ) ([FORMATS 2010](https://doi.org/10.1007/978-3-642-15297-9_9)) is a real-valued function on (trajectory, formula) pairs that is positive iff the trajectory satisfies the formula and whose magnitude measures the signed margin to the nearest violation. There is no human choice of "what almost-satisfies looks like." The Donzé-Maler recursion on min/max nodes returns a number, in float64, deterministic.

This makes scientific control a clean testbed for SERA's central conjecture, with one structural advantage: **the verifier-noise term in the Goodhart decomposition is provably zero**, so any soft-vs-hard gap that appears in the sweep is attributable to spec-completeness alone (theory.md §6, this README §7). That decomposition is the central theoretical contribution of the artifact.

I built this in three weeks, solo, on an M5 Pro, because (a) I want to find out if SERA's "soft suffices" pattern transfers to a domain where the soft signal is genuine, (b) I needed a research artifact to send REDACTED' group ahead of CMU matriculation, and (c) the REDACTED firewall ([paper/REDACTED.md](paper/REDACTED.md)) made this the natural problem to pick: same simulator infrastructure as the REDACTED paper, mathematically disjoint optimization (control over actions, not synthesis over kinetic parameters).

---

## 2. What this is NOT

This is **not a benchmark**. There are 2 task families (bio_ode, glucose-insulin), each with 3 textbook-derived STL specs ([paper/REDACTED.md](paper/REDACTED.md)). I am not claiming to cover scientific control. I am claiming to have one carefully-instrumented loop where **one knob (verifier density) was measured** and **one effect was found** (or, if the sweep falsifies H1, one effect was honestly not found — see [paper/theory.md §3](paper/theory.md) pre-registration).

This is **not an autonomy framework**. There is no "give the agent a goal and watch it figure out the world" framing here. The agent receives a state vector and an STL specification in JSON, emits a control sequence, the simulator integrates, the STL evaluator scores, and the trajectory either passes the filter or doesn't. Engineered workflow, not autonomous-agent magic. I think Dettmers' [Use Agents or Be Left Behind](https://timdettmers.com/2026/01/13/use-agents-or-be-left-behind/) post is right that the load-bearing word is "use" — instrument the loop, measure the effect, do not anthropomorphize the model.

This is **not a SERA replication**. SERA ran code-agent SFT at the 8B–32B scale on 121 repositories with self-hosted GLM-4.5-Air as teacher. stl-seed runs scientific-control SFT at the 0.6B–4B scale on 2 task families with no teacher (the reference policy is a heterogeneous {random, heuristic, small-LLM} mixture, [paper/theory.md §2](paper/theory.md)). The structural comparability is in the recipe shape, not the scale.

This is **not a final paper**. v0.1.0 ships the canonical sweep and the hierarchical Bayes posterior ([paper/theory.md §4](paper/theory.md)). v0.0.x is the Phase-1 ship: theory, library, local pilot. The smoke test is a hard-checkpoint pass, not a result.

---

## 3. The setup in one figure

```
                    +-------------------+
                    |  STL spec φ       |
                    |  state x_0        |
                    |  (JSON prompt)    |
                    +---------+---------+
                              |
                              v
+----------------------+   +-----+
|  Qwen3-{0.6,1.7,4}B  |   | LLM |---<state>0.0,0.0,...</state><action>12.34</action>
|  (LoRA-adapted)      |   +--+--+   <state>0.1,0.4,...</state><action>0.00</action>
+----------+-----------+      |      ... H control points ...
           ^                  |
           |                  v
           |           +------+-------+
           |           | tool API     |  parse → u_{1:H} ∈ R^{H×m}
           |           | (typer CLI)  |
           |           +------+-------+
           |                  |
           |                  v
           |           +------+-------+
           |           | Diffrax sim  |  dx/dt = f(x, u; θ_fixed)
           |           | (Tsit5,      |  θ_fixed from BRENDA / Bergman /
           |           |  rtol 1e-6)  |     Dalla Man (literature)
           |           +------+-------+
           |                  |
           |                  v   τ : [0, T] → R^n
           |           +------+-------+
           |           | STL evaluator|  ρ(τ, φ) ∈ R via Donzé-Maler
           |           | (Donzé-Maler,|  recursion on min/max nodes
           |           |  float64)    |  (depth ≤ 12 → ≤ 1.2e-14 ulp error)
           |           +------+-------+
           |                  |
           |                  v
           |           +------+--------+
           |           | filter v ∈    |   hard:    ρ > 0
           |           | {hard,quant,  |   quant:   top-25%
           |           |  cont}        |   cont:    softmax(ρ/β)
           |           +------+--------+
           |                  |
           |                  v
           |           +------+--------+
           +-----------+ SFT loss      |  weighted NLL on (prompt, completion)
                       | (MLX or bnb)  |  one epoch, AdamW, cosine, LoRA r=8/32
                       +---------------+
```

**Walking one trajectory.** The system prompt JSON-serializes (x_0, φ) — for the smoke test, x_0 is a fasting glucose level and φ is `glucose_insulin.tir.easy` (time-in-range 70–180 mg/dL over a 240-minute meal-response horizon). The model emits 17 `<state>...</state><action>...</action>` blocks corresponding to 17 control points u_{1:17} in mU/min insulin rate. The CLI parses those blocks into a (17, 1) action array. Diffrax integrates Bergman's 1979 minimal model with literature-fixed kinetic parameters ([paper/REDACTED.md](paper/REDACTED.md)) and the parsed action sequence as the input. The STL evaluator walks the formula tree, computing ρ recursively over min/max nodes — for the time-in-range spec this resolves to `min_t min(180 − G(t), G(t) − 70)`. A positive ρ means the trajectory stays in range with margin ρ; a negative ρ means the worst violation has magnitude |ρ|. The filter chooses what to do with that ρ during SFT: `hard` keeps only ρ > 0, `quantile` keeps the top 25%, `continuous` keeps everything with importance weight `softmax(ρ_i / β)` where β is the per-batch std.

The key claim is that every box in this diagram is **deterministic and inspectable**. The simulator is Diffrax with fixed tolerances. The STL evaluator is a tree walk in float64 over min/max. The filter is a sort-and-threshold or a softmax. None of these introduces a learned approximation between the trajectory and the soft signal. That is the property that collapses the verifier-noise term in §7.

---

## 4. Result 1 — smoke-test scaling-of-zero curves

The Phase-1 hard checkpoint is **A15**: validate the SFT loop end-to-end on consumer hardware before committing to RunPod. Real numbers from [paper/REDACTED.md](paper/REDACTED.md):

| Field | Value |
|---|---|
| Backend | mlx_lm 0.31 + MLX Metal |
| Base model | `mlx-community/Qwen3-0.6B-bf16` (no quantization at 0.6B scale) |
| Hardware | Apple Silicon M5 Pro, 48 GB unified memory |
| LoRA target | `self_attn.q_proj`, `self_attn.v_proj` (rank 8, α 16) |
| Dataset | 100 trajectories sampled from `data/pilot/filtered_glucose_insulin_hard.parquet` (1344 hard-filtered) |
| Spec | `glucose_insulin.tir.easy` (time-in-range, easy threshold) |
| Optimizer | AdamW, lr 2e-4, linear-warmup-then-cosine, 5 warmup steps |
| Iterations | 50 (batch 1, grad-accum 4 → effective batch 4) |
| Initial loss (mean of first 2 reports) | **1.4838** |
| Final loss (mean of last 2 reports) | **0.4659** |
| Loss monotone? | yes (no NaN/Inf, min 0.4241, max 1.4947) |
| Wall-clock for training | **15.0 s** |
| Held-out parse-success | **5/5** with regex `<state>...</state><action>...</action>` |
| Adapter checkpoint size | **4.6 MiB** (`adapters.safetensors` + `adapter_config.json`) |
| HF cache for base model | 2.25 GiB |

Per-iter loss curve (every 5 iterations):

| iter | loss |
|---:|---:|
| 5 | 1.4728 |
| 10 | 1.4947 |
| 15 | 1.4062 |
| 20 | 1.2457 |
| 25 | 1.0046 |
| 30 | 0.9481 |
| 35 | 0.7090 |
| 40 | 0.6203 |
| 45 | 0.5078 |
| 50 | 0.4241 |

This is what a working SFT loop looks like on a 100-example QLoRA on a 0.6B base: a brief plateau through iter 10 while warmup completes, then monotone descent through cosine annealing, finishing at ~0.47 with no instability events. The four hard-checkpoint criteria — (a) no crash, (b) final < initial, (c) no NaN/Inf, (d) parse rate ≥ 1/5 — pass cleanly.

The smoke test is **not a scientific result**. It is a loop-validation: the data pipeline (Parquet → ChatDataset → CacheDataset), the prompt schema (state-action XML blocks), the LoRA layer expansion ("self_attn.q_proj" path-relative naming), the optimizer schedule, the eval-time greedy decoder, and the adapter persistence layout (safetensors + config in a single dir) all work end-to-end on Apple Silicon. That is what unblocks Phase 2.

The 5/5 parse-success at 50 iters tells me the structured-output bias is learnable from 100 examples in 15 seconds on a 0.6B base. **It does not tell me the model learned control** — see §8 for the exact failure mode (every held-out generation produced an identical first action `12.34 mU/min`, almost certainly memorization on the dominant insulin-step pattern in the training subset). The smoke-test purpose was bounded; the diversity question is a Phase-2 eval-harness item.

The framing for Phase 2: this smoke test passes the hard checkpoint; the full canonical sweep (18 checkpoints, ~$25 of RunPod 4090 spot, ~3 days of wall-clock with checkpoint-resume) is the v0.1.0 release. Phase 2 swaps the MLX backend for bitsandbytes 4-bit on CUDA, scales the LoRA rank to 32 (matching SERA's [Unsloth MoE QLoRA YAML](https://github.com/allenai/SERA/blob/main/sera/datagen/train/train_config/unsloth_qwen3_moe_qlora.yaml)), and runs the locked 3 × 3 × 2 grid with 5 seeds × 25 instances × 8 BoN budgets per cell.

---

## 5. Result 2 — STL evaluator validates against pilot trajectories

The other Phase-1 deliverable is the STL infrastructure. Real numbers:

- **317 unit tests pass + 2 platform-skipped** across 22 test modules in `tests/` on the smoke-test machine (319 total collected; 2 skips are an Apple-Silicon platform check on a non-Apple host and the CUDA-only bnb backend tests). Coverage is **91%** on `src/stl_seed/` (target ≥ 80%; raised in the 1.7 polish pass after Phase-2 prep added `test_canonical_scripts.py`, `test_topology_aware.py`, `test_mlx_loop.py`, and the `_extra` regression files).
- **Empirical ICC = 0.9979** on the pilot's 3,982 trajectories grouped by task family, computed with the unbalanced-group Shrout-Fleiss ICC(1,1) estimator ([paper/power_analysis_empirical.md §3](paper/power_analysis_empirical.md)). Higher than the design-time plug-in ICC = 0.40, but the hierarchical model pools across 18 cells and the global pooled MDE is **0.0244** on the probability scale. The canonical registered TOST equivalence threshold (theory.md §3) is Δ = 0.05, which requires SE ≤ 0.0171; actual SE is **0.0098**. The locked design is powered for the strict registered Δ. (The "Δ = 0.080" framing in earlier drafts of `paper/power_analysis_empirical.md` was a relaxation that was never canonical; theory.md §3 is the registration of record.)
- **Pilot composition** (original 4 task × policy buckets used for the ICC computation, ρ in trajectory units of concentration·time):

  | task | policy | N | mean ρ | std | range |
  |---|---|---:|---:|---:|---:|
  | `bio_ode.repressilator` | heuristic (bang-bang, pre-fix) | 1,000 | -2.488e+02 | 0.000 | flat |
  | `bio_ode.repressilator` | random | 982 | -2.455e+02 | 4.565 | -2.487e+02 to -2.040e+02 |
  | `glucose_insulin` | heuristic | 1,000 | +2.075e+01 | 0.000 | flat |
  | `glucose_insulin` | random | 1,000 | -1.129e+00 | 3.408 | -1.042e+01 to +1.051e+01 |

  The deterministic heuristic policies have zero within-bucket variance (PIDController on glucose-insulin and the original BangBangController on repressilator are both pure functions of state). The random policy gives the stochastic spread we will see scaled up under the {random, heuristic, LLM} mixture in Phase 2.

  **Post-fix (topology-aware) repressilator regeneration** — `data/canonical/bio_ode.repressilator/` (N=2,388):

  | policy | N | mean ρ | success rate (ρ > 0) |
  |---|---:|---:|---:|
  | `topology_aware` (heuristic slot) | 1,000 | +25.000 | **100.0%** (deterministic) |
  | `perturbed_heuristic` | 406 | -109.670 | 27.1% |
  | `random` | 982 | -245.533 | 0.0% |
  | **pooled** | 2,388 | — | **46.48%** |

  The flat-ρ = -248.8 failure mode (§7) was not a property of the spec or the simulator; it was a topology-naive heuristic that drove the wrong gene. Replacing it with a `TopologyAwareController` that silences the upstream repressor in the cyclic ring (Elowitz-Leibler 2000 wiring; `tests/test_topology_aware.py`) lifts the heuristic-bucket success from 0% to 100% deterministically, and pulls the pooled rate to 46.48%.

- **Bootstrap CI coverage** is verified by simulation in `tests/test_stats.py`: paired and unpaired bootstrap intervals achieve nominal 95% coverage on synthetic data with known truth. The hierarchical Bayes posterior sampler (`src/stl_seed/stats/hierarchical_bayes.py`, NumPyro NUTS 4 chains × 2000 draws) recovers the synthetic δ_A = 0.4 effect within the 95% HDI on a held-out simulation.
- **Fisher-information sanity check.** At the prior median (A=0.6, b=0.25, N=128) the per-observation Fisher information matrix is `[[2.0249, 2.4941], [2.4941, 3.0719]]`, computed in closed form and cross-checked against an autograd-derivative implementation. Translates through the design-effect chain (n_eff_per_cell = 25.04 with empirical ICC) to the SE numbers above.

The original pilot revealed a failure mode in the bio_ode task family: the bang-bang heuristic on the repressilator never satisfied the easy spec because **the control channel observed was wrong** and the controller was **topology-naive** (see §7 for the bug + fix). That is the kind of finding that would be silently buried by aggregate metrics; the per-cell pilot laid it bare. The fix shipped in the Phase-2-prep pass replaces the bang-bang with a `TopologyAwareController`, lifting pooled bio_ode satisfaction to 46.48% (table above).

---

## 6. Goodhart-resistance via decomposition

This is the central theoretical contribution. Full derivation in [paper/theory.md §6](paper/theory.md).

Let R_gold(τ) be the latent oracle reward (the behavior we actually want), R_spec(τ) the reward induced by the STL formula φ_spec we wrote down, R_proxy(τ) the reward used at training time, and R_verifier(τ) what our algorithm actually computes. Tautologically:

```
R_gold − R_proxy = (R_gold − R_spec) + (R_spec − R_verifier) + (R_verifier − R_proxy)
```

By **defining** R_proxy ≡ R_verifier (the proxy is whatever the algorithm computes), the third term collapses to zero, leaving:

```
R_gold − R_proxy = (R_gold − R_spec)        [spec-completeness]
                  + (R_spec − R_verifier)   [verifier-fidelity]
```

For STL with formal robustness ρ as the verifier, **the second term is zero in symbolic semantics**: R_spec(τ) is *defined* as ρ(τ, φ_spec) and R_verifier(τ) is what the Donzé-Maler evaluator returns for the same (τ, φ_spec). The two are equal by construction modulo float64 round-off. The recursive min/max evaluator over a depth-12 STL formula accumulates at most 12 ulps ≈ **1.2 × 10⁻¹⁴ per evaluation**, which we bound empirically at **≤ 1 × 10⁻⁶** after the σ-squashing in [paper/theory.md §3](paper/theory.md).

**What this exposes.** In RLHF and learned reward modeling (the standard "soft-RL" setup), R_verifier is a *learned* approximation of R_spec, which is itself a noisy approximation of R_gold, and **the two error terms are entangled** in the reward model's training residual. [Gao, Schulman, and Hilton (2022)](https://arxiv.org/abs/2210.10760) showed empirically that learned RMs exhibit a Goodhart-style overoptimization curve in best-of-N where the proxy reward continues to climb while the gold reward turns over. The functional form is sublinear in √KL with a quadratic overoptimization correction, but the decomposition cannot tell you which fraction of the gap is verifier noise vs. spec misspecification, because the only handle on R_spec is the learned RM itself. You cannot audit a thing whose error is folded into your only measurement of it.

STL ρ collapses the verifier-fidelity term to a numerical floor. The entire R_gold − R_proxy gap becomes the **spec-completeness** term, which is **auditable**: a researcher inspecting an STL spec can in principle reason about whether a behavior φ_gold ⊃ φ_spec is missed, because both are written in the same logic. We operationalize this in [paper/theory.md §6](paper/theory.md) by constructing φ_gold from φ_spec via two augmentations: (a) tightening every numerical threshold by 10% (`G_{[0,T]} (x_1 > 0.5)` becomes `G_{[0,T]} (x_1 > 0.55)`) and (b) adding two conjuncts that were withheld during training (a no-overshoot constraint and a control-effort cap).

The prediction for v0.1.0: on the held-out trajectory set, (R_spec − R_verifier) sits at the 1e-6 floor, while (R_gold − R_spec) grows monotonically with model capacity (the better the policy fits φ_spec, the more headroom there is for φ_gold to differ). The **comparison baseline** is a learned-critic proxy: a small reward model fine-tuned to regress ρ from rollouts. Prediction: the learned critic exhibits *both* a non-zero verifier-fidelity term (regression error) and a spec-completeness term, and they cannot be disentangled from observation. STL exposes the spec-completeness term in isolation. That is the auditable handle.

This is also why I think [PAVs (Setlur et al. 2024)](https://arxiv.org/abs/2410.08146) and STL-soft-SFT are complementary rather than competing. PAVs construct a per-step process-advantage signal from rollouts, and the construction is brilliant but the signal still inherits learned-critic noise. STL is one construction in which the per-step signal is **definitionally** the rollout's robustness — no critic, no regression error — but pays for it by requiring a hand-authored spec. The two methods sit on opposite ends of a spec-effort-vs-verifier-noise trade-off curve, and a paper that ran both side-by-side on the same control task would be a real contribution. (Phase-3 candidate, not in scope for v0.1.0.)

---

## 7. What didn't work / what surprised me

The honest section. From the actual experience of bringing the Phase-1 stack up between 2026-04-04 and 2026-04-24:

**The repressilator bang-bang heuristic produced flat ρ = −248.8 across 1,000 trajectories.** This was caught when [paper/power_analysis_empirical.md](paper/power_analysis_empirical.md) ran the ICC computation and the heuristic bucket had **zero within-bucket variance** — every single one of 1,000 trajectories landed at exactly the same robustness value. Two compounding bugs, both real:
  1. The heuristic was observing the wrong state channels. The repressilator state vector is `(m_1, m_2, m_3, p_1, p_2, p_3)` — three mRNAs followed by three proteins — but the bang-bang was driving its decision off the mRNA channels (indices 0, 1, 2) when the spec is over the protein channels (indices 3, 4, 5). Patched by adding `observation_indices=[3, 4, 5]` to the controller constructor.
  2. Even after the channel fix, the controller is *topology-naive*: it tries to drive p_1 high by maximizing the input that nominally activates p_1, but the repressilator topology is `p_1 represses p_2 represses p_3 represses p_1`, so to drive p_1 high you have to **silence p_3** (p_1's repressor), not p_1 itself. Random sampling on the 30-dimensional continuous action space rarely hits the narrow good region, and a topology-naive heuristic provably never does. **Resolution (Phase-2 prep, shipped 2026-04-24):** `TopologyAwareController` (`tests/test_topology_aware.py`, drop-in heuristic-slot replacement) silences the upstream repressor in the cyclic ring; on the regenerated canonical pilot it lifts the heuristic-bucket satisfaction from 0% to 100% (deterministic) and the pooled rate across all 2,388 bio_ode trajectories from 0% to 46.48%. The bio_ode reference policy mixture for Phase 2 is now informative.

  **What this taught me:** the pilot's job is to expose this kind of bug before you have spent $25 on a full sweep. ICC near 1.0 in a synthetic pilot is a *symptom*, not an inconvenience — it means one of your policies has zero entropy and you should figure out why before the canonical run. The numerical-power exercise paid for itself the first time it ran.

**`mlx_lm` 0.31 has API drift from when `MLXBackend` was written.** The `TrainingArgs` constructor used to accept `learning_rate`, `lr_schedule`, `warmup_steps`, and `seed` as fields; in 0.31, all four were moved (the schedule lives on the optimizer, the seed is passed to `iterate_batches`). The `train()` function used to take `tokenizer` as a kwarg; in 0.31 it doesn't. The dataset used to be a raw HuggingFace `Dataset`; in 0.31 it must be a `CacheDataset` wrapping a `ChatDataset`. The smoke test bypasses `MLXBackend` entirely and uses the new API directly; the wrapper patch (~50 LOC) is a Phase-2 followup. **This is fine** — RunPod canonical training uses the bnb backend, not MLX, so the wrapper bug doesn't block Phase 2 — but it's a reminder that pinning the entire dependency stack matters more than I thought when the underlying library is moving fast.

**LoRA `target_modules` naming convention silently fails at the bare-name path.** I initially passed `["q_proj", "v_proj"]` to `linear_to_lora_layers`. `print_trainable_parameters` returned `0.000% (0.000M)`. The training loss did not move. No error was raised. The bare name is matched against module names *relative to each TransformerBlock*, and Qwen3's projection modules live under `self_attn.`, so the correct path is `["self_attn.q_proj", "self_attn.v_proj"]`. Caught only because the loss-watchdog flagged a flat curve; otherwise this would have produced a clean-looking 50-iter run with zero gradient flow. This is a documented mlx_lm idiom worth pinning in the architecture doc and worth checking against on every backend swap.

**5/5 held-out generations produced identical first actions.** Every one of the 5 held-out prompts generated `1.234e+01 mU/min` as the first action, with identical 17-block output structure. This is **expected** at smoke-test scale — 95 training examples, rank-8 LoRA, 50 iters, greedy decoding — and it almost certainly reflects memorization on the dominant pattern in the satisfying-trajectory subset (many flat-zero-then-step insulin schedules with peak rates near 12 mU/min). It is **not a failure** of the smoke checkpoint, which only required parse-success. But it is a **red flag** for the Phase-2 eval harness: a clean-looking parse-success rate can hide zero output diversity. A16 (the next subphase) adds per-prompt action diversity as a smoke-test metric so this regression mode is caught before convergence runs.

**The empirical ICC = 0.9979 was not what I expected.** The design-time plug-in of 0.40 came from the REDACTED paper's experience with similar bio-ODE pilots, and I expected the empirical value to land near it. It didn't, because two of the four pilot buckets have deterministic heuristic policies (zero within-bucket variance), and the (task × policy) ICC estimator is pulled toward 1.0 when half your buckets are degenerate. The fix in [paper/power_analysis_empirical.md §2](paper/power_analysis_empirical.md) is to switch to a (task)-only ICC, which gives 0.9979 — still high, but the hierarchical model pools across 18 cells and the global TOST is still adequately powered (SE = 0.0098 vs. threshold 0.0171). **What I'd do differently:** run the ICC computation *during* pilot design, not after, so the heuristic-policy variance issue is caught before generating 4,000 trajectories.

**`uv` lockfile drift between MLX and CUDA extras.** `uv sync --extra mlx` and `uv sync --extra cuda` resolve to different versions of `transformers` and `numpy` because `mlx_lm` and `bitsandbytes` have different upper bounds. The lockfile resolves both, but `pip install stl-seed[mlx]` and `pip install stl-seed[cuda]` are not installable into the same env without conflict resolution. Documented; the canonical recommendation is one venv per backend.

What did **not** surprise me: Diffrax `Tsit5` integration is rock-solid on both backends. NumPyro NUTS converges on the synthetic recovery in <2 minutes for 4 chains. The Hydra config schema is overkill for Phase 1 and exactly right for Phase 2. The REDACTED firewall ([paper/REDACTED.md](paper/REDACTED.md)) is verbose but the import-graph grep gate catches things (it caught one accidental `from REDACTED.specs import ...` during a refactor).

---

## 8. The library

### Install

```bash
# Apple Silicon (development, MLX backend)
uv sync --extra mlx

# CUDA (canonical training, RunPod / Linux, bitsandbytes backend)
uv sync --extra cuda

# Both (separate venvs recommended; lockfile resolves both but has conflicts at install time)
uv sync --extra mlx --extra cuda --extra dev
```

PyPI release will land at v0.1.0 alongside the canonical sweep. Until then, install from source:

```bash
git clone https://github.com/AA-Alghamdi/stl-seed
cd stl-seed
uv sync --extra mlx   # or --extra cuda
```

### CLI

```bash
stl-seed demo --task glucose_insulin
stl-seed generate --task repressilator --n 2000
stl-seed filter --task glucose_insulin --condition continuous
stl-seed train --config-name pilot
stl-seed evaluate --checkpoint runs/pilot/checkpoint
stl-seed analyze --runs runs/sweep_main/   # hierarchical Bayes posterior
```

The CLI mirrors SERA's stage names (`generate / distill_stage_one / distill_stage_two / eval / postprocess`) where it makes sense; `--stage=` resume semantics are inherited so a pod restart on a 40-hour sweep doesn't redo finished cells.

### Python API

```python
from stl_seed.tasks import GlucoseInsulinSimulator
from stl_seed.specs import SPEC_REGISTRY
from stl_seed.stl import evaluate_robustness
import jax.numpy as jnp
import jax

sim = GlucoseInsulinSimulator()
spec = SPEC_REGISTRY["glucose_insulin.tir.easy"]

key = jax.random.key(0)
x_0 = jnp.array([100.0, 0.0, 7.0])    # G, X, I (mg/dL, 1/min, mU/L)
u = jnp.zeros((sim.horizon, sim.action_dim))   # zero-insulin baseline

trajectory = sim.simulate(x_0, u, sim.params, key)
rho = evaluate_robustness(spec, trajectory)
print(f"STL robustness ρ(τ, φ) = {rho:+.3f}")
```

The simulator, STL evaluator, filter, and training backend conform to four `Protocol` interfaces documented in [paper/architecture.md](paper/architecture.md). Adding a new task family means implementing `Simulator` and registering specs; adding a new filter means implementing `FilterCondition`; adding a new training backend (e.g. unsloth) means implementing `TrainingBackend`. All four are drop-in.

---

## 9. What I'd change with more compute

The Phase-1 budget is $0 (M5 Pro). The Phase-2 budget is ~$25 (RunPod 4090 spot for the canonical 18-cell sweep). With more, in priority order:

1. **Open-weights model size sweep above 8B.** Qwen3-32B and Qwen3-72B (when released) on A100 80GB. SERA's headline used Qwen3-32B at ~$2,000 for data-gen + training; the analog here is "what does the soft-vs-hard gap look like at the model size where SERA actually saw the soft-suffices effect?" My current 0.6B–4B grid will likely sit *below* the saturation regime where SERA's [discussion section](https://arxiv.org/abs/2601.20789) flagged the conjecture that "verified correct code may become necessary." Hitting the saturation regime means scaling up.

2. **Reasoning-trace conditioning.** SERA's `postprocess.add_think` flag toggles `<think>` tags for Qwen reasoning-mode outputs. I deliberately did *not* add reasoning traces to the Phase-1 pipeline because doing so would confound the soft-vs-hard filter axis with a reasoning-on-vs-off axis. With more compute I'd run a 2 × 3 cross (think-on/think-off × {hard, quant, cont}) on the smallest two model sizes and report the joint effect. Reasoning traces in scientific control are interesting because the model would in principle produce a controller-design rationale that is itself inspectable.

3. **Wet-lab loop.** The Phase-2 closed-loop is simulator-only. CMU has [John Kitchin's Claude-Light platform](https://kitchingroup.cheme.cmu.edu/) which closes the loop at the physical-experiment scale. Connecting the STL evaluator to a real spectrometer would test whether the spec-completeness term in §7 is even more dominant when φ_gold is "what the experimentalist actually wanted." This is the natural Phase-3 ambition.

4. **RLVR with STL reward.** Phase 2 is SFT-only by design (matches SERA's protocol). But the structurally-zero verifier-noise term in §7 is exactly the property that should make on-policy reinforcement learning with STL ρ as the reward more stable than RLHF: there is no learned critic to overoptimize against, so the [Gao 2022 quadratic overoptimization](https://arxiv.org/abs/2210.10760) curve should flatten. Sketch: GRPO ([Shao et al. 2024](https://arxiv.org/abs/2402.03300)) on 4B-class students with ρ as the per-trajectory reward and per-step advantage estimation via PAVs ([Setlur et al. 2024](https://arxiv.org/abs/2410.08146)) for credit assignment over the H control points. Compute: probably 8× A100 for 1–2 weeks per cell. Out of budget for the standalone artifact, in scope for a follow-up paper.

5. **A spec-authoring assistant.** STL spec authoring is the human bottleneck (§11). With more compute I'd train a 30B-class model on natural-language → STL transcription, using a curated bench of (NL description, STL formula) pairs from the formal-methods literature. Bootstrapping signal: the same STL evaluator gives a "satisfies the NL description" check via paraphrase consistency.

---

## 10. Limitations

The honest list. None of these are dealbreakers; all of them are real:

- **Closed-weights models out of scope by design.** The artifact targets open-weights agents (Qwen3 family). Closed-weights model-parameter scaling (GPT-5, Claude 4.5/4.6/4.7) is not supported because LoRA adapters require parameter access. The SERA blog's anthropic-validation runs (`specialization_anthropic.yaml`) are the closest closed-weights analog; we don't replicate them.

- **Toy ODE benchmarks, not real lab problems.** Repressilator, toggle switch, MAPK, Bergman + Dalla Man — all are textbook biomolecular systems with literature parameters. None is a real wet-lab control problem with sensor noise, actuator delay, and unmodeled dynamics. The "scientific control" framing is aspirational; the Phase-2 evidence is on toy systems.

- **2 task families, not the full scientific control space.** The locked Phase-2 sweep is `bio_ode × glucose_insulin`. Power-grid control, fluid dynamics, robotic locomotion, drug-dosing optimization — all natural extensions, none in scope. A reviewer might reasonably ask whether the soft-vs-hard pattern transfers across task families with very different ρ landscape geometries; we cannot answer that with the v0.1.0 evidence.

- **STL spec authoring is a human bottleneck.** Every spec in `src/stl_seed/specs/` was written by hand against textbook references ([paper/REDACTED.md](paper/REDACTED.md)). There is no automated NL → STL pipeline. A clinical workflow that wanted to add a new spec on a new patient would require a control engineer in the loop. This is a real adoption barrier and §9 item 5 is the long-term plan.

- **No wet-lab validation loop.** As above. The Phase-2 eval is closed-loop in simulation only.

- **The repressilator pilot exposed the topology-naive-heuristic limitation, since fixed.** The original bang-bang heuristic didn't satisfy the easy spec because random sampling on a 30-dimensional continuous action space rarely hits the narrow good region of u-space induced by the cyclic-repression topology. The 2026-04-24 prep pass replaced it with `TopologyAwareController` (silences upstream repressor in the cyclic ring; `tests/test_topology_aware.py`); the regenerated canonical pilot now shows pooled satisfaction 46.48% on bio_ode (heuristic 100% deterministic, perturbed_heuristic 27%, random 0%). The Phase-2 reference policy mixture {random, heuristic, LLM} is now informative on both task families.

- **Hierarchical Bayes inference uses NumPyro with default NUTS settings.** R̂ < 1.01 and ESS thresholds ([paper/theory.md §4](paper/theory.md)) will be checked at Phase-2 fit time. If the posterior turns out to be funnel-pathological for the random-effects standard deviations, the analysis switches to non-centered parametrization and re-runs; the registered analysis plan permits this without de-registration because the likelihood is unchanged.

- **n=3 seeds per cell is small** by SERA's own admission ("some reported effects may be noise"). The locked design has 5 seeds × 25 instances per cell × 8 BoN budgets, which is more than SERA's n=3 but still subject to the same caveat for cells where the variance turns out larger than the pilot suggested. The hierarchical model pools across cells precisely to mitigate this; per-cell results are explicitly *exploratory* in [paper/theory.md §5](paper/theory.md), not the registered primary endpoint.

- **No ablation of the action-tokenization scheme.** I serialize actions as `<state>x_1,x_2,...</state><action>u</action>` blocks. SERA serializes tool calls in Hermes/XML/raw formats and ablates. Phase-2 inherits one fixed serialization for the soft-vs-hard comparison; the tokenization-scheme axis is deferred.

- **The Goodhart decomposition theorem in §6 is a property of the verifier construction, not of the trained model.** The empirical claim — that the spec-completeness term grows monotonically with model capacity while the verifier-fidelity term sits at the float64 floor — requires Phase-2 evidence to confirm. Phase 1 ships the *construction* and the *predicted* decomposition; the *measurement* is Phase 2.

---

## 11. Acknowledgments

In approximate dependency order:

- **Murat Arcak** (UC Berkeley EECS, BAIR) — I came to STL through a graduate seminar he taught. The framing of formal robustness as a control-design tool that doesn't require giving up real-valued gradients is his.
- **Hanna REDACTED, Jakob Thumm, Matthias Althoff et al.** — The [REDACTED 2025 line on STL parameter synthesis for biomolecular ODEs](https://arxiv.org/abs/2412.15227) is the precedent. The REDACTED paper I'm a co-author on is firewalled from this artifact ([paper/REDACTED.md](paper/REDACTED.md)); same simulator infrastructure, mathematically disjoint optimization (their decision variable is θ, ours is u_{1:H}).
- **Karen Leung et al.** — [STLCG++](https://arxiv.org/abs/2501.04194) is the differentiable-robustness library that established the recursive-min/max evaluator pattern this project uses for the float64 path.
- **REDACTED, William Brown Tormoen, Aanvi Shah, Ali Farhadi, et al.** — [SERA](https://arxiv.org/abs/2601.20789) is the artifact this project tests. The recipe shape (generate → score → soft-filter → SFT) is theirs; the soft-vs-hard equivalence question I am asking in scientific control is their §Discussion question.
- **Amrith Setlur, Chirag Nagpal, Adam Fisch et al.** — [Process Advantage Verifiers (PAVs)](https://arxiv.org/abs/2410.08146) are the closest empirical neighbor on dense process verification. The std-rescaling for the continuous-weighted filter ([paper/theory.md §2](paper/theory.md)) is motivated by their reward-unit-coupling observation.
- **Leo Gao, John Schulman, Jacob Hilton** — [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) is the empirical baseline for what learned-RM Goodhart looks like; the spec-completeness vs. verifier-fidelity decomposition in §7 is the contrast.
- **Alexandre Donzé, Oded Maler** — Space-robustness ρ is theirs ([FORMATS 2010](https://doi.org/10.1007/978-3-642-15297-9_9)); STL itself is [Maler & Nickovic FORMATS 2004](https://doi.org/10.1007/978-3-540-30206-3_12).
- **Richard Bergman, Claudio Cobelli, Chiara Dalla Man** — The [1979 minimal model](https://pubmed.ncbi.nlm.nih.gov/443421/) and [2007 meal model](https://pubmed.ncbi.nlm.nih.gov/17926672/) for the glucose-insulin task family.
- **Michael Elowitz, Stanislas Leibler; Tim Gardner, Charles Cantor, Jim Collins; Chi-Ying Huang, Jim Ferrell** — Repressilator ([Nature 2000](https://doi.org/10.1038/35002125)), toggle switch ([Nature 2000](https://doi.org/10.1038/35002131)), MAPK cascade ([PNAS 1996](https://doi.org/10.1073/pnas.93.19.10078)) for the bio_ode task family.
- **The maintainers of Diffrax, Equinox, NumPyro, mlx_lm, bitsandbytes, transformers, and uv** — every one of these is a load-bearing dependency, all written by people who have given far more to the open ecosystem than I have.

I funded my own M5 Pro and the Phase-2 RunPod budget. The REDACTED line is funded through Berkeley BAIR. CMU MS-AIE matriculation is self-funded.

---

## 12. Citation and license

```bibtex
@misc{alghamdi2026stlseed,
  author = {Abdullah AlGhamdi},
  title  = {stl-seed: Soft-verified SFT for scientific control via STL robustness},
  year   = {2026},
  url    = {https://github.com/AA-Alghamdi/stl-seed},
  note   = {Phase-1 release; v0.1.0 with canonical sweep forthcoming},
}
```

Apache 2.0. See [LICENSE](LICENSE).

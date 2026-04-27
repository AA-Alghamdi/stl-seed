# Coding-Agent Task Cell — Design

A minimal task cell that bridges `stl-seed` from biomolecular ODE control to LLM coding agents, so the artifact's central claim — *STL robustness as a soft verifier for iterative LLM agents* — can be demonstrated outside its native ODE habitat. This is a design document, not an implementation; the implementation is scoped at the end as a 2-3 day TODO.

Author: Abdullah AlGhamdi. Date: 2026-04-26.

## 1. Problem framing and motivation

The `stl-seed` artifact today ([README.md](../README.md)) is built on five biomolecular ODE families. Its abstract claim is more general: STL robustness ρ is a *formal soft verifier* — continuous and differentiable in the simulator, zero-residual against its own spec by construction ([theory.md §"Goodhart decomposition theorem"](theory.md)). This claim only carries weight if it ports to a domain where the natural soft verifiers are *not* formal — where the practitioner has to hand-engineer one. SERA's coding-agent setting (Shen et al. 2026, \[arXiv:2601.20789\]; [allenai/SERA](https://github.com/allenai/SERA)) is the canonical such case: SERA's verifier is the line-overlap recipe `r = |P_2 ∩ P_1| / |P_1|` between two rollout patches, accepted into training when `r ≥ τ` for thresholds `τ ∈ {0, 0.25, 0.50, 0.75, 1.0}`. Their ablation finds all thresholds perform similarly, which is itself the hypothesis we want to test in our framework: is verifier-density (soft vs hard) actually a degenerate axis once the verifier is *formally constructed*?

The cell adds a single dataset, a single edit-vocabulary, and a small set of STL specs over coding-agent trajectories. It does not aim to compete with SERA on SWE-Bench; it aims to demonstrate that the same Donzé-Maler evaluator and the same 9 samplers run unmodified against a coding-agent simulator, with a soft-vs-hard comparison whose verifier-fidelity term is provably zero.

## 2. The simulator analog

In bio-ODE the simulator is `Sim_ODE: (x_0 ∈ R^n, u_{1:H} ∈ R^{H×m}) → Trajectory`, where `Trajectory.states` has shape `(T, n)` and is the integrated ODE solution ([\_trajectory.py](../src/stl_seed/tasks/_trajectory.py)). The coding-agent analog is

`Sim_Code: (c_0 ∈ Σ*, e_{1:H} ∈ V^H) → Trajectory_code`

where `c_0` is the initial buggy source, `e_h` is a discrete edit drawn from a finite vocabulary `V` (Section 4), and `Trajectory_code` packages a sequence of intermediate code states `c_1, ..., c_H` along with a *measurement vector* `m_h ∈ R^d_meas` per step. Concretely:

```
Trajectory_code:
  states  : Float[Array, "T d_meas"]   # measurement vector at each step
  actions : Int[Array,   "H"]          # discrete edit indices into V
  times   : Float[Array, " T"]         # step indices, 0..H, in unit "edit steps"
  meta    : { n_apply_failures, n_test_timeouts, ... }
```

The measurement vector `m_h` for a candidate code state `c_h` is computed by a *deterministic build-and-measure callback* `Measure: Σ* → R^d_meas`. The d_meas channels for the prototype:

m_h\[0\] = test_pass_rate ∈ \[0, 1\] fraction of unit tests that pass m_h\[1\] = lint_violation_count ∈ Z_≥0 ruff/pyflakes count m_h\[2\] = type_check_pass ∈ {0, 1} pyright/mypy boolean m_h\[3\] = ast_parse_ok ∈ {0, 1} 1 iff `ast.parse(c_h)` succeeds m_h\[4\] = num_new_imports_vs_c0 ∈ Z_≥0 |imports(c_h) \\ imports(c_0)| m_h\[5\] = patch_lines_changed ∈ Z_≥0 `git diff --stat` total

These are all measurable on a sandboxed subprocess against the dataset's test runner. The state vector lives in `R^d_meas` so the existing STL evaluator ([evaluator.py](../src/stl_seed/stl/evaluator.py)) works unchanged: it indexes `states[:, c]` for channel `c` and returns `states[:, c] - threshold` per the `_gt`/`_lt` predicate convention used by every spec module ([bio_ode_specs.py](../src/stl_seed/specs/bio_ode_specs.py) lines 76-92).

**Differentiability.** `Measure` is a black-box subprocess call (compiler, linter, test runner). It is *not* differentiable in `e_h`. The simulator therefore lacks the smooth `∂ρ/∂u` channel that gradient-guided sampling exploits in the bio-ODE family. This rules out the `STLGradientGuidedSampler`, the `HybridGradientBoNSampler`, and the `CMAESGradientSampler` cells of the existing 9-sampler grid. We *do not* manufacture a differentiable surrogate (e.g., a learned code-quality regressor whose gradient stands in for `∂ρ`); doing so would re-introduce a verifier-fidelity term against a learned model, defeating the cell's central comparison. The honest finding is that gradient guidance fundamentally cannot run on this simulator family — and that is itself a structural distinction between bio-ODE control and coding-agent control worth naming in the paper. The discrete samplers (`StandardSampler`, `BestOfNSampler`, `ContinuousBoNSampler`, `BeamSearchWarmstartSampler`, `RolloutTreeSampler`, `HorizonFoldedSampler`) all run unmodified — see Section 5.

**Determinism / sentinel policy.** When `Measure` fails (build error, test runner timeout, subprocess crash) the policy mirrors the bio-ODE NaN guard ([bio_ode.py `_sanitize_states`](../src/stl_seed/tasks/bio_ode.py)): replace `m_h` with a sentinel "all-fail" vector `(0, 9999, 0, 0, 0, 0)`, increment the failure counter in `meta`. This keeps `ρ` finite and signed (negative on failure) and never NaN. Test timeout is hard-capped at 30 s per evaluation; total wall-time per trajectory is `H × 30 s ≤ 6 min` at `H = 12`.

## 3. The dataset

I considered four candidates and recommend **HumanEval mutated** (Chen et al. 2021, \[arXiv:2107.03374\]) as the primary cell, with a built-from-scratch synthetic toy as fallback if HumanEval-mutated does not yield a reasonable bug-density.

| Candidate                                | Size         | License    | Test runner ready? | Pedagogical clarity |
| ---------------------------------------- | ------------ | ---------- | ------------------ | ------------------- |
| HumanEval (Chen 2021) + mutations        | 164 problems | MIT        | Yes (in-paper)     | Highest             |
| BugsInPy (Widyasari 2020)                | 493 bugs     | MIT        | Heavy (Docker)     | Moderate            |
| CodeNet defects subset                   | ~thousands   | Apache 2.0 | Heavy              | Low (CP idioms)     |
| MBPP+ (Liu 2023) + mutations             | 378 problems | MIT        | Yes                | High                |
| Synthetic toy (~30 hand-written defects) | ~30 problems | n/a        | Trivial            | Highest             |

**Recommendation: HumanEval-mutated, scope to ~50 problems.** Rationale:

1. *Lifecycle.* Each HumanEval problem ships with a `check(candidate)` function that runs the test suite in \<1 s, no Docker, no fixture juggling. The `Measure.test_pass_rate` channel is a one-liner: catch exceptions from `check`, count assertion successes vs total via test instrumentation.
1. *Mutation pipeline.* We generate the buggy starting `c_0` by applying a single canonical mutation drawn from a fixed catalog: `{flip_comparison_op, off_by_one_in_range, swap_args, negate_boolean, replace_+_with_-}`. This catalog is small (5 entries), each mutation is semantics-changing for almost all HumanEval functions, and the fix-set has high overlap with the action vocabulary in Section 4. The mutation script is ~50 LOC of `ast` rewriting and is deterministic given a `(problem_id, mutation_type, seed)` triple.
1. *Difficulty calibration.* Per [theory.md §"Failure mode taxonomy" FM1](theory.md), the cell wants `Pr[some sampler succeeds at N=128] ∈ [0.3, 0.7]`. HumanEval at single-line mutations is approximately at this band for small open-weight models per the SWE-Bench Lite difficulty curve; if a pilot of `~10` problems × `8` samples shows `Standard` saturating outside the band, we either tighten or relax the mutation catalog.
1. *License.* MIT, redistributable. Mutated copy can be checked into `data/coding_he_mutated/`.
1. *Size.* 50 problems × ~12 H-step trajectories × ~50 wall-seconds-per-trajectory ≈ 8 wall-CPU-h for a full sampler×problem grid at `N=8`, well within an afternoon's budget on a single M5 Pro.

The synthetic toy is the contingency. If for some reason HumanEval-mutated produces zero diversity in `Measure` outputs (every problem either passes or fails outright with no intermediate test_pass_rate), we fall back to ~30 hand-written Python functions with explicit multi-test suites (3-5 tests per function) so `test_pass_rate` is genuinely continuous in `[0, 1]` rather than binary. This contingency is documented in the implementation checklist.

The deliberate non-choices: BugsInPy is too heavy (Docker fixtures per bug, 30 GB of checkouts); CodeNet defects pull in competitive-programming idioms that small LLMs struggle with for unrelated stylistic reasons; MBPP+ is a defensible alternative but HumanEval has the cleaner test-instrumentation hook, so we pick the one with less infrastructure risk.

## 4. The action vocabulary

The vocabulary `V` must be (a) finite for beam-search and rollout-tree, (b) expressive enough that some `e_{1:H} ∈ V^H` actually fixes the bug in a non-trivial fraction of HumanEval-mutated problems, (c) plausible as something a small open-weight LLM can emit token-by-token without fine-tuning. The candidates I considered:

1. *Free-form code edit.* The LLM emits a complete replacement file, scored by `Measure`. The action space is essentially all of `Σ*`, so `|V| = ∞`. This is the SERA-faithful action space but kills beam-search and rollout-tree (no enumeration possible). **Rejected.**
1. *Patch-type vocabulary `V_patch` (≈ 50 entries).* Tokens like `flip_comparison_op@line_N`, `add_null_check@line_N`, `swap_args@line_N`, `replace_op_+_with_-@line_N`, with `line_N` ranging over the function body. **Recommended.** Discussion below.
1. *Token-level edit DSL.* Roughly `{insert_token(t, pos), delete_token(pos), replace_token(t, pos)}` with `t` ranging over a Python token vocabulary. This is finer-grained but `|V|` blows up with the token alphabet, and most edits are syntactically catastrophic. **Rejected.**

**Recommendation: `V_patch`, structured as two factors.** Each edit is a pair `(operator, location)`:

operator ∈ V_op = { flip_comparison_op, # \< ↔ > ↔ \<= ↔ >= flip_boolean_op, # and ↔ or, not insertion/removal off_by_one_inc, off_by_one_dec, swap_first_two_args, replace\_+_with_-, replace\_-_with_+, replace\_*_with_+, replace\_/_with_*, negate_return, add_zero_check, # `if x == 0: return 0` shim before main expr null_op, # explicit no-op (lets the policy "wait") revert_to_c0, # rollback } location ∈ V_loc = { line_1, line_2, ..., line_L }

with `L` capped at the maximum function-body length in the dataset (≈ 30 lines after parsing the HumanEval-mutated set, padded). `|V_op| = 13`, `|V_loc| = 30` → `|V| = 390`. Edits that are syntactic no-ops at a given location are not pruned at sampling time (cheaper to let `Measure` see them and report `ast_parse_ok = 1, test_pass_rate unchanged`), but the beam-search expansion deduplicates by post-edit code hash (caching `Measure` outputs per `(c_h, e_h)` pair lets repeated proposals be free).

This vocabulary covers every mutation in the catalog of Section 3 by construction (each catalog mutation has an inverse in `V_op`), so the satisfying sequence *exists in the vocabulary* for every HumanEval-mutated problem. This is the same structural design choice as the `bio_ode.repressilator.easy` task ([README.md line 12](../README.md)) — vocabulary tuning is transparent in code rather than hidden in tuning. The cell is therefore not a free win against beam-search; it is a controlled comparison of how each sampler navigates a vocabulary that provably contains a satisfier.

The `null_op` and `revert_to_c0` entries are deliberate. `null_op` lets the agent pause without "eating a budget step" on a syntactic dead-end; `revert_to_c0` is a panic button that the rollout tree sampler in particular uses to escape local minima ([rollout_tree.py](../src/stl_seed/inference/rollout_tree.py)).

## 5. STL specs

Three specs of escalating difficulty, each composed exclusively from `Always`, `Eventually`, n-ary `And`, and predicate-level `Negation` per the firewall §C.1 convention enforced in [specs/__init__.py](../src/stl_seed/specs/__init__.py). Channel indices follow the measurement vector ordering of Section 2 (m\[0\] = test_pass_rate, m\[1\] = lint_violations, m\[2\] = type_check_pass, m\[3\] = ast_parse_ok, m\[4\] = num_new_imports, m\[5\] = patch_lines_changed).

**Easy: `coding.fix.easy`.** formula: `F_[0,H] (test_pass_rate >= 0.5)` meaning: at some debugging step, more than half of unit tests pass. rationale: smooth eventual reachability — every sampler should be able to find an edit that flips a single mutation back at *some* step. If a sampler fails this, the failure is structural (vocabulary or simulator) not an STL artifact.

**Medium: `coding.fix.medium`.** formula: `G_[H/2, H] (test_pass_rate >= 0.8) AND G_[0, H] (lint_violations <= LINT_INITIAL)` meaning: from the back half of the trajectory onward, ≥80% tests pass; AND lint violations never exceed the initial buggy code's count. rationale: a *sustained* high pass-rate (the agent must not break the fix in later steps) plus a no-regression guard on a separate measurement channel — exactly the conjunction-of-tracking + safety pattern the `bio_ode.toggle.medium` spec uses ([bio_ode_specs.py lines 231-289](../src/stl_seed/specs/bio_ode_specs.py)). `LINT_INITIAL` is computed per problem at simulator-construction time, not hard-coded, mirroring how `TOGGLE_HIGH_NM` is per-task.

**Hard: `coding.fix.hard`.** formula: `F_[0, H] (test_pass_rate >= 1.0) AND G_[0, H] (num_new_imports = 0) AND G_[0, H] (ast_parse_ok = 1)` meaning: full pass at some step, no new imports introduced anywhere along the trajectory, and every intermediate state must be a parseable Python file. rationale: import-discipline is the coding-agent analog of the bio_ode safety constraint `G_[0,T] (x1 < TOGGLE_UNSAFE_NM)`. The reach-then-hold-shape mirrors `bio_ode.mapk.hard`'s reach-then-settle structure but on a different channel pairing. The `ast_parse_ok` clause rules out "lucky" trajectories that emit syntactically broken code in early steps — every intermediate state must be a real code state.

The robustness `ρ(spec, traj)` is computed by the existing [`stl_seed.stl.evaluator.compile_spec`](../src/stl_seed/stl/evaluator.py) unchanged. The verifier-fidelity term `R_spec - R_verifier` therefore inherits the same depth-bounded float64 round-off floor (≤ 4 ulp ≈ 4×10^-16, [theory.md §"Goodhart decomposition theorem"](theory.md)) that the bio-ODE specs achieve. This is the central transferable property: the moment the measurement vector and the spec are both fixed, the verifier is *constructed*, not learned, and its fidelity term collapses to a numerical floor — independent of whether the underlying trajectory is an ODE solve or a sequence of `git apply` calls.

## 6. Sampler-by-sampler integration

| Sampler                      | Runs?                              | Notes                                                                                                                                                                                                                                                                                                                                                                     |
| ---------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `StandardSampler`            | Yes                                | Categorical over `V` (390 entries) at each step. The LLM's logits come from a small open-weight model (Qwen3-0.6B is the default in the existing real-LLM cell, [real_llm_comparison.md](real_llm_comparison.md)) projected onto `V` via the same nearest-vocabulary embedding-lookup trick used by `LLMProposal` ([protocol.py](../src/stl_seed/inference/protocol.py)). |
| `BestOfNSampler`             | Yes                                | N parallel rollouts, pick argmax-ρ. Same projection.                                                                                                                                                                                                                                                                                                                      |
| `ContinuousBoNSampler`       | Yes                                | Softmax-weighted aggregation over rollouts in the *trajectory* dimension. Discrete actions are fine; only the score is continuous.                                                                                                                                                                                                                                        |
| `BeamSearchWarmstartSampler` | Yes — natural fit                  | The B×K sweep at each step is \`B ×                                                                                                                                                                                                                                                                                                                                       |
| `RolloutTreeSampler`         | Yes                                | Tree expansion over the discrete vocabulary; the cell's natural high-budget sampler.                                                                                                                                                                                                                                                                                      |
| `HorizonFoldedSampler`       | Yes (degraded)                     | The horizon-fold trick is gradient-based ([horizon_folded.py](../src/stl_seed/inference/horizon_folded.py)). On a non-differentiable simulator it degrades to BoN with horizon-staggered re-sampling. We document this honestly.                                                                                                                                          |
| `STLGradientGuidedSampler`   | **No** — by structural distinction | No `∂ρ/∂e_h` exists in any meaningful sense. `Measure` is a subprocess. We do *not* manufacture a learned surrogate; instead we report this as a finding ("gradient guidance fundamentally requires a differentiable simulator; coding-agent simulators are not"). The cell highlights the bio-ODE-vs-coding distinction.                                                 |
| `HybridGradientBoNSampler`   | **No**                             | Same.                                                                                                                                                                                                                                                                                                                                                                     |
| `CMAESGradientSampler`       | **No**                             | Same.                                                                                                                                                                                                                                                                                                                                                                     |

Of 9 samplers, 6 run cleanly, 1 runs degraded, 2 are excluded by structural constraint. The exclusion is not a flaw — it is the headline finding the bridge produces. SERA's soft-verifier is also non-differentiable; their setup faces the same constraint, and they handle it by *not* running gradient methods at all. Our cell makes this distinction explicit and quantifiable (`coding.fix.{easy,medium,hard}` ρ-curves vs `glucose_insulin.tir.easy` ρ-curves on the same sampler grid).

## 7. Connection to SERA

| Axis                  | SERA                                          | This cell                                                                                              |
| --------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Soft signal           | \`r =                                         | P_2 ∩ P_1                                                                                              |
| Construction          | Hand-engineered overlap recipe                | Formal STL semantics over measurement channels                                                         |
| Fidelity term         | Unmeasurable (no `R_spec` to compare against) | `R_spec - R_verifier` ≤ 4 ulp by construction                                                          |
| Filtering at training | `r ≥ τ` threshold filter                      | Same SERA recipe (hard / quantile / continuous, [theory.md §"Soft-filtered SFT formalism"](theory.md)) |
| Action space          | SWE-agent tool calls, ≤115 steps              | `V_patch`, `H = 12` steps                                                                              |
| Dataset               | 200k synthetic from 121 codebases             | 50 HumanEval-mutated problems                                                                          |

SERA's central empirical finding (§Ablation: "all verification thresholds perform similarly, suggesting verification primarily determines which samples to include versus exclude") is exactly the H1 equivalence claim in [theory.md §"Pre-registered hypotheses"](theory.md): hard, quantile, and continuous filters yield within-Δ-equivalent SFT outcomes. Their finding is on a learned- overlap signal; our cell tests whether the same equivalence holds when the signal is *formally constructed*. The two failure modes are different in kind: SERA could fail equivalence because their overlap recipe is too lossy at low-r; we could fail equivalence because STL ρ is too discriminative at high-ρ. Both are testable and both are publishable.

The unique handle this cell offers is the *separability* of the verifier-fidelity term. SERA cannot ablate `R_spec - R_verifier` because they have no `R_spec` other than their recipe. We can: by training the same SFT pipeline against the *same trajectories* with two different specs (`coding.fix.medium` and `coding.fix.medium` with the lint clause dropped), the spec-completeness contribution can be empirically isolated, exactly as in the bio-ODE goldspec construction ([theory.md §"Goodhart decomposition theorem"](theory.md)). This is a contribution SERA cannot make on its own — and it is sharper to make on the same coding domain than to argue it transfers from biomolecules.

## 8. Honest scope

This cell is *not* a coding-agent system. It does not:

- Compete with SERA, SWE-agent, or any production coding-agent on SWE-Bench Verified.
- Train a LoRA on coding rollouts (the SFT half of [theory.md](theory.md)'s pre-registered sweep is bio-ODE-only; this cell is inference-time only at first).
- Cover repository-scale edits, multi-file changes, or anything beyond single-function defects.
- Exercise the full SWE-agent tool surface (no shell, no test-running-as-a-tool, no file tree navigation).

It *does* demonstrate that the same nine-sampler grid, the same Donzé-Maler evaluator, and the same firewall-compliant STL spec form run unmodified against a coding-agent simulator on HumanEval-mutated bugs, with six samplers producing finite ρ-curves and three ruled out by a structural distinction (non-differentiable simulator) that is itself a contribution to name.

The wider claim the cell unlocks is this: every result in the existing [`unified_comparison_results.md`](unified_comparison_results.md) — "different samplers dominate different task structures" — is now demonstrably not specific to ODEs. A coding-agent task with a narrow vocabulary attractor (a single satisfying edit pattern) should be dominated by beam-search warmstart by the same structural argument that dominates `bio_ode.repressilator.easy`. The cell is the falsification opportunity: if beam-search *doesn't* dominate `coding.fix.hard`, the "structural-search vs continuous-search" distinction is narrower than the headline claims.

## 9. Caveats and failure modes

Five modes ranked by ex-ante probability with detection criteria.

**FM1 (~30%): Vocabulary too narrow.** No `e_{1:H} ∈ V_patch^H` actually fixes a meaningful fraction of HumanEval-mutated problems, because real bugs need multi-token edits the vocabulary can't compose. *Detect:* before any sampler comparison, run `BeamSearchWarmstart` with `B = |V|` (exhaustive width-1 enumeration over the next edit) for `H = 1` step on each problem; require ≥30% of problems to have *some* one-edit fix. If \<30%, expand `V_op` to ~25 entries with literature-grounded mutations from CodeNet's defect taxonomy; do not silently move on.

**FM2 (~25%): Differentiable samplers unfixable, halving the comparison.** This is by design (Section 6), but if the resulting bar chart looks asymmetric — only six bars per task instead of nine — readers may mistake the absence for a failure. *Detect:* not a runtime issue; a presentation issue. Mitigation is to plot the bio-ODE 9-sampler grid and the coding 6-sampler grid side-by-side with a clear "N/A by structural constraint" annotation on the three excluded samplers, and to call out the distinction in the caption.

**FM3 (~25%): Test-pass-rate is too coarse.** HumanEval problems often have 3-5 unit tests per function, so `test_pass_rate ∈ {0, 0.25, 0.5, 0.75, 1.0}` is a 5-level discrete signal. STL temporal patterns over 5-level signals give a low-resolution `ρ`; the soft-vs-hard comparison may collapse to "trivially equivalent because both filters see the same coarse landscape." *Detect:* run a pilot of 100 trajectories per problem and measure the entropy of the trajectory- level `ρ` distribution; require entropy ≥ 2 nats (i.e., the distribution actually has spread). If not, fall back to the synthetic toy with denser test suites (Section 3 contingency).

**FM4 (~15%): 50 problems too small for statistical claims.** The bio-ODE sweep budgets 25 instances × 5 seeds × 8 BoN budgets per cell ([theory.md §"Statistical analysis plan"](theory.md)); 50 HumanEval problems × 3 seeds × 8 budgets gives 1200 trials per (sampler, spec) cell. *Detect:* this is an analysis-time problem, not a detection-time one. The hierarchical model from [theory.md §"Statistical analysis plan"](theory.md) absorbs the smaller sample size cleanly via partial pooling across problems; the right framing is "exploratory cross-domain finding" rather than "claim of equivalence on coding," and the abstract should say so.

**FM5 (~10%): Mutation catalog leaks the answer.** If `V_op` contains the exact inverse of every mutation in the dataset's mutation script (as the recommendation in Section 4 *deliberately arranges*), then beam-search can trivially win by enumeration. *Detect:* this is a feature, not a bug, in the same way `k_per_dim=5` containing `u=(0,0,1)` is a feature of `bio_ode.repressilator.easy` ([README.md line 12](../README.md), the structural-search distinction). The honest framing is already in place: "the vocabulary is constructed to contain the satisfier; the contribution is the sampler-wise navigation comparison, not a free win." This must be in the README and the abstract, not just the design doc.

## 10. Implementation checklist (2-3 day estimate)

Day 1: simulator + dataset.

- `src/stl_seed/tasks/coding_he.py`: implements `CodingHumanEvalSimulator` conforming to the existing `Simulator` protocol ([bio_ode.py lines 119-148](../src/stl_seed/tasks/bio_ode.py)). Constructor takes a problem id, a mutation seed, a vocabulary `V`. `simulate(...)` applies the edit sequence step-by-step and runs `Measure` after each, returning a `Trajectory` with the measurement vector as `states`.
- `src/stl_seed/tasks/coding_he_params.py`: per-problem metadata (problem id, mutation type, initial lint count `LINT_INITIAL`).
- `data/coding_he_mutated/`: ~50 mutated problems checked in, generated by `scripts/generate_he_mutations.py` (deterministic given a seed list).
- `src/stl_seed/tasks/coding_he_vocabulary.py`: defines `V_patch` and the `apply_edit` AST rewrite logic (~150 LOC of `ast.NodeTransformer`).
- Self-test: `python -m stl_seed.tasks.coding_he` runs a single problem with a known fix sequence and asserts `test_pass_rate[H] == 1.0`.

Day 2: specs + sampler integration.

- `src/stl_seed/specs/coding_specs.py`: registers `coding.fix.easy/medium/hard` per Section 5.
- `tests/test_coding_he.py`: protocol compliance + sentinel policy + STL-roundtrip tests mirroring [test_bio_ode.py](../tests/test_bio_ode.py).
- `tests/test_coding_specs.py`: spec-format compliance + ρ-on-known-trajectory unit tests.
- Verify the 6 supported samplers run on a single problem; document the 3 excluded samplers' failure modes (clean exception with a `NotImplementedError("non-differentiable simulator")` rather than silent NaN).

Day 3: comparison harness + paper integration.

- `scripts/run_coding_unified_comparison.py`: drops into the same harness pattern as `scripts/run_unified_comparison.py`, produces a parquet artifact.
- `paper/coding_results.md`: numbers + bar chart in the same template as [unified_comparison_results.md](unified_comparison_results.md).
- README headline update: extend "5 biomolecular ODE systems" to "5 biomolecular ODE systems plus one coding-agent cell" with a one-line summary of the key finding.

Tests to add: ≥6 (protocol compliance, sentinel-on-build-failure, sentinel-on-test-timeout, spec registration, ρ-roundtrip on a hand-crafted satisfying trajectory, gradient-sampler-raises- NotImplementedError). Coverage target: 90% line on the new modules, matching the project's existing 91% baseline.

Compute: ~1 wall-CPU-h for the full 6-sampler × 3-spec × 50-problem × 3-seed grid at `N=8` BoN. Single afternoon on M5 Pro. No GPU required for the inference-time cell (LLM proposal can be mocked via a uniform-prior `LLMProposal` for the first pass, matching the [unified_comparison flat-prior caveat](../README.md) on the bio-ODE side). Real-LLM head-to-head follows the same template as [real_llm_comparison.md](real_llm_comparison.md) and lands as a follow-up day.

## Decision summary

| Choice                 | Picked                                                       | Why                                                                                                  |
| ---------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| Simulator              | Stepwise AST-edit + subprocess `Measure`; non-differentiable | Honest match to coding-agent setting; rules out gradient samplers as a feature, not a bug            |
| Dataset                | HumanEval-mutated, 50 problems                               | Lifecycle simplicity (no Docker), MIT license, single-line mutations match the action vocabulary     |
| Action vocabulary      | `V_patch` = `V_op × V_loc`, \`                               | V                                                                                                    |
| STL specs              | `coding.fix.{easy, medium, hard}` per Section 5              | Mirror the `bio_ode.{*.easy, *.medium, *.hard}` difficulty schedule with the same firewall-§C.1 form |
| Excluded samplers      | `gradient_guided`, `hybrid`, `cmaes_gradient`                | Structural — non-differentiable simulator. Excluded explicitly, named in the paper as a contribution |
| Pilot calibration band | `Pr[some sampler succeeds at N=128] ∈ [0.3, 0.7]`            | Matches bio-ODE [FM1 mitigation in theory.md](theory.md); same calibration logic for the same reason |
| Paper artifact         | `paper/coding_results.md` + bar chart in unified-comparison  | Single new figure, slot into the existing artifact rather than spinning a parallel paper             |

## References

\[arXiv:2107.03374\]: Chen, M. *et al.* "Evaluating Large Language Models Trained on Code." 2021. HumanEval dataset.

\[arXiv:2601.20789\]: Shen, E.; Tormoen, D. *et al.* "SERA: Soft-Verified Efficient Repository Agents." 2026. Soft verifier construction `r = |P_2 ∩ P_1| / |P_1|`. Source: AI2.

Donzé, A. & Maler, O. "Robust Satisfaction of Temporal Logic over Real-Valued Signals." FORMATS 2010, LNCS 6246: 92-106. DOI: 10.1007/978-3-642-15297-9_9. STL space-robustness.

Maler, O. & Nickovic, D. "Monitoring Temporal Properties of Continuous Signals." FORMATS 2004. arXiv:cs/0408019. STL syntax and semantics.

Widyasari, R. *et al.* "BugsInPy: A Database of Existing Bugs in Python Programs." 2020. ISSTA.

Liu, J. *et al.* "Is Your Code Generated by ChatGPT Really Correct?" 2023. arXiv:2305.01210 (MBPP+).

\[allenai/SERA\]: https://github.com/allenai/SERA — SERA project repo, 2026.

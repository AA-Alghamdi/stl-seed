# stl-seed — extended abstract

*Phase-1 release, 2026-04-24. v0.1.0 (canonical sweep) ships next.*

**Problem.** Soft-verified SFT — accepting trajectories with a continuous proxy reward instead of a binary pass/fail — is the dominant recipe for instructing small open-weights LLMs on verifiable tasks, but the soft signal is almost always *constructed* (line-overlap on a patch, learned RM, judge-LLM rubric), so the Goodhart decomposition `R_gold − R_proxy = (R_gold − R_spec) + (R_spec − R_verifier)` is unauditable: the spec-completeness and verifier-fidelity error terms are entangled in the verifier's training residual.

**Approach.** stl-seed is a SERA-mimic (Shen et al. 2026, arXiv:2601.20789) where the soft signal is mathematically primitive: Signal Temporal Logic robustness ρ(τ, φ) (Donzé–Maler 2010) on simulated trajectories from a closed-form ODE. Because R_verifier is *defined* as the recursive Donzé–Maler evaluator on the same (τ, φ) that R_spec is defined over, the verifier-fidelity term collapses to a float64 floor (≤ 1 × 10⁻⁶ after σ-squashing), and the entire R_gold − R_proxy gap becomes the auditable spec-completeness term.

**Method.** One epoch of weighted-NLL SFT over a {random, heuristic, small-LLM} mixture-policy reference, on QLoRA adapters of Qwen3-{0.6B, 1.7B, 4B} bases, with three filter conditions {hard (ρ > 0), top-quartile, ρ-weighted softmax}. Scaling axis: 3 × 3 × 2 grid (model size × filter × {bio_ode, glucose_insulin}) with 5 seeds × 25 instances × 8 best-of-N budgets per cell — 36,000 trials, ~$25 of RunPod 4090. Comparison axis: pre-registered H1 (TOST equivalence at Δ = 0.05, hard vs soft) and H2 (size-monotone improvement), with H3 measuring the spec-completeness term against a learned-critic baseline.

**Pre-Phase-2 results.** A 100-trajectory MLX QLoRA smoke on Qwen3-0.6B-bf16 drove training loss 1.484 → 0.466 monotonically in 15.0 s on M5 Pro, hit 5/5 held-out parse-success, produced a 4.6 MiB adapter — all for $0 cloud compute. Empirical pilot ICC of ρ across task families is 0.9979; the hierarchical pooled SE on the global TOST contrast is 0.0098 against the registered-Δ-0.05 threshold of 0.0171, so the locked design is powered. The post-fix `TopologyAwareController` lifts repressilator pooled satisfaction from 0% to **46.48%** (N = 2,388). 317 unit tests pass at 91% coverage; REDACTED firewall green.

**Interpretation.** Success looks like H1 holding within Δ = 0.05 across all 18 cells, with a non-zero spec-completeness term (H3) empirically separable from learned-critic baseline noise. Failure looks like soft significantly underperforming hard in at least one cell — the first negative result on SERA's central conjecture, in the direction the literature has not measured.

**Link.** https://github.com/AA-Alghamdi/stl-seed (Apache 2.0). One-command Phase-2 launch: `python scripts/run_canonical_sweep.py --confirm`.

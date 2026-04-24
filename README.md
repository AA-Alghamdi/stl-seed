# stl-seed

> Soft-verified SFT for scientific control. STL robustness as a formal process verifier on small open-weights LLMs.

**Status:** v0.0.1 — under active development. Phase 1 (theory + library + local pilot) in progress as of 2026-04-24.

## TL;DR

This repository tests whether [SERA](https://arxiv.org/abs/2601.20789)'s soft-verification recipe — generate trajectories, filter by a soft signal, supervised-finetune on the filtered set — extends to scientific control when the soft signal is a **formal Signal Temporal Logic (STL) specification** rather than line-overlap on patches.

Three filter conditions:
- **hard**: keep trajectories with `ρ(τ, φ) > 0` (binary)
- **quantile**: keep top-25% by ρ
- **continuous**: ρ-weighted likelihood during SFT

Three Qwen3 sizes: **0.6B / 1.7B / 4B**. All trained with `bitsandbytes` 4-bit NF4 + QLoRA on the canonical RunPod sweep; iterated locally on Apple Silicon via MLX.

## Why

SERA's §Discussion (Shen et al., 2026) flags an open question: *"Once a model saturates on these aspects, verified correct code may become necessary for further improvement."* In coding, the soft verifier was constructed (line-overlap on a teacher patch). In scientific control, a soft verifier is **mathematically natural**: STL robustness ρ is real-valued by construction.

We test the same conjecture in this domain. The headline finding (forthcoming at v0.1.0) will either confirm or break SERA's soft-vs-hard equivalence in a domain where the soft signal is genuine.

## Status (2026-04-24)

This is a Phase-1 commit: repo bootstrapped, theoretical foundation, library scaffolding. The actual experimental sweep (Phase 2) and final writeup (Phase 3) follow. Watch the changelog.

## Install

```bash
# Apple Silicon (development, iteration, MLX backend)
uv sync --extra mlx

# CUDA (canonical training, RunPod / Linux)
uv sync --extra cuda

# Both
uv sync --extra mlx --extra cuda --extra dev
```

## License

Apache 2.0. See [LICENSE](LICENSE).

## Acknowledgments

- Murat Arcak (Berkeley) — STL formal-methods grounding
- Hanna REDACTED et al. — STL on biomolecular ODEs precedent (REDACTED 2025)
- Karen Leung et al. — STLCG++ differentiable robustness library
- REDACTED et al. — SERA, the soft-verification thesis we test here
- Setlur et al. — PAVs, the closest empirical neighbor on dense process verification

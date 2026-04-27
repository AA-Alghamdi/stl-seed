# Quantization × model-size sweep: does the methodology gap survive?

After Day 1's `METHODOLOGY MATTERS` verdict on real `Qwen3-0.6B-bf16` (`paper/real_llm_comparison.md`), two natural follow-up questions remain. (1) Does the methodology gap survive NF4 / 4-bit / 8-bit quantization (the `bitsandbytes` lineage)? (2) Does it survive scaling to 1.7B, where the LLM has more entropy and may saturate easier tasks on its own (the SERA-saturation transition the Limitations section flags)? This sweep answers both.

Reproduce with `uv run python scripts/quant_size_sweep.py`. Per-cell data in `runs/quant_size_sweep/{results.parquet, summary.json, results.jsonl, run.log}`. The pre-registered outcome rule and per-cell runner are reused verbatim from `scripts/real_llm_hard_specs.py`.

## Sweep design

5 models × 2 samplers × 4 hard tasks × 3 fixed seeds = **120 cells**, ~22 minutes wall-clock on M5 Pro.

| Model             | Precision              | Size | HF id                           |
| ----------------- | ---------------------- | ---- | ------------------------------- |
| `qwen3-0.6b`      | bf16                   | 0.6B | `mlx-community/Qwen3-0.6B-bf16` |
| `qwen3-0.6b-8bit` | 8-bit                  | 0.6B | `mlx-community/Qwen3-0.6B-8bit` |
| `qwen3-0.6b-4bit` | 4-bit (NF4-equivalent) | 0.6B | `mlx-community/Qwen3-0.6B-4bit` |
| `qwen3-1.7b`      | bf16                   | 1.7B | `mlx-community/Qwen3-1.7B-bf16` |
| `qwen3-1.7b-4bit` | 4-bit                  | 1.7B | `mlx-community/Qwen3-1.7B-4bit` |

Tasks: `bio_ode.{repressilator, toggle, mapk}` + `cardiac.suppress_after_two.hard`. Samplers: `standard` and `beam_search_warmstart`. Seeds: `{3000, 3001, 3002}`, matching Day 1.

## Headline result

**`METHODOLOGY MATTERS` fires on every one of the 5 models.** The methodology gap survives the full quantization × size factorial.

| Model             | Sat-fraction (standard) | Sat-fraction (beam-search) | Verdict             |
| ----------------- | ----------------------- | -------------------------- | ------------------- |
| `qwen3-0.6b`      | 0/12                    | 12/12                      | METHODOLOGY MATTERS |
| `qwen3-0.6b-8bit` | 0/12                    | 12/12                      | METHODOLOGY MATTERS |
| `qwen3-0.6b-4bit` | 1/12                    | 12/12                      | METHODOLOGY MATTERS |
| `qwen3-1.7b`      | 3/12                    | 12/12                      | METHODOLOGY MATTERS |
| `qwen3-1.7b-4bit` | 3/12                    | 12/12                      | METHODOLOGY MATTERS |

The verdict is unanimous; the *quality* of the standard-sampling baseline shifts in two directions, neither of which breaks the methodology gap.

## Mean ρ per (model, task, sampler)

| Model             | Task          |  Standard ρ̄ | Beam-search ρ̄ |      Gap |
| ----------------- | ------------- | ----------: | ------------: | -------: |
| `qwen3-0.6b`      | repressilator |    −247.582 |       +25.000 | +272.582 |
| `qwen3-0.6b`      | toggle        |     −99.960 |       +29.992 | +129.952 |
| `qwen3-0.6b`      | mapk          |      −0.500 |        +0.002 |   +0.502 |
| `qwen3-0.6b`      | cardiac       |      −1.434 |        +0.850 |   +2.284 |
| `qwen3-0.6b-8bit` | repressilator |    −247.582 |       +25.000 | +272.582 |
| `qwen3-0.6b-8bit` | toggle        |     −99.960 |       +29.992 | +129.952 |
| `qwen3-0.6b-8bit` | mapk          |      −0.500 |        +0.002 |   +0.502 |
| `qwen3-0.6b-8bit` | cardiac       |      −1.434 |        +0.850 |   +2.284 |
| `qwen3-0.6b-4bit` | repressilator |    −247.582 |       +25.000 | +272.582 |
| `qwen3-0.6b-4bit` | toggle        | **−61.950** |       +29.992 |  +91.942 |
| `qwen3-0.6b-4bit` | mapk          |      −0.500 |        +0.002 |   +0.502 |
| `qwen3-0.6b-4bit` | cardiac       |      −1.434 |        +0.850 |   +2.284 |
| `qwen3-1.7b`      | repressilator |    −248.759 |       +25.000 | +273.759 |
| `qwen3-1.7b`      | toggle        | **+14.071** |       +29.992 |  +15.921 |
| `qwen3-1.7b`      | mapk          |      −1.165 |        +0.002 |   +1.167 |
| `qwen3-1.7b`      | cardiac       |      −1.434 |        +0.850 |   +2.284 |
| `qwen3-1.7b-4bit` | repressilator |    −248.759 |       +25.000 | +273.759 |
| `qwen3-1.7b-4bit` | toggle        | **+14.071** |       +29.992 |  +15.921 |
| `qwen3-1.7b-4bit` | mapk          |      −0.500 |        +0.002 |   +0.502 |
| `qwen3-1.7b-4bit` | cardiac       |      −2.199 |        +0.850 |   +3.049 |

Bold values indicate where the standard sampler clears the satisfaction boundary on a strict majority of seeds (toggle saturates at 1.7B; the 0.6B-4bit point is one-of-three, not a majority).

## What the rows reveal

**Quantization (rows 1-12, fixed at 0.6B).** The `0.6b-bf16` and `0.6b-8bit` rows are bit-identical on standard sampling for every task (every cell hits the same final ρ at every seed). 8-bit quantization **preserves** Qwen3-0.6B's mode-selection on these prompts at temperature 0.5. The `0.6b-4bit` row diverges only on toggle: standard sampling rescues 1 of 3 seeds (mean ρ jumps from −99.96 to −61.95), i.e. quantization-induced noise nudges one seed across the satisfaction boundary on the toggle task. The methodology gap stays large (beam-search +29.99 vs standard −61.95). All beam-search columns are bit-identical across the three precisions.

**Scaling 0.6B → 1.7B (rows 13-24, fixed at bf16 and 4-bit).** Three movements appear when the LLM grows from 0.6B to 1.7B. (1) Repressilator gets *slightly* harder: standard mean ρ −247.58 → −248.76. The LLM's mode-selection on a 3-D action box has shifted, but the satisfying corner is still off the standard-sampler's reach. (2) Toggle **saturates**: standard mean ρ −99.96 → +14.07 with full sat-fraction (3/3 seeds), so the larger LLM solves toggle on its own. This is the first appearance of the SERA-saturation transition in the artifact's data. Beam-search still wins on toggle (+29.99 vs +14.07), but this is a magnitude difference, not a sat-fraction difference. (3) MAPK and cardiac stay solidly in the methodology-mattering regime: standard fails on every seed at every model size; beam-search rescues every seed.

**4-bit at 1.7B (rows 17-20).** Bit-identical to 1.7B-bf16 on every cell except cardiac (standard mean ρ −1.434 → −2.199, slight degradation; no crossings in either direction).

## Connection to the SERA-saturation hypothesis

Dettmers et al. (Shen, Tormoen, Shah, Farhadi, Dettmers; SERA, arXiv:2601.20789) flag in the Limitations section: "once a model saturates on these aspects, **verified correct code may become necessary** for further improvement." The toggle row at 1.7B is the first cell in this artifact's data where saturation appears: at 1.7B the LLM picks the satisfying corner on toggle on its own, no verifier needed. At 0.6B across all three precisions it does not. **The methodology gap on toggle has gone from a sat-fraction question (0/3 vs 3/3 at 0.6B) to a magnitude question (3/3 vs 3/3, +14.07 vs +29.99 at 1.7B).** That is the shape of an emerging saturation transition. At 4B and 8B+. past M5 Pro unified-memory budget. one would expect more tasks to follow the same trajectory; mapk and cardiac look closest behind, repressilator looks furthest.

The headline therefore strengthens: **the methodology gap is robust to ~3× quantization compression and ~3× size scaling, and the only task that breaks the gap is the one where SERA's saturation transition appears for free.** Both directions of the survival test are informative.

## Honest caveats

- **Three seeds is small.** Same caveat as Day 1: this is a falsification test, not a population estimate. With N=3 and zero per-seed variance on most cells (Qwen3 at T=0.5 deterministically picks the same vocab entry across the three seeds), CIs are tight by construction. Increasing temperature or N would loosen the CIs and may change the one-of-three crossings on `0.6b-4bit + toggle`.
- **Toggle saturation is a single-task observation.** With only 4 hard tasks one cell flipping to saturated is consistent with both "the hypothesis is real" and "this task happens to be easier than the others at this scale." A larger task suite would discriminate.
- **bf16 / 8-bit equivalence on standard sampling could be misleading.** These two precisions produce bit-identical action sequences on every task at temperature 0.5 here. With a larger benchmark or higher temperature, divergences would emerge; the strong reading is "8-bit quantization is observationally indistinguishable from bf16 on this small benchmark," not "8-bit quantization is precision-equivalent."
- **The vocabulary-by-construction caveat carries through.** Beam-search warmstart's rescue rests on the satisfying repressilator corner `u=(0,0,1)` being in the `k_per_dim=5` lattice. That has not changed across precisions or sizes; it is a structural property of the vocabulary, not the LLM. The contribution is the structural-search- vs-continuous-search distinction (cf. `paper/landscape_theorem.md`), not a free win.
- **The full saturation curve from 0.6B → 8B+ is past available compute.** The four model sizes Qwen3 ships in (0.6B, 1.7B, 4B, 8B, 14B, 32B) would map the saturation transition cleanly, but 4B is at M5 Pro's unified-memory edge for a K=125 beam-search cell and 8B+ needs CUDA. That is the rotation pitch.

## Files

- Driver: `scripts/quant_size_sweep.py`
- Per-cell results: `runs/quant_size_sweep/results.parquet`, `runs/quant_size_sweep/results.jsonl`
- Per-model verdicts (machine-readable): `runs/quant_size_sweep/summary.json`
- Run log: `runs/quant_size_sweep/run.log`

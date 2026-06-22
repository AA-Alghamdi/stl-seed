# Addendum: Qwen3-4B-4bit extends the scaling axis to 3 sizes

The original quant-size sweep (`paper/quant_size_results.md`) covered 5 model variants at 0.6B and 1.7B. This addendum extends to a 6th variant: `Qwen3-4B-4bit` (the only 4B configuration that fits M5 Pro unified memory at our K=125 beam-search vocabulary). Run on 2026-04-27, ~7 minutes wall on M5 Pro, 24 cells. Reproduce with `uv run python scripts/quant_size_sweep.py --models qwen3-4b-4bit --out-dir runs/quant_size_sweep_4b`.

## Result

**`METHODOLOGY MATTERS` fires on `Qwen3-4B-4bit` too.** The size-scaling pattern from `0.6B → 1.7B` extends cleanly to `4B`. We now have 6 (model, precision) configurations confirming the methodology gap, spanning 3 model sizes (0.6B, 1.7B, 4B) and 3 precisions (bf16, 8-bit, 4-bit).

| Task                              | Standard sat / n |  Standard ρ̄ | Beam sat / n |  Beam ρ̄ | Note                     |
| --------------------------------- | ---------------- | ----------: | ------------ | ------: | ------------------------ |
| `bio_ode.repressilator.easy`      | 0 / 3            |     −248.34 | 3 / 3        | +25.000 | full rescue              |
| `bio_ode.toggle.medium`           | **3 / 3**        | **+14.071** | 3 / 3        | +29.992 | saturated (same as 1.7B) |
| `bio_ode.mapk.hard`               | 0 / 3            |      −1.188 | 3 / 3        |  +0.002 | full rescue              |
| `cardiac.suppress_after_two.hard` | 0 / 3            |      −1.434 | 3 / 3        |  +0.850 | full rescue              |

## What this confirms

The toggle saturation that first appeared at 1.7B persists at 4B-4bit (3/3 standard sat, identical numerical value `+14.071`). Repressilator, MAPK, and cardiac stay solidly methodology-mattering at 4B: standard sampling fails on every seed, beam-search rescues every seed. The saturation transition is task-specific, not size-monotone across all tasks: scaling from 0.6B → 1.7B → 4B-4bit moves toggle from regime II to regime I, but the other three hard tasks remain in regime II at all three sizes tested.

## Three-size scaling table (sat-fraction summary across all 6 configurations)

| Model               | repressilator | toggle        | mapk      | cardiac   |
| ------------------- | ------------- | ------------- | --------- | --------- |
| `qwen3-0.6b` (bf16) | 0/3 → 3/3     | 0/3 → 3/3     | 0/3 → 3/3 | 0/3 → 3/3 |
| `qwen3-0.6b-8bit`   | 0/3 → 3/3     | 0/3 → 3/3     | 0/3 → 3/3 | 0/3 → 3/3 |
| `qwen3-0.6b-4bit`   | 0/3 → 3/3     | 1/3 → 3/3     | 0/3 → 3/3 | 0/3 → 3/3 |
| `qwen3-1.7b` (bf16) | 0/3 → 3/3     | **3/3 → 3/3** | 0/3 → 3/3 | 0/3 → 3/3 |
| `qwen3-1.7b-4bit`   | 0/3 → 3/3     | **3/3 → 3/3** | 0/3 → 3/3 | 0/3 → 3/3 |
| `qwen3-4b-4bit`     | 0/3 → 3/3     | **3/3 → 3/3** | 0/3 → 3/3 | 0/3 → 3/3 |

Standard sat-fraction → beam-search sat-fraction. Bold cells are saturated by the LLM alone (no methodology needed).

## Implications

1. **Size scaling does not collapse the methodology gap on 3 of 4 hard tasks.** At 4B-4bit, repressilator / mapk / cardiac all show the maximum methodology gap (0/3 → 3/3 sat). The SERA-saturation transition predicted by the Limitations section happens on only 1 of 4 tasks at this scale.

1. **The toggle saturation reproduces deterministically across (1.7B-bf16, 1.7B-4bit, 4B-4bit).** Standard sat-fraction = 3/3 at the same `ρ̄ = +14.071` on all three. The LLM's mode-selection on toggle at temperature 0.5 collapses to the same satisfying corner across model sizes once the scale is sufficient.

1. **The remaining open question (the rotation pitch).** What happens at 8B / 14B / 32B? The artifact predicts the saturation transition spreads to mapk and cardiac before repressilator (cardiac and mapk have shallower failure floors `-1.43` and `-1.19` than repressilator's `-248.34`, suggesting the LLM is closer to the satisfying region in those cases). Falsifiable; testable on $1k of H100; past M5 Pro budget.

## Files

- Driver: `scripts/quant_size_sweep.py --models qwen3-4b-4bit --out-dir runs/quant_size_sweep_4b`
- Per-cell results: `runs/quant_size_sweep_4b/results.parquet`, `results.jsonl`
- Per-model verdict (machine-readable): `runs/quant_size_sweep_4b/summary.json`
- Run log: `runs/quant_size_sweep_4b/run.log`

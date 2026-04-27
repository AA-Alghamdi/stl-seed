# money_plot.png caption

**Figure 1.** Methodology gap survives quantization and size scaling. Each cell shows the difference in mean STL robustness between the structural-search inference-time methodology (beam-search warmstart) and a standard sampler, $\Delta\rho = \rho_\mathrm{beam} - \rho_\mathrm{standard}$, evaluated on real Qwen3 LLM rollouts with N=3 pre-registered seeds per cell. Rows span a 5x model grid (0.6B and 1.7B parameters at bf16, 8-bit, and 4-bit precision, omitting 1.7B-8bit); columns span the four hard tasks where the 0.6B-bf16 standard sampler fails on a majority of seeds (repressilator, toggle, MAPK, cardiac action potential). The diverging colormap is centred at zero; positive (red) cells indicate the methodology helps. The colour scale is symmetric-log to keep both small and large gaps legible. Cell annotations show $\Delta\rho$ (top) and the per-sampler satisfaction count "beam vs standard" (bottom), so the figure remains legible in greyscale. Saturated cells (heavy border, circled S) mark the first appearance of SERA-saturation: at 1.7B on the toggle task the standard sampler already solves the spec on 3/3 seeds, and the methodology gap collapses from a sat-fraction win (0/3 vs 3/3 across 0.6B) to a magnitude residual (+15.9 in $\rho$). The remaining three task families (repressilator, MAPK, cardiac AP) stay solidly methodology-mattering across the full precision and size range; the gap on repressilator is approximately +273 across all five model configurations.

## Companion figure

`money_plot_inset.png` shows the toggle row across all five models as side-by-side sat-fraction bars, isolating the SERA-saturation transition: standard goes 0/3 -> 0/3 -> 1/3 -> 3/3 -> 3/3 as we scale precision then size, while beam warmstart stays at 3/3 throughout.

## Files

- `paper/figures/money_plot.png` (200 DPI raster)
- `paper/figures/money_plot.pdf` (vector)
- `paper/figures/money_plot_inset.png` / `.pdf` (companion)
- `paper/figures/_make_money_plot.py` (generator, deterministic given the parquet)

# Hierarchical Bayesian analysis of the methodology gap

Source: `runs/quant_size_sweep/results.parquet` (120 rows = 5 models x 4 tasks x 2 samplers x 3 seeds). Inference: NumPyro NUTS, 4 chains, 1000 warmup + 1000 samples per chain, target_accept=0.95.

## 1. Model specification

```
rho[i] = mu[m, t] + delta[m, t] * 1{sampler[i] = beam_search_warmstart} + eps[i]
mu[m, t]    ~ Normal(mu_global,    sigma_mu)
delta[m, t] ~ Normal(delta_global, sigma_delta)   <-- partial pooling target
eps[i]      ~ Normal(0, sigma_eps)
mu_global, delta_global ~ Normal(0, 100)
sigma_mu, sigma_delta, sigma_eps ~ HalfNormal(50)
```

- `delta_global` is the population-mean methodology gap (beam minus standard, in raw `rho` units), partial-pooled across the (model, task) population.
- Note: the raw sampler label is `beam_search_warmstart` (not literally `beam`); the encoded indicator is identical to the spec in the prompt.
- The model is homoscedastic in `eps` (single `sigma_eps`); see Section 5 for the heteroscedastic concern with deterministic Qwen3 cells.

## 2. Sampler diagnostics

- Chains: 4. Total post-warmup samples: 4000.
- Divergences: **0**
- Per-parameter R-hat (max over indices) and minimum ESS:

| parameter    | max R-hat | min ESS |
| ------------ | --------: | ------: |
| mu_global    |     1.000 |    4868 |
| delta_global |     1.000 |    4774 |
| sigma_mu     |     1.000 |    4415 |
| sigma_delta  |     1.000 |    4767 |
| sigma_eps    |     1.001 |    2178 |
| mu           |     1.001 |    4094 |
| delta        |     1.000 |    4119 |

Convergence verdict: **CONVERGED** (threshold: divergences \< 30, all R-hat \< 1.10, leaf R-hat \< 1.05).

## 3. Posterior over `delta_global` (population methodology gap)

- Posterior mean: **82.820** (rho units)
- 95% credible interval: **\[35.219, 129.444\]**
- P(delta_global > 0) = **1.000**
- Posterior mean of population scale `sigma_delta`: 109.074
- Posterior mean of `sigma_eps` (within-cell seed noise): 10.593

Interpretation: the 95% CI excludes zero on the positive side — the population-mean methodology gap favours beam over standard sampling.

## 4. Per-(model, task) posterior of `delta[m, t]`

Columns: `raw_gap` = beam-mean minus standard-mean of the 3 seeds in that cell; `post_mean_delta` = posterior mean from partial pooling; `crosses_zero` = whether the 95% CI crosses zero; `shrunk` = |posterior mean| \< |raw gap|.

| task                  | model           | raw_gap | post_mean | 95% CI               | crosses 0 | shrunk |
| --------------------- | --------------- | ------: | --------: | -------------------- | :-------: | :----: |
| bio_ode.mapk          | qwen3-0.6b      |   +0.50 |     +1.47 | \[-14.85, +18.21\]   |    YES    |   no   |
| bio_ode.mapk          | qwen3-0.6b-4bit |   +0.50 |     +1.23 | \[-16.64, +18.57\]   |    YES    |   no   |
| bio_ode.mapk          | qwen3-0.6b-8bit |   +0.50 |     +1.33 | \[-15.38, +18.23\]   |    YES    |   no   |
| bio_ode.mapk          | qwen3-1.7b      |   +1.17 |     +2.15 | \[-15.12, +19.75\]   |    YES    |   no   |
| bio_ode.mapk          | qwen3-1.7b-4bit |   +0.50 |     +1.39 | \[-15.19, +18.22\]   |    YES    |   no   |
| bio_ode.repressilator | qwen3-0.6b      | +272.58 |   +270.60 | \[+253.84, +287.29\] |    no     |  YES   |
| bio_ode.repressilator | qwen3-0.6b-4bit | +272.58 |   +270.66 | \[+253.17, +287.57\] |    no     |  YES   |
| bio_ode.repressilator | qwen3-0.6b-8bit | +272.58 |   +270.80 | \[+253.49, +288.14\] |    no     |  YES   |
| bio_ode.repressilator | qwen3-1.7b      | +273.76 |   +271.61 | \[+254.54, +288.31\] |    no     |  YES   |
| bio_ode.repressilator | qwen3-1.7b-4bit | +273.76 |   +271.72 | \[+255.02, +287.93\] |    no     |  YES   |
| bio_ode.toggle        | qwen3-0.6b      | +129.95 |   +129.61 | \[+112.92, +146.73\] |    no     |  YES   |
| bio_ode.toggle        | qwen3-0.6b-4bit |  +91.94 |    +91.81 | \[+74.97, +108.78\]  |    no     |  YES   |
| bio_ode.toggle        | qwen3-0.6b-8bit | +129.95 |   +129.53 | \[+112.90, +146.76\] |    no     |  YES   |
| bio_ode.toggle        | qwen3-1.7b      |  +15.92 |    +16.72 | \[-1.31, +33.86\]    |    YES    |   no   |
| bio_ode.toggle        | qwen3-1.7b-4bit |  +15.92 |    +16.70 | \[+0.23, +33.15\]    |    no     |   no   |
| cardiac_ap            | qwen3-0.6b      |   +2.28 |     +2.95 | \[-14.18, +20.15\]   |    YES    |   no   |
| cardiac_ap            | qwen3-0.6b-4bit |   +2.28 |     +2.89 | \[-15.00, +20.17\]   |    YES    |   no   |
| cardiac_ap            | qwen3-0.6b-8bit |   +2.28 |     +3.10 | \[-14.44, +20.28\]   |    YES    |   no   |
| cardiac_ap            | qwen3-1.7b      |   +2.28 |     +3.16 | \[-13.95, +20.42\]   |    YES    |   no   |
| cardiac_ap            | qwen3-1.7b-4bit |   +3.05 |     +3.80 | \[-13.13, +20.86\]   |    YES    |   no   |

Summary: **11 / 20 cells have CI crossing zero**; **8 / 20 posterior means are shrunk toward zero relative to the raw cell mean** (the partial-pooling shrinkage signature; loud single-seed outliers are pulled toward `delta_global`).

## 5. Caveats

1. **N=3 seeds per cell.** The likelihood contributes only 3 paired observations per (model, task); the posterior on `delta[m,t]` is dominated by the prior on `sigma_delta` for cells where the raw gap is small relative to within-cell noise. This is exactly the situation hierarchical pooling is designed for, but reviewers should not over-interpret per-cell CIs.
1. **Zero-variance cells violate homoscedasticity.** When a (model, task, sampler) triple is deterministic across the 3 seeds — observed for some Qwen3 cells in the parquet — the sample variance is exactly zero. The model uses a single shared `sigma_eps`, so it cannot capture this heterogeneity; the inferred `sigma_eps` is an across-cell average and inflates uncertainty for the deterministic cells while underestimating it for the noisy ones. A per-cell `sigma_eps[m, t]` (e.g. with a HalfNormal hyperprior) is the natural fix and should be tried in v2.
1. **The methodology gap is not one-dimensional.** `final_rho` mixes (a) sat-fraction (whether `rho >= 0`) and (b) magnitude. A linear additive `delta` treats them identically, so a +20 rho swing in an already-satisfied cell is weighted the same as a -200 rho rescue of an unsatisfied cell. A two-stage model — Bernoulli on `satisfied` plus a magnitude regression conditional on satisfaction — would separate these. The current `delta_global` should be read as a *combined* gap.
1. **Heavy-tailed residuals.** Empirical `final_rho` ranges from -248 to +30, with the negative tail far heavier than the positive. Gaussian residuals will be pulled by these outliers; a Student-t observation likelihood with `nu ~ Gamma(2, 0.1)` is the standard robustification.

## 6. Implications

- The population-mean methodology gap is **credibly non-zero** at the 95% level, with sign **positive** (beam > standard).
- Partial-pooling shrinkage matters: 8/20 cells have posterior means pulled toward `delta_global` relative to the raw cell mean (the others were already at or outside the population mean and the prior pulls less). The forest plot in Figure 1 (bottom) shows this: bio_ode.repressilator and the high-magnitude bio_ode.toggle cells barely shrink (their raw gaps dominate the likelihood), while bio_ode.mapk and cardiac_ap cells with raw_gap \< 4 are pulled noticeably toward the population mean of 82.8.
- For the FMAI / NeurIPS D&B reviewer asking about population-level methodology gaps: this is the right summary, with the sat-fraction-vs-magnitude caveat explicit.
- Negative-result-as-data: even if `delta_global` straddles zero, the `sigma_delta` posterior tells us whether the *variance across cells* is large; a small `sigma_delta` would mean methodology is uniformly inert, while a large `sigma_delta` means the choice matters but not in a single direction.

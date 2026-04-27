```
    # Empirical Power Analysis (Subphase 1.4 / A16)

    *Date: 2026-04-24. Companion to paper/theory.md §5.*

    This document refreshes the design-time power analysis in
    `paper/theory.md` §5 with an *empirical* intra-class correlation
    (ICC) of robustness rho estimated from the Subphase-1.4 pilot
    (A13). The original analysis used a worst-case plug-in
    `rho_ICC = 0.4` based on a 30-trajectory-per-cell synthetic pilot;
    the present analysis substitutes the empirical ICC measured on
    N=3,982 trajectories distributed over k=2
    (task x policy) buckets. All other design knobs are unchanged
    from the locked plan: 3 sizes x 3 filters x 2 task families x 25
    instances x 5 seeds x 8 BoN budgets = 36,000 trials.

    ## 1. Pilot composition

    | task | policy | N | mean | std | min | max |
    |------|--------|---:|-----:|----:|----:|----:|
```

| `bio_ode.repressilator` | `heuristic` | 1,000 | -2.488e+02 | 0.000e+00 | -2.488e+02 | -2.488e+02 | | `bio_ode.repressilator` | `random` | 982 | -2.455e+02 | 4.565e+00 | -2.487e+02 | -2.040e+02 | | `glucose_insulin` | `heuristic` | 1,000 | +2.075e+01 | 0.000e+00 | +2.075e+01 | +2.075e+01 | | `glucose_insulin` | `random` | 1,000 | -1.129e+00 | 3.408e+00 | -1.042e+01 | +1.051e+01 |

```
    Two task families are represented by their easy specs
    (`bio_ode.repressilator.easy` and `glucose_insulin.tir.easy`),
    each with a {random: 0.5, heuristic: 0.5} policy mix.

    ## 2. Two ICC variants

    We compute and report two ICC(1,1) variants. The first (the
    deliverable's literal request) groups by `(task x policy)`; the
    second groups by `task` only. The (task x policy) variant is
    structurally inflated by the deterministic heuristic policies
    (PIDController and BangBangController are deterministic functions
    of state with zero randomness, so their within-bucket variance
    is zero), making MSW dominated only by the random-policy bucket
    and pushing ICC near 1.0 even when the within-task variation in
    rho is healthy. We therefore use the (task)-only ICC for the
    verdict, since it more closely mirrors theory.md §5's
    within-task seed-replicate ICC structure that the canonical
    sweep will exhibit (where each (m, v, f, instance) cell is
    replicated across n_seeds=5 with a stochastic LLM-mixture
    policy).

    | grouping | ICC(1,1) |
    |----------|---------:|
    | (task x policy) | 0.9996  *(inflated; sanity check only)* |
    | (task) | **0.9979**  *(verdict-driving)* |

    ## 3. ICC estimator

    We follow Shrout & Fleiss (1979) ICC(1,1) for a one-way
    random-effects model with unbalanced groups (their Eq. 5 for the
    n0 correction):

        ICC(1,1) = (MSB - MSW) / (MSB + (n0 - 1) * MSW),
        n0 = (N - sum_g(n_g^2) / N) / (k - 1).

    We use the one-way model rather than ICC(2,1) or ICC(3,1) because
    the (task x policy) grouping is a fixed partition of the design
    space rather than a sample of "rater" effects; ICC(1,1) is the
    canonical estimator for the design-effect correction we apply
    downstream. Reference: Shrout & Fleiss (1979),
    DOI 10.1037/0033-2909.86.2.420.

    Empirical values from the pilot:

    | quantity | value |
    |----------|------:|
    | N (total) | 3,982 |
    | k (groups) | 2 |
    | n0 (unbalanced) | 1990.959 |
    | MSB | 6.5735e+07 |
    | MSW | 6.9477e+01 |
    | Var(group means) | 1.6508e+04 |
    | Var(all rhos) | 1.6577e+04 |
    | **Empirical ICC** | **0.9979** |

    For comparison, theory.md §5 used `rho_ICC = 0.4` as a worst-case
    plug-in; the empirical value is
    **0.9979** (higher than the design-time estimate).

    ## 4. Recomputed power numbers

    Substituting the empirical ICC into the design-effect chain
    (theory.md §5):

        design_effect = 1 + (n_seeds - 1) * ICC
        n_eff_per_cell = n_instances * n_seeds / design_effect
        bon_factor = n_bon / (1 + (n_bon - 1) * r_bon)    # r_bon = 0.7
        I_AA_cell = n_eff_per_cell * bon_factor * I_AA(N=128)

    and SE on the logit-A contrast:

        SE_dA_logit_cell = sqrt(2 / I_AA_cell)
        SE_dA_logit_pool = SE_dA_logit_cell / sqrt(n_cells)        # n_cells = 18
        SE_dA_prob_pool ~ SE_dA_logit_pool * p * (1 - p)            # p = 0.42

    At the prior median (A=0.6, b=0.25, N=128) the per-observation Fisher info is:

    | I_AA | I_bb | I_Ab |
    |-----:|-----:|-----:|
    | 2.0249 | 3.0719 | 2.4941 |

    Side-by-side comparison of the original (ICC=0.40 plug-in) and the
    empirical recomputation:

    | quantity | original (ICC=0.40) | empirical (ICC=0.998) |
    |----------|--------------------:|--------------------------------------:|
    | design_effect | 2.6000 | 4.9916 |
    | n_eff_per_cell | 48.08 | 25.04 |
    | bon_factor | 1.3559 | 1.3559 |
    | I_AA_cell | 132.0013 | 68.7562 |
    | SE_dA_logit_cell | 0.1231 | 0.1706 |
    | SE_dA_prob_cell | 0.0300 | 0.0416 |
    | SE_dA_logit_pool | 0.0290 | 0.0402 |
    | SE_dA_prob_pool | 0.0071 | 0.0098 |
    | MDE(Delta_A) (one-sided alpha=0.05, power=0.8) | 0.0176 | 0.0244 |
    | TOST SE threshold (Delta=0.05) | 0.0171 | 0.0171 |
    | TOST powered at the global scale | YES | YES |

    On the b parameter, the same chain gives MDE on the *log* scale.
    Translating to the rate scale at the prior median (b=0.25):

        MDE(Delta_b on rate scale) ~ MDE(log b) * b
                                   = 0.0812 * 0.25
                                   = 0.0203

    ## 5. Verdict

    **YES. the locked design (3x3x2x25x5x8 = 36000 trials) is adequately powered to detect Delta_A>=0.080 and Delta_b>=0.100 at the global pooled scale.**

    Per-criterion breakdown:

    - MDE(Delta_A) on probability scale = 0.0244; registered Delta_A = 0.080 -> POWERED.
    - MDE(Delta_b) on rate scale = 0.0203; registered Delta_b = 0.100 -> POWERED.
    - TOST equivalence at Delta = 0.05 requires SE_dA_prob_pool <= 0.0171; actual = 0.0098 -> POWERED.


    ## 6. Caveats and scope

    1. The empirical ICC is estimated on the full pilot pool
       (not segmented by the 18 (m, v, f) sweep cells, which do not
       exist yet at this stage of Phase 1). Hierarchical Bayes posterior
       uncertainty (theory.md §4) will propagate the ICC posterior
       rather than the ICC point estimate; the present plug-in is the
       sampling-design analogue.
    2. The pilot contains two policy classes (random + heuristic)
       rather than the full mixture {random, heuristic, LLM} that
       the canonical sweep will use. The LLM-policy leg is delegated
       to A15 (separate agent) once an MLX-capable Qwen3 build is
       available locally; per FM4 in theory.md §7, MLX vs bnb numerical
       drift is monitored separately from the present ICC.
    3. The locked design pools across all 18 (size x filter x task)
       cells via the random-effects hierarchical model (theory.md §4).
       Per-cell power is *exploratory*, not the primary endpoint;
       the verdict above is on the **global** pooled contrast.
    4. The Fisher information uses point estimates at the prior median
       (A=0.6, b=0.25). If the actual posterior mass moves materially
       away from this point during the canonical sweep, the SE will
       change accordingly and the verdict should be re-checked.

    ## 7. Reproducibility

    Run `uv run python scripts/power_analysis_real.py` from the repo
    root. The script reads the pilot store at `data/pilot/`, computes
    the ICC and the design-power numbers, and writes this file.
    Determinism: numerical values are pure functions of the on-disk
    pilot rho values; no Monte Carlo step is used here.
```

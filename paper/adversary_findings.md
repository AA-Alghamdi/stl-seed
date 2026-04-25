# Trajectory adversary findings

Empirical operationalization of the Goodhart spec-completeness gap (`paper/theory.md` §6). For each task family + proxy spec, the trajectory adversary searches for control sequences that satisfy the proxy STL spec (high $\\rho$) yet violate an unstated gold objective (low gold score). A successful find is a direct lower bound on the per-task spec-completeness gap.

Generated 2026-04-25 03:18:55 UTC; seed 2026; 8 restarts × 80 inner iterations; random population 200.

## glucose_insulin · `glucose_insulin.tir.easy`

Adversary: best spec $\\rho = +36.0450$, best gold = $-1.9229$. Per-restart finals: $(36.076, -1.742)$, $(36.003, -1.913)$, $(20.0, -1.477)$, $(20.0, -1.809)$, $(20.0, -1.581)$, $(36.045, -1.923)$, $(36.047, -1.81)$, $(36.079, -1.715)$. NaN/Inf events: 12,861. Adversary did not converge (8 restarts at 80 iter each). 42.20 s wall.

Random population reference (200 trajectories, all spec-satisfying): Pearson $r(\\rho, \\text{gold}) = +0.0058$; Spearman $+0.0185$; regression slope per $\\sigma$ = $+0.0016$; top-decile gap (gold) = $-0.0018$; min gold among satisfying = $-0.5116$; mean gold = $+0.3498$. FM2 flagged (Spearman \< 0.3). 0.61 s.

Empirical gap lower bound: adv gold − mean(satisfying random gold) = $-2.2727$; adv gold − min(satisfying random gold) = $-1.4114$.

## bio_ode.repressilator · `bio_ode.repressilator.easy`

Adversary: best spec $\\rho = +25.0000$, best gold = $+0.9685$. Per-restart finals: $(25.0, 0.991)$, $(25.0, 0.998)$, $(25.0, 0.997)$, $(25.0, 1.0)$, $(25.0, 0.997)$, $(25.0, 0.995)$, $(25.0, 0.968)$, $(25.0, 1.0)$. No NaN/Inf events. Converged. 8.78 s wall.

Random population (200 trajectories, **0 spec-satisfying**): Pearson $+0.1510$; Spearman $+0.1380$; slope $+0.0226$; top-decile gap $+0.0482$. FM2 flagged. 1.26 s.

Gap lower bound: NaN (no random-satisfying baseline; the adversary verifies the spec is satisfiable but cannot exhibit a *gap* against a reference distribution that does not exist).

## bio_ode.toggle · `bio_ode.toggle.medium`

Adversary: best spec $\\rho = -40.0000$, best gold = $+0.4613$. All eight restart finals collapse to $(-40.0, 0.461)$. Converged. 9.04 s wall. The adversary did not even find a spec-satisfying trajectory at this restart × iter budget; the toggle spec is hard.

Random population (200 trajectories, 0 spec-satisfying): Pearson $+0.0278$; Spearman $-0.0237$; slope $+0.0003$; top-decile gap $+0.0016$. FM2 flagged. 4.85 s.

Gap lower bound: NaN.

## Interpretation

Per `paper/theory.md` §6, the spec-completeness term $R\_\\text{gold}(\\tau) - R\_\\text{spec}(\\tau)$ is the Goodhart-relevant residual once the verifier-fidelity term is collapsed by the choice of formal STL robustness as the verifier. The adversary above provides a constructive lower bound on $\\sup\_\\tau \[R\_\\text{spec}(\\tau) - R\_\\text{gold}(\\tau)\]$ restricted to the spec-satisfying half-space. A negative gap (adv gold \< mean satisfying random gold) means the spec is *locally* exploitable: there exist high-$\\rho$ trajectories that score worse under the unstated gold than a random spec-satisfying baseline.

Only glucose_insulin produces a finite gap. Repressilator and toggle have zero spec-satisfying random trajectories at this pool size, so the gap is undefined. The −2.27 number on glucose_insulin is a single-instance lower bound from one optimizer run, not a confidence interval — framed correctly as existence-style, not population-mean.

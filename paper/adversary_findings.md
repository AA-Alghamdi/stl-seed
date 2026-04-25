# Trajectory adversary findings

Empirical operationalization of the Goodhart spec-completeness gap (see `paper/theory.md` S6). For each task family + proxy spec, the trajectory adversary searches for control sequences that satisfy the proxy STL spec (high rho) yet violate an unstated gold objective (low gold score). A successful find is a direct lower-bound on the per-task spec-completeness gap.

## Configuration

- Generated: 2026-04-25 03:18:55 UTC
- Random seed: 2026
- Adversary restarts: 8
- Adversary iterations per restart: 80
- Random population size: 200

## glucose_insulin -- glucose_insulin.tir.easy

### Adversary

- best spec rho       : `+36.0450`
- best gold score     : `-1.9229`
- restarts            : `8` (iter/restart: `80`)
- NaN/Inf events      : `12861`
- adversary converged : `False`
- per-restart finals  : [(36.076, -1.742), (36.003, -1.913), (20.0, -1.477), (20.0, -1.809), (20.0, -1.581), (36.045, -1.923), (36.047, -1.81), (36.079, -1.715)]
- wall time (s)       : `42.20`

### Random-population reference

- n trajectories               : `200`
- n spec-satisfying            : `200`
- Pearson r(rho, gold)         : `+0.0058`
- Spearman r(rho, gold)        : `+0.0185`
- regression slope (per sigma) : `+0.0016`
- top-decile gap (gold)        : `-0.0018`
- min gold among satisfying    : `-0.5116`
- mean gold among satisfying   : `+0.3498`
- FM2 flagged (Spearman < 0.3) : `True`
- wall time (s)                : `0.61`

### Empirical gap lower bound

- adv gold - mean(satisfying random gold): `-2.2727`
- adv gold - min(satisfying random gold) : `-1.4114`

---

## bio_ode.repressilator -- bio_ode.repressilator.easy

### Adversary

- best spec rho       : `+25.0000`
- best gold score     : `+0.9685`
- restarts            : `8` (iter/restart: `80`)
- NaN/Inf events      : `0`
- adversary converged : `True`
- per-restart finals  : [(25.0, 0.991), (25.0, 0.998), (25.0, 0.997), (25.0, 1.0), (25.0, 0.997), (25.0, 0.995), (25.0, 0.968), (25.0, 1.0)]
- wall time (s)       : `8.78`

### Random-population reference

- n trajectories               : `200`
- n spec-satisfying            : `0`
- Pearson r(rho, gold)         : `+0.1510`
- Spearman r(rho, gold)        : `+0.1380`
- regression slope (per sigma) : `+0.0226`
- top-decile gap (gold)        : `+0.0482`
- min gold among satisfying    : `+nan`
- mean gold among satisfying   : `+nan`
- FM2 flagged (Spearman < 0.3) : `True`
- wall time (s)                : `1.26`

### Empirical gap lower bound

- adv gold - mean(satisfying random gold): `+nan`
- adv gold - min(satisfying random gold) : `+nan`

---

## bio_ode.toggle -- bio_ode.toggle.medium

### Adversary

- best spec rho       : `-40.0000`
- best gold score     : `+0.4613`
- restarts            : `8` (iter/restart: `80`)
- NaN/Inf events      : `0`
- adversary converged : `True`
- per-restart finals  : [(-40.0, 0.461), (-40.0, 0.461), (-40.0, 0.461), (-40.0, 0.461), (-40.0, 0.461), (-40.0, 0.461), (-40.0, 0.461), (-40.0, 0.461)]
- wall time (s)       : `9.04`

### Random-population reference

- n trajectories               : `200`
- n spec-satisfying            : `0`
- Pearson r(rho, gold)         : `+0.0278`
- Spearman r(rho, gold)        : `-0.0237`
- regression slope (per sigma) : `+0.0003`
- top-decile gap (gold)        : `+0.0016`
- min gold among satisfying    : `+nan`
- mean gold among satisfying   : `+nan`
- FM2 flagged (Spearman < 0.3) : `True`
- wall time (s)                : `4.85`

### Empirical gap lower bound

- adv gold - mean(satisfying random gold): `+nan`
- adv gold - min(satisfying random gold) : `+nan`

---

## Interpretation

Per `paper/theory.md` S6, the spec-completeness term `R_gold(tau) - R_spec(tau)` is the Goodhart-relevant residual once the verifier-fidelity term is collapsed by the choice of formal STL robustness as the verifier. The adversary above provides a constructive lower bound on `sup_tau [R_spec(tau) - R_gold(tau)]` restricted to the spec-satisfying half-space. A negative gap (adv gold < mean satisfying random gold) indicates the spec is *locally* exploitable: there exist high-rho trajectories that score WORSE under the unstated gold than a random spec-satisfying baseline.

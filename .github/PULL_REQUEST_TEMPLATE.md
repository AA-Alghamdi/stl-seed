## Summary

<!-- One or two sentences. What does this PR change and why? -->

## Related issues

<!-- e.g. Closes #12, Refs #34. Use "Closes #N" so the issue auto-closes on merge. -->

## Testing done

- [ ] `make lint` passes
- [ ] `make typecheck` reviewed (warn-only; no new regressions)
- [ ] `make firewall` passes
- [ ] `make test` passes locally
- [ ] New / changed code has unit tests, or is exempt with justification:

<!-- Paste relevant pytest output, benchmark numbers, or before/after metrics. -->

## Breaking-change checklist

- [ ] No public API removals or signature changes, OR
- [ ] Breaking changes are documented in this PR description AND in `CHANGELOG`
      / docs, AND downstream callers in this repo are updated.

## REDACTED firewall checklist

(See `paper/REDACTED.md`, Parts E and F.)

- [ ] No new imports of `REDACTED`, `REDACTED`, `REDACTED`,
      `REDACTED`, `REDACTED` (or `*_v2` variants).
- [ ] No new mention in source / configs of: `CEGAR`, augmented Lagrangian,
      residual NN / NeuralODE, landscape smoothing, counter-example buffer,
      `_FILTER_WEIGHT`, `_NO_RESIDUAL_NN`, `_PARAM_SPACE`, `_JIT_TRACE_FIRED`,
      `PhysicalFilter`, `PhysicalPrior`.
- [ ] All new STL specs map to one of the allowed forms in
      `paper/REDACTED.md` §C.1.
- [ ] All new numerical kinetic constants are cited inline to a non-REDACTED
      external source (paper, database, derivation).

## Notes for reviewers

<!-- Anything reviewers should pay particular attention to (subtle invariants,
     numerical stability, JIT-trace ordering, GPU-only paths, etc.). -->

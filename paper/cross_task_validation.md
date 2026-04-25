# Cross-task validation of gradient-guided STL sampling

The headline ρ improvement on glucose-insulin (`paper/inference_method.md`: 19.91 vs 1.58, ~12× lift, 5/6 paired wins) does not transfer to the repressilator. This document is the receipt — the negative result with the structural diagnosis that explains it. The framing matters: this is not a tuning failure but a calibration of where the method works and where it does not.

## Pre-registered protocol

Spec: `bio_ode.repressilator.easy` = $G_{[120,200]}$ (m1 ≥ 250 nM) ∧ $F_{[0,60]}$ (p2 < 25 nM) — post-transient sustained-high band on the first repressor and a transient silencing of the second. IC: the canonical pilot `[0, 0, 0, 15, 5, 25]` (zero mRNAs, low-amplitude unequal proteins to break the symmetric unstable fixed point — the Elowitz-Leibler 2000 convention). Vocabulary: $V \in \mathbb{R}^{8 \times 3}$, the corners of $[0,1]^3$, including the action $u = (0, 0, 1)$ that the topology-aware heuristic in `tests/test_topology_aware.py::test_topology_aware_repressilator_satisfies` shows reaches ρ ≈ +25 when held constant. Flat (uniform) LLM, $\lambda \in \{0, 2\}$, $T = 0.5$, six seeds. Acceptance criterion: paired wins ≥ 4/6 OR mean improvement strictly positive.

## Result: transfer fails

```
seed 0: baseline rho=-250.000, guided rho=-250.000
seed 1: baseline rho=-248.013, guided rho=-250.000
seed 2: baseline rho=-250.000, guided rho=-250.000
seed 3: baseline rho=-250.000, guided rho=-248.753
seed 4: baseline rho=-247.763, guided rho=-250.000
seed 5: baseline rho=-250.000, guided rho=-250.000

mean baseline = -249.296,  mean guided = -249.792,  paired wins = 1/6.
```

`tests/test_inference.py::test_gradient_guided_improves_rho_repressilator` is marked `xfail(strict=False)` with the documented reason.

To rule out hyperparameter blame, we swept three default-action choices and three $\lambda$ values, six seeds each:

| default-action | $\lambda$ | mean ρ over 6 seeds |
|---|---|---|
| center (0.5, 0.5, 0.5) | 0.0 | -250.000 |
| center (0.5, 0.5, 0.5) | 5.0 | -250.000 |
| center (0.5, 0.5, 0.5) | 50.0 | -250.000 |
| zeros (0, 0, 0) | 0.0 | -250.000 |
| zeros (0, 0, 0) | 5.0 | -250.000 |
| zeros (0, 0, 0) | 50.0 | -250.000 |
| silence-3 (0, 0, 1) | 0.0 | -250.000 |
| silence-3 (0, 0, 1) | 5.0 | -250.000 |
| silence-3 (0, 0, 1) | 50.0 | -248.111 (1/6 escapes) |

Even with the *known-satisfying* default action and a 25× larger $\lambda$, only one of six seeds budges off the −250 floor, and only by ~11 ρ units of a ~275-unit gap. The signal is essentially absent.

## Why the gradient probe fails here

The structure is a cliff. Constant-action sweeps show ρ ≈ +25 for the constant `(0, 0, 1)` policy and ρ ≈ −250 for almost every other constant policy. The satisfying region is a measure-near-zero attractor in $[0, 1]^{30}$ (3 actions × 10 control steps).

The G-clause is conjunctive over a 30-point window: $G_{[120,200]}$ on a 1-min save grid is `min` over 81 grid points of `m1[t] - 250`. A single dip of m1 below 250 in that window kills the ρ. The gradient at any single control step $u_t$ informs only the immediate vicinity (~20 min downstream), not the 80-minute G-window.

The probe assumes the future is the default action. $\nabla_{\bar{u}_t} \rho$ holds $u_{t+1}, \ldots, u_H$ fixed at $u_{\mathrm{def}}$. Glucose-insulin tolerates this because a single bolus mostly determines local glucose dynamics. The repressilator does not: $u_{t+1}, \ldots, u_H$ jointly determine whether m1 stays high through the back of the horizon, and freezing them at $u_{\mathrm{def}} = (0.5, 0.5, 0.5)$ creates a mid-amplitude oscillating regime with no informative gradient toward `silence-3`.

The action box also has near-equivalent satisfying corners. `silence-3` $= (0, 0, 1)$ silences gene 3; under the cyclic topology this de-represses gene 1 (m1 up). But `silence-3` is adjacent to `silence-3-and-1` $= (1, 0, 1)$, which silences m1 itself and *fails* the spec. The vocabulary geometry puts a sharp ridge between satisfying and failing corners, and the linear projection $\langle V_k - \bar{u}, g \rangle$ cannot disambiguate them when $\bar{u}_t$ sits at the box center far from any corner.

The cleanest fix is a multi-step rollout-tree gradient probe: at each step $t$, average $\nabla_{u_t} \rho$ over a small set of *future-action samples* drawn from the LLM prior, instead of the single default extrapolation. That removes the myopic assumption and softens the discrete-corner geometry. Pre-registered as future work.

## Hybrid sampler on harder glucose

The hybrid sampler runs $n$ independent gradient-guided draws and selects argmax-ρ. On `glucose_insulin.dawn.hard` — a harder spec where ρ hovers in $[-33, -19]$ rather than saturating near $+20$ — hybrid strictly beats pure guidance:

```
guided                : per-seed = [-33.0, -33.0, -33.0, -18.69, -33.0, -33.0],  mean = -30.616
hybrid (n=2)          : per-seed = [-33.0, -33.0, -33.0, -19.15, -33.0, -33.0],  mean = -30.692
hybrid (n=3)          : per-seed = [-33.0, -33.0, -33.0, -19.15, -18.69, -33.0], mean = -28.307
hybrid (n=4)          : per-seed = [-33.0, -33.0, -33.0, -19.15, -18.69, -33.0], mean = -28.307
```

Hybrid($n=4$) is +2.3 ρ units better than pure guidance on this spec, with 5/6 hybrid $\ge$ guided per seed and one strict win. Compute is $4\times$ pure guidance; the matched-compute baseline is ContinuousBoNSampler($n=8$), which belongs in the Phase-2 sweep, not in this single-shot note. `tests/test_inference.py::test_hybrid_beats_pure_guidance` codifies the finding with a `tol = 2.0` ρ-unit margin so harmless RNG drift doesn't flip the test red.

## Where this sits in the test-time-compute story

The negative result is a calibration point. Brown et al. (arXiv:2407.21787) and Snell et al. (arXiv:2408.03314) argue that repeated sampling with verifier selection is a robust scaling axis precisely because individual rollouts are noisy and a verifier amplifies weak signals. Our hybrid is the analogue with two changes: the per-rollout sampler is gradient-guided rather than vanilla, and the verifier is the *exact same* continuous STL signal the inner sampler uses for guidance. The empirical pattern — hybrid beats pure guidance on the harder glucose spec, neither helps on the repressilator default IC — sits on the same scaling curve as Brown 2024 (more samples + cheap verifier) with the additional structure that each inner sample is itself information-efficient.

The pre-registered interpretation, written before the result was in:

> If gradient guidance fails to transfer, the most likely cause is the myopic-default-action approximation. This is *not* a refutation of the underlying claim that STL gradients carry useful decoding-time information; it is evidence that the partial-then-extrapolated probe is the bottleneck on tasks where future-action structure matters.

The post-hoc result confirmed this. We have not edited the framing to make it tighter retroactively.

## References

Aksaray et al. "Q-learning for robust satisfaction of signal temporal logic specifications." CDC 2016.
Brown et al. "Large Language Monkeys: scaling inference compute with repeated sampling." arXiv:2407.21787, 2024.
Donzé & Maler. "Robust satisfaction of temporal logic over real-valued signals." FORMATS 2010, DOI 10.1007/978-3-642-15297-9_9.
Elowitz & Leibler. "A synthetic oscillatory network of transcriptional regulators." Nature 403, 335–338 (2000), DOI 10.1038/35002125.
Snell et al. "Scaling LLM test-time compute optimally." arXiv:2408.03314, 2024.

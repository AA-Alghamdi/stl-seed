# Getting started with `stl-seed`

This guide walks a reader who has never seen `stl-seed` from a clean
clone to a working filtered-SFT pipeline. Read it top-to-bottom; every
command is meant to be pasted directly.

## What is `stl-seed`?

`stl-seed` is a research artifact testing whether SERA-style soft
verification (Shen et al., arXiv:2601.20789) extends to scientific
control. The agent (a small open-weights LLM) emits a piecewise-constant
control schedule `u_{1:H}` for an ODE-driven biomolecular or
physiological system. The simulator integrates the trajectory; the STL
evaluator scores it under a textbook spec; and the SFT loss is filtered
or weighted by the resulting Donzé-Maler robustness margin `rho` so that
gradient updates favour schedules that satisfy the formal spec. Two
training backends are interchangeable: MLX on Apple Silicon (local
iteration) and bitsandbytes on CUDA (RunPod / canonical sweep).

## 1. Install

`stl-seed` uses [uv](https://docs.astral.sh/uv/) for environment
management and Python 3.11. The two backends are optional extras —
install the one matching your hardware. The `dev` extra adds the test
runner, linter, type checker, and pre-commit.

On Apple Silicon (M-series Mac):

```bash
git clone https://github.com/AA-Alghamdi/stl-seed
cd stl-seed
uv sync --extra mlx --extra dev
```

On Linux with CUDA:

```bash
git clone https://github.com/AA-Alghamdi/stl-seed
cd stl-seed
uv sync --extra cuda --extra dev
```

On any host (CPU only — examples 01 / 02 still work, example 03 will
fail-fast with a clear error):

```bash
uv sync --extra dev
```

Verify the install:

```bash
uv run stl-seed version
# 0.0.1
```

## 2. Run the demo CLI

```bash
uv run stl-seed demo --task glucose_insulin
```

The Phase 1 CLI is a stub that exits with `Demo not yet implemented`;
the canonical end-to-end demo lives in
[`examples/01_basic_simulation.py`](../examples/01_basic_simulation.py)
through [`examples/03_mlx_training_minimal.py`](../examples/03_mlx_training_minimal.py),
covered in §4 below. (The Hydra-configured `stl-seed train`,
`stl-seed evaluate`, and `stl-seed analyze` subcommands are wired in
Phase 2.)

## 3. The four core concepts

### 3.1 `Simulator`

A `Simulator` is a pure-JAX/Diffrax ODE integrator that takes an
initial state, a piecewise-constant control schedule, a literature-fixed
parameter object, and a PRNG key, and returns a `Trajectory` pytree
(states, actions, times, NaN-replacement count). Every task family
implements the same `simulate(initial_state, control_sequence, params,
key) -> Trajectory` signature defined in
[`paper/architecture.md`](../paper/architecture.md) §"Simulator
interface".

```python
from stl_seed.tasks.bio_ode import (
    RepressilatorSimulator,
    default_repressilator_initial_state,
)
from stl_seed.tasks.bio_ode_params import RepressilatorParams
import jax, jax.numpy as jnp

sim = RepressilatorSimulator()              # 200-min horizon, 10 control points
params = RepressilatorParams()              # Elowitz & Leibler 2000 defaults
x0 = default_repressilator_initial_state(params)
u = jnp.zeros((sim.n_control_points, sim.action_dim))   # no inducer
traj = sim.simulate(x0, u, params, jax.random.key(0))
print(traj.states.shape, traj.actions.shape, traj.times.shape)
# (201, 6) (10, 3) (201,)
```

Four simulators ship today: `GlucoseInsulinSimulator` (Bergman 1979 +
Dalla Man 2007), `RepressilatorSimulator` (Elowitz & Leibler 2000),
`ToggleSimulator` (Gardner et al. 2000), and `MAPKSimulator` (reduced
Huang & Ferrell 1996 / Markevich et al. 2004 form).

### 3.2 `STLNode` / `STLSpec`

An `STLSpec` wraps a backend-agnostic AST built from `Predicate`,
`Negation`, `And`, `Always[a, b]`, and `Eventually[a, b]` nodes (the
conjunction-only fragment from
[`paper/REDACTED.md`](../paper/REDACTED.md) §C.1). Six
specs are registered today, three per family, calibrated to the
textbook clinical / biological thresholds documented in
[`paper/REDACTED.md`](../paper/REDACTED.md).

```python
from stl_seed.specs import REGISTRY

spec = REGISTRY["glucose_insulin.tir.easy"]
print(spec.formula_text)
# G_[30,120] (G >= 70) AND G_[30,120] (G < 180)
print(spec.citations[0])
# ADA 2024 Standards of Care, Recommendation 6.5b ...
```

The full registered set is `glucose_insulin.{tir.easy, no_hypo.medium,
dawn.hard}` and `bio_ode.{repressilator.easy, toggle.medium, mapk.hard}`.

### 3.3 `Policy`

A `Policy` is any callable conforming to `(state, spec, history, key) ->
action`. The library ships `RandomPolicy`, `ConstantPolicy`,
`PIDController`, `BangBangController`, `MLXModelPolicy` (LLM-backed
on Apple Silicon), and a `HeuristicPolicy` router that selects a
sensible hand-coded controller per task family.

```python
from stl_seed.generation.policies import HeuristicPolicy

policy = HeuristicPolicy(task_family="glucose_insulin")
# routes to a PID with literature-anchored gains
```

The policy contract is open-loop in the SERA setting: the runner asks
the policy for `H` actions in sequence (each conditioned on the initial
state plus the running history of `(state, action)` pairs), assembles
them into the schedule, and integrates the trajectory in one shot.

### 3.4 `FilterCondition`

A `FilterCondition` consumes a list of trajectories and a parallel
vector of robustness scores, and returns the kept subset plus a
per-trajectory weight. Three implementations exist:

* `HardFilter(threshold)` — keep `rho > threshold`, weights uniform.
* `QuantileFilter(top_k_pct)` — keep the top-K%, weights uniform.
* `ContinuousWeightedFilter(temperature)` — keep all trajectories,
  weights proportional to `softmax(rho / beta)` then rescaled to mean 1.

```python
from stl_seed.filter.conditions import HardFilter

hf = HardFilter(rho_threshold=0.0)
kept, weights = hf.filter(trajectories, rhos)
```

All three raise `FilterError` rather than silently degrade when the
kept subset is below `min_kept` — the project rule "no silent
calibration failure" from
[`paper/theory.md`](../paper/theory.md) §7 (FM2).

## 4. End-to-end pipeline

The three runnable examples implement the full pipeline. Each is
self-contained; later examples reuse the patterns from earlier ones.

### 4.1 Single trajectory + STL score (example 01)

```bash
uv run python examples/01_basic_simulation.py
```

Expected first lines (numerical values reproducible from `seed=0`):

```
Bergman 1979 + Dalla Man 2007 (single-meal challenge)
  horizon         : 120.0 min, 12 control points
  ...
Open-loop schedule A: zero infusion (no exogenous insulin)
  glucose range : [90.0, 158.8] mg/dL
  robustness rho: +21.227 mg/dL  (SATISFIES with margin 21.227)
```

### 4.2 Generate + filter (example 02)

```bash
uv run python examples/02_stl_filtering.py
```

This generates 50 glucose-insulin trajectories under a `random + heuristic`
policy mix, scores each, and applies all three filter conditions side by
side. The output is a comparison table that reads top-to-bottom across
filter densities.

### 4.3 Train an adapter (example 03 — Apple Silicon only)

```bash
uv run python examples/03_mlx_training_minimal.py
```

Generates 50 glucose-insulin trajectories, filters with `HardFilter`,
renders each as a chat conversation, runs 10 LoRA iterations on
`mlx-community/Qwen3-0.6B-bf16`, saves the adapter under
`runs/example_03/`, and reloads + decodes one held-out sample. Wall
clock ~1-2 min on M5 Pro / 48 GB unified memory.

For a longer (50-iter) version with parse-checking and a written
report, run [`scripts/smoke_test_mlx.py`](../scripts/smoke_test_mlx.py)
instead.

### 4.4 The Phase 1 pilot drivers (`scripts/`)

The `scripts/` directory holds the canonical pilot pipeline:

```bash
# A13 — generate a 4000-trajectory store under data/pilot/
uv run python scripts/generate_pilot.py

# A14 — apply HardFilter to the glucose-insulin slice
uv run python scripts/filter_pilot.py

# A15 — MLX QLoRA on the filtered subset (HARD CHECKPOINT)
uv run python scripts/smoke_test_mlx.py
```

The Phase 1 status (which scripts have run cleanly, with what numbers)
is recorded in [`paper/REDACTED.md`](../paper/REDACTED.md).

## 5. Where to look next

* **Module-by-module API**: [`docs/api_reference.md`](api_reference.md).
* **Architecture lock**: [`paper/architecture.md`](../paper/architecture.md)
  — the canonical interface contracts that every module implements.
* **Theoretical claim + statistical model**:
  [`paper/theory.md`](../paper/theory.md).
* **Spec design + threshold sourcing**:
  [`paper/REDACTED.md`](../paper/REDACTED.md).
* **REDACTED firewall posture**:
  [`paper/REDACTED.md`](../paper/REDACTED.md).
* **SERA recipe mapping**:
  [`paper/REDACTED.md`](../paper/REDACTED.md).
* **Tests as documentation**: `tests/test_stl_evaluator.py` is the most
  pedagogical entry point — it builds tiny ASTs by hand and asserts
  Donzé-Maler space-robustness equalities one node at a time.

## 6. Reading the registered specs

The six registered STL specifications cover three difficulties per task
family. The metadata is machine-readable and citation-anchored — every
threshold has a literature source recorded inline.

```python
from stl_seed.specs import REGISTRY

for key in sorted(REGISTRY):
    spec = REGISTRY[key]
    print(f"{key:42s}  ({spec.metadata['difficulty']})")
    print(f"  formula: {spec.formula_text}")
    print(f"  citations: {len(spec.citations)} sources")
```

The conjunction-only fragment used here (`Always`, `Eventually`,
n-ary `And`, predicate-level `Negation`) is documented in
[`paper/REDACTED.md`](../paper/REDACTED.md) §C.1; the
threshold sourcing convention is documented in
[`paper/REDACTED.md`](../paper/REDACTED.md). When you add a new
spec (see `CONTRIBUTING.md`), the AST builders ship in
`stl_seed.specs.{Predicate, Negation, And, Always, Eventually,
Interval}` and the predicate convention is
`lambda traj, t, c=CHANNEL, th=THRESHOLD: traj[t, c] - th`. Do not break
the convention — the STL evaluator's introspection path keys off the
lambda's `__defaults__` to compile a JIT-traceable JAX expression.

## 7. Reading and interpreting the robustness margin

The Donzé-Maler space-robustness `rho(spec, trajectory)` returned by
`stl_seed.stl.evaluate_robustness` is the signed *margin* by which the
trajectory satisfies (or violates) the spec, in the spec's natural
units:

* `rho > 0`: trajectory satisfies `spec` with margin `rho`. The
  smallest predicate-level signed margin over the spec's temporal
  intervals.
* `rho < 0`: trajectory violates `spec` with margin `|rho|`. The most
  negative predicate-level signed margin over the spec's temporal
  intervals.
* `rho == 0`: trajectory is on the boundary.

Margins are in the units of the dominant predicate. For
`glucose_insulin.tir.easy` with `formula = G_[30, 120] (G >= 70) AND
G_[30, 120] (G < 180)`, the unit is mg/dL — example 01's
`rho = +21.227` says the worst-case glucose excursion in the [30, 120]
window is 21.2 mg/dL inside the [70, 180] band. For
`bio_ode.repressilator.easy` the unit is monomers per cell. For
heterogeneous-unit specs the smallest-unit predicate dominates, which
is why every spec is calibrated against `paper/REDACTED.md`'s
"random-policy success rate by difficulty" target before it is
registered.

## 8. Common gotchas

* `GlucoseInsulinSimulator.simulate(...)` returns a 3-tuple
  `(states, times, meta)` — not a `Trajectory` pytree. The
  `TrajectoryRunner` adapter wraps it; if you call the simulator
  directly (as example 01 does), wrap by hand.
* The bio_ode simulators DO return a `Trajectory` directly. The
  asymmetry is documented and load-bearing — do not "fix" it without
  updating the runner adapter (`generation/runner.py:_simulate_one`).
* Glucose-insulin's control input is a 1-D `(H,)` array of insulin
  rates in U/h; bio_ode's are 2-D `(H, m)` arrays of dimensionless
  inducer fractions in `[0, 1]`. The simulators clip on entry, so
  out-of-range policies do not crash but lose every signal beyond the
  clip.
* `MLXModelPolicy` raises a clear `RuntimeError` on non-Apple-Silicon
  hosts at construction time. Do not import it on Linux/CUDA — use
  the bnb backend's eval shim instead.
* JIT-sensitive globals: there are none in `stl-seed` (in contrast to
  the REDACTED `REDACTED.py` codebase). The STL evaluator compiles
  the AST once and the closure is JIT-compatible iff every predicate
  matches the `_gt`/`_lt` introspection convention; check the
  `_FALLBACK_USED` attribute on the compiled function to verify.

# `stl-seed` API reference

Stub-style reference for the eight public subpackages under
`src/stl_seed/`. For full docstrings, follow the source links — every
public symbol is documented in-place. The contracts below summarize the
shape of each module without duplicating its docstrings.

---

## `stl_seed.tasks`

ODE simulators + parameter dataclasses + the canonical `Trajectory`
pytree.

### Imports

```python
from stl_seed.tasks import (
    # Trajectory pytree
    Trajectory, TrajectoryMeta,
    # Glucose-insulin family
    BergmanParams, GlucoseInsulinSimulator, MealSchedule,
    default_normal_subject_initial_state, single_meal_schedule,
    U_INSULIN_MIN_U_PER_H, U_INSULIN_MAX_U_PER_H,
    # Bio_ode family
    Simulator,                               # runtime-checkable Protocol
    RepressilatorSimulator, RepressilatorParams,
    ToggleSimulator, ToggleParams,
    MAPKSimulator, MAPKParams,
    default_repressilator_initial_state,
    default_toggle_initial_state,
    default_mapk_initial_state,
)
```

### Key types

* `Trajectory(states, actions, times, meta)` — `equinox.Module` pytree,
  shapes `(T, n)`, `(H, m)`, `(T,)`, plus `TrajectoryMeta`. Defined in
  `stl_seed/tasks/_trajectory.py`. The canonical type passed between
  every pipeline stage.
* `TrajectoryMeta(n_nan_replacements, final_solver_result, used_stiff_fallback)`
  — all fields are 0-d JAX arrays, jit-friendly.
* `Simulator` — runtime-checkable `Protocol` requiring
  `simulate(initial_state, control_sequence, params, key) -> Trajectory`
  plus `state_dim`, `action_dim`, `horizon` properties.

### Public functions / classes

* `GlucoseInsulinSimulator(horizon_min=120.0, n_control_points=12,
  n_save_points=121, rtol=1e-6, atol=1e-9, max_steps=16384)` — Bergman
  1979 minimal model with a Dalla Man 2007 oral-meal disturbance.
  Action is a 1-D `(H,)` insulin infusion schedule in U/h; clipped to
  `[U_INSULIN_MIN_U_PER_H, U_INSULIN_MAX_U_PER_H] = [0, 5]`.
  **Returns a 3-tuple `(states, times, meta)`, NOT a `Trajectory`** —
  the runner adapter wraps it.
* `RepressilatorSimulator(horizon_minutes=200.0, n_control_points=10,
  n_save_points=201, solver="tsit5", ...)` — Elowitz & Leibler 2000
  6-state oscillator. Action is `(H, 3)` per-gene inducer fractions in
  `[0, 1]`. Returns `Trajectory`.
* `ToggleSimulator(horizon_minutes=100.0, ...)` — Gardner-Cantor-Collins
  2000 2-state toggle. Action is `(H, 2)` IPTG/aTc fractions in `[0, 1]`.
  Returns `Trajectory`.
* `MAPKSimulator(horizon_minutes=60.0, ...)` — Reduced Huang & Ferrell
  1996 / Markevich 2004 6-state Michaelis-Menten cascade. Action is
  `(H, 1)` E1-stimulus fraction in `[0, 1]`. Returns `Trajectory`.
* `BergmanParams`, `RepressilatorParams`, `ToggleParams`, `MAPKParams`
  — `equinox.Module` parameter containers; every default is cited
  inline to a published paper or biological database
  (paper/REDACTED.md).
* `MealSchedule(onset_times_min, carb_mass_mg)` — NamedTuple of
  per-meal events; `MealSchedule.empty(n_slots=4)` for the no-meal
  case; `single_meal_schedule(onset_min, carb_grams)` for one meal.
* `default_*_initial_state(params=None)` — literature-default initial
  state for each family. For the bio_ode family, the proteins / mRNA
  start at the convention used in the source paper's Fig. 1 / 2.

Source: [`src/stl_seed/tasks/`](../src/stl_seed/tasks/).

---

## `stl_seed.specs`

Backend-agnostic STL AST + per-family registered specifications.

### Imports

```python
from stl_seed.specs import (
    # AST nodes (conjunction-only fragment per firewall §C.1)
    Predicate, Negation, And, Always, Eventually, Interval, Node,
    # Spec wrapper + registry
    STLSpec, REGISTRY, register,
)
```

### AST nodes

All nodes are frozen dataclasses:

* `Predicate(name, fn)` — atomic, `fn(states, t) >= 0` is the satisfaction
  condition. The signed value `fn(states, t)` is the predicate-level
  Donzé-Maler robustness.
* `Negation(inner)` — wraps a `Predicate` only (raises `TypeError` on a
  non-`Predicate` inner; firewall §C.1 forbids De Morgan-induced
  disjunction).
* `And(children)` — n-ary conjunction; requires `len(children) >= 2`.
* `Always(inner, interval)` — `G_[t_lo, t_hi] inner`.
* `Eventually(inner, interval)` — `F_[t_lo, t_hi] inner`.
* `Interval(t_lo, t_hi)` — closed bounded interval in minutes.
* `Node = Predicate | Negation | Always | Eventually | And` — type alias.

### Registry

`REGISTRY: dict[str, STLSpec]` is populated at import time. The six
keys are:

| Key | Family | Difficulty | Form |
|---|---|---|---|
| `glucose_insulin.tir.easy` | glucose-insulin | easy | TIR conjunction |
| `glucose_insulin.no_hypo.medium` | glucose-insulin | medium | TIR + hypo / hyper avoidance |
| `glucose_insulin.dawn.hard` | glucose-insulin | hard | reach + track + 3 avoidance |
| `bio_ode.repressilator.easy` | repressilator | easy | per-gene tracking |
| `bio_ode.toggle.medium` | toggle | medium | switch + safety |
| `bio_ode.mapk.hard` | MAPK | hard | reach saturation |

Each `STLSpec` carries `formula: Node`, `signal_dim`, `horizon_minutes`,
`description`, `citations`, `formula_text`, and a `metadata` dict.
Every numerical threshold is cited inline at the point of use to the
relevant biological / clinical literature.

Source: [`src/stl_seed/specs/`](../src/stl_seed/specs/).

---

## `stl_seed.stl`

JAX-native Donzé-Maler STL space-robustness evaluator + streaming +
worst-subformula localization.

### Imports

```python
from stl_seed.stl import (
    Trajectory,                      # duck-typed Protocol (states + times)
    compile_spec,                    # AST -> JAX closure
    evaluate_robustness,             # full-trajectory rho
    evaluate_streaming,              # partial-trajectory rho (lower bound)
    worst_violating_subformula,      # localization
)
```

### Public functions

* `evaluate_robustness(spec, trajectory) -> Float[Array, ""]`
  — Donzé-Maler `rho` at trajectory time `t = 0`. Recursive on the AST,
  O(T) per node, JIT-compatible iff every predicate matches the
  `_gt`/`_lt` introspection convention (which holds for every spec in
  `REGISTRY`). Returns a scalar JAX array; positive `rho` iff the
  trajectory satisfies the spec with margin `rho`.
* `compile_spec(spec) -> CompiledSpec` — returns a closure
  `(states, times) -> rho_scalar`. Inspect the `_FALLBACK_USED`
  attribute to verify JIT compatibility.
* `evaluate_streaming(spec, trajectory, current_time) -> Float[Array, ""]`
  — partial-trajectory rho usable inside the agent loop. Returns a
  *lower bound* on the eventual rho so the LLM can abort or
  course-correct mid-trajectory. Returns `+inf` for not-yet-activated
  Always operators and `-inf` for not-yet-activated Eventually
  operators (by design — see `streaming.py` module docstring for the
  algebraic identity argument).
* `worst_violating_subformula(spec, trajectory) -> tuple[Node, float, float]`
  — locates the subformula with the lowest rho and the time-of-min,
  used by `MLXModelPolicy` to format natural-language verifier
  feedback ("Spec G_[120, 200] (p1 >= 250) violated at t = 145.2 by
  margin 0.234").

Source: [`src/stl_seed/stl/`](../src/stl_seed/stl/).

---

## `stl_seed.generation`

Trajectory generation pipeline: policies, runner, store.

### Imports

```python
from stl_seed.generation import (
    # Policies (all conform to (state, spec, history, key) -> action)
    RandomPolicy, ConstantPolicy, PIDController, BangBangController,
    HeuristicPolicy, MLXModelPolicy,
    # Orchestration
    TrajectoryRunner, TrajectoryStore,
)
```

### Policies

* `RandomPolicy(action_dim, action_low, action_high)` — uniform
  `[low, high]` per step, deterministic in `key` via `fold_in(key, step)`.
* `ConstantPolicy(value)` — same action every step; useful as a
  zero-control baseline.
* `PIDController(setpoint, kp, ki, kd, action_clip, action_dim)` —
  classical PID on `state[0]`; literature defaults targeted at the
  glucose-insulin task (gains from Marchetti et al. 2008 IEEE TBME).
* `BangBangController(threshold, low_action, high_action, action_dim,
  observation_indices)` — per-channel bang-bang; default for bio_ode.
* `HeuristicPolicy(task_family, overrides=None)` — routes to the
  literature-anchored hand-coded controller for each task family
  (`pid` for glucose-insulin, `bangbang` for the three bio_ode
  families).
* `MLXModelPolicy(model_path, tokenizer_path, prompt_template,
  action_dim, max_tokens)` — LLM-backed via `mlx_lm.load` + chat
  prompting. Apple Silicon only — raises `RuntimeError` at
  construction time on non-Darwin/arm64. Gracefully returns a zero
  action on parse failure (the simulator's clip is the authoritative
  bound).

### Orchestration

* `TrajectoryRunner(simulator, spec_registry, output_store, *,
  initial_state, horizon, action_dim, aux, nan_fraction_threshold=0.1,
  sim_params)` — composes simulator + spec + initial state + meal
  schedule (or other aux) and exposes
  `generate_trajectories(task, n, policy_mix, key, *, spec_key=None,
  policy_factories=None) -> (trajectories, metadata)`. Tracks
  `last_stats: _RunnerStats` (n_requested, n_kept, n_nan_dropped,
  n_failed, nan_rate). Drops trajectories whose
  `meta.n_nan_replacements / T > nan_fraction_threshold`.
* `TrajectoryStore(root)` — append-only Parquet shards of trajectories
  with `save(trajectories, metadata) -> Path`,
  `load() -> list[(Trajectory, dict)]`,
  `get_by_id(trajectory_id) -> (Trajectory, dict) | None`.
  Each shard is one Parquet file; concurrent reads are safe;
  per-trajectory IDs are looked up via an in-memory index built by
  `load(...)`.

Source: [`src/stl_seed/generation/`](../src/stl_seed/generation/).

---

## `stl_seed.filter`

STL-driven SFT-density filter conditions + HuggingFace `Dataset` builder.

### Imports

```python
from stl_seed.filter import (
    HardFilter, QuantileFilter, ContinuousWeightedFilter,
    FilterError, build_sft_dataset,
)
```

### Filter conditions

All implement `filter(trajectories, robustness) -> (kept, weights)`
and raise `FilterError` (a `ValueError`) when the kept subset would
be below `min_kept` (default 10) — no silent fallback.

* `HardFilter(rho_threshold=0.0, min_kept=10)` — keep `rho > threshold`,
  weights = 1.0. SERA / RFT baseline.
* `QuantileFilter(top_k_pct=25.0, min_kept=10)` — keep top-K% by rho,
  weights = 1.0. Uses Hamilton's largest-remainders rounding.
* `ContinuousWeightedFilter(temperature=None, min_kept=10)` — keep all,
  weights = `N * softmax(rho / beta)` with `beta = std(rho)` if
  `temperature is None`. Rescaled so the *expected* per-trajectory
  weight is 1.0, matching the L_hard / L_quantile loss scale.

### Dataset builder

* `build_sft_dataset(filtered, weights, spec, task, *, formatter=None)
  -> datasets.Dataset` — converts a filtered trajectory list +
  weights into a HuggingFace `Dataset` of records with at least
  `prompt: str`, `completion: str`, `weight: float`,
  `trajectory_id: str`, `spec_key: str` columns. The text rendering
  uses `stl_seed.training.tokenize.format_trajectory_as_text` when
  available, otherwise an in-package fallback formatter that produces
  the same JSON-schema output.

Source: [`src/stl_seed/filter/`](../src/stl_seed/filter/).

---

## `stl_seed.training`

Backend-agnostic SFT training layer — MLX (Apple Silicon) and
bitsandbytes (CUDA).

### Imports

```python
from stl_seed.training import (
    TrainingBackend, TrainingConfig, TrainedCheckpoint,
    train_with_filter,
)
from stl_seed.training.backends.mlx import MLXBackend
from stl_seed.training.backends.bnb import BNBBackend
from stl_seed.training.tokenize import format_trajectory_as_text
from stl_seed.training.prompts import render_system_prompt, list_tasks
```

### Key types

* `TrainingConfig(base_model="Qwen/Qwen3-0.6B-Instruct",
  learning_rate=5e-5, lr_schedule="cosine", warmup_ratio=0.1,
  num_epochs=3, batch_size=1, gradient_accumulation_steps=4,
  max_seq_length=8192, lora_rank=32, lora_alpha=128.0,
  lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
  "gate_proj", "up_proj", "down_proj"], lora_dropout=0.0, seed=42,
  output_dir=Path("runs/default"), weight_format="nf4",
  use_8bit_optimizer=True, weight_decay=0.01)` — frozen dataclass; every
  default is sourced to SERA's `unsloth_qwen3_moe_qlora.yaml`
  (`paper/REDACTED.md` §C.3) or to the QLoRA paper [Dettmers et al.,
  arXiv:2305.14314].
* `TrainedCheckpoint(backend, model_path, base_model,
  training_loss_history, wall_clock_seconds, metadata)` — frozen
  dataclass returned by every backend.
* `TrainingBackend` — runtime-checkable `Protocol` with
  `name: str`, `train(base_model, dataset, config, output_dir) -> TrainedCheckpoint`,
  `load(checkpoint) -> Callable[..., Any]`.

### Backends

* `MLXBackend()` — Apple Silicon via `mlx_lm.tuner.trainer.train`.
  Lazy-imports `mlx` / `mlx_lm` inside `train(...)`; constructing the
  object on a CPU / CUDA host does not raise. The wrapper currently
  targets the pre-0.20 `mlx_lm` API and has a known follow-up patch
  documented in `paper/REDACTED.md` §"Phase-2 followups";
  the canonical MLX path that the smoke test exercises is in
  [`scripts/smoke_test_mlx.py`](../scripts/smoke_test_mlx.py).
* `BNBBackend()` — CUDA via `trl.SFTTrainer` + `bitsandbytes` 4-bit
  (NF4 + bf16 compute). Lazy-imports `torch` / `transformers` /
  `trl` / `bitsandbytes` inside `train(...)`; raises a clear error on
  non-Linux / non-CUDA hosts.

### Helpers

* `train_with_filter(filter_condition, task, model, backend,
  config=None, *, dataset=None) -> TrainedCheckpoint` — the single
  end-to-end entry point used by the CLI (Phase 2). Loads the
  filtered dataset for `(filter_condition, task)` if `dataset is None`
  and dispatches to the named backend.
* `format_trajectory_as_text(trajectory, spec, task) -> dict` —
  renders a `Trajectory` into a `{"system", "user", "assistant"}`
  chat dict consumed by both backends. Each per-step record is
  `<state>v1,...,vn</state><action>u1,...,um</action>` in 4-sig-fig
  scientific notation.
* `render_system_prompt(task)` / `list_tasks()` — task-specific system
  prompts surface in `stl_seed.training.prompts`.

Source: [`src/stl_seed/training/`](../src/stl_seed/training/).

---

## `stl_seed.evaluation`

Held-out eval harness + metrics + parallel runner.

### Imports

```python
from stl_seed.evaluation import (
    EvalHarness, EvalResults, EvalRunner,
    success_rate, bon_success, bon_success_curve, rho_margin,
    goodhart_gap,
)
```

### Key types

* `EvalHarness(simulator, stl_evaluator, specs, n_samples_per_spec,
  bon_budgets=DEFAULT_BON_BUDGETS)` — `DEFAULT_BON_BUDGETS = (1, 2, 4,
  8, 16, 32, 64, 128)`. Provides
  `evaluate(checkpoint, key) -> EvalResults`. Operates on
  `SimulatorProtocol` + `CheckpointProtocol` so it is unit-testable
  against synthetic stand-ins.
* `EvalResults` — frozen dataclass with `per_spec: dict[str,
  PerSpecResult]`, `aggregate_success_rate: float`, `wall_clock_seconds`.
* `PerSpecResult(spec_name, n_samples, rhos, seeds, ...)` — per-spec
  rho vector, seeds, BoN curves at the registered budgets.
* `EvalRunner(harness, output_dir)` — parallel multi-checkpoint driver
  emitting `RunRecord` entries to a JSONL log under `output_dir`.

### Metrics

* `success_rate(rhos) -> float` — fraction with `rho > 0`; non-finite
  values excluded.
* `bon_success(rhos_per_seed, n) -> float` — `Pr[max of n samples > 0]`,
  computed by sample reuse on the first `n` columns of a
  `(n_seeds, K)` array.
* `bon_success_curve(rhos_per_seed, budgets) -> dict[int, float]` —
  the BoN-success vector at all budgets in one pass.
* `rho_margin(rhos) -> tuple[float, float]` — `(mean, IQR)`.
* `goodhart_gap(rhos_proxy, rhos_gold) -> float` — `mean(σ(rho_proxy))
  − mean(σ(rho_gold))`, the operational definition of the
  spec-completeness term in `paper/theory.md` §6.

Source: [`src/stl_seed/evaluation/`](../src/stl_seed/evaluation/).

---

## `stl_seed.stats`

Bootstrap CIs, NumPyro hierarchical Bayes, TOST equivalence test.

### Imports

```python
from stl_seed.stats import (
    BootstrapCI, bootstrap_mean_ci, bootstrap_diff_ci,
    bootstrap_proportion_ci,
    HierarchicalData, model, fit, summarize, convergence_check,
    TOSTResult, tost_equivalence,
)
```

### Bootstrap

* `BootstrapCI(estimate, lower, upper, level, method, n_resamples,
  bootstrap_distribution)` — frozen dataclass.
* `bootstrap_mean_ci(x, level=0.95, n_resamples=10_000, method="bca",
  rng=None) -> BootstrapCI` — three methods: `"bca"`
  (bias-corrected and accelerated, preferred), `"percentile"`,
  `"basic"` / `"pivotal"`.
* `bootstrap_diff_ci(x, y, level=0.95, n_resamples=10_000,
  method="bca", paired=False, rng=None) -> BootstrapCI` — paired or
  unpaired difference of means.
* `bootstrap_proportion_ci(successes, n, level=0.95, n_resamples=10_000,
  method="bca", rng=None) -> BootstrapCI` — proportion CI from
  Bernoulli outcomes; `proportion_wilson_ci(...)` is the closed-form
  Wilson alternative.

### Hierarchical Bayes

* `HierarchicalData(...)` — input container for the multi-level model.
* `model(data)` — the NumPyro model from `paper/theory.md` §4:
  trial-level Bernoulli outcomes linked to a saturating power-law
  BoN curve `p(N) = A * (1 - N^{-b})`, with `logit A` and `log b`
  decomposed into model-size, task-family, filter-condition,
  model×family interaction, and instance-level effects.
* `fit(data, *, num_warmup, num_samples, num_chains, ...) ->
  arviz.InferenceData` — runs NUTS via `numpyro.infer.MCMC`.
* `summarize(idata) -> pandas.DataFrame` — per-parameter HDI / mean / sd.
* `convergence_check(idata) -> dict[str, float]` — R-hat / ESS gates.

### TOST

* `TOSTResult(p_lower, p_upper, p_value, ci_lower, ci_upper,
  equivalent, ...)` — frozen dataclass.
* `tost_equivalence(x, y, *, lower_bound, upper_bound, alpha=0.05)
  -> TOSTResult` — Schuirmann's two one-sided tests for equivalence
  (the formal test for hypothesis H1, `paper/theory.md` §3).

Source: [`src/stl_seed/stats/`](../src/stl_seed/stats/).

---

## Module dependency graph

```
tasks  -->  generation  -->  filter  -->  training
specs  -->  stl  -----^                   |
                       \                  v
                        ----- evaluation  --> stats
```

Every module is importable on a CPU-only laptop without `mlx` or
`bitsandbytes`. Heavy native dependencies (mlx_lm, torch, trl, bnb) are
imported lazily inside `train(...)` / `MLXModelPolicy.__init__`. The
project's REDACTED firewall (`paper/REDACTED.md` §F) is enforced by a
single grep over `src/`:

```bash
git grep -nIE 'REDACTED|REDACTED|REDACTED|REDACTED|REDACTED|CEGAR|residual.nn|augmented.lagrang' src/
```

— must return zero hits.

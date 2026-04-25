# Contributing to `stl-seed`

Thank you for your interest in `stl-seed`. This document covers the
development setup, the project's quality gates, and the two most common
extension paths (adding a task family and adding an STL spec). Read it
end-to-end before opening your first PR.

## Code of conduct

`stl-seed` is released under the Apache 2.0 license and follows the
ecosystem-standard Apache code of conduct: be respectful, assume good
faith, and prioritise constructive technical discussion. Harassment,
ad hominem attacks, and intellectual dishonesty are not tolerated.

## Development setup

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.11. The
two backends are optional extras — install whichever matches your
hardware. The `dev` extra adds the test runner, linter, type checker,
and pre-commit hooks.

On Apple Silicon (M-series Mac):

```bash
uv sync --extra mlx --extra dev
```

On Linux with CUDA (RunPod, on-prem H100, etc.):

```bash
uv sync --extra cuda --extra dev
```

Once dependencies are installed, set up the pre-commit hooks once per
clone:

```bash
uv run pre-commit install
```

This installs `ruff` formatting + lint, `pyright` type-check, the
forbidden-string firewall grep, and the standard end-of-file / trailing
whitespace fixers as commit-time gates.

## Running the quality gates

Everything below should be runnable from the repo root with no extra
flags.

Tests (the default suite skips slow / GPU / MLX / CUDA-marked tests):

```bash
uv run pytest tests/ -q
```

Run only the tests you care about:

```bash
uv run pytest tests/test_stl_evaluator.py -q
uv run pytest -m "not slow" tests/ -q
```

Lint:

```bash
uv run ruff check src tests
uv run ruff format --check src tests
```

Type check:

```bash
uv run pyright src
```

Coverage (target: 80% line coverage on `src/`):

```bash
uv run pytest --cov=stl_seed --cov-report=term-missing tests/
```

## REDACTED firewall enforcement

`stl-seed` ships under a hard firewall against the unpublished REDACTED
parameter-synthesis algorithm developed in the sibling REDACTED
project. The full audit posture is documented in
[`paper/REDACTED.md`](paper/REDACTED.md). Two grep checks
gate every PR; they MUST return zero hits before merge.

Forbidden strings (`paper/REDACTED.md` §F.1):

```bash
git grep -nIE \
  '(physics[_ ]?filter|physics[_ ]?prior|PhysicalFilter|PhysicalPrior|\b_FILTER\b|\b_FILTER_WEIGHT\b|\b_NO_RESIDUAL_NN\b|\b_PARAM_SPACE\b|\b_JIT_TRACE_FIRED\b|CEGAR|counter-?example buffer|residual NeuralODE|landscape smoothing|spec[ _]form ablation|conjunction vs implication|\bC1\b|\bC2\b|\bC3\b|\bC4\b|\bC5\b|\bC6\b|\bC7\b|\bC8\b|\bC9\b|\bC10\b|\bC11\b)' \
  src/ tests/ scripts/ configs/
```

Forbidden imports (`paper/REDACTED.md` §F.5):

```bash
git grep -nIE \
  'from (REDACTED|REDACTED|REDACTED|REDACTED)|import (REDACTED|REDACTED|REDACTED|REDACTED)|/.superset/worktrees/REDACTED'
```

Hits inside the audit document itself or hits in negative form
("must NOT contain CEGAR") are inspected manually and recorded as
exceptions in your PR description.

## Branch strategy and commits

* Branch from `main` for every change. Name the branch
  `<topic>/<short-description>`; e.g. `task/heart-rate-pid` or
  `spec/glucose-overnight-fast`.
* Open a pull request against `main`. PRs require at least one review
  approval and a green CI run.
* No force-pushes to `main`. No `--no-verify`, `--no-edit`, or
  `--amend` on a published commit. If the pre-commit hooks fail,
  fix the issue and create a NEW commit; do not amend the failed one.

Commit messages follow the standard imperative-mood form: a short
title (≤ 72 chars), a blank line, and an optional body explaining the
*why* (not the *what* — the diff already shows that). The first line
is the headline. Examples drawn from the existing `git log`:

```
T2-E v2: progress-check enables honest cross-benchmark evaluation
Unit tests: 125 tests across 4 MSS modules (paper-reproducibility)
Paper-ready LaTeX tables: 7 tables auto-generated from source files
```

Do not include emoji. Do not use Conventional Commits prefixes
(`feat:`, `fix:`, etc.) — they conflict with the existing log style.

## Extending the library

### Adding a new task family

`stl-seed`'s simulator interface is the runtime-checkable `Simulator`
Protocol from [`paper/architecture.md`](paper/architecture.md)
§"Simulator interface". A new task family must:

1. Add a `<family>_params.py` under `src/stl_seed/tasks/` defining one
   or more `equinox.Module` parameter dataclasses, each field
   citing a published source inline.
2. Add a `<family>.py` defining a simulator class:

   ```python
   import diffrax as dfx, equinox as eqx, jax.numpy as jnp
   from stl_seed.tasks._trajectory import Trajectory, TrajectoryMeta

   class MySimulator(eqx.Module):
       horizon_minutes: float = eqx.field(static=True, default=...)
       n_control_points: int = eqx.field(static=True, default=...)
       n_save_points: int = eqx.field(static=True, default=...)
       solver: str = eqx.field(static=True, default="tsit5")
       rtol: float = eqx.field(static=True, default=1.0e-6)
       atol: float = eqx.field(static=True, default=1.0e-9)

       @property
       def state_dim(self) -> int: ...
       @property
       def action_dim(self) -> int: ...
       @property
       def horizon(self) -> int: ...

       def simulate(self, initial_state, control_sequence, params, key) -> Trajectory:
           # 1. clip control to declared bounds
           # 2. build dfx.ODETerm + dfx.Tsit5() + dfx.PIDController
           # 3. dfx.diffeqsolve(..., throw=False)
           # 4. NaN-replace with literature sentinel; count replacements
           # 5. return Trajectory(states, actions, times, meta)
   ```

3. Export the new symbols from `tasks/__init__.py`.
4. Add at least one heuristic policy entry to
   `_HEURISTIC_DEFAULTS` in `generation/policies.py`.
5. Add tests in `tests/test_<family>.py`. The existing
   `tests/test_bio_ode.py` is the reference pattern: protocol
   conformance + a Diffrax integration regression + a NaN-policy test
   + a round-trip through `Trajectory`.

### Adding a new STL spec

Specs live under `src/stl_seed/specs/`, one file per task family.
A new spec must:

1. Use only the conjunction-only fragment (`Always`, `Eventually`,
   `And`, predicate-level `Negation`) per
   [`paper/REDACTED.md`](paper/REDACTED.md) §C.1. No
   top-level disjunction. No implication. No `Until`.
2. Cite every numerical threshold inline to a textbook, paper, or
   biological/clinical database — no REDACTED-tuned values.
3. Build the AST by composition:

   ```python
   from stl_seed.specs import (
       Always, And, Eventually, Interval, Negation, Predicate,
       STLSpec, register,
   )

   def _gt(name, channel, threshold):
       return Predicate(
           f"{name}>{threshold}",
           fn=lambda traj, t, c=channel, th=threshold: float(traj[t, c]) - th,
       )

   spec = STLSpec(
       name="<family>.<short_handle>.<difficulty>",
       formula=And(children=(...)),
       signal_dim=...,
       horizon_minutes=...,
       description="One-paragraph English statement",
       citations=("...", "..."),     # one per threshold
       formula_text="G_[a,b] (...) AND ...",
       metadata={"difficulty": "easy" | "medium" | "hard", ...},
   )
   register(spec)
   ```

4. The lambda convention `lambda traj, t, c=CHANNEL, th=THRESHOLD: ...`
   is load-bearing — the STL evaluator's predicate introspector reads
   `(channel, threshold)` from `fn.__defaults__` and falls back to a
   slow Python evaluator if the convention is broken.
5. Add a calibration / smoke test under `tests/test_<family>.py`
   asserting the spec's rho on a known-good and known-bad trajectory
   sits on the expected sides of zero with the expected magnitude.

## Reporting issues

Open a GitHub issue at
[`AA-Alghamdi/stl-seed/issues`](https://github.com/AA-Alghamdi/stl-seed/issues)
with:

* The exact command you ran and its full output (or stack trace).
* Your platform (`uname -a` on macOS / Linux) and Python version
  (`uv run python --version`).
* Whether the issue is on the MLX or the bnb path (or backend-agnostic).
* A minimal reproducer if at all possible.

Pre-tag the issue with `bug`, `feature`, `docs`, or `firewall` so
review can be routed appropriately.

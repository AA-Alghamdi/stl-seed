# Contributing

`stl-seed` is a single-author research artifact. This file documents the dev loop, the firewall, and the two extension paths a reviewer or external researcher is most likely to need.

## Dev setup

```bash
uv sync --extra mlx --extra dev      # Apple Silicon
uv sync --extra cuda --extra dev     # Linux + CUDA
uv run pre-commit install
```

## Quality gates

```bash
uv run pytest tests/ -q
uv run ruff check src tests
uv run ruff format --check src tests
uv run pyright src
```

Coverage target on `src/`: 80% line.

## REDACTED firewall

`stl-seed` is hard-firewalled against the unpublished REDACTED parameter-synthesis algorithm in the sibling REDACTED codebase. Audit posture: [`paper/REDACTED.md`](paper/REDACTED.md). Every PR must pass:

```bash
bash scripts/REDACTED.sh
```

Two grep patterns gate the merge — forbidden strings (§F.1) and forbidden imports (§F.5). Both must return zero hits.

## Branches and commits

Branch from `main`. Imperative-mood commit titles ≤ 72 chars. No force-push to `main`. No `--amend` on a published commit; if a pre-commit hook fails, fix and create a new commit.

## Adding a task family

1. Add `<family>_params.py` and `<family>.py` under `src/stl_seed/tasks/`. The simulator implements the runtime-checkable `Simulator` Protocol from [`paper/architecture.md`](paper/architecture.md).
1. Cite every parameter inline to a published source.
1. Replace NaN ODE outputs with literature sentinels and count the replacements in `TrajectoryMeta`.
1. Register a heuristic policy in `generation/policies.py`.
1. Mirror the test pattern in `tests/test_bio_ode.py`.

## Adding an STL spec

Specs live under `src/stl_seed/specs/`, one file per family. The conjunction-only fragment (`Always`, `Eventually`, `And`, predicate-level `Negation`) is mandatory per firewall §C.1 — no top-level disjunction, no implication, no `Until`. Every numerical threshold cites a textbook, paper, or biological/clinical database inline. Predicates use the lambda convention `lambda traj, t, c=CHANNEL, th=THRESHOLD: ...`; the STL evaluator's introspection path keys off `fn.__defaults__`.

## Issues

Open at [`AA-Alghamdi/stl-seed/issues`](https://github.com/AA-Alghamdi/stl-seed/issues) with the exact command, full output, and platform. Pre-tag with `bug`, `feature`, `docs`, or `firewall`.

# `stl-seed` documentation

`stl-seed` is a research artifact for soft-verified supervised fine-tuning
of small open-weights LLMs on scientific control problems. The agent (an
LLM policy) emits a piecewise-constant control schedule
`u_{1:H}` for an ODE-driven biomolecular or physiological system; the
trajectory is scored by a Signal Temporal Logic (STL) robustness margin
`rho`; and the SFT loss is filtered or weighted by `rho` to bias
gradient updates toward control schedules that satisfy the formal spec.

The project tests whether SERA's soft-verification thesis (Shen et al.,
arXiv:2601.20789) extends to scientific-control LLM agents when the
soft signal is formal STL robustness.

## Where to start

| If you want to ... | Read |
|---|---|
| Install and run a demo in five minutes | [`docs/getting_started.md`](getting_started.md) |
| Understand a specific module | [`docs/api_reference.md`](api_reference.md) |
| See a runnable end-to-end pipeline | [`examples/`](../examples/README.md) |
| Read the formal claim, hypotheses, and statistical model | [`paper/theory.md`](../paper/theory.md) |
| Read the locked module layout and shared interfaces | [`paper/architecture.md`](../paper/architecture.md) |
| Verify the REDACTED firewall (audit posture) | [`paper/REDACTED.md`](../paper/REDACTED.md) |
| Contribute code or specs | [`CONTRIBUTING.md`](../CONTRIBUTING.md) |
| Read the GitHub-facing project overview | [README](../README.md) |

## Layout

```
src/stl_seed/        Library code (8 subpackages: tasks, specs, stl,
                     generation, filter, training, evaluation, stats).
examples/            Three runnable, self-contained examples.
scripts/             A13-A16 pilot drivers (generate/filter/train).
paper/               Theory, architecture, firewall audit, spec design,
                     SERA recipe, smoke-test report.
docs/                Getting-started, API reference, this index.
tests/               Pytest suite mirroring the src layout.
configs/             Hydra YAMLs (Phase 2; pilot-only today).
runs/                Output checkpoints + logs (git-ignored).
data/                Generated trajectory stores + filtered manifests.
```

## Reproducibility

Every script seeds JAX, NumPy, and (on Apple Silicon) MLX from a single
integer; every checkpoint records its config + git SHA in a sibling
`provenance.json`; every results file is append-only by contract.
Pinned versions live in `pyproject.toml` and `uv.lock`.

## License

Apache 2.0. See [`LICENSE`](../LICENSE).

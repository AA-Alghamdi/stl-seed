# stl-seed v0.1.0

First public PyPI release. Tracks the Day-1 `METHODOLOGY MATTERS` result shipped 2026-04-26.

## What ships in v0.1.0

- Public API surface frozen for the four canonical samplers: `StandardSampler`, `BestOfNSampler`, `ContinuousBoNSampler`, `STLGradientGuidedSampler`. Plus the four extended samplers: `HybridGradientBoNSampler`, `BeamSearchWarmstartSampler`, `cmaes_gradient`, `horizon_folded`, `rollout_tree`.
- Five task families: `glucose_insulin` (Bergman 1979), `bio_ode.repressilator`, `bio_ode.toggle`, `bio_ode.mapk`, and `cardiac_ap` (FitzHugh-Nagumo).
- Streaming Donze-Maler 2010 STL evaluator with vmap and JIT support; spec registry exposed at `stl_seed.specs.REGISTRY`.
- Unified comparison harness (`scripts/run_unified_comparison.py`) and the real-LLM falsification harness (`scripts/real_llm_hard_specs.py`).
- Optional dependency groups: `mlx` (Apple Silicon, MLX backend) and `cuda` (Linux, bitsandbytes + TRL backend for the canonical sweep).
- Colab-runnable demo at `notebooks/demo_colab.ipynb`.

## Build artifacts

The wheel and sdist live in `dist/` (gitignored):

```
dist/stl_seed-0.1.0-py3-none-any.whl
dist/stl_seed-0.1.0.tar.gz
```

Both pass `twine check` cleanly.

## Reproducing the build locally

```bash
# from the repo root, with .venv active
.venv/bin/python -m pip install --quiet build twine
rm -f dist/stl_seed-0.1.0*
.venv/bin/python -m build
.venv/bin/python -m twine check dist/stl_seed-0.1.0*
```

## PyPI upload command (operator runs manually)

```bash
.venv/bin/python -m twine upload dist/stl_seed-0.1.0*
```

The maintainer should set `TWINE_USERNAME=__token__` and `TWINE_PASSWORD=<pypi-token>` (or rely on `~/.pypirc`) before running this. Test upload first with the TestPyPI repository:

```bash
.venv/bin/python -m twine upload --repository testpypi dist/stl_seed-0.1.0*
```

## Post-release checklist

1. Tag and push: `git tag -s v0.1.0 -m 'stl-seed v0.1.0' && git push origin v0.1.0`.
1. Cut a GitHub release pointing at the tag; attach the wheel and sdist as release assets.
1. Verify `pip install stl-seed` from a clean environment resolves to 0.1.0 and `python -c 'import stl_seed; print(stl_seed.__version__)'` prints `0.1.0`.
1. Flip the install line in `notebooks/demo_colab.ipynb` from the GitHub-clone fallback to `!pip install stl-seed`.

## Version bump rationale

Previous version was `0.0.1` (Phase 1 scaffolding only, marked explicitly as such in `src/stl_seed/__init__.py`). Phase 1 audit gates have all passed; the public API is now stable enough to make unannounced changes a breaking-change event, which is the entry criterion for `0.1.0` per semver. The `Development Status :: 4 - Beta` classifier is now set to match.

## Links

- GitHub: https://github.com/AA-Alghamdi/stl-seed
- Issues: https://github.com/AA-Alghamdi/stl-seed/issues
- Day-1 narrative: `paper/real_llm_comparison.md`
- Blog post: `docs/blog/stl_seed_intro.md`

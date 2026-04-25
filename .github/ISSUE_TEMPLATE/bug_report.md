---
name: Bug report
about: Report a defect in stl-seed (incorrect output, crash, regression, doc error)
title: "[BUG] "
labels: bug
assignees: ''
---

## Summary

A one-line description of the bug.

## Environment

- stl-seed version / commit SHA: `git rev-parse HEAD` →
- Python version: `python --version` →
- OS / arch: (e.g. macOS 15.4 / arm64, Ubuntu 22.04 / x86_64)
- Backend in use: ☐ MLX (Apple Silicon)  ☐ bnb / CUDA (Linux)  ☐ CPU-only
- JAX / jaxlib version: `uv run python -c "import jax; print(jax.__version__, jax.devices())"` →
- Other relevant package versions (transformers, diffrax, mlx, bitsandbytes):

## Steps to reproduce

A minimal sequence that triggers the bug. Prefer a runnable command.

```bash
# example
uv run python -m stl_seed.cli ...
```

If the bug requires a specific config or input file, include it inline or attach
to the issue.

## Expected behavior

What should happen.

## Actual behavior

What actually happens. Include the full traceback (not just the last line).

```
<paste traceback or unexpected output here>
```

## Logs / artifacts

- Run log file (if any):
- Generated trajectory store (if relevant):
- Pyright / ruff output (if a static-check regression):

## Additional context

- Does the bug reproduce on a clean clone (`uv sync --extra dev && uv run pytest`)?
- Is it a regression from an earlier commit? If so, suspected commit:
- Anything that may have triggered it (recent dependency update, config change,
  switching backends, switching model size)?

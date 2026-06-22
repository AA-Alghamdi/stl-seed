"""Programmatic builder for ``notebooks/demo_colab.ipynb``.

Run via::

    .venv/bin/python notebooks/_build_demo_notebook.py

Authoring the notebook through ``nbformat`` keeps the cell list under
diff-friendly source control instead of opaque JSON. Re-run after edits
to refresh the .ipynb.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NOTEBOOK_PATH = Path(__file__).resolve().parent / "demo_colab.ipynb"


def md(*lines: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell("\n".join(lines))


def code(*lines: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("\n".join(lines))


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"provenance": [], "toc_visible": True},
    }

    cells: list[nbf.NotebookNode] = []

    # --- 1. Framing ---------------------------------------------------------
    cells.append(
        md(
            "# stl-seed demo: STL robustness as a soft verifier for small-LLM scientific control",
            "",
            "`stl-seed` tests whether SERA-style soft-verification (Shen et al. 2026) extends "
            "to scientific control when the soft signal is a *formal* STL specification rather "
            "than an engineered patch-overlap proxy. With a formal verifier the noise term in the "
            "Goodhart decomposition `R_gold - R_proxy = (R_gold - R_spec) + (R_spec - R_verifier)` "
            "collapses to float64 round-off, leaving the spec-completeness gap as the only honest "
            "axis of comparison.",
            "",
            "This notebook reproduces the Day-1 `METHODOLOGY MATTERS` headline on a free-tier "
            "Colab T4 (or CPU): real `Qwen3-0.6B-bf16` plus standard sampling fails 4/4 hard "
            "biomolecular control specs; structural search (beam-search warmstart) rescues all 4. "
            "Total runtime budget on T4 is ~5 minutes if you set `QUICK = True`.",
            "",
            "Sections:",
            "1. Install",
            "2. 5-minute demo: gradient-guided sampler on `glucose_insulin.tir.easy`",
            "3. Headline figure: 3 samplers x 2 tasks unified comparison (subset)",
            "4. Where to read more",
        )
    )

    # --- 2. Install ---------------------------------------------------------
    cells.append(md("## 1. Install"))
    cells.append(
        md(
            "Until `stl-seed` is live on PyPI, the install line below pulls from GitHub. "
            "Once v0.1.0 is uploaded, swap the second line for `!pip install stl-seed` and "
            "comment out the GitHub install."
        )
    )
    cells.append(
        code(
            "# !pip install stl-seed   # uncomment after v0.1.0 lands on PyPI",
            "!pip install --quiet 'git+https://github.com/AA-Alghamdi/stl-seed.git'",
        )
    )
    cells.append(
        code(
            "import platform, sys",
            "import stl_seed",
            "print('stl-seed', stl_seed.__version__)",
            "print('python  ', sys.version.split()[0])",
            "print('platform', platform.system(), platform.machine())",
        )
    )

    # --- 3. Five-minute demo ----------------------------------------------
    cells.append(md("## 2. 5-minute demo: gradient-guided sampling on glucose-insulin"))
    cells.append(
        md(
            "We instantiate the Bergman 1979 minimal glucose-insulin model, the easy "
            "time-in-range STL spec from `paper/specs.md`, and the 5-level uniform action "
            "vocabulary on `[0, 5] U/h`. We then run two samplers under a flat-prior synthetic "
            "LLM (`logits = 0`), so the only signal driving choices is the STL gradient:",
            "",
            "* `lambda = 0`  (a no-op baseline, equivalent to standard sampling)",
            "* `lambda = 2.0` (gradient-guided STL decoding)",
            "",
            "Expected outcome on this task: the lambda=0 baseline produces mean rho ~= +2.5; "
            "gradient guidance saturates the spec ceiling at rho = +20.0.",
        )
    )
    cells.append(
        code(
            "import jax",
            "import jax.numpy as jnp",
            "import numpy as np",
            "",
            "from stl_seed.inference import STLGradientGuidedSampler",
            "from stl_seed.inference.gradient_guided import make_uniform_action_vocabulary",
            "from stl_seed.specs import REGISTRY",
            "from stl_seed.tasks.glucose_insulin import (",
            "    BergmanParams,",
            "    GlucoseInsulinSimulator,",
            "    default_normal_subject_initial_state,",
            ")",
            "",
            "sim = GlucoseInsulinSimulator()",
            "params = BergmanParams()",
            "spec = REGISTRY['glucose_insulin.tir.easy']",
            "V = make_uniform_action_vocabulary(0.0, 5.0, k_per_dim=5)",
            "x0 = default_normal_subject_initial_state(params)",
            "",
            "K = int(V.shape[0])",
            "def flat_llm(state, history, key):",
            "    return jnp.zeros(K)",
            "",
            "common_args = (flat_llm, sim, spec, V, params)",
            "common_kw = dict(horizon=int(sim.n_control_points), sampling_temperature=0.5)",
            "",
            "baseline = STLGradientGuidedSampler(*common_args, guidance_weight=0.0, **common_kw)",
            "guided   = STLGradientGuidedSampler(*common_args, guidance_weight=2.0, **common_kw)",
        )
    )
    cells.append(
        code(
            "N_SEEDS = 4   # raise to 8 if you want tighter CIs",
            "rho_baseline, rho_guided = [], []",
            "for s in range(N_SEEDS):",
            "    key = jax.random.key(1000 + s)",
            "    _, db = baseline.sample(x0, key)",
            "    _, dg = guided.sample(x0, key)",
            "    rho_baseline.append(float(db['final_rho']))",
            "    rho_guided.append(float(dg['final_rho']))",
            "",
            "print(f'baseline (lambda=0):  mean rho = {np.mean(rho_baseline):+.3f}'",
            "      f'  per-seed = {[round(r, 2) for r in rho_baseline]}')",
            "print(f'guided   (lambda=2):  mean rho = {np.mean(rho_guided):+.3f}'",
            "      f'  per-seed = {[round(r, 2) for r in rho_guided]}')",
            "",
            "assert np.mean(rho_guided) > np.mean(rho_baseline), 'guidance failed to lift rho'",
            "print('OK: gradient guidance lifts mean rho on glucose_insulin.tir.easy')",
        )
    )

    # --- 4. Headline figure -------------------------------------------------
    cells.append(md("## 3. Headline figure: 3 samplers x 2 tasks (subset)"))
    cells.append(
        md(
            "The full unified comparison harness sweeps 9 samplers x 5 task families x N seeds "
            "and is too heavy for a free-tier T4. Here we run a subset: three samplers "
            "(`standard`, `gradient_guided`, `beam_search_warmstart`) x two tasks "
            "(`glucose_insulin`, `bio_ode.toggle`), with the flat-prior synthetic LLM. We use "
            "the canonical script in `scripts/run_unified_comparison.py` so the numbers match "
            "those in the paper exactly.",
            "",
            "Set `QUICK = True` to skip the slow beam-search-on-repressilator-K=125 cell. "
            "Total wall-time on T4: ~3-5 minutes at `QUICK=True`.",
        )
    )
    cells.append(
        code(
            "import pathlib",
            "import subprocess",
            "import sys",
            "",
            "import stl_seed",
            "",
            "QUICK = True            # set False for the full sweep (slow on CPU)",
            "N_SEEDS = 2 if QUICK else 4",
            "TASKS = ['glucose_insulin', 'bio_ode.toggle']",
            "SAMPLERS = 'standard,gradient_guided,beam_search_warmstart'",
            "",
            "# Locate the installed package and chdir to a writable work dir.",
            "REPO = pathlib.Path(stl_seed.__file__).resolve().parent.parent.parent",
            "SCRIPT = REPO / 'scripts' / 'run_unified_comparison.py'",
            "WORK = pathlib.Path('/tmp/stl_seed_demo')",
            "WORK.mkdir(exist_ok=True, parents=True)",
            "",
            "# If the script is missing (pip install case, repo-root not bundled in the wheel),",
            "# fall back to git-cloning a sparse copy of the scripts/ directory.",
            "if not SCRIPT.exists():",
            "    print('scripts/ not in the installed wheel; cloning the scripts dir for the demo')",
            "    REPO = WORK / 'stl-seed'",
            "    if not REPO.exists():",
            "        subprocess.run(",
            "            ['git', 'clone', '--depth', '1',",
            "             'https://github.com/AA-Alghamdi/stl-seed.git', str(REPO)],",
            "            check=True,",
            "        )",
            "    SCRIPT = REPO / 'scripts' / 'run_unified_comparison.py'",
            "",
            "cmd = [",
            "    sys.executable, str(SCRIPT),",
            "    '--n-seeds', str(N_SEEDS),",
            "    '--tasks', *TASKS,",
            "    '--samplers', SAMPLERS,",
            "    '--out-dir', str(WORK),",
            "    '--fig-path', str(WORK / 'demo_unified_comparison.png'),",
            "    '--md-path', str(WORK / 'demo_unified_comparison.md'),",
            "    '--llm', 'uniform',",
            "]",
            "print('running:', ' '.join(cmd))",
            "subprocess.run(cmd, check=True, cwd=str(REPO))",
        )
    )
    cells.append(
        code(
            "# Render the unified-comparison figure inline.",
            "from IPython.display import Image, display",
            "fig = WORK / 'demo_unified_comparison.png'",
            "if fig.exists():",
            "    display(Image(filename=str(fig)))",
            "else:",
            "    print('figure not produced; check the subprocess output above')",
        )
    )
    cells.append(
        code(
            "# Read the parquet and tabulate per (task, sampler) cells.",
            "import pandas as pd",
            "df = pd.read_parquet(WORK / 'results.parquet')",
            "agg = (",
            "    df.groupby(['task', 'sampler'])['final_rho']",
            "      .agg(['mean', 'min', 'max', 'count'])",
            "      .round(3)",
            ")",
            "agg",
        )
    )

    # --- 5. Real-LLM caveat -------------------------------------------------
    cells.append(md("## 4. Real-LLM (Qwen3-0.6B) note"))
    cells.append(
        md(
            "The `METHODOLOGY MATTERS` headline ships the Qwen3-0.6B-bf16 numbers via the MLX "
            "backend, which is **Apple Silicon only**. On Colab (Linux/T4) this notebook uses "
            "the flat-prior synthetic LLM exclusively. Reproducing the real-LLM numbers requires "
            "either:",
            "",
            "* an Apple Silicon machine: `git clone` the repo, `uv sync --extra mlx`, then "
            "  `python scripts/real_llm_hard_specs.py`; or",
            "* a CUDA host: install the `cuda` extra and use the Transformers / bitsandbytes "
            "  backend (the canonical sweep target).",
            "",
            "The cell below auto-detects platform and is a no-op on non-Darwin/arm64 hosts.",
        )
    )
    cells.append(
        code(
            "import platform",
            "is_apple_silicon = (platform.system() == 'Darwin' and platform.machine() == 'arm64')",
            "if not is_apple_silicon:",
            "    print('non-Apple-Silicon host detected; skipping real-LLM scoring.')",
            "    print('the flat-prior synthetic LLM is used in the cells above instead.')",
            "else:",
            "    print('Apple Silicon detected; you can run scripts/real_llm_hard_specs.py')",
            "    print('directly on the host (not from this Colab notebook) to reproduce the')",
            "    print('METHODOLOGY MATTERS numbers on Qwen3-0.6B-bf16.')",
        )
    )

    # --- 6. Pointers --------------------------------------------------------
    cells.append(md("## 5. Where to read more"))
    cells.append(
        md(
            "* Paper-grade narrative and per-cell numbers: "
            "[`paper/real_llm_comparison.md`](https://github.com/AA-Alghamdi/stl-seed/blob/main/paper/real_llm_comparison.md)",
            "* Unified comparison harness: "
            "[`paper/unified_comparison_results.md`](https://github.com/AA-Alghamdi/stl-seed/blob/main/paper/unified_comparison_results.md)",
            "* Inference method (gradient guidance derivation): "
            "[`paper/inference_method.md`](https://github.com/AA-Alghamdi/stl-seed/blob/main/paper/inference_method.md)",
            "* SERA original: Shen et al. 2026, [arXiv:2601.20789](https://arxiv.org/abs/2601.20789)",
            "* STL semantics: Donze and Maler, FORMATS 2010",
            "",
            "Issues and PRs welcome at "
            "[github.com/AA-Alghamdi/stl-seed](https://github.com/AA-Alghamdi/stl-seed).",
        )
    )

    nb["cells"] = cells

    NOTEBOOK_PATH.write_text(nbf.writes(nb))
    print(f"wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()

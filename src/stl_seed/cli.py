"""stl-seed CLI entry point.

Subcommands:

* ``stl-seed version`` — print the package version.
* ``stl-seed demo`` — placeholder for an end-to-end demo (P1.3).
* ``stl-seed sample`` — run inference-time decoding under the chosen
  sampler (Subphase 2.x: gradient-guided sampling).
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import typer

from stl_seed import __version__

app = typer.Typer(
    name="stl-seed",
    help="Soft-verified SFT for scientific control with STL robustness.",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print the stl-seed version."""
    typer.echo(__version__)


@app.command()
def demo() -> None:
    """Run an end-to-end demo (placeholder for Phase 1.3)."""
    typer.echo("Demo not yet implemented. Phase 1 scaffolding only.")
    raise typer.Exit(code=0)


@app.command()
def sample(
    task: str = typer.Option(
        "glucose_insulin",
        "--task",
        help=(
            "Task family: 'glucose_insulin', 'bio_ode.repressilator', "
            "'bio_ode.toggle', 'bio_ode.mapk'."
        ),
    ),
    sampler: str = typer.Option(
        "gradient_guided",
        "--sampler",
        help=(
            "One of {standard, bon, bon_continuous, gradient_guided, hybrid}. "
            "'standard' is vanilla LLM decoding; 'bon' is binary-STL "
            "Best-of-N; 'bon_continuous' is argmax-rho BoN; "
            "'gradient_guided' is the gradient-guided contribution; "
            "'hybrid' runs n gradient-guided draws and selects argmax-rho."
        ),
    ),
    n: int = typer.Option(
        8,
        "--n",
        help=(
            "Sample budget for the BoN samplers (ignored for 'standard' "
            "and 'gradient_guided', which use one sample per call)."
        ),
    ),
    guidance_weight: float = typer.Option(
        1.0,
        "--guidance-weight",
        help="Lambda hyperparameter for gradient-guided sampling.",
    ),
    spec_key: str | None = typer.Option(
        None,
        "--spec",
        help=(
            "STL spec key from stl_seed.specs.REGISTRY. Defaults to the "
            "first registered spec for the task family."
        ),
    ),
    seed: int = typer.Option(0, "--seed", help="PRNG seed."),
    k_per_dim: int = typer.Option(
        5,
        "--k-per-dim",
        help="Action vocabulary granularity (per axis).",
    ),
    sampling_temperature: float = typer.Option(
        1.0,
        "--temperature",
        help="Sampling temperature on biased logits. 0 = greedy argmax.",
    ),
    llm: str = typer.Option(
        "uniform",
        "--llm",
        help=(
            "LLM proposal backend. 'uniform' (default) uses a flat-prior "
            "synthetic LLM. 'qwen3-0.6b' / 'qwen3-1.7b' / 'qwen3-4b' wrap "
            "the corresponding mlx-community Qwen3-bf16 checkpoint via "
            "MLXLLMProposal (Apple Silicon only)."
        ),
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        help="Optional JSON file to write the diagnostics dict to.",
    ),
) -> None:
    """Run inference-time decoding and report STL robustness.

    By default the CLI uses a *uniform* LLM proxy (every action equally
    likely under the LLM prior) so the result isolates the contribution
    of gradient guidance vs BoN. Pass ``--llm qwen3-0.6b`` (or
    ``qwen3-1.7b`` / ``qwen3-4b``) to use a real Qwen3 base model via
    :class:`MLXLLMProposal` -- this is the methodology-honest baseline
    introduced in 2026-04-25 to falsify the original "+128x" claim from
    the uniform-proxy regime. See ``paper/inference_method.md``.
    """
    from stl_seed.inference import (
        BestOfNSampler,
        ContinuousBoNSampler,
        HybridGradientBoNSampler,
        StandardSampler,
        STLGradientGuidedSampler,
    )
    from stl_seed.inference.gradient_guided import make_uniform_action_vocabulary
    from stl_seed.specs import REGISTRY

    # Resolve task -> (simulator, params, action box, default spec key).
    if task == "glucose_insulin":
        from stl_seed.tasks.glucose_insulin import (
            U_INSULIN_MAX_U_PER_H,
            U_INSULIN_MIN_U_PER_H,
            BergmanParams,
            GlucoseInsulinSimulator,
            default_normal_subject_initial_state,
        )

        sim = GlucoseInsulinSimulator()
        params = BergmanParams()
        x0 = default_normal_subject_initial_state(params)
        lo, hi = float(U_INSULIN_MIN_U_PER_H), float(U_INSULIN_MAX_U_PER_H)
        action_dim = 1
        default_spec = "glucose_insulin.tir.easy"
    elif task in {"bio_ode.repressilator", "bio_ode.toggle", "bio_ode.mapk"}:
        from stl_seed.tasks import bio_ode
        from stl_seed.tasks.bio_ode_params import (
            MAPKParams,
            RepressilatorParams,
            ToggleParams,
        )

        if task == "bio_ode.repressilator":
            sim = bio_ode.RepressilatorSimulator()
            params = RepressilatorParams()
            x0 = bio_ode._repressilator_initial_state(params)
            action_dim = bio_ode.REPRESSILATOR_ACTION_DIM
            default_spec = "bio_ode.repressilator.easy"
        elif task == "bio_ode.toggle":
            sim = bio_ode.ToggleSimulator()
            params = ToggleParams()
            x0 = bio_ode._toggle_initial_state(params)
            action_dim = bio_ode.TOGGLE_ACTION_DIM
            default_spec = "bio_ode.toggle.medium"
        else:
            sim = bio_ode.MAPKSimulator()
            params = MAPKParams()
            x0 = bio_ode._mapk_initial_state(params)
            action_dim = bio_ode.MAPK_ACTION_DIM
            default_spec = "bio_ode.mapk.hard"
        lo, hi = 0.0, 1.0
    else:
        typer.echo(f"unknown task: {task!r}", err=True)
        raise typer.Exit(code=2)

    spec_name = spec_key or default_spec
    if spec_name not in REGISTRY:
        typer.echo(
            f"spec key {spec_name!r} not in registry; available: "
            f"{sorted(k for k in REGISTRY if k.startswith(task.split('.')[0]))}",
            err=True,
        )
        raise typer.Exit(code=2)
    spec = REGISTRY[spec_name]

    # Build vocabulary.
    if action_dim == 1:
        V = make_uniform_action_vocabulary(lo, hi, k_per_dim=k_per_dim)
    else:
        V = make_uniform_action_vocabulary(
            [lo] * action_dim, [hi] * action_dim, k_per_dim=k_per_dim
        )

    horizon = int(sim.n_control_points)

    # LLM proposal: synthetic uniform (default) or real Qwen3 via MLX.
    if llm == "uniform":

        def uniform_llm(state, history, key):
            return jnp.zeros(int(V.shape[0]))

        llm_callable = uniform_llm
    elif llm in {"qwen3-0.6b", "qwen3-1.7b", "qwen3-4b"}:
        from stl_seed.inference.mlx_llm_proposal import MLXLLMProposal

        llm_callable = MLXLLMProposal(
            action_vocabulary=V,
            spec=spec,
            task=task,
            initial_state=x0,
            horizon=horizon,
            state_dim=int(jnp.asarray(x0).shape[0]),
            model_id=llm,
        )
    else:
        typer.echo(
            f"unknown llm backend: {llm!r}; choose one of "
            "{uniform, qwen3-0.6b, qwen3-1.7b, qwen3-4b}",
            err=True,
        )
        raise typer.Exit(code=2)

    common = dict(
        llm=llm_callable,
        simulator=sim,
        spec=spec,
        action_vocabulary=V,
        sim_params=params,
        horizon=horizon,
        sampling_temperature=sampling_temperature,
    )
    if sampler == "standard":
        s = StandardSampler(**common)
    elif sampler == "bon":
        s = BestOfNSampler(n=n, **common)
    elif sampler == "bon_continuous":
        s = ContinuousBoNSampler(n=n, **common)
    elif sampler == "gradient_guided":
        s = STLGradientGuidedSampler(guidance_weight=guidance_weight, **common)
    elif sampler == "hybrid":
        s = HybridGradientBoNSampler(n=n, guidance_weight=guidance_weight, **common)
    else:
        typer.echo(
            f"unknown sampler: {sampler!r}; choose one of "
            "{standard, bon, bon_continuous, gradient_guided, hybrid}",
            err=True,
        )
        raise typer.Exit(code=2)

    key = jax.random.key(seed)
    _, diag = s.sample(x0, key)

    typer.echo(f"task={task} spec={spec_name} sampler={sampler}")
    typer.echo(f"final_rho = {diag['final_rho']:.4f}")
    if "n_steps_changed_by_guidance" in diag:
        typer.echo(
            f"steps_changed_by_guidance = {diag['n_steps_changed_by_guidance']} / {diag['n_steps']}"
        )
    if "max_rho" in diag:
        typer.echo(f"max_rho_in_batch = {diag['max_rho']:.4f}")

    if output is not None:
        # Cast all JAX arrays to plain Python before json-dumping.
        def _coerce(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                if isinstance(v, list):
                    return [_coerce(x) for x in v]
                return v

        clean = {k: _coerce(v) for k, v in diag.items()}
        with open(output, "w") as f:
            json.dump(clean, f, indent=2, default=str)
        typer.echo(f"wrote diagnostics to {output}")


if __name__ == "__main__":
    app()

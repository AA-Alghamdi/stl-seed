"""stl-seed CLI entry point.

Phase 1 stub. Real subcommands (demo, generate, filter, train, evaluate) are
wired in subphase 1.3. For now, `stl-seed --version` works.
"""

from __future__ import annotations

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


if __name__ == "__main__":
    app()

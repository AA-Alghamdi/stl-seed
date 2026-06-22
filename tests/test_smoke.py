"""Basic smoke tests."""

from __future__ import annotations


def test_import() -> None:
    """Package imports cleanly."""
    import stl_seed

    assert stl_seed.__version__ == "0.1.0"


def test_cli_version() -> None:
    """CLI version command runs."""
    from typer.testing import CliRunner

    import stl_seed
    from stl_seed.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert stl_seed.__version__ in result.stdout

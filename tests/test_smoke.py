"""Phase 1 smoke tests. Real test suite lands in subphase 1.6."""

from __future__ import annotations


def test_import() -> None:
    """Package imports cleanly."""
    import stl_seed

    assert stl_seed.__version__ == "0.0.1"


def test_cli_version() -> None:
    """CLI version command runs."""
    from typer.testing import CliRunner

    from stl_seed.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.0.1" in result.stdout

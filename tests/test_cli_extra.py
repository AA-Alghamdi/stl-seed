"""Supplemental coverage tests for ``stl_seed.cli``.

The CLI is currently 77% covered (only the ``demo`` placeholder and the
``__main__`` guard are untested). These tests exercise the demo command
via ``CliRunner`` and the no-args-is-help behaviour.
"""

from __future__ import annotations

from typer.testing import CliRunner

from stl_seed.cli import app

runner = CliRunner()


def test_cli_demo_command_exits_zero() -> None:
    """The placeholder ``demo`` command must exit 0 with a clear message."""
    result = runner.invoke(app, ["demo"])
    assert result.exit_code == 0
    assert "not yet implemented" in result.stdout.lower() or "phase 1" in result.stdout.lower()


def test_cli_help_lists_subcommands() -> None:
    """Top-level --help must list both ``version`` and ``demo`` subcommands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "version" in result.stdout
    assert "demo" in result.stdout


def test_cli_no_args_shows_help() -> None:
    """``no_args_is_help=True`` -> invoking with no args prints help and exits."""
    result = runner.invoke(app, [])
    # Typer with no_args_is_help returns a non-zero exit code (typically 2)
    # but should print the help text — verify it didn't crash with a stack trace.
    assert "Usage" in result.stdout or "version" in result.stdout


def test_cli_unknown_command_fails() -> None:
    result = runner.invoke(app, ["nonexistent-subcommand"])
    assert result.exit_code != 0

"""CLI tests for `merton excel ...` commands (via typer.testing)."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from merton.cli.main import app

runner = CliRunner()


@pytest.fixture
def sideload_tmp(tmp_path, monkeypatch):
    """Redirect the installer's sideload directory at the function level."""
    target = tmp_path / "wef"
    # Monkeypatch the resolver so install/uninstall use the temp directory.
    from merton.excel import installer

    monkeypatch.setattr(installer, "sideload_directory", lambda: target)
    return target


class TestExcelInstall:
    def test_install_command(self, sideload_tmp) -> None:
        result = runner.invoke(app, ["excel", "install", "--url", "http://localhost:9100"])
        assert result.exit_code == 0, result.output
        assert (sideload_tmp / "merton-manifest.xml").exists()

    def test_uninstall_command(self, sideload_tmp) -> None:
        runner.invoke(app, ["excel", "install", "--url", "http://localhost:9100"])
        result = runner.invoke(app, ["excel", "uninstall"])
        assert result.exit_code == 0
        assert not (sideload_tmp / "merton-manifest.xml").exists()

    def test_uninstall_when_not_installed(self, sideload_tmp) -> None:
        result = runner.invoke(app, ["excel", "uninstall"])
        assert result.exit_code == 0
        assert "not installed" in result.output.lower()


class TestExcelStatus:
    def test_status_runs(self, sideload_tmp) -> None:
        result = runner.invoke(app, ["excel", "status"])
        assert result.exit_code == 0
        assert "Manifest installed" in result.output


class TestExcelSampleWorkbook:
    def test_writes_a_workbook(self, tmp_path) -> None:
        out = tmp_path / "sample.xlsx"
        result = runner.invoke(app, ["excel", "sample-workbook", "--out", str(out)])
        assert result.exit_code == 0
        assert out.exists()


class TestServerSubcommand:
    def test_help_lists_subcommands(self) -> None:
        result = runner.invoke(app, ["excel", "server", "--help"])
        assert result.exit_code == 0
        assert "start" in result.output
        assert "stop" in result.output
        assert "status" in result.output

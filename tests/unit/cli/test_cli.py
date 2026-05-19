"""Smoke tests for the merton CLI."""

from __future__ import annotations

import pandas as pd
from typer.testing import CliRunner

from merton import __version__
from merton.cli.main import app

runner = CliRunner()


class TestCLIRoot:
    def test_version(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.stdout

    def test_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "doctor" in result.stdout
        assert "config" in result.stdout
        assert "fit" in result.stdout


class TestDoctor:
    def test_doctor_runs(self) -> None:
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        assert "Python" in result.stdout
        assert "numpy" in result.stdout


class TestConfig:
    def test_show(self) -> None:
        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0
        assert "backend" in result.stdout

    def test_set_and_reset(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("MERTON_CONFIG_DIR", str(tmp_path))
        result = runner.invoke(app, ["config", "set", "default_horizon", "2.5"])
        assert result.exit_code == 0
        cfg_file = tmp_path / "config.toml"
        assert cfg_file.exists()
        assert "default_horizon" in cfg_file.read_text()
        result = runner.invoke(app, ["config", "reset"])
        assert result.exit_code == 0
        assert not cfg_file.exists()


class TestFit:
    def test_single_row_csv(self, tmp_path) -> None:
        csv = tmp_path / "firm.csv"
        pd.DataFrame(
            {
                "equity": [100.0],
                "debt_short": [20.0],
                "debt_long": [30.0],
                "equity_vol": [0.30],
                "rf": [0.04],
            }
        ).to_csv(csv, index=False)
        result = runner.invoke(app, ["fit", str(csv), "--method", "jmr_iterative"])
        assert result.exit_code == 0
        assert "Distance-to-default" in result.stdout

    def test_panel_csv(self, tmp_path) -> None:
        csv = tmp_path / "panel.csv"
        pd.DataFrame(
            {
                "ticker": ["A", "B"],
                "equity": [100.0, 200.0],
                "debt_short": [20.0, 40.0],
                "debt_long": [30.0, 60.0],
                "equity_vol": [0.30, 0.25],
                "rf": [0.04, 0.04],
            }
        ).to_csv(csv, index=False)
        out = tmp_path / "results.csv"
        result = runner.invoke(
            app, ["fit", str(csv), "--method", "jmr_iterative", "--out", str(out)]
        )
        assert result.exit_code == 0
        assert out.exists()
        df = pd.read_csv(out)
        assert len(df) == 2
        assert "dd" in df.columns

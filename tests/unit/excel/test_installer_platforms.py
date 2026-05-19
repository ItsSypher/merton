"""Per-platform sideload-directory tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from merton.excel import installer


@pytest.fixture
def fake_home(tmp_path, monkeypatch) -> Path:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))  # type: ignore[arg-type]
    return tmp_path


class TestSideloadDirectory:
    def test_macos_path(self, fake_home: Path, monkeypatch) -> None:
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        d = installer.sideload_directory()
        assert "Containers/com.Microsoft.Excel" in str(d)

    def test_windows_path(self, fake_home: Path, monkeypatch) -> None:
        monkeypatch.setattr("platform.system", lambda: "Windows")
        monkeypatch.setenv("LOCALAPPDATA", str(fake_home / "AppData/Local"))
        d = installer.sideload_directory()
        assert "Microsoft/Office/16.0/Wef" in str(d).replace("\\", "/")

    def test_windows_without_localappdata(self, fake_home: Path, monkeypatch) -> None:
        monkeypatch.setattr("platform.system", lambda: "Windows")
        monkeypatch.delenv("LOCALAPPDATA", raising=False)
        d = installer.sideload_directory()
        assert "AppData/Local" in str(d).replace("\\", "/")

    def test_linux_fallback(self, fake_home: Path, monkeypatch) -> None:
        monkeypatch.setattr("platform.system", lambda: "Linux")
        d = installer.sideload_directory()
        assert ".config/merton/excel" in str(d).replace("\\", "/")

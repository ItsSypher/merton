"""Tests for the manifest installer."""

from __future__ import annotations

from pathlib import Path

from merton.excel.installer import (
    MANIFEST_FILENAME,
    install,
    is_installed,
    sideload_directory,
    uninstall,
)


class TestSideloadDirectory:
    def test_returns_a_path(self) -> None:
        d = sideload_directory()
        assert isinstance(d, Path)


class TestInstallFlow:
    def test_install_writes_manifest(self, tmp_path: Path) -> None:
        path = install(base_url="http://localhost:9000", sideload_dir=tmp_path)
        assert path.exists()
        assert path.name == MANIFEST_FILENAME
        assert "http://localhost:9000" in path.read_text()

    def test_is_installed_returns_true_after_install(self, tmp_path: Path) -> None:
        install(sideload_dir=tmp_path)
        assert is_installed(sideload_dir=tmp_path) is True

    def test_is_installed_returns_false_when_missing(self, tmp_path: Path) -> None:
        assert is_installed(sideload_dir=tmp_path) is False

    def test_uninstall_returns_true_when_removed(self, tmp_path: Path) -> None:
        install(sideload_dir=tmp_path)
        assert uninstall(sideload_dir=tmp_path) is True
        assert not is_installed(sideload_dir=tmp_path)

    def test_uninstall_returns_false_when_absent(self, tmp_path: Path) -> None:
        assert uninstall(sideload_dir=tmp_path) is False

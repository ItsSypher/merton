"""Sideload-directory installer for the Office.js manifest.

Excel reads Office.js add-in manifests from a platform-specific
*sideload directory*:

- macOS: ``~/Library/Containers/com.Microsoft.Excel/Data/Documents/wef``
- Windows: ``%LOCALAPPDATA%\\Microsoft\\Office\\16.0\\Wef``
- Linux: not supported (Excel itself is unavailable; users on Linux need
  Excel for the web and a manual upload).

This module writes the manifest XML to the right place; users can also
just install via Excel's *Upload My Add-in* UI by pointing at any local
path.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Final

from .manifest import deterministic_add_in_id, render_manifest

MANIFEST_FILENAME: Final[str] = "merton-manifest.xml"


def sideload_directory() -> Path:
    """Best-effort detection of Excel's sideload directory for this OS."""
    system = platform.system()
    home = Path.home()
    if system == "Darwin":
        return home / "Library/Containers/com.Microsoft.Excel/Data/Documents/wef"
    if system == "Windows":
        import os

        local = os.environ.get("LOCALAPPDATA", str(home / "AppData/Local"))
        return Path(local) / "Microsoft/Office/16.0/Wef"
    # Linux / others: write to ~/.config/merton/excel/ as a manual-upload location.
    return home / ".config/merton/excel"


def install(
    *,
    base_url: str = "http://localhost:8000",
    sideload_dir: Path | None = None,
) -> Path:
    """Write the manifest into the sideload directory. Returns its path."""
    target_dir = sideload_dir or sideload_directory()
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = target_dir / MANIFEST_FILENAME
    manifest_path.write_text(render_manifest(base_url=base_url))
    return manifest_path


def uninstall(*, sideload_dir: Path | None = None) -> bool:
    """Remove the manifest. Returns ``True`` if a file was actually deleted."""
    target_dir = sideload_dir or sideload_directory()
    manifest_path = target_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        manifest_path.unlink()
        return True
    return False


def is_installed(*, sideload_dir: Path | None = None) -> bool:
    """Whether the manifest is currently in the sideload directory."""
    target_dir = sideload_dir or sideload_directory()
    return (target_dir / MANIFEST_FILENAME).exists()


__all__ = [
    "MANIFEST_FILENAME",
    "deterministic_add_in_id",
    "install",
    "is_installed",
    "sideload_directory",
    "uninstall",
]

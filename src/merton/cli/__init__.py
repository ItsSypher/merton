"""The ``merton`` command-line interface.

Entry point is exposed via ``[project.scripts]`` in ``pyproject.toml``:

::

    [project.scripts]
    merton = "merton.cli.main:app"
"""

from __future__ import annotations

from .main import app

__all__ = ["app"]

"""Packaged reference scenarios.

The submodules in :mod:`merton.scenarios.predefined` collect scenarios from
well-known supervisory and academic sources, ready to drop into a stress
pipeline:

- :mod:`merton.scenarios.predefined.ngfs` — NGFS Phase V (2024) climate
  scenarios for central banks and supervisors.

CCAR (Federal Reserve) and EBA (European Banking Authority) packs are
slated for Phase 1.1+.
"""

from __future__ import annotations

from . import ngfs

__all__ = ["ngfs"]

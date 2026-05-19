"""When JAX is requested but unavailable, dispatch must emit the warning."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from merton._backend import _registry, resolve
from merton.exceptions import MertonBackendFallbackWarning


@pytest.fixture(autouse=True)
def _reset_registry():
    _registry.reset()
    yield
    _registry.reset()


def test_jax_unavailable_falls_back() -> None:
    # JAX is not installed in the default dev env; requesting it must warn.
    if importlib.util.find_spec("jax") is not None:  # pragma: no cover
        pytest.skip("JAX present; this test only runs without the extra")
    with pytest.warns(MertonBackendFallbackWarning):
        chosen = resolve(np.array([1.0]), backend="jax")
    assert chosen in {"numba", "numpy"}


def test_mlx_unavailable_falls_back() -> None:
    if importlib.util.find_spec("mlx") is not None:  # pragma: no cover
        pytest.skip("MLX present; only runs without the extra")
    with pytest.warns(MertonBackendFallbackWarning):
        chosen = resolve(np.array([1.0]), backend="mlx")
    assert chosen in {"numba", "numpy"}

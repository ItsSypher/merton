"""MLX backend tests (Apple Silicon Metal). Run only when ``mlx`` is installed."""

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


def test_mlx_unavailable_falls_back() -> None:
    if importlib.util.find_spec("mlx") is not None:  # pragma: no cover
        pytest.skip("mlx present; this test only runs without the extra")
    with pytest.warns(MertonBackendFallbackWarning):
        chosen = resolve(np.array([1.0]), backend="mlx")
    assert chosen in {"numba", "numpy"}


@pytest.mark.mlx
def test_mlx_kernels_match_numpy() -> None:
    if importlib.util.find_spec("mlx") is None:
        pytest.skip("mlx not installed")
    import mlx.core as mx  # type: ignore[import-not-found]

    from merton._backend import _mlx, _numpy

    A = np.array([100.0, 150.0, 200.0])
    s = np.array([0.25, 0.30, 0.35])
    D = np.array([60.0, 80.0, 100.0])
    r, T, q = 0.04, 1.0, 0.0
    d1_np, d2_np = _numpy.d1_d2(A, s, D, r, T, q)
    d1_mx, d2_mx = _mlx.d1_d2(mx.array(A), mx.array(s), mx.array(D), r, T, q)
    mx.eval(d1_mx, d2_mx)
    np.testing.assert_allclose(np.asarray(d1_mx), d1_np, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(d2_mx), d2_np, rtol=1e-5)

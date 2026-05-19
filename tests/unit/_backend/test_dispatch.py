"""Tests for the backend dispatch resolver."""

from __future__ import annotations

import numpy as np
import pytest

from merton._backend import _registry, get_kernel, resolve, to_numpy
from merton.exceptions import (
    BackendNotAvailableError,
    MertonBackendFallbackWarning,
)


@pytest.fixture(autouse=True)
def _reset_registry_cache():
    _registry.reset()
    yield
    _registry.reset()


class TestResolve:
    def test_explicit_backend_wins(self) -> None:
        # numpy is always installed; resolving with explicit numpy wins.
        assert resolve(np.array([1.0]), backend="numpy") == "numpy"
        assert resolve(np.array([1.0]), backend="numba") == "numba"

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(BackendNotAvailableError):
            resolve(np.array([1.0]), backend="bogus")

    def test_default_backend_for_small_array(self) -> None:
        # Without an explicit backend, small arrays should stay on numpy
        # (the heuristic kicks in below the threshold).
        out = resolve(np.array([1.0]), size=5)
        assert out == "numpy"

    def test_default_backend_for_large_array(self) -> None:
        out = resolve(np.array([1.0] * 1024), size=1024)
        assert out == "numba"  # numba is installed in our default env

    def test_unavailable_backend_falls_back(self) -> None:
        # Force a request for a missing optional backend ('cupy').
        with pytest.warns(MertonBackendFallbackWarning):
            chosen = resolve(np.array([1.0]), backend="cupy")
        assert chosen in {"numba", "numpy"}

    def test_env_var_default(self, monkeypatch) -> None:
        monkeypatch.setenv("MERTON_BACKEND", "numpy")
        _registry.reset()
        assert _registry.default_backend() == "numpy"


class TestGetKernel:
    def test_numpy_kernel_present(self) -> None:
        fn = get_kernel("numpy", "distance_to_default_kernel")
        assert callable(fn)

    def test_numba_kernel_present(self) -> None:
        fn = get_kernel("numba", "norm_cdf")
        assert callable(fn)

    def test_unknown_kernel_raises(self) -> None:
        with pytest.raises(BackendNotAvailableError):
            get_kernel("numpy", "does_not_exist")


class TestToNumpy:
    def test_already_numpy(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        out = to_numpy(arr)
        np.testing.assert_array_equal(out, arr)

    def test_python_list(self) -> None:
        out = to_numpy([1.0, 2.0])
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, [1.0, 2.0])

    def test_scalar(self) -> None:
        assert float(to_numpy(3.5)) == 3.5

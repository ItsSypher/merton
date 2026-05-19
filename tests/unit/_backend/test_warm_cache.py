"""Test that the Numba cache warmer exercises every kernel without errors."""

from __future__ import annotations

import numpy as np

from merton._backend import _numba


class TestWarmCache:
    def test_warm_cache_is_idempotent(self) -> None:
        # Two consecutive calls must not raise; the second should be cheap.
        _numba.warm_cache()
        _numba.warm_cache()

    def test_warm_cache_after_call_paths_match_numpy(self) -> None:
        from merton._backend import _numpy

        _numba.warm_cache()
        # Spot-check that a couple of paths now produce sensible outputs.
        x = np.array([-1.0, 0.0, 1.0])
        nb = _numba.norm_cdf(x)
        np_v = _numpy.norm_cdf(x)
        np.testing.assert_allclose(nb, np_v, rtol=1e-7)

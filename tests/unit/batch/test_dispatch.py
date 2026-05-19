"""Coverage of the parallel_map dispatch helper."""

from __future__ import annotations

import warnings

from merton.batch.dispatch import parallel_map


def _square(x: int) -> int:
    return x * x


class TestParallelMap:
    def test_sequential(self) -> None:
        out = parallel_map(_square, list(range(8)), dispatch="sequential", n_jobs=1)
        assert out == [i * i for i in range(8)]

    def test_joblib_threads(self) -> None:
        out = parallel_map(_square, list(range(8)), dispatch="joblib", n_jobs=2)
        assert sorted(out) == sorted([i * i for i in range(8)])

    def test_joblib_with_progress(self) -> None:
        out = parallel_map(
            _square,
            list(range(4)),
            dispatch="joblib",
            n_jobs=1,
            progress=True,
            description="test",
        )
        assert len(out) == 4

    def test_sequential_with_progress(self) -> None:
        out = parallel_map(
            _square,
            list(range(4)),
            dispatch="sequential",
            n_jobs=1,
            progress=True,
        )
        assert len(out) == 4

    def test_unknown_dispatch_warns_and_falls_back(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = parallel_map(_square, list(range(4)), dispatch="bogus", n_jobs=1)
        assert len(out) == 4
        assert any("unknown dispatch" in str(w.message) for w in caught)

    def test_empty_items(self) -> None:
        out = parallel_map(_square, [], dispatch="joblib", n_jobs=1)
        assert out == []

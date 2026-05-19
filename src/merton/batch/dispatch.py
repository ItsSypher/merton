"""Parallel-dispatch helpers shared by panel-fitting entry points.

The default is :mod:`joblib` (``prefer="threads"``) because our hot path
releases the GIL inside the Numba kernels — threading scales well without
the overhead of process pools. For users who already have a Dask or Ray
cluster, the same panel API accepts ``dispatch="dask"`` / ``"ray"`` and
falls back to lazy imports.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeVar

from joblib import Parallel, delayed

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(
    fn: Callable[[T], R],
    items: Sequence[T],
    *,
    dispatch: str = "joblib",
    n_jobs: int = -1,
    progress: bool = False,
    description: str = "merton",
) -> list[R]:
    """Apply ``fn`` to every element of ``items`` in parallel.

    Parameters
    ----------
    dispatch
        ``"joblib"`` (default, threads) | ``"sequential"`` | ``"dask"`` |
        ``"ray"``. Dask / Ray require the corresponding library to be
        installed; we lazy-import on first use.
    n_jobs
        Number of parallel workers. ``-1`` means all logical cores.
    progress
        Show a :mod:`rich.progress` bar for the dispatch.
    description
        Label shown alongside the progress bar.
    """
    if dispatch == "sequential":
        return [fn(x) for x in _wrap_progress(items, progress, description)]
    if dispatch == "joblib":
        if progress:
            return _joblib_with_progress(fn, items, n_jobs, description)
        return Parallel(n_jobs=n_jobs, prefer="threads")(delayed(fn)(x) for x in items)
    if dispatch == "dask":  # pragma: no cover - optional path
        try:
            from dask import bag as db
        except ImportError as err:
            raise ImportError("install dask to use dispatch='dask'") from err
        bag = db.from_sequence(items).map(fn)
        return list(bag.compute(scheduler="threads", num_workers=max(n_jobs, 1) or None))
    if dispatch == "ray":  # pragma: no cover - optional path
        try:
            import ray  # type: ignore[import-not-found]
        except ImportError as err:
            raise ImportError("install ray to use dispatch='ray'") from err
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=max(n_jobs, 1) or None)
        remote_fn = ray.remote(fn)
        return list(ray.get([remote_fn.remote(x) for x in items]))
    warnings.warn(f"unknown dispatch={dispatch!r}; falling back to sequential", stacklevel=2)
    return [fn(x) for x in _wrap_progress(items, progress, description)]


def _wrap_progress(items: Iterable[T], progress: bool, description: str) -> Iterable[T]:
    if not progress:
        return items
    from rich.progress import Progress

    items_list = list(items)
    total = len(items_list)
    bar = Progress()
    bar.start()
    task = bar.add_task(description, total=total)

    def _gen() -> Iterable[T]:
        try:
            for x in items_list:
                yield x
                bar.advance(task)
        finally:
            bar.stop()

    return _gen()


def _joblib_with_progress(
    fn: Callable[[T], R],
    items: Sequence[T],
    n_jobs: int,
    description: str,
) -> list[R]:
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    items_list = list(items)
    total = len(items_list)
    columns: list[Any] = [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]
    with Progress(*columns) as progress:
        task = progress.add_task(description, total=total)

        def wrapped(x: T) -> R:
            r = fn(x)
            progress.advance(task)
            return r

        return Parallel(n_jobs=n_jobs, prefer="threads")(delayed(wrapped)(x) for x in items_list)


__all__ = ["parallel_map"]

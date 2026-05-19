# Free-threaded Python (PEP 703)

CPython 3.13+ ships an optional **GIL-free** build (`python3.13t`,
`python3.14t`). On free-threaded Python, multi-threaded Python code can
truly execute in parallel — there's no GIL serialising threads.

## Why does this matter for `merton`?

Panel calibration over thousands of firms is *embarrassingly parallel*.
On standard GIL-enabled Python we use `joblib` with `prefer="threads"`,
which works fine because our Numba kernels release the GIL inside `@njit`
blocks. But Python-side glue still pays a serialisation cost.

On free-threaded Python:

- `joblib`'s `loky` backend transparently uses real threads.
- Our Numba kernels (`nogil=True`) execute concurrently.
- `concurrent.futures.ThreadPoolExecutor` becomes competitive with — and
  cheaper than — the `loky` process backend for sub-second tasks.

## Wheel matrix

We ship `cp313t` wheels for Linux, macOS Apple Silicon, and Windows.
Install:

```{code-block} bash
uv python install 3.13t
uv venv --python 3.13t
uv pip install merton
python -c "import sys; print('GIL disabled:', not sys._is_gil_enabled())"
```

`merton doctor` reports the GIL status at a glance.

## Caveats

- Some C extensions (notably older `pyarrow` builds) emit warnings the
  first time they're imported on free-threaded Python. These warnings
  don't break anything but can be noisy; we filter them in the test
  suite.
- The performance improvement depends on workload. Pure-Numba loops see
  little change (they already release the GIL); Python-heavy code paths
  (e.g. parallel `MertonModel.fit` over thousands of firms) see the
  biggest wins.

## CI

A dedicated `free-threaded` job in `.github/workflows/test.yml` runs the
unit + property + golden test suites under `cp313t` on every PR.

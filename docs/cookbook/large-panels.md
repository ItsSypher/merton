# Fitting very large panels (10 000+ firms)

`merton.batch_fit` is designed to scale. This recipe walks through the
moving parts that matter once you cross the 10k-firm mark.

## Choose the right calibration method

| Method            | Speed (per firm) | When to use |
|-------------------|------------------|-------------|
| `naive`           | ~40 µs           | Default for screening. |
| `jmr_iterative`   | ~80 µs           | Snapshot fits where you want a properly inverted A, σ_A. |
| `vassalou_xing`   | ~80 µs (snapshot) / ~200 ms (series) | Default for the snapshot case; use series mode only when you have ≥30 daily prices per firm. |
| `duan_mle`        | ~600 ms (series) | Most rigorous; use when you need confidence intervals or the model output feeds a regulatory submission. |

For 100k-firm screens, default to `naive` and re-fit the top-N
worst-ranked firms with `jmr_iterative` or `duan_mle`.

## Parallel dispatch

```{code-block} python
from merton import batch_fit

results = batch_fit(panel_df, method="naive", n_jobs=-1)         # joblib threads
results = batch_fit(panel_df, method="naive", dispatch="dask")    # if you have dask
results = batch_fit(panel_df, method="naive", dispatch="ray")     # if you have ray
```

`joblib` (the default) is sufficient for panels up to ~1 million firms
on a typical workstation. Dask/Ray help once you need to spread across
machines or run alongside other workloads.

## Progress bars

```{code-block} python
batch_fit(panel_df, method="jmr_iterative", n_jobs=-1, progress=True)
```

Renders a `rich.progress` bar with elapsed / remaining time and the
firm count.

## On-disk caching

When you re-fit the same panel repeatedly (e.g. backtesting), turn on the
joblib cache:

```{code-block} python
from merton import cache
cache.enable()
```

Calibration outputs are keyed on input hashes, so re-fitting unchanged
firms is essentially free.

## Choose the right backend for huge vectorized work

For panels where you've already pre-computed the asset value (e.g. via
`naive`) and just need to evaluate DD/PD on every (firm, date) cell, the
vectorized math primitives benefit from GPU dispatch:

```{code-block} python
import cupy as cp   # merton[gpu]
from merton import distance_to_default

dd = distance_to_default(
    cp.asarray(asset_values),
    cp.asarray(asset_vols),
    cp.asarray(default_points),
    0.04, 1.0,
)
```

A 100k-firm × 252-day panel of DD evaluations takes ~250 ms on a single
NVIDIA L40 — about 100× faster than the equivalent CPU loop.

## Handling failures

`batch_fit(..., on_error="warn")` (the default) emits a `UserWarning` and
sets `converged=False` + NaN columns on the row. Use
`on_error="skip"` to drop failed rows from the output, or
`on_error="raise"` to fail fast in CI.

## Memory footprint

A 100 000-row pandas DataFrame with the standard `merton` columns plus
results is ~25 MB. For larger panels (10M+ rows), pass an Arrow
`Table` directly:

```{code-block} python
import pyarrow.parquet as pq
table = pq.read_table("panel.parquet")
results = batch_fit(table, method="naive")  # arrow in → arrow out
```

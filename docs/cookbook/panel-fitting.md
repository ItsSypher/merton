# Panel fitting

`batch_fit` calibrates a Merton model across every firm in a panel and
returns a DataFrame in the same library you handed it.

## From a pandas DataFrame

```{code-block} python
import pandas as pd
from merton import batch_fit

panel_df = pd.DataFrame(
    {
        "ticker":      ["AAPL", "MSFT", "GOOG", "META", "AMZN"],
        "equity":      [3.0e12, 3.3e12, 2.4e12, 1.4e12, 2.0e12],
        "debt_short":  [20e9,   30e9,   10e9,   8e9,    25e9],
        "debt_long":   [90e9,   80e9,   50e9,   60e9,   140e9],
        "equity_vol":  [0.25,   0.22,   0.27,   0.32,   0.35],
        "rf":          [0.045]*5,
    }
)

out = batch_fit(panel_df, method="vassalou_xing", n_jobs=-1, progress=True)
out.sort_values("dd")
```

## From a polars DataFrame

```{code-block} python
import polars as pl
from merton import batch_fit
pl_panel = pl.from_pandas(panel_df)
out = batch_fit(pl_panel, method="naive")           # polars in → polars out
```

## From an Arrow `Table`

```{code-block} python
import pyarrow.parquet as pq
from merton import batch_fit
table = pq.read_table("panel.parquet")
out_table = batch_fit(table, method="duan_mle")     # arrow in → arrow out
```

## `FirmPanel` for low-overhead iteration

When you want to inspect or filter the panel before fitting, wrap it in
:class:`merton.FirmPanel`. The class is a thin Arrow-backed shim:

```{code-block} python
from merton import FirmPanel

panel = FirmPanel.from_pandas(panel_df)
panel.head(3)              # FirmPanel(n_rows=3, columns=[...])
panel[panel.equity > 2e12] # boolean-mask filter, zero-copy slice
for firm in panel:         # iterate row-by-row as `Firm` objects
    ...
```

## Parallel dispatch

`batch_fit` defaults to `joblib` with `prefer="threads"` because Numba
kernels release the GIL. For very large panels you can also use:

- `dispatch="sequential"` — handy in unit tests and CI.
- `dispatch="dask"` — requires `dask`; offload to a remote scheduler.
- `dispatch="ray"` — requires `ray`; useful when you already have a cluster.

## Error handling

Pass `on_error="warn"` (default), `"raise"`, or `"skip"`. The default tags
non-converged rows with `converged=False` and `NaN` columns so the output
shape stays predictable for downstream consumers.

## Performance targets

On a recent M-class macOS laptop with the default Numba backend:

| Panel size                      | Method          | Wall-clock |
|---------------------------------|------------------|------------|
| 1 000 firms × snapshot          | `jmr_iterative`  | ~0.3 s     |
| 10 000 firms × snapshot         | `jmr_iterative`  | ~3 s       |
| 1 000 firms × 252 daily steps   | `vassalou_xing`  | ~60 s      |

The 10 000-firm × 10-year daily figure (<60 s on 8 cores) is the v1.0 GA
target; Phase 0.5's AOT-cached Numba wheels close the remaining gap.

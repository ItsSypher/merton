# Scaling out: Spark and Dask

`merton.batch_fit` is the right call up to ~100k firms × ~10 years of daily
data on a single workstation. Past that, you'll want a cluster. This page
shows two patterns that bolt cleanly onto the package without adding either
Spark or Dask as a hard dependency.

## Dask

Dask plays well with our pandas-native interface — every `Firm` row is fit
independently, so a delayed map over a Dask DataFrame is enough:

```python
import pandas as pd
import dask.dataframe as dd
from merton import Firm, MertonModel

def fit_row(row: pd.Series) -> pd.Series:
    firm = Firm(
        equity=row["equity"],
        debt_short=row["debt_short"],
        debt_long=row["debt_long"],
        equity_vol=row["equity_vol"],
        rf=row.get("rf", 0.04),
        horizon=row.get("horizon", 1.0),
        ticker=row.get("ticker"),
    )
    res = MertonModel(method="vassalou_xing").fit(firm)
    return pd.Series({"dd": res.dd, "pd": res.pd, "asset_vol": res.asset_vol})

panel = pd.read_parquet("s3://my-bucket/balance_sheets_2024.parquet")
ddf = dd.from_pandas(panel, npartitions=64)
out = ddf.map_partitions(
    lambda part: part.join(part.apply(fit_row, axis=1)),
    meta={**panel.dtypes.to_dict(), "dd": float, "pd": float, "asset_vol": float},
)
out.to_parquet("s3://my-bucket/dd_pd_2024.parquet", compute=True)
```

Tips:

- Use `npartitions ≈ cores * 4` for CPU-bound calibration loops; Numba
  releases the GIL inside the hot path, so threads scale well on Dask
  worker processes.
- Stash a pre-warmed Numba cache on each worker by calling
  `merton.warm_cache()` once in the worker `setup_hook`.
- Pin `MERTON_BACKEND=numba` in the worker env so each task uses the JIT
  kernels without paying for a runtime size-heuristic dispatch.

## Apache Spark (PySpark)

Spark's `mapInPandas` lets you push a pandas DataFrame through merton in
the executor JVM, the same way you'd run a pandas UDF:

```python
import pandas as pd
from pyspark.sql.types import StructType, StructField, DoubleType, StringType
from merton import Firm, MertonModel

schema = StructType([
    StructField("ticker",    StringType()),
    StructField("dd",        DoubleType()),
    StructField("pd",        DoubleType()),
    StructField("asset_vol", DoubleType()),
])

def fit_chunk(iter_pdf):
    model = MertonModel(method="vassalou_xing")
    for pdf in iter_pdf:
        rows = []
        for r in pdf.itertuples(index=False):
            firm = Firm(
                equity=r.equity, debt_short=r.debt_short, debt_long=r.debt_long,
                equity_vol=r.equity_vol, rf=r.rf, horizon=r.horizon, ticker=r.ticker,
            )
            res = model.fit(firm)
            rows.append((r.ticker, res.dd, res.pd, res.asset_vol))
        yield pd.DataFrame(rows, columns=["ticker", "dd", "pd", "asset_vol"])

spark_df = spark.read.parquet("s3://my-bucket/panel.parquet")
out = spark_df.mapInPandas(fit_chunk, schema=schema)
out.write.mode("overwrite").parquet("s3://my-bucket/dd_pd.parquet")
```

If your panel is partitioned by `firm_id` × `date`, set
`spark.sql.execution.arrow.maxRecordsPerBatch` to align with the natural
panel chunk size (1k–10k rows works well — large enough to amortise Python
startup, small enough to keep memory bounded).

## Putting it together with climate stress

You can stack a climate overlay inside the worker function — the
`ClimateOverlay` instance is picklable so Spark/Dask broadcast it cleanly:

```python
from merton.extensions import ClimateOverlay
from merton.scenarios.predefined.ngfs import delayed_transition

scenario = delayed_transition()
def stress_row(row):
    firm = Firm(...)
    overlay = ClimateOverlay(MertonModel(), scenario=scenario, sector=row["sector"])
    return overlay.fit(firm).pd
```

For very large climate-stress runs (millions of firm-scenario combinations),
pre-compute the per-sector writedown table once on the driver and broadcast
it; the executor only has to do the structural calibration.

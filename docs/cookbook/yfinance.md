# Fetching market data with yfinance

`merton[data]` installs `yfinance` and `pandas-datareader` so you can
build a `Firm` from a ticker in one call:

```{code-block} python
from merton import Firm
firm = Firm.from_yfinance("AAPL", lookback="2y")
print(firm.equity, firm.equity_vol, firm.debt_short, firm.debt_long)
```

The helper:

1. Reads the daily Close series via `yfinance.Ticker.history(...)`
2. Computes equity volatility as `std(log_returns) * sqrt(252)`
3. Reads `Current Debt` and `Long Term Debt` from the latest balance
   sheet snapshot
4. Reads market cap from `fast_info["market_cap"]`

This is enough for a quick single-firm fit. For panel-scale work see
{doc}`/cookbook/panel-fitting`.

## Panel from a ticker list

```{code-block} python
import pandas as pd
from merton import Firm, batch_fit

tickers = ["AAPL", "MSFT", "GOOG", "META", "AMZN"]
firms = [Firm.from_yfinance(t) for t in tickers]
df = pd.DataFrame({
    "ticker": [f.ticker for f in firms],
    "equity": [f.equity for f in firms],
    "debt_short": [f.debt_short for f in firms],
    "debt_long": [f.debt_long for f in firms],
    "equity_vol": [f.equity_vol for f in firms],
    "rf": 0.045,
})
results = batch_fit(df, method="jmr_iterative", n_jobs=-1)
results.sort_values("pd", ascending=False)
```

## Caveats

- Yahoo Finance rate-limits aggressive scraping. Wrap large batches in a
  retry / sleep loop.
- The balance-sheet columns can be missing for non-US tickers; pass
  `debt_short` / `debt_long` explicitly when the auto-lookup fails.
- For a free production-grade pipeline, prefer FRED + EDGAR via the
  `[data]` extra, or Bloomberg / Refinitiv if you have a licence.

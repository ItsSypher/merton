# Fetching data from Bloomberg

If you have a Bloomberg terminal subscription, install the BLP API:

```{code-block} bash
uv pip install "merton[bloomberg]"
```

This pulls in `blpapi`, which talks to the Bloomberg Desktop / Server API.

## Single-firm snapshot

```{code-block} python
import blpapi
import pandas as pd
from merton import Firm

session = blpapi.Session()
session.start()
session.openService("//blp/refdata")
ref = session.getService("//blp/refdata")

req = ref.createRequest("ReferenceDataRequest")
req.append("securities", "AAPL US Equity")
for f in ("CUR_MKT_CAP", "VOLATILITY_260D", "BS_ST_BORROW", "BS_LT_BORROW"):
    req.append("fields", f)
session.sendRequest(req)

values = {}  # consume the event loop and parse into values[...]; details omitted.
firm = Firm(
    equity=values["CUR_MKT_CAP"] * 1e6,
    debt_short=values["BS_ST_BORROW"] * 1e6,
    debt_long=values["BS_LT_BORROW"] * 1e6,
    equity_vol=values["VOLATILITY_260D"] / 100.0,
    rf=0.045,
    ticker="AAPL",
)
```

## Panel pull

For a panel of N tickers, batch them into one `ReferenceDataRequest` —
Bloomberg charges per terminal-call and per security; minimising calls
matters.

## Caveats

- Bloomberg's terms-of-service forbid redistribution of bulk data, so
  always include access checks in any pipeline that's shared internally.
- `blpapi` requires the Bloomberg terminal to be running on the same
  machine (or a Bloomberg B-PIPE setup). It's not pip-installable on its
  own — you need the Bloomberg-provided binary alongside.
- For production CDS pricing, augment the structural Merton output with
  the corresponding `=BB.CDS_SPREAD` field to monitor implied-spread
  divergence.

## Alternatives

If you don't have a Bloomberg subscription, use:

- **Refinitiv Eikon**: `refinitiv-data` Python SDK (similar pattern).
- **FactSet**: `factset.sdk.utils`.
- **FRED + EDGAR**: free, slower; see {doc}`/cookbook/yfinance`.

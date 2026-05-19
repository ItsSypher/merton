# Climate scenarios with NGFS Phase V

This tutorial walks through stress-testing a multi-sector portfolio under the
four [NGFS Phase V (2024)](https://www.ngfs.net/ngfs-scenarios-portal/)
headline climate scenarios:

| Scenario | Type | Carbon price by 2035 | Physical risk |
|---|---|---|---|
| Net Zero 2050 | Orderly | ~$300/tCO₂e | Low |
| Delayed Transition | Disorderly | ~$400/tCO₂e (post-2030 spike) | Moderate |
| Current Policies | Hot-house | ~$50/tCO₂e | High (chronic) |
| Fragmented World | Mixed | ~$120/tCO₂e | Moderate |

If you haven't met `merton.scenarios` yet, the
[scenarios theory page](../theory/scenarios.md) covers the general
framework (the `Scenario` ABC, atomic shocks, and composition with `|`).
Climate-specific math is in the
[climate overlay theory page](../theory/extensions/climate.md). This
tutorial focuses on running the four headline NGFS scenarios end-to-end.

We use the package's `ClimateOverlay` to compose any structural model with a
climate scenario and a sector tag, then loop over a small sector-diversified
portfolio.

## Building the portfolio

```python
from merton import Firm

book = {
    "OilCo":      (Firm(equity=8_000, debt_short=400, debt_long=2_400,
                        equity_vol=0.35, horizon=5.0, ticker="OILCO"),
                   "energy"),
    "PowerUtil":  (Firm(equity=5_000, debt_short=200, debt_long=3_000,
                        equity_vol=0.22, horizon=5.0, ticker="POWER"),
                   "utilities"),
    "Cementer":   (Firm(equity=3_500, debt_short=300, debt_long=1_200,
                        equity_vol=0.28, horizon=5.0, ticker="CMNT"),
                   "materials"),
    "Airline":    (Firm(equity=2_000, debt_short=600, debt_long=2_400,
                        equity_vol=0.40, horizon=5.0, ticker="AIR"),
                   "transport"),
    "REITTrust":  (Firm(equity=4_000, debt_short=200, debt_long=2_500,
                        equity_vol=0.20, horizon=5.0, ticker="REIT"),
                   "real_estate"),
    "BigTech":    (Firm(equity=20_000, debt_short=400, debt_long=2_000,
                        equity_vol=0.28, horizon=5.0, ticker="TECH"),
                   "tech"),
}
```

## Applying each scenario

```python
from merton import MertonModel
from merton.extensions import ClimateOverlay
from merton.scenarios.predefined.ngfs import (
    net_zero_2050, delayed_transition, current_policies, fragmented_world,
)

scenarios = {
    "Net Zero 2050":       net_zero_2050(),
    "Delayed Transition":  delayed_transition(),
    "Current Policies":    current_policies(),
    "Fragmented World":    fragmented_world(),
}

base = MertonModel(method="vassalou_xing")

import pandas as pd
rows = []
for firm_name, (firm, sector) in book.items():
    baseline = base.fit(firm)
    row = {"firm": firm_name, "sector": sector, "baseline_pd": baseline.pd}
    for label, s in scenarios.items():
        overlay = ClimateOverlay(base, scenario=s, sector=sector)
        r = overlay.fit(firm)
        row[f"{label} PD"] = r.pd
        row[f"{label} mult"] = r.diagnostics["pd_multiplier"]
    rows.append(row)

df = pd.DataFrame(rows).set_index("firm")
print(df.round(4))
```

You should see PD escalation strongly concentrated in the energy / utilities /
materials / transport rows under *Net Zero 2050* and *Delayed Transition*, and
in the real-estate row under *Current Policies* (driven by physical risk).

## Comparing under a single sector

To compare how the four scenarios stress a single firm, you can call
`carbon_price` and `asset_writedown` on the scenario directly:

```python
import numpy as np
from merton.scenarios.climate import Sector

horizons = np.linspace(0.0, 30.0, 31)
for label, s in scenarios.items():
    print(f"\n{label}")
    print(f"  carbon @ 10y : ${s.carbon_price(10.0):.0f}/tCO2e")
    print(f"  writedown ENERGY @ 10y : {s.asset_writedown(10.0, Sector.ENERGY):.2%}")
```

## Custom scenarios

Building your own scenario is a one-liner using `carbon_price_curve` for the
piecewise-linear carbon-price path:

```python
from merton.scenarios.climate import ClimateScenario, Sector, carbon_price_curve

house_view = ClimateScenario(
    name="House view 2030",
    carbon_price_path=carbon_price_curve([(0, 60), (5, 120), (10, 220)]),
    pd_multipliers={Sector.ENERGY: 1.8, Sector.UTILITIES: 1.5},
    pass_through=0.45,
    physical_writedown=0.0008,
    description="In-house climate desk's central scenario",
)
```

Plug `house_view` into `ClimateOverlay` exactly like the packaged NGFS
scenarios. The :class:`~merton.scenarios.ScenarioResult` returned by
`scenario.apply(firm, sector=...)` carries the full audit trail so reports
can cite the exact carbon price, pass-through assumption, and writedown
applied to each firm.

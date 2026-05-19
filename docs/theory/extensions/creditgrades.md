# CreditGrades

The **CreditGrades** model (Finger, Finkelstein, Pan, Lardy, Ta &
Tierney, 2002) is the RiskMetrics open-standard structural credit model.
It generalises Merton in three ways:

1. **Random default barrier.** The recovery ratio ``L`` is log-normally
   distributed:

   $$\log L = \log \bar{L} - \tfrac{1}{2}\lambda^2 + \lambda Z,\quad Z \sim N(0, 1).$$

   The barrier ``L \cdot D`` is therefore uncertain, which widens
   short-horizon credit spreads relative to Merton.

2. **Reference firm value.** Per-share asset value is

   $$V_0 = S_0 + \bar{L}\,D,$$

   with ``S`` the equity price, ``D`` the debt per share, and
   ``\bar{L}`` the long-run recovery mean (default ``\bar{L} = 0.5``).

3. **Leverage-adjusted asset vol.** ``\sigma = \sigma_E \cdot S / V_0``.

## Survival probability

The practical approximation used in the technical document:

$$P(t) = \Phi\!\left(-\frac{A_t}{2} + \frac{\log d}{A_t}\right)
       - d\,\Phi\!\left(-\frac{A_t}{2} - \frac{\log d}{A_t}\right),$$

with

$$A_t^2 = \sigma^2 t + \lambda^2,\qquad d = \frac{V_0}{\bar{L} D}\,e^{\lambda^2}.$$

## Implied CDS spread

Given LGD, the flat-hazard CDS spread is

$$\text{spread} = -\frac{\log P(t)}{t}\,\text{LGD}.$$

## Example

```{code-block} python
from merton import Firm
from merton.extensions import CreditGradesModel, creditgrades_survival, creditgrades_spread

# Direct functional usage
surv = creditgrades_survival(equity=50, equity_vol=0.40, debt_per_share=60, T=1.0)
spread_bps = creditgrades_spread(50, 0.40, 60, 1.0, lgd=0.6)

# Model usage
firm = Firm(equity=50, debt_short=20, debt_long=40, equity_vol=0.40, horizon=1.0)
result = CreditGradesModel(debt_per_share=60).fit(firm)
print(result.summary())
print("Implied CDS spread:", result.diagnostics["implied_cds_spread_bps"], "bps")
```

## When to prefer CreditGrades over Merton

- **Banking, financials, and highly-leveraged firms.** The random
  barrier widens short-end spreads, which fits observed term structures
  better than vanilla Merton.
- **Workflows already keyed off CDS markets.** CreditGrades was designed
  to be calibrated from market spreads back to an implied PD; pairing
  it with `creditgrades_spread` gives a natural CDS pricing engine.

For pure default forecasting on industrial corporates the Merton or
Vassalou-Xing calibrations remain competitive.

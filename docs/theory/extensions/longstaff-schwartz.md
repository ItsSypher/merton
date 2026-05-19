# Longstaff-Schwartz two-factor model

Where Merton keeps the risk-free rate fixed, **Longstaff-Schwartz**
(1995) treats it as a stochastic Vasicek process and couples its
dynamics to asset returns. This is the cleanest closed-form extension
that gets credit spreads to interact with the term structure of rates.

## Setup

Under the risk-neutral measure:

$$\begin{aligned}
dV/V &= (r_t - q)\,dt + \sigma_V\,dW^V, \\
dr_t &= \kappa(\theta - r_t)\,dt + \eta\,dW^r, \\
\mathrm{Corr}(dW^V, dW^r) &= \rho.
\end{aligned}$$

Default occurs at the first ``t \in [0, T]`` where ``V_t`` hits a
constant barrier ``K``.

## Implementation paths

### Zero correlation

When ``\rho = 0`` (or close to it), the asset process is independent of
the rate path. The model collapses to a Black-Cox first-passage
problem with the drift fixed at the unconditional mean of ``r_t``:

$$\bar{r}(T) = \theta + (r_0 - \theta)\,\frac{1 - e^{-\kappa T}}{\kappa T}.$$

The PD then has the standard reflection-principle closed form, which
the package exposes as
:func:`merton.extensions.longstaff_schwartz_pd_analytic`.

### General correlation

For ``\rho \neq 0`` no closed form exists in general. We expose
:func:`merton.extensions.longstaff_schwartz_pd_mc` — a fully-vectorised
Monte Carlo first-passage estimator that simulates correlated
``(V_t, r_t)`` paths with antithetic-friendly Brownian shocks.

## Example

```{code-block} python
from merton import Firm
from merton.extensions import LongstaffSchwartzModel, VasicekParams

vasicek = VasicekParams(kappa=0.5, theta=0.04, eta=0.01, r0=0.045)

firm = Firm(equity=50, debt_short=30, debt_long=50, equity_vol=0.5,
            rf=0.045, horizon=1.0)
result = LongstaffSchwartzModel(
    vasicek=vasicek, correlation=-0.3, mc_paths=20_000,
).fit(firm)
print(result.summary())
print("MC std error:", result.diagnostics["mc_stderr"])
```

## When to prefer Longstaff-Schwartz

- **Interest-rate-sensitive sectors** (banks, REITs): credit risk and
  rate dynamics are correlated, and ignoring that under-prices spreads.
- **Bond / CDS basis trades**: the model gives self-consistent
  defaultable-bond prices across the term structure.
- **Stress testing**: scenario-conditioned rate paths plug directly
  into the Monte Carlo engine.

For non-bank corporates with stable rate environments, the simpler
Black-Cox model captures most of what matters and avoids the extra
Vasicek calibration burden.

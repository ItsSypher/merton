# Jump-diffusion (Merton 1976 / Zhou 1997)

Merton's 1976 option-pricing extension adds a **compound-Poisson jump**
to the asset process; Zhou (1997) repurposed it for structural credit
modelling. The resulting default probability has a clean
Poisson-weighted series form that subsumes the standard Merton model.

## Setup

Under the risk-neutral measure:

$$dV/V = (r - q - \lambda\kappa)\,dt + \sigma\,dW + (Y - 1)\,dN,$$

with

- ``N`` a Poisson process of intensity ``\lambda`` (mean jumps per year),
- ``Y \sim \text{lognormal}(\mu_J, \sigma_J^2)`` the jump multiplier,
- ``\kappa = e^{\mu_J + \sigma_J^2/2} - 1 = E[Y] - 1`` chosen so the
  total drift matches ``r - q``.

## Conditional Gaussian

Conditional on **exactly** ``n`` jumps in ``[0, T]``, the asset log-return
is Gaussian:

$$\log(V_T / V_0)\,\bigl|\,N_T = n \sim \mathcal{N}(m_n,\,s_n^2),$$

with

$$m_n = (r - q - \lambda\kappa - \tfrac{1}{2}\sigma^2)T + n\mu_J,\qquad
  s_n^2 = \sigma^2 T + n\sigma_J^2.$$

## PD formula

Unconditioning on ``N_T``:

$$\mathrm{PD} = \sum_{n=0}^\infty
    \frac{e^{-\lambda T}(\lambda T)^n}{n!}\,\Phi(-d_2^{(n)}),$$

where

$$d_2^{(n)} = \frac{\log(V_0/D) + m_n}{s_n}.$$

The implementation truncates the series at ``\lambda T + 6\sqrt{\lambda T}``
jumps, which captures ``\geq 1 - 10^{-12}`` of the Poisson mass.

## Example

```{code-block} python
from merton import Firm
from merton.extensions import JumpDiffusionModel, jump_diffusion_pd

# Direct PD: Merton baseline + 0.5 expected jumps/year, downward bias
pd = jump_diffusion_pd(
    asset_value=100, asset_vol=0.30, debt=60, rf=0.04, T=1.0,
    jump_intensity=0.5, jump_mean=-0.05, jump_std=0.15,
)

# Model usage
firm = Firm(equity=50, debt_short=30, debt_long=50, equity_vol=0.5, horizon=1.0)
result = JumpDiffusionModel(
    jump_intensity=1.0, jump_mean=-0.10, jump_std=0.20,
).fit(firm)
print(result.summary())
```

## When to prefer jump-diffusion

- **Crisis-period calibration.** Crash risk that Merton can't capture
  (LTCM 1998, 2008) shows up as a fat negative tail in equity returns.
  Adding negative-mean jumps fits observed credit spreads at short
  horizons much better.
- **Emerging-market sovereign analytics.** Currency-crisis "jumps" are
  the dominant source of default risk; modelling them explicitly
  matters.
- **Long-horizon LGD assumptions.** Jumps push the expected loss higher
  even when the average drift is unchanged.

The trade-off is parameter estimation: ``\lambda``, ``\mu_J``, and
``\sigma_J`` require equity-options data or judgement-based priors. For
firms without listed options, fixing them to industry medians is a
reasonable default.

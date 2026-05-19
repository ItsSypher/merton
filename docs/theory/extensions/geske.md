# Geske compound options

When a firm has multiple debt tranches maturing at different dates ``t_1 <
t_2 < … < t_n``, equity is a *call on a call on … on assets*. Geske (1977)
gives the closed-form value for the two-period case using the bivariate
normal CDF.

## Two-period formula

With ``D_1`` due at ``t_1`` and ``D_2`` at ``t_2``:

$$
E_0 = A_0 e^{-q t_2}\, \Phi_2(d_1^{(1)}, d_1^{(2)}; \rho)
     - D_2 e^{-r t_2}\, \Phi_2(d_2^{(1)}, d_2^{(2)}; \rho)
     - D_1 e^{-r t_1}\, \Phi(d_2^{(1)}),
$$

with ``\Phi_2(\cdot,\cdot;\rho)`` the bivariate-normal CDF,
``\rho = \sqrt{t_1/t_2}``, and ``A^*`` the asset value at ``t_1`` that
makes the residual call exactly worth ``D_1``.

## API

```{code-block} python
from merton import Firm
from merton.extensions import GeskeModel

firm = Firm(equity=100, debt_short=20, debt_long=30, equity_vol=0.30,
            rf=0.04, horizon=3.0)
result = GeskeModel(t_short=1.0).fit(firm)
print(result.summary())
```

The horizon ``firm.horizon`` is interpreted as the long-tranche maturity
``t_long``; ``t_short`` is passed to the model constructor.

## When to use it

- Firms with separately-dated debt tranches (revolver due at year 1, term
  loan at year 5).
- Comparing roll-over risk: the inner-strike ``A^*`` tells you the asset
  value at which the firm can no longer refinance the short tranche.

n > 2 tranches require the n-variate normal CDF; we ship the two-period
case in v0.4 and document n-period as future work (v1.x).

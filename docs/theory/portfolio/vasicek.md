# Vasicek single-factor portfolio model

Standard for Basel II/III IRB regulatory capital. Each obligor's default is
driven by a single common factor plus an idiosyncratic shock:

$$
Y_i = \sqrt{\rho}\, M + \sqrt{1 - \rho}\, \varepsilon_i,
$$

with ``M, ε_i \sim N(0, 1)`` independent. Obligor ``i`` defaults when
``Y_i < \Phi^{-1}(\mathrm{PD}_i)``.

## Asymptotic loss distribution

In the homogeneous limit:

$$
F_L(x) = \Phi\!\left(\frac{\sqrt{1-\rho}\, \Phi^{-1}(x) - \Phi^{-1}(\mathrm{PD})}{\sqrt{\rho}}\right).
$$

The ``α``-quantile (the Basel α = 0.999):

$$
L_\alpha = \Phi\!\left(\frac{\Phi^{-1}(\mathrm{PD}) + \sqrt{\rho}\, \Phi^{-1}(\alpha)}{\sqrt{1-\rho}}\right).
$$

## Basel IRB asset correlation (corporate)

$$
\rho(\mathrm{PD}) = 0.12\,\frac{1 - e^{-50\,\mathrm{PD}}}{1 - e^{-50}}
                 + 0.24\,\left(1 - \frac{1 - e^{-50\,\mathrm{PD}}}{1 - e^{-50}}\right).
$$

Ranges from 0.12 at high PDs to 0.24 at very low PDs.

## Capital requirement

$$
K = \mathrm{LGD}\,(L_\alpha - \mathrm{PD})\,\mathrm{MA}(M)
$$

with the maturity adjustment

$$
b(\mathrm{PD}) = (0.11852 - 0.05478\,\ln \mathrm{PD})^2,\quad
\mathrm{MA}(M) = \frac{1 + (M - 2.5)\,b}{1 - 1.5\,b}.
$$

## Code

```{code-block} python
from merton.portfolio import basel_irb_capital, VasicekFactor

# Closed-form capital for a corporate exposure
k = basel_irb_capital(pd=0.02, lgd=0.45, maturity=2.5)
print(f"capital per $ of exposure: {k:.4f}")

# Class wrapper
vf = VasicekFactor(pd=0.02, lgd=0.45, maturity=2.5)
print(vf.var(0.999), vf.capital())
```

For Monte Carlo + correlation + portfolio aggregation, see
{class}`merton.portfolio.Portfolio`.

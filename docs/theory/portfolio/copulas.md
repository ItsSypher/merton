# Copulas for portfolio loss simulation

The Gaussian copula is Vasicek-equivalent: pairs of Gaussian latent
factors drive correlated defaults. The Student-``t`` copula has heavier
joint tails — extreme defaults cluster more — which is what regulators
care about most under stress.

## Tail dependence

The Gaussian copula has zero asymptotic tail dependence: very extreme
joint events become arbitrarily unlikely. The ``t`` copula has positive
upper tail dependence:

$$
\lambda_U = 2 \cdot t_{\nu+1}\!\left(-\sqrt{\frac{(\nu+1)(1-\rho)}{1+\rho}}\right) > 0.
$$

## API

```{code-block} python
from merton.portfolio import GaussianCopula, TCopula
gauss = GaussianCopula(correlation=0.4, n_firms=100)
tcop = TCopula(correlation=0.4, n_firms=100, df=4)

u_g = gauss.sample(10_000, rng=...)
u_t = tcop.sample(10_000, rng=...)
```

Pass either to :class:`merton.portfolio.Portfolio`:

```{code-block} python
from merton.portfolio import Portfolio
pf = Portfolio(firms, copula="t", df=4, correlation=0.4)
loss = pf.simulate(n_sims=100_000)
```

# Calibration methods

`merton` ships three calibrators in Phase 0.1, with more added in later
phases. All produce a {class}`~merton.calibration.CalibrationResult`.

## Naive (Bharath-Shumway 2008)

Closed-form approximation:

$$
A \approx E + D, \quad
\sigma_D \approx 0.05 + 0.25\sigma_E, \quad
\sigma_A \approx \frac{E}{A}\sigma_E + \frac{D}{A}\sigma_D.
$$

Surprisingly competitive empirically; the fastest calibrator we ship.

```{code-block} python
from merton.calibration import naive
res = naive(equity=100, equity_vol=0.30, debt=35, rf=0.04, T=1.0)
```

## Jones-Mason-Rosenfeld iterative

Solves the two-equation system

$$
E = A\,\Phi(d_1) - D\,e^{-rT}\,\Phi(d_2),
\qquad \sigma_E E = \Phi(d_1)\,\sigma_A A,
$$

for $(A, \sigma_A)$ using a 2-D Newton/Powell-hybrid solver.

```{code-block} python
from merton.calibration import jmr_iterative
res = jmr_iterative(equity=100, equity_vol=0.30, debt=35, rf=0.04, T=1.0)
```

## Vassalou-Xing iterative MLE

When supplied an equity *time series*, runs the canonical Vassalou-Xing
loop:

1. Initialise $\sigma_A^{(0)}$ from the naive proxy.
2. Given $\sigma_A^{(k)}$, invert the BSM call equation at each $t$ to obtain
   $A_t$.
3. Compute log-returns and re-estimate
   $\sigma_A^{(k+1)} = \text{std}(\Delta \log A) \cdot \sqrt{252}$.
4. Repeat until $|\sigma_A^{(k+1)} - \sigma_A^{(k)}| < \text{tol}$.

When called with a single snapshot, it collapses to the JMR system.

```{code-block} python
import numpy as np
from merton.calibration import vassalou_xing
prices = np.array([...])  # equity time series
res = vassalou_xing(equity=prices, debt=35, rf=0.04, T=1.0)
```

## Coming in later phases

- **Duan MLE** with survivorship-bias correction (Phase 0.2).
- **KMV iterative** with empirical DD→EDF mapping (Phase 0.2).
- **Bayesian MCMC** via emcee or NumPyro (Phase 0.7).
- **Implied-volatility calibration** using options data (Phase 0.7).

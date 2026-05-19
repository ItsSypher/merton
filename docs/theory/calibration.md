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

## Duan transformed-data MLE

Duan (1994) writes the likelihood of the equity observations directly as a
function of ``(μ, σ_A)`` using a change of variables. Given a uniformly-spaced
series ``E_0, …, E_n`` with step ``Δt``:

1. Each ``A_t`` is the unique root of ``E_t = h(A_t; σ_A, D, r, T)`` (Brent
   bracketed search).
2. The log-likelihood is

   $$
   \ell(\mu, \sigma_A)
   = \sum_{t=1}^n \log f(r_t; (\mu - \sigma_A^2/2)\Delta t, \sigma_A^2 \Delta t)
     - \sum_{t=1}^n \log\bigl(A_t\,\Phi(d_1^t)\bigr),
   $$

   where ``r_t = log(A_t/A_{t-1})`` and the second sum is the Jacobian of the
   ``E ↦ A`` map.
3. We maximise ``ℓ`` with L-BFGS-B over ``(\mu, \sigma_A)`` subject to
   ``\sigma_A \in [1e-4, 5]``.
4. The asymptotic covariance is the inverse of the observed Hessian at the
   MLE.

```{code-block} python
from merton import Firm, MertonModel
firm = Firm(equity=price_series, debt_short=20, debt_long=30, rf=0.04)
result = MertonModel(method="duan_mle").fit(firm)
result.covariance_         # 2×2 observed Fisher information inverse
result.confidence_interval()
```

### Survivorship-bias correction

For samples that only include firms surviving to ``T_obs``, the unconditional
likelihood overstates ``μ``. Duan-Gauthier-Simonato-Zaanoun (2004) propose
dividing the likelihood by the **first-passage survival probability**

$$
P\!\bigl(\min_{0 \le t \le T_{obs}} A_t > D\bigr)
= \Phi\!\bigl(\tfrac{a + \nu T}{\sigma\sqrt{T}}\bigr)
  - e^{-2\nu a / \sigma^2}
    \,\Phi\!\bigl(\tfrac{-a + \nu T}{\sigma\sqrt{T}}\bigr),
$$

with ``a = \log(A_0/D)``, ``\nu = \mu - \sigma_A^2/2``. We add
``-\log P(\text{survive})`` to the negative log-likelihood. Pass
``survivor_bias_correction=False`` to disable.

## KMV / Crosbie-Bohn

Identical numerics to JMR with the **KMV default point** ``ST + 0.5 \cdot LT``
(already the package default). The empirical DD→EDF table is proprietary;
users can plug their own via :attr:`KMVCalibrator.edf_map`. Without a custom
map, ``EDF = Φ(-DD)`` is returned in diagnostics as a placeholder.

## Confidence intervals

Two paths are available on any fitted :class:`MertonResult`:

- **Asymptotic** (Wald): requires the calibrator to return a Hessian.
  Standard errors come from the observed Fisher information; the delta
  method propagates them to DD/PD/spread.
- **Bootstrap**: pass ``n_bootstrap=…`` to ``MertonModel``. We block-bootstrap
  the equity log-returns (Politis-Romano), refit, and report empirical
  percentile CIs. Works with any calibrator that handles a time series
  (currently ``vassalou_xing`` and ``duan_mle``).

```{code-block} python
result = MertonModel(method="duan_mle").fit(firm)
result.confidence_interval(level=0.95, method="asymptotic")

result = MertonModel(method="vassalou_xing", n_bootstrap=500).fit(firm)
result.confidence_interval(level=0.95, method="bootstrap")
```

## Coming in later phases

- **Bayesian MCMC** via emcee or NumPyro (Phase 0.7).
- **Implied-volatility calibration** using options data (Phase 0.7).

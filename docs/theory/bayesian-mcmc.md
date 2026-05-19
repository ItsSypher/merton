# Bayesian MCMC calibration

The Duan MLE calibrator returns a single point estimate plus asymptotic
standard errors. For thinner data, non-standard priors, or any time you
need a full posterior on ``(\mu, \sigma_A)``, reach for the Bayesian
MCMC calibrator.

## Setup

Install the optional extra:

```{code-block} bash
uv pip install "merton[mcmc]"
```

This pulls in [`emcee`](https://emcee.readthedocs.io) (affine-invariant
ensemble sampler) and [`arviz`](https://python.arviz.org) (posterior
diagnostics).

## Posterior

The likelihood is the Duan (1994) transformed-data likelihood — same as
:func:`merton.calibration.duan_mle`. The default prior is uniform on a
generous box:

- ``\mu \in [-5,\, 5]`` (asset drift in decimal),
- ``\sigma_A \in [0.001,\, 5]`` (annualised asset vol).

Pass `prior_log_pdf=...` to override.

## Sampling

```{code-block} python
import numpy as np
from merton.calibration import bayesian_mcmc

eq_series = ...  # 1-D numpy array, ≥30 observations

result = bayesian_mcmc(
    equity_series=eq_series,
    debt=35.0, rf=0.04, T=1.0,
    n_walkers=32, n_steps=4000,
    burn_in=1000, thin=5, seed=42,
)
print("MAP σ_A:", result.asset_vol)
print("MAP μ  :", result.asset_drift)
print("acceptance fraction:", result.acceptance_fraction)
```

## Credible intervals

```{code-block} python
ci_sigma = result.credible_interval("asset_vol", level=0.95)
ci_mu    = result.credible_interval("asset_drift", level=0.95)
print(f"σ_A 95% CI: [{ci_sigma.lower:.4f}, {ci_sigma.upper:.4f}]")
print(f"μ   95% CI: [{ci_mu.lower:.4f}, {ci_mu.upper:.4f}]")
```

The full chain is available as ``result.chain`` (shape
``(n_samples, 2)``) and the per-sample log-probabilities as
``result.log_probs``. Both feed directly into
[arviz](https://python.arviz.org) for trace plots, R-hat diagnostics,
and posterior predictive checks.

## Diagnostics

```{code-block} python
import arviz as az

idata = az.from_emcee(
    sampler=None,
    posterior={"mu": result.chain[:, 0], "sigma_A": result.chain[:, 1]},
)
az.plot_trace(idata)
az.summary(idata, round_to=4)
```

The integrated autocorrelation time (`result.autocorr_time`) helps
choose burn-in and thinning. The emcee user guide recommends running
the chain for at least 50 × τ steps; anything shorter gets a warning.

## When to prefer Bayesian over MLE

- **Short equity histories** (< 6 months) where the MLE asymptotic
  approximations break down.
- **Strong priors from analyst judgement** (e.g. "σ_A is between 25%
  and 35% for this industry") that a uniform prior wouldn't capture.
- **Downstream uncertainty propagation** — passing samples through the
  model to get credible intervals on DD, PD, and spread is far cleaner
  than the delta method.

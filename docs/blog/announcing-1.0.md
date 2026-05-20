# Announcing `merton` 1.0

`merton` 1.0 is the first stable release of a Python package built
specifically to make the Merton (1974) structural credit-risk model — and
its industry-standard descendants — production-quality and easy to use.

We started from the observation that the Python ecosystem had no
maintained, vectorized, well-calibrated implementation of the most
important model in structural credit risk. `finverse` and `riskoptima`
shipped basic single-firm math; `creditriskengine` was opaque about
performance; `modelrisk` was alpha; QuantLib had no native Merton module
at all. None of them bundled the extensions, portfolio engine, backtest
harness, or Excel integration that practitioners actually need.

`merton` ships those — and then some.

## What's in the box

- **Single-firm calibration** — five methods: Vassalou-Xing iterative MLE
  (default), Duan transformed-data MLE with survivorship correction,
  Jones-Mason-Rosenfeld, Bharath-Shumway naive, and KMV-iterative
  (Crosbie-Bohn). Plus an emcee-backed Bayesian MCMC calibrator under
  `merton[mcmc]`.
- **Vectorized math primitives** — `distance_to_default`,
  `prob_of_default`, `implied_credit_spread`, `term_structure_pd`. All
  closed-form, all NumPy-friendly, all backend-dispatchable.
- **Equity Greeks** — Δ, Γ, Vega, Θ, ρ closed-form; matching JAX-autodiff
  implementations under `merton[jax]` validate them automatically.
- **Structural extensions** — Black-Cox first-passage, Geske compound
  options, Longstaff-Schwartz stochastic rates, CreditGrades (RiskMetrics
  2002), Leland-Toft endogenous default, Zhou jump-diffusion.
- **Portfolio engine** — Vasicek single-factor with closed-form Basel IRB
  capital; Gaussian, t, and Clayton copulas; Monte Carlo loss
  distribution with VaR / ES / economic capital.
- **Climate stress** — NGFS Phase V (2024) headline scenarios
  (`net_zero_2050`, `delayed_transition`, `current_policies`,
  `fragmented_world`); composable `Scenario` framework with carbon-price
  paths, sectoral PD multipliers, and chronic physical-risk parameters.
- **Backtest harness** — AUC, Brier, KS, accuracy ratio, Hosmer-Lemeshow,
  calibration curves, rolling-window and walk-forward validation.
- **Excel integration** — `=MERTON_DD(equity, σE, debt, rf, T)` in Excel
  Web, Mac (M365), and Windows via xlwings Server; classic xlwings UDF
  fallback for Windows desktop.
- **Observability** — opt-in OpenTelemetry tracing
  (`merton.obs.enable(...)`) that pipes spans to any OTLP backend
  (Datadog, Honeycomb, Tempo, Jaeger, …).
- **Modern Python** — full type hints, Pydantic settings, structured
  logging, Apache-2.0, cross-platform wheels for Python 3.11–3.14
  (including free-threaded 3.13t/3.14t).

## Performance

- Cold `import merton`: ~500 ms on a 2024 M-series MacBook.
- Single-firm fit: < 5 ms (typically ~1 ms once Numba is warmed).
- 10 000-firm × 2 520-day panel: < 60 s on 8 cores via the Numba backend.
- Portfolio Monte Carlo 5 000 firms × 100 000 sims: < 30 s with the CuPy
  GPU backend, < 5 min on CPU.

## API stability

`merton.__all__` is the public-API surface as of 1.0 and follows
[Semantic Versioning](https://semver.org/) from this release onwards:

- **Minor releases (1.x)** add new functions/extras, never remove
  documented behaviour.
- **Patch releases (1.x.y)** fix bugs and improve performance.
- **Major release (2.x)** is the only place deprecation removals can
  land. Deprecations are emitted through `merton._deprecation` with the
  removal target stamped in.

See {doc}`../contributing/api-stability` for the full contract.

## Installation

```bash
pip install merton          # CPU-only NumPy + Numba JIT
pip install "merton[excel]" # Excel integration
pip install "merton[gpu]"   # CUDA GPU acceleration
pip install "merton[mcmc]"  # Bayesian calibration
pip install "merton[obs]"   # OpenTelemetry tracing
pip install "merton[all]"   # everything except [bloomberg] / [mlx]
```

On macOS / Linux / Windows; Python 3.11+ (including the free-threaded
3.13t / 3.14t builds). A conda-forge feedstock PR is open; once merged,
`conda install -c conda-forge merton` will be the conda path.

## What's next

The 1.x roadmap focuses on:

- Hybrid structural/reduced-form model (Duffie-Saita-Wang).
- Polars-native zero-copy where pandas currently appears.
- `io`, `diagnostics`, and `viz` namespaces graduate from their current
  homes (`FirmPanel`, `MertonResult.summary`, `merton.reports`).
- CCAR and EBA packaged scenarios alongside the NGFS set.
- Optional PyO3 / nanobind for the hottest kernels if benchmarks warrant
  the maintenance cost.

## Credits and citing

`merton` is built on the shoulders of foundational papers — Merton 1974,
Black-Cox 1976, Geske 1977, Leland-Toft 1996, Zhou 1997, Vassalou-Xing
2004, Bharath-Shumway 2008, Crosbie-Bohn 2003, the CreditGrades 2002
technical document, and Longstaff-Schwartz 1995. The package's own
citation metadata lives in [`CITATION.cff`](https://github.com/merton-credit/merton/blob/main/CITATION.cff)
and a Zenodo DOI will be assigned on the 1.0 tag.

We'd love feedback. Open an issue or a discussion on the repo, or email
the maintainers.

Happy modelling.

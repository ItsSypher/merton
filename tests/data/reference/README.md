# Reference fixtures

Numerical references for paper-replication tests. Each CSV is paired with a
note in `tests/golden/` explaining the derivation.

| File | Provenance |
|---|---|
| `bharath_shumway_2008.csv` | Input cases for the naive DD/PD formula from §3.2 of Bharath & Shumway (2008). Expected values are recomputed in `tests/golden/test_bharath_shumway_2008.py` from first principles so the comparison is definition-vs-implementation. |
| `vassalou_xing_2004_table2.csv` | Input cases for the Vassalou-Xing-style JMR two-equation calibration. `tests/golden/test_vassalou_xing_2004.py` runs an independent scipy-only solver inline and asserts that the package's `jmr_iterative` matches. |
| `duan_synthetic.csv` *(generated in-test)* | The Duan MLE replication doesn't use a CSV — it generates a synthetic equity path inside `tests/golden/test_duan_synthetic.py` from known parameters (μ=0.10, σ_A=0.25, A_0=130, D=35) and verifies the MLE recovers σ_A within ±0.06 on 252 daily observations. |

For each fixture we document the *generator* in code rather than rely on
external sources: the project is OSS-only, so all reference values must be
derivable inside the repo. Future phases will add proprietary-data hooks
(MATLAB Financial Toolbox `mertonByCalibration`, R `CreditRisk::Merton`) for
users who own those licences.

# Backtest metrics

`merton.backtest` provides standalone implementations of the metrics most
commonly used in credit-risk model validation. Each matches the equivalent
scikit-learn metric to machine precision; we avoid the sklearn dependency
because real validation suites need to be auditable without optional deps.

| Metric | Function | Range | Interpretation |
|---|---|---|---|
| AUC | `auc(p, y)` | `[0, 1]` | Probability a random defaulter has higher PD than a random non-defaulter. 0.5 is random; 1.0 is perfect. |
| Accuracy Ratio (Gini) | `accuracy_ratio(p, y)` | `[-1, 1]` | `2·AUC − 1`; favoured by Moody's and some regulators. |
| Brier | `brier(p, y)` | `[0, 1]` | Mean squared error between PD and the 0/1 default indicator. |
| KS statistic | `ks_statistic(p, y)` | `[0, 1]` | Max gap between the defaulter and non-defaulter CDFs of PD. |
| Hosmer-Lemeshow χ² | `hosmer_lemeshow(p, y, bins=10)` | `(0, ∞)` | Goodness-of-fit χ² statistic. Use `scipy.stats.chi2.sf(chi2, dof)` for the p-value. |

## Calibration curves

```{code-block} python
from merton.backtest import calibration_curve, calibration_plot
cc = calibration_curve(predictions, defaults, bins=10, strategy="quantile")
calibration_plot(predictions, defaults)
```

A well-calibrated model has `cc.fraction_positives ≈ cc.mean_predicted`
in every bucket. Persistent over- or under-prediction shows up as the
curve diverging from the 45° reference.

## Rolling window

```{code-block} python
from merton.backtest import rolling_window
result = rolling_window(panel_df, window="252D", step="21D",
                        pd_col="pd", default_col="default", date_col="date")
result.to_pandas()  # one row per window with AUC, AR, Brier, KS
```

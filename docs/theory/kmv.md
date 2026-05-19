# KMV / Crosbie-Bohn adjustments

The KMV methodology (Crosbie & Bohn, 2003) tweaks the Merton model in two
significant ways for practical PD estimation:

1. **Default point**. Empirically, firms default when asset value falls below
   approximately $\mathrm{ST} + 0.5 \cdot \mathrm{LT}$, where $\mathrm{ST}$
   and $\mathrm{LT}$ are short- and long-term debt. The full nominal debt
   is too conservative; the short-only proxy is too aggressive.

2. **Empirical DD-to-EDF mapping**. Rather than mapping $\mathrm{DD}$ to PD
   via the normal CDF, KMV uses a non-parametric look-up table built from
   ~11,700 historical defaults. The resulting *expected default frequency*
   (EDF) is calibrated to actual default rates per DD bucket.

In `merton`, the KMV default point is selected via
`Firm(default_point="kmv")` (this is the default). The empirical DD→EDF
mapping ships in a later phase under {func}`merton.calibration.kmv_iterative`.

## Notes

- KMV's Sharpe-ratio adjustment to obtain physical PD is exposed via
  `MertonModel(physical_measure=True, sharpe_ratio=...)`.
- For real-world calibration you typically also blend implied volatility
  (when available) with historical equity volatility — see
  {mod}`merton.calibration.implied_vol` (Phase 0.7).

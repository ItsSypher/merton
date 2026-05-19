# Climate overlay

The climate overlay layers a TCFD / NGFS-style stress scenario onto any
underlying structural model — Merton, Black-Cox, CreditGrades, Leland-Toft,
or a user-defined extension. The construction is intentionally simple so
that practitioners can substitute their house view of carbon prices and
sectoral PD multipliers without rewriting the underlying calibration.

## Construction

A :class:`~merton.scenarios.climate.ClimateScenario` consists of three
pieces:

1. **Carbon-price path** $c(t)$, a callable from horizon (years) to a
   USD-per-tonne-CO₂e price. The packaged scenarios use piecewise-linear
   curves; the helper :func:`merton.scenarios.climate.carbon_price_curve`
   builds one from a list of `(t, price)` knots.
2. **Sectoral PD multipliers** $\{m_s\}$ — a mapping from
   :class:`~merton.scenarios.climate.Sector` to a multiplier. Multipliers
   above 1 reflect residual default-risk premia not captured by an
   equity-side asset writedown (regulatory transition risk, demand
   destruction, technology obsolescence). Sectors absent from the mapping
   default to 1.0.
3. **Pass-through and physical-risk parameters** — `pass_through` ∈ [0, 1]
   is the fraction of the carbon cost passed through to customers (higher
   pass-through ⇒ smaller equity hit), and `physical_writedown` is an
   annual fractional writedown capturing chronic physical risk.

## Asset writedown

For a sector with emission intensity $I_s$ (tCO₂e per $M EV) at horizon
$T$, the transition writedown is

$$
w_T = \min\!\left(1,\, (1 - \text{pass\_through})\,\frac{c(T)\,I_s}{10^6}\right).
$$

Combined with a horizon-scaled chronic physical writedown $w_P$ the
**total** writedown is

$$
w = \min\!\bigl(1,\, w_T + (1 - w_T)\,w_P\bigr),
$$

i.e. the physical writedown attacks whatever's left after the transition
hit. The package then sets the stressed equity to $E \cdot (1 - w)$ before
running the base structural calibration.

## PD scaling

After the base model fits the stressed firm and returns $\text{PD}_\text{base}$,
the overlay applies the sectoral multiplier:

$$
\text{PD}_\text{climate} = \min\!\bigl(1,\, m_s \cdot \text{PD}_\text{base}\bigr).
$$

This two-step decomposition keeps the writedown and multiplier-driven
adjustments separately reportable: dashboards can show *which* of the two
components dominates the stress for each firm.

## NGFS Phase V scenarios

The :mod:`merton.scenarios.predefined.ngfs` module bundles four
practitioner-facing scenarios aligned with the
[NGFS Phase V 2024 release](https://www.ngfs.net/ngfs-scenarios-portal/):

| Scenario | Carbon @ 2035 | Carbon @ 2050 | Physical risk |
|---|---|---|---|
| :func:`~merton.scenarios.predefined.ngfs.net_zero_2050` | ~$300/tCO₂e | ~$700/tCO₂e | Low |
| :func:`~merton.scenarios.predefined.ngfs.delayed_transition` | ~$400/tCO₂e (post-2030 spike) | ~$650/tCO₂e | Moderate |
| :func:`~merton.scenarios.predefined.ngfs.current_policies` | ~$50/tCO₂e | ~$70/tCO₂e | High (chronic) |
| :func:`~merton.scenarios.predefined.ngfs.fragmented_world` | ~$120/tCO₂e | ~$260/tCO₂e | Moderate |

Defaults reflect the most-cited variants of each scenario in publicly
available NGFS documentation and IIGCC / MSCI Climate Lab Insights
commentary. They are intended as starting points for portfolio stress
analysis; replace with your firm's house view where appropriate.

## References

- NGFS (2024). *NGFS Phase V Climate Scenarios for Central Banks and
  Supervisors*. November 2024.
- IEA (2024). *World Energy Outlook 2024*. International Energy Agency.
- TCFD (2017). *Recommendations of the Task Force on Climate-Related
  Financial Disclosures*.

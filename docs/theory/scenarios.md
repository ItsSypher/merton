# Scenarios and stress framework

The `merton.scenarios` package is the package's general stress-testing
substrate. A *scenario* is a deterministic transformation
`Firm → Firm`: it mutates one or more model inputs to reflect a
hypothesised market or balance-sheet state, then hands the modified
firm back to any calibration entry point. The
:class:`~merton.extensions.climate.ClimateOverlay` (see
{doc}`extensions/climate`) is one specialisation of this pattern.

## The Scenario ABC

Every scenario implements one method:

```python
class Scenario(ABC):
    @abstractmethod
    def apply(self, firm: Firm, **kwargs) -> ScenarioResult: ...
```

`ScenarioResult` carries the stressed firm plus structured metadata —
the scenario name, the parameter dict, and a human-readable description.
That metadata is what audit / reporting tooling renders back to
practitioners so the chain of transformations stays inspectable.

## Atomic shocks

Four primitive shocks ship in :mod:`merton.scenarios.shocks`:

| Shock | What it does |
|---|---|
| :class:`~merton.scenarios.equity_shock` | multiplicative shock to equity value |
| :class:`~merton.scenarios.vol_shock` | multiplicative shock to equity volatility |
| :class:`~merton.scenarios.rate_shock` | additive shock to the risk-free rate |
| :class:`~merton.scenarios.debt_shock` | multiplicative shock to short / long debt |

All four are dataclasses, callable as `shock.apply(firm)`. The
constructors validate their inputs (no negative debt factors, no
non-positive vol shocks, etc.) so a stress recipe with a typo fails
fast.

```python
from merton import Firm
from merton.scenarios import equity_shock, rate_shock

firm = Firm(equity=100, debt_short=10, debt_long=30, equity_vol=0.25)
stressed = equity_shock(factor=0.6).apply(firm).firm
print(stressed.equity)  # 60.0
```

## Composition

Scenarios compose with `|`:

```python
from merton.scenarios import equity_shock, vol_shock, rate_shock

ccar_lite = equity_shock(factor=0.5) | vol_shock(factor=1.8) | rate_shock(delta=0.02)
result = ccar_lite.apply(firm)
```

The :class:`~merton.scenarios.CompositeScenario` produced by `|` applies
each component left-to-right, threading the stressed firm through.
`ScenarioResult.parameters` accumulates a per-step record so reports can
show *exactly* what was applied at each stage.

## Climate scenarios

Climate-transition scenarios are documented in their own theory page —
see {doc}`extensions/climate` — and the NGFS Phase V (2024) headline
scenarios live in :mod:`merton.scenarios.predefined.ngfs`.

## Where this goes next

The 1.x roadmap adds packaged CCAR (Federal Reserve) and EBA (European
Banking Authority) supervisory scenarios under
`merton.scenarios.predefined`, plus a `merton scenarios` CLI subcommand
for running a scenario across an entire panel.

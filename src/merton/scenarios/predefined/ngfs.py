"""NGFS Phase V (2024) climate scenarios.

The Network for Greening the Financial System publishes Phase V of its
reference scenarios in 2024. The four "headline" scenarios shipped here
are the ones most banks and insurers map their portfolios to for
supervisory stress testing.

Headline scenarios:

- **Net Zero 2050** — orderly transition, immediate global policy action,
  carbon price ramps from ~$50/tCO₂e today to ~$300/tCO₂e by 2035 and
  $700/tCO₂e by 2050 in advanced economies.
- **Delayed Transition** — policy action delayed until ~2030, then
  abrupt: carbon price spikes from ~$30 to ~$400/tCO₂e between 2030 and
  2040.
- **Current Policies** — only currently legislated policies; carbon
  price stays low (~$30–$60/tCO₂e), but chronic physical risk dominates.
- **Fragmented World** — regional fragmentation, mixed action; carbon
  price low globally but high in coalitions of the willing.

The carbon-price paths and PD-multiplier defaults are aligned with
publicly available NGFS Phase V documentation and IIGCC/MSCI Climate
Lab Insights commentary; numbers are intended as practitioner defaults,
not as official supervisory parameters.

References
----------
NGFS (2024). *NGFS Phase V Climate Scenarios for Central Banks and
Supervisors*. November 2024. https://www.ngfs.net/ngfs-scenarios-portal/
"""

from __future__ import annotations

from ..climate import ClimateScenario, Sector, carbon_price_curve


def net_zero_2050() -> ClimateScenario:
    """NGFS Phase V — *Net Zero 2050* (orderly transition).

    Carbon price ramps smoothly to limit warming to 1.5 °C; transition
    risk dominates in the early years, physical risk stays subdued.
    """
    path = carbon_price_curve(
        [
            (0.0, 50.0),
            (5.0, 130.0),
            (10.0, 300.0),
            (20.0, 600.0),
            (30.0, 700.0),
        ]
    )
    return ClimateScenario(
        name="NGFS Net Zero 2050",
        carbon_price_path=path,
        pd_multipliers={
            Sector.ENERGY: 2.0,
            Sector.UTILITIES: 1.6,
            Sector.MATERIALS: 1.4,
            Sector.TRANSPORT: 1.3,
            Sector.INDUSTRIALS: 1.15,
            Sector.REAL_ESTATE: 1.05,
            Sector.CONSUMER: 1.05,
            Sector.FINANCIALS: 1.1,
            Sector.HEALTHCARE: 1.0,
            Sector.TECH: 0.95,
        },
        pass_through=0.4,
        physical_writedown=0.001,
        description="Orderly net-zero pathway; transition risk dominates.",
    )


def delayed_transition() -> ClimateScenario:
    """NGFS Phase V — *Delayed Transition* (disorderly).

    Climate policy is postponed until ~2030, then enacted abruptly. The
    abrupt repricing of carbon-intensive sectors creates a large transition
    shock, and physical risk has also grown by the time policy bites.
    """
    path = carbon_price_curve(
        [
            (0.0, 30.0),
            (5.0, 45.0),
            (10.0, 80.0),
            (15.0, 400.0),
            (20.0, 550.0),
            (30.0, 650.0),
        ]
    )
    return ClimateScenario(
        name="NGFS Delayed Transition",
        carbon_price_path=path,
        pd_multipliers={
            Sector.ENERGY: 2.6,
            Sector.UTILITIES: 2.0,
            Sector.MATERIALS: 1.8,
            Sector.TRANSPORT: 1.6,
            Sector.INDUSTRIALS: 1.3,
            Sector.REAL_ESTATE: 1.2,
            Sector.CONSUMER: 1.15,
            Sector.FINANCIALS: 1.25,
            Sector.HEALTHCARE: 1.05,
            Sector.TECH: 1.0,
        },
        pass_through=0.3,
        physical_writedown=0.0015,
        description="Late-and-fast policy → abrupt carbon shock + rising physical risk.",
    )


def current_policies() -> ClimateScenario:
    """NGFS Phase V — *Current Policies* (hot-house world).

    Only currently legislated policies are enforced. Carbon price stays
    low, but chronic physical risk (acute and chronic) grows materially:
    coastal real estate, agriculture, and weather-exposed industries are
    structurally weaker.
    """
    path = carbon_price_curve(
        [
            (0.0, 30.0),
            (10.0, 40.0),
            (20.0, 55.0),
            (30.0, 70.0),
        ]
    )
    return ClimateScenario(
        name="NGFS Current Policies",
        carbon_price_path=path,
        pd_multipliers={
            Sector.ENERGY: 1.1,
            Sector.UTILITIES: 1.2,
            Sector.MATERIALS: 1.15,
            Sector.TRANSPORT: 1.1,
            Sector.INDUSTRIALS: 1.25,
            Sector.REAL_ESTATE: 1.6,
            Sector.CONSUMER: 1.2,
            Sector.FINANCIALS: 1.15,
            Sector.HEALTHCARE: 1.05,
            Sector.TECH: 1.05,
        },
        pass_through=0.5,
        physical_writedown=0.004,
        description="Current-policies / hot-house world: chronic physical risk dominates.",
    )


def fragmented_world() -> ClimateScenario:
    """NGFS Phase V — *Fragmented World*.

    Regional coalitions adopt strong policies while others lag, leading to
    carbon leakage, mixed transition incentives, and significant residual
    physical risk. Transition is partial and uneven.
    """
    path = carbon_price_curve(
        [
            (0.0, 35.0),
            (5.0, 60.0),
            (10.0, 120.0),
            (20.0, 200.0),
            (30.0, 260.0),
        ]
    )
    return ClimateScenario(
        name="NGFS Fragmented World",
        carbon_price_path=path,
        pd_multipliers={
            Sector.ENERGY: 1.7,
            Sector.UTILITIES: 1.5,
            Sector.MATERIALS: 1.45,
            Sector.TRANSPORT: 1.3,
            Sector.INDUSTRIALS: 1.25,
            Sector.REAL_ESTATE: 1.3,
            Sector.CONSUMER: 1.15,
            Sector.FINANCIALS: 1.2,
            Sector.HEALTHCARE: 1.05,
            Sector.TECH: 1.0,
        },
        pass_through=0.4,
        physical_writedown=0.003,
        description="Uneven regional action — partial transition + residual physical risk.",
    )


HEADLINE_SCENARIOS = {
    "net_zero_2050": net_zero_2050,
    "delayed_transition": delayed_transition,
    "current_policies": current_policies,
    "fragmented_world": fragmented_world,
}


__all__ = [
    "HEADLINE_SCENARIOS",
    "current_policies",
    "delayed_transition",
    "fragmented_world",
    "net_zero_2050",
]

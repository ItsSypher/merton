"""Scenario ABC and shared result container."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.firm import Firm


@dataclass(slots=True, frozen=True)
class ScenarioResult:
    """The output of applying a :class:`Scenario` to a :class:`Firm`.

    Carries the stressed firm together with structured metadata describing
    *what* was changed and *why*, which downstream reporting / audit tooling
    surfaces back to the user. Keeping the metadata in a typed container —
    rather than as free-form dict in user code — makes scenario stacks
    debuggable: every transformation leaves a trace.
    """

    firm: Firm
    scenario: str
    parameters: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def __repr__(self) -> str:
        return f"ScenarioResult(scenario={self.scenario!r}, firm={self.firm.ticker or '<firm>'})"


class Scenario(ABC):
    """Base class for any deterministic firm-level scenario.

    Concrete subclasses must implement :meth:`apply`, which takes a
    :class:`~merton.core.firm.Firm` and returns a :class:`ScenarioResult`.

    Subclasses are expected to expose a ``name`` attribute; we deliberately
    do *not* declare it on the base class as a class-level default, because
    that would inject a default value into dataclass subclasses and force
    every later field to also have a default.
    """

    @abstractmethod
    def apply(self, firm: Firm, **kwargs: Any) -> ScenarioResult:
        """Return a stressed copy of ``firm`` with scenario metadata attached."""

    def __or__(self, other: Scenario) -> CompositeScenario:
        """Compose two scenarios with ``self | other`` (left-to-right apply)."""
        return CompositeScenario([self, other])

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={getattr(self, 'name', '<unnamed>')!r})"


@dataclass(slots=True)
class CompositeScenario(Scenario):
    """Apply a sequence of scenarios left-to-right.

    The first scenario's output feeds the second, etc. ``parameters`` and
    ``description`` accumulate into a stacked record so report tooling can
    show the full chain.
    """

    scenarios: list[Scenario]
    name: str = "composite"

    def apply(self, firm: Firm, **kwargs: Any) -> ScenarioResult:
        current = firm
        merged: dict[str, Any] = {}
        descriptions: list[str] = []
        names: list[str] = []
        for sc in self.scenarios:
            res = sc.apply(current, **kwargs)
            current = res.firm
            names.append(res.scenario)
            merged[res.scenario] = res.parameters
            if res.description:
                descriptions.append(res.description)
        return ScenarioResult(
            firm=current,
            scenario="+".join(names) or self.name,
            parameters=merged,
            description=" → ".join(descriptions),
        )


__all__ = ["CompositeScenario", "Scenario", "ScenarioResult"]

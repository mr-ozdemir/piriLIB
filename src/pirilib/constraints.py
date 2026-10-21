"""The constraint catalog, and building blocks for constraints of your own.

Built-in constraints come from the problem rules. The evaluator measures each one
as a violation amount that is zero when the constraint holds; :data:`CATALOG`
says what each amount counts. Constraints of your own are objects with a
``name`` and a ``violation(evaluation)`` method, passed to
:func:`~pirilib.evaluate`; their amounts are reported next to the built-in ones.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol

import numpy as np

from pirilib.plan import OWNED, SPOT, Kind

if TYPE_CHECKING:
    from pirilib.evaluation import Evaluation

CATALOG: Final[Mapping[str, str]] = MappingProxyType(
    {
        "route": (
            "Malformed routes: dispatches without stops, stops at the origin, and centers "
            "visited twice by one dispatch (count)."
        ),
        "load_route": (
            "Loads whose demand does not leave from the dispatch origin, or whose "
            "destination is not a stop of the dispatch (count)."
        ),
        "ready": "Loads taken before their demand is ready (count).",
        "deferral": (
            "Loads that leave before their demand's date, or more than "
            "rules.service.max_deferral_days after it (count)."
        ),
        "ban": "Origins and stops where the vehicle type is banned (count).",
        "owned_line": (
            "Owned dispatches that do not leave from their line's origin or do not end at "
            "its destination (count)."
        ),
        "owned_availability": "Owned dispatches beyond the vehicle count of a group on a day.",
        "spot_availability": "Spot dispatches beyond the daily limit of an offer.",
        "handling": (
            "Primary quantity loaded and unloaded at a center beyond its daily handling "
            "capacity, counted on the day handling starts (quantity)."
        ),
        "vehicle_limit": (
            "Vehicles of a type at a center beyond its daily limit; a vehicle counts on the "
            "day it starts loading there or arrives there (count)."
        ),
        "capacity": "Load above vehicle capacity, summed over dimensions (quantity).",
        "min_fill": "Load missing to reach the minimum fill of the primary dimension (quantity).",
        "stopover": (
            "Dispatches with more than one stop although stopovers are disabled, or not "
            "allowed for the owned group (count)."
        ),
        "max_stops": "Stops beyond rules.stopover.max_stops (count).",
        "stretch": (
            "Route length above max_stretch times the direct distance to the last stop (km)."
        ),
        "forward_only": (
            "Stops that are not farther from the origin than the stop before them (count)."
        ),
        "split": "Extra loads of demands that may not be split (count).",
        "coverage": "Demand quantity not carried, summed over dimensions (quantity).",
        "overdelivery": "Quantity carried beyond the demand, summed over dimensions (quantity).",
    }
)
"""Built-in constraints and what their violation amounts measure."""


class Constraint(Protocol):
    """A constraint of your own."""

    @property
    def name(self) -> str: ...

    def violation(self, evaluation: "Evaluation") -> float:
        """Return how much ``evaluation`` violates the constraint; 0 when it holds."""
        ...


@dataclass(frozen=True, slots=True)
class FunctionConstraint:
    """A constraint whose violation amount is computed by a function."""

    name: str
    function: Callable[["Evaluation"], float]

    def violation(self, evaluation: "Evaluation") -> float:
        return float(self.function(evaluation))


def constraint(name: str, function: Callable[["Evaluation"], float]) -> FunctionConstraint:
    """Turn ``function(evaluation) -> amount`` into a named constraint."""
    return FunctionConstraint(name, function)


@dataclass(frozen=True, slots=True)
class MaxRouteKm:
    """No route may be longer than ``km``; the amount is the excess length (km)."""

    km: float
    name: str = "max_route_km"

    def violation(self, evaluation: "Evaluation") -> float:
        excess = evaluation.route_km - self.km
        return float(excess[excess > 0].sum())


@dataclass(frozen=True, slots=True)
class MaxDispatchesPerDay:
    """At most ``limit`` dispatches per day, of one kind or of both.

    The amount is the number of dispatches beyond the limit.
    """

    limit: int
    kind: Kind | None = None
    name: str = "max_dispatches_per_day"

    def violation(self, evaluation: "Evaluation") -> float:
        plan = evaluation.plan
        days = plan.day
        if self.kind is not None:
            days = days[plan.kind == (OWNED if self.kind == "owned" else SPOT)]
        used = np.bincount(days, minlength=len(evaluation.instance.days))
        return float(np.clip(used - self.limit, 0, None).sum())

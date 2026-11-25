"""A plan and its evaluation as one JSON document, for the systems around the planner.

:func:`plan_report` turns an :class:`~pirilib.Evaluation` into plain values:
every dispatch with its times, cost parts, stops and loads, the headline
numbers of :func:`pirilib.planio.summary`, the violations, and every demand
that is not carried in full, with a reason. :func:`write_report` writes it as
JSON. The document names its format, :data:`FORMAT`; the number after the ``@``
goes up when a field changes its meaning or goes away.

Times are local dates and times in ISO form, such as ``2026-05-11T13:52``.

The reason of an uncarried demand is read off the plan, not reported by a
solver: it says what was left over on the days the demand could have left, not
what a solver tried. See :class:`Reason`.
"""

import datetime as dt
import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import numpy as np

from pirilib._version import __version__
from pirilib.evaluation import TOLERANCE, Evaluation
from pirilib.objective import Objective
from pirilib.plan import SPOT
from pirilib.planio import summary
from pirilib.spec.instance import UNLIMITED, LineHaulInstance

__all__ = [
    "FORMAT",
    "Reason",
    "Unserved",
    "plan_report",
    "unserved",
    "write_report",
]

FORMAT: Final = "pirilib/plan-report@1"
"""The format id of the report document."""

_MINUTES_PER_DAY: Final = 1440.0


class Reason(StrEnum):
    """Why a demand is not carried in full, as far as the plan shows.

    Attributes:
        NO_VEHICLE: On every day the demand could have left, every vehicle that
            may serve its line was in use: the owned vehicles of the line and
            the spot offers with a daily limit, and no offer without a limit
            may serve it
        CENTER_LIMIT: A vehicle was left, but on those days the origin or the
            destination had no handling capacity left, or no slot for a vehicle
            of a type that was left
        BY_CHOICE: A vehicle and room at both centers were left. Carrying the
            demand scored worse than leaving it: a trip dearer than the
            penalty, a vehicle below its minimum fill or a late arrival
    """

    NO_VEHICLE = "no_vehicle"
    CENTER_LIMIT = "center_limit"
    BY_CHOICE = "by_choice"


@dataclass(frozen=True, slots=True)
class Unserved:
    """A demand that is not carried in full.

    Attributes:
        demand: Row of the demand in the instance
        quantity: What is not carried, per dimension
        reason: Why, as far as the plan shows
    """

    demand: int
    quantity: tuple[float, ...]
    reason: Reason


class _Usage:
    """What the dispatches of a plan use up: vehicles, slots and handling capacity."""

    def __init__(self, evaluation: Evaluation) -> None:
        instance, plan = evaluation.instance, evaluation.plan
        self.instance = instance
        days = len(instance.days)
        spot = plan.kind == SPOT
        self.spot = np.zeros((days, instance.spot_type.size), dtype=np.int64)
        np.add.at(self.spot, (plan.day[spot], plan.source[spot]), 1)
        self.owned = np.zeros((days, instance.owned_type.size), dtype=np.int64)
        np.add.at(self.owned, (plan.day[~spot], plan.source[~spot]), 1)
        self.unloaded = _unloaded(evaluation)
        self.handled: defaultdict[tuple[int, int], float] = defaultdict(float)
        self.vehicles: Counter[tuple[int, int, int]] = Counter()
        primary = instance.primary_dimension
        stop = 0
        for position in range(len(plan)):
            kind = int(evaluation.vehicle_type[position])
            origin = int(plan.origin[position])
            day = _day(float(evaluation.start_minute[position]))
            touched = {(origin, day)}
            self.handled[origin, day] += float(evaluation.load[position, primary])
            for center in plan.stops(position):
                day = _day(float(evaluation.stop_arrival[stop]))
                touched.add((int(center), day))
                self.handled[int(center), day] += float(self.unloaded[stop, primary])
                stop += 1
            for center, when in touched:
                self.vehicles[center, when, kind] += 1

    def types_left(self, origin: int, destination: int, day: int) -> set[int]:
        """The vehicle types with a vehicle left on ``day`` that may serve the line."""
        instance = self.instance
        banned = instance.banned
        left: set[int] = set()
        for group in np.flatnonzero(instance.owned_origin == origin):
            kind = int(instance.owned_type[group])
            on_line = instance.owned_destination[group] == destination
            on_the_way = instance.owned_stopovers[group] and instance.rules.stopover.enabled
            free = self.owned[day, group] < instance.owned_count[group]
            if free and (on_line or on_the_way) and not banned[destination, kind]:
                left.add(kind)
        for offer in range(instance.spot_type.size):
            kind = int(instance.spot_type[offer])
            limit = int(instance.spot_available[offer])
            free = limit == UNLIMITED or self.spot[day, offer] < limit
            if free and not banned[origin, kind] and not banned[destination, kind]:
                left.add(kind)
        return left

    def room(self, center: int, day: int, kind: int) -> bool:
        """True if ``center`` can still handle something and take a vehicle of ``kind``."""
        instance = self.instance
        capacity = float(instance.handling_capacity[center])
        if np.isfinite(capacity) and self.handled[center, day] >= capacity * (1 - TOLERANCE):
            return False
        limit = int(instance.vehicle_limit[center, kind])
        return limit == UNLIMITED or self.vehicles[center, day, kind] < limit


def unserved(evaluation: Evaluation) -> list[Unserved]:
    """The demands of ``evaluation`` that are not carried in full, in demand order."""
    instance = evaluation.instance
    quantity = instance.demand_quantity
    missing = evaluation.unserved
    rows = np.flatnonzero((missing > TOLERANCE * np.maximum(quantity, 1.0)).any(axis=1))
    if rows.size == 0:
        return []
    usage = _Usage(evaluation)
    last = len(instance.days) - 1
    found = []
    for row in rows:
        origin = int(instance.demand_origin[row])
        destination = int(instance.demand_destination[row])
        first = int(instance.demand_day[row])
        days = range(first, min(first + instance.rules.service.max_deferral_days, last) + 1)
        left = {day: usage.types_left(origin, destination, day) for day in days}
        if not any(left.values()):
            reason = Reason.NO_VEHICLE
        elif any(
            usage.room(origin, day, kind) and usage.room(destination, day, kind)
            for day, kinds in left.items()
            for kind in kinds
        ):
            reason = Reason.BY_CHOICE
        else:
            reason = Reason.CENTER_LIMIT
        found.append(Unserved(int(row), tuple(float(value) for value in missing[row]), reason))
    return found


def plan_report(
    evaluation: Evaluation,
    *,
    objective: Objective | None = None,
    stats: Mapping[str, float | int | str] | None = None,
) -> dict[str, Any]:
    """The plan of ``evaluation`` and everything computed about it, as plain values.

    With ``objective`` the report holds its value, and the value of every term
    when the objective can break it down. ``stats`` are the statistics of the
    solver that made the plan (:attr:`pirilib.SolveResult.stats`).
    """
    instance = evaluation.instance
    document: dict[str, Any] = {
        "format": FORMAT,
        "pirilib": __version__,
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "problem": {
            "centers": len(instance.centers),
            "demands": len(instance.demand_ids),
            "first_day": instance.days[0].isoformat(),
            "last_day": instance.days[-1].isoformat(),
            "dimensions": list(instance.dimensions),
            "primary_dimension": instance.dimensions[instance.primary_dimension],
        },
        "summary": summary(evaluation),
        "hard": sorted(evaluation.hard),
    }
    if objective is not None:
        document["objective"] = {"value": float(objective(evaluation))}
        breakdown = getattr(objective, "breakdown", None)
        if breakdown is not None:
            document["objective"]["terms"] = dict(breakdown(evaluation))
    if stats is not None:
        document["stats"] = dict(stats)
    document["dispatches"] = _dispatches(evaluation)
    document["unserved"] = [
        {
            "demand": instance.demand_ids[item.demand],
            "origin": instance.centers[int(instance.demand_origin[item.demand])],
            "destination": instance.centers[int(instance.demand_destination[item.demand])],
            "date": instance.days[int(instance.demand_day[item.demand])].isoformat(),
            "quantity": _by_dimension(instance, item.quantity),
            "reason": item.reason.value,
        }
        for item in unserved(evaluation)
    ]
    return document


def write_report(
    evaluation: Evaluation,
    path: str | os.PathLike[str],
    *,
    objective: Objective | None = None,
    stats: Mapping[str, float | int | str] | None = None,
) -> Path:
    """Write :func:`plan_report` as JSON to ``path`` and return the path."""
    file = Path(path)
    document = plan_report(evaluation, objective=objective, stats=stats)
    file.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return file


def _dispatches(evaluation: Evaluation) -> list[dict[str, Any]]:
    instance, plan = evaluation.instance, evaluation.plan
    unloaded = _unloaded(evaluation)
    primary = instance.primary_dimension
    found = []
    stop = 0
    for position, dispatch in enumerate(plan.dispatches()):
        names = instance.spot_names if dispatch.kind == "spot" else ()
        load = evaluation.load[position]
        onboard = load.copy()
        stops = []
        for center in dispatch.stops:
            onboard = onboard - unloaded[stop]
            stops.append(
                {
                    "center": instance.centers[center],
                    "arrival": _moment(instance, float(evaluation.stop_arrival[stop])),
                    "finish": _moment(instance, float(evaluation.stop_finish[stop])),
                    "unloaded": _by_dimension(instance, unloaded[stop]),
                    "onboard": _by_dimension(instance, np.clip(onboard, 0.0, None)),
                }
            )
            stop += 1
        loads = []
        for row in np.flatnonzero(plan.load_dispatch == position):
            demand = int(plan.load_demand[row])
            finish = float(evaluation.load_finish[row])
            due = float(instance.demand_due[demand])
            loads.append(
                {
                    "demand": instance.demand_ids[demand],
                    "destination": instance.centers[int(instance.demand_destination[demand])],
                    "quantity": _by_dimension(instance, plan.load_quantity[row]),
                    "unloaded": None if np.isnan(finish) else _moment(instance, finish),
                    "due": _moment(instance, due) if np.isfinite(due) else None,
                    "late_hours": int(evaluation.load_late_hours[row]),
                }
            )
        found.append(
            {
                "dispatch": position + 1,
                "date": instance.days[dispatch.day].isoformat(),
                "kind": dispatch.kind,
                "vehicle_type": instance.vehicle_types[int(evaluation.vehicle_type[position])],
                "source": f"fleet.{dispatch.kind}[{dispatch.source}]",
                "offer": (names[dispatch.source] or None) if names else None,
                "origin": instance.centers[dispatch.origin],
                "start": _moment(instance, float(evaluation.start_minute[position])),
                "end": _moment(instance, float(evaluation.end_minute[position])),
                "hours": float(evaluation.hours[position]),
                "route_km": float(evaluation.route_km[position]),
                "cost": {
                    "total": float(evaluation.dispatch_cost[position]),
                    "fixed": float(evaluation.cost_fixed[position]),
                    "km": float(evaluation.cost_km[position]),
                    "hours": float(evaluation.cost_hours[position]),
                    "units": float(evaluation.cost_units[position]),
                },
                "load": _by_dimension(instance, load),
                "capacity": _by_dimension(instance, evaluation.capacity[position]),
                "fill": float(load[primary] / evaluation.capacity[position, primary]),
                "stops": stops,
                "loads": loads,
            }
        )
    return found


def _unloaded(evaluation: Evaluation) -> np.ndarray[Any, np.dtype[np.float64]]:
    """(stops, dimensions): what the loads of each dispatch leave at each of its stops.

    A center visited twice by one dispatch gets its loads at the first visit.
    """
    instance, plan = evaluation.instance, evaluation.plan
    amounts = np.zeros((plan.stop_center.size, len(instance.dimensions)))
    first_visit: dict[tuple[int, int], int] = {}
    stop = 0
    for position in range(len(plan)):
        for center in plan.stops(position):
            first_visit.setdefault((position, int(center)), stop)
            stop += 1
    destinations = instance.demand_destination[plan.load_demand]
    for row, (position, center) in enumerate(zip(plan.load_dispatch, destinations, strict=True)):
        where = first_visit.get((int(position), int(center)))
        if where is not None:
            amounts[where] += plan.load_quantity[row]
    return amounts


def _by_dimension(instance: LineHaulInstance, values: Any) -> dict[str, float]:
    return {name: float(value) for name, value in zip(instance.dimensions, values, strict=True)}


def _moment(instance: LineHaulInstance, minute: float) -> str:
    """``minute`` after 00:00 of the first calendar day, as a local date and time."""
    start = dt.datetime.combine(instance.days[0], dt.time())
    moment = start + dt.timedelta(minutes=round(minute * 60.0, 6) / 60.0)
    return moment.isoformat(timespec="minutes" if moment.second == 0 else "seconds")


def _day(minute: float) -> int:
    return int(minute // _MINUTES_PER_DAY)

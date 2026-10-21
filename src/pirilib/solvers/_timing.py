"""When the routes of a subproblem run, as :func:`pirilib.evaluate` times them.

A route starts loading when its loads are ready (and not before 00:00 of its
day), loads everything at the origin, then drives every leg and unloads at every
stop. Every handling and every leg takes whole minutes, rounded up. A load is
late by the started hours between its due time and the end of the unloading at
its stop. :func:`timeline` times every route of a colony at once, with the
segment sums of :mod:`pirilib.evaluation`; :func:`route_timing` and
:func:`route_lateness` time a single route for the local search.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
from numpy.typing import NDArray

from pirilib.solvers.decompose import MINUTES_PER_DAY, Subproblem

if TYPE_CHECKING:
    from pirilib.solvers._colony import Batch, Colony, Shared

LATE: Final = 1e-9
"""Minutes after the due time below which a load is on time."""


def whole_minutes(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Whole minutes, rounded up (ignoring float noise), as in the evaluator."""
    return np.asarray(np.ceil(np.round(values, 9)), dtype=np.float64)


def late_hours(late: NDArray[np.float64]) -> NDArray[np.float64]:
    """Started hours of lateness, from minutes after the due time."""
    with np.errstate(invalid="ignore"):
        hours = np.where(late > LATE, np.ceil(np.round(late / 60.0, 9)), 0.0)
    return np.asarray(hours, dtype=np.float64)


@dataclass(frozen=True, eq=False, slots=True)
class Timeline:
    """Times and handling events of the routes of a colony.

    Attributes:
        route_row: (routes,) row of the colony
        route_type: (routes,) vehicle type
        start: (routes,) minute loading starts at the origin
        end: (routes,) minute the last unloading ends
        load: (routes,) primary quantity loaded at the origin
        stop_route: (stops,) route of every stop
        stop_node: (stops,) node of every stop
        arrival: (stops,) minute the vehicle reaches the stop
        finish: (stops,) minute unloading there ends
        unload: (stops,) primary quantity unloaded
        service: (routes,) lateness penalty of the route's loads
    """

    route_row: NDArray[np.int64]
    route_type: NDArray[np.int64]
    start: NDArray[np.float64]
    end: NDArray[np.float64]
    load: NDArray[np.float64]
    stop_route: NDArray[np.int64]
    stop_node: NDArray[np.int64]
    arrival: NDArray[np.float64]
    finish: NDArray[np.float64]
    unload: NDArray[np.float64]
    service: NDArray[np.float64]

    @property
    def hours(self) -> NDArray[np.float64]:
        return np.asarray((self.end - self.start) / 60.0, dtype=np.float64)

    @property
    def start_day(self) -> NDArray[np.int64]:
        return np.floor(self.start / MINUTES_PER_DAY).astype(np.int64)

    @property
    def arrival_day(self) -> NDArray[np.int64]:
        return np.floor(self.arrival / MINUTES_PER_DAY).astype(np.int64)


def timeline(batch: "Batch", colony: "Colony", shared: "Shared") -> Timeline:
    """Time every route of ``colony``."""
    primary = shared.primary
    of = colony.route_row // colony.ants
    option = colony.route_option
    counts = np.diff(colony.stop_offsets)
    has_stops = counts > 0
    stop_route = np.repeat(np.arange(option.size, dtype=np.int64), counts)
    stop_of = of[stop_route]
    node = colony.stop_node
    first = colony.stop_offsets[:-1]
    opening = np.zeros(node.size, dtype=np.bool_)
    opening[first[has_stops]] = True
    previous = np.zeros_like(node)
    previous[1:] = node[:-1]
    previous[opening] = 0
    legs = np.where(opening, batch.origin_km[stop_of, node], batch.node_km[stop_of, previous, node])
    speed = batch.speed[of, option][stop_route]
    with np.errstate(invalid="ignore", divide="ignore"):
        travel = np.where(np.isfinite(speed), whole_minutes(legs / speed * 60.0), 0.0)
    per_unit = shared.minutes_per_unit
    unload = colony.stop_amount[:, primary]
    load = colony.route_load[:, primary]
    unloading = whole_minutes(per_unit * unload)
    loading = whole_minutes(per_unit * load)

    ready = np.full(option.size, -np.inf)
    if shared.rules.split_demand:
        carried = colony.stop_amount.max(axis=1, initial=0.0) > 0
        np.maximum.at(ready, stop_route[carried], batch.node_ready[stop_of[carried], node[carried]])
    else:
        item_route = stop_route[colony.item_stop]
        np.maximum.at(
            ready,
            item_route,
            batch.item_ready[of[item_route], node[colony.item_stop], colony.item_slot],
        )
    start = np.maximum(batch.day[of] * MINUTES_PER_DAY, ready)
    steps = np.cumsum(travel + unloading)
    before = np.concatenate([[0.0], steps])[first]
    finish = start[stop_route] + loading[stop_route] + steps - before[stop_route]
    end = start + loading
    end[has_stops] = finish[colony.stop_offsets[1:][has_stops] - 1]

    if shared.rules.split_demand:
        demand = batch.demand[stop_of, node, primary]
        share = np.where(demand > 0, unload / np.where(demand > 0, demand, 1.0), 0.0)
        quantity = batch.item_quantity[stop_of, node, :, primary] * share[:, None]
        hours = late_hours(finish[:, None] - batch.item_due[stop_of, node])
        late = np.sum(quantity * hours, axis=1)
        late_route = stop_route
    else:
        stop = colony.item_stop
        where = (stop_of[stop], node[stop], colony.item_slot)
        quantity = batch.item_quantity[(*where, np.full(stop.size, primary))]
        late = quantity * late_hours(finish[stop] - batch.item_due[where])
        late_route = stop_route[stop]
    service = shared.penalty * np.bincount(late_route, weights=late, minlength=option.size)
    return Timeline(
        route_row=colony.route_row,
        route_type=batch.vehicle_type[of, option],
        start=start,
        end=end,
        load=load,
        stop_route=stop_route,
        stop_node=node,
        arrival=finish - unloading,
        finish=finish,
        unload=unload,
        service=service,
    )


def route_timing(
    problem: Subproblem,
    shared: "Shared",
    option: int,
    stops: Sequence[int],
    unloads: Sequence[float],
    ready: float,
) -> tuple[float, list[float]]:
    """Start of one route and the end of the unloading at each stop, in minutes."""
    per_unit = shared.minutes_per_unit
    speed = float(problem.speed[option])
    start = max(problem.day * MINUTES_PER_DAY, ready)
    clock = start + _minutes(per_unit * sum(unloads))
    finishes = []
    here = -1
    for stop, unload in zip(stops, unloads, strict=True):
        km = float(problem.origin_km[stop] if here < 0 else problem.node_km[here, stop])
        if math.isfinite(speed):
            clock += _minutes(km / speed * 60.0)
        clock += _minutes(per_unit * unload)
        finishes.append(clock)
        here = stop
    return start, finishes


def route_lateness(
    problem: Subproblem,
    shared: "Shared",
    stops: Sequence[int],
    amounts: Sequence[float],
    items: Sequence[Sequence[int]],
    finishes: Sequence[float],
) -> float:
    """Lateness penalty of one route; ``items`` are the item slots of each stop (whole items)."""
    primary = shared.primary
    total = 0.0
    for stop, amount, slots, finish in zip(stops, amounts, items, finishes, strict=True):
        if shared.rules.split_demand:
            demand = float(problem.demand[stop, primary])
            if demand <= 0:
                continue
            quantity = problem.item_quantity[stop, :, primary] * (amount / demand)
            hours = late_hours(finish - problem.item_due[stop])
        else:
            chosen = list(slots)
            quantity = problem.item_quantity[stop, chosen, primary]
            hours = late_hours(finish - problem.item_due[stop, chosen])
        total += float(np.sum(quantity * hours))
    return shared.penalty * total


def _minutes(value: float) -> float:
    return float(math.ceil(round(value, 9)))

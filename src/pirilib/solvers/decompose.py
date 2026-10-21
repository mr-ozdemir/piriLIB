"""Decomposition of a line-haul instance into (day, origin) subproblems.

Vehicles leave one origin on one day, so the dispatches of different (day,
origin) pairs only interact through the daily limits of spot offers. Solvers
work on one :class:`Subproblem` at a time and :class:`PlanBuilder` puts the
pieces back together.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from pirilib.plan import OWNED, SPOT, Plan
from pirilib.spec.instance import UNLIMITED, LineHaulInstance

UNBOUNDED: Final = 1 << 40
"""Availability used for spot offers without a daily limit."""


@dataclass(frozen=True, eq=False, slots=True)
class Subproblem:
    """The demand leaving one origin on one day and the vehicles that may carry it.

    *Nodes* are the destinations with demand, followed by the line ends of owned
    groups that have no demand. *Options* are the owned groups of the origin and
    the spot offers allowed there. *Items* are the demand rows of each node,
    largest first, padded with -1.

    Attributes:
        nodes: (k,) center of each node
        demand: (k, D) quantity per node
        item_row: (k, I) demand row, -1 where there is none
        item_quantity: (k, I, D), zero where there is no item
        origin_km: (k,) from the origin to each node
        node_km: (k, k) between nodes
        kind: (n,) OWNED or SPOT
        source: (n,) owned group or spot offer
        vehicle_type: (n,)
        capacity: (n, D)
        fixed: (n,) charged per dispatch
        per_km: (n,)
        available: (n,) vehicles per day, UNBOUNDED when unlimited
        line_end: (n,) node where an owned route must end, -1 for spot
        stopovers: (n,) may serve several nodes on one route
        allowed: (n, k) the vehicle type may stop at the node
    """

    day: int
    origin: int
    nodes: NDArray[np.int64]
    demand: NDArray[np.float64]
    item_row: NDArray[np.int64]
    item_quantity: NDArray[np.float64]
    origin_km: NDArray[np.float64]
    node_km: NDArray[np.float64]
    kind: NDArray[np.int8]
    source: NDArray[np.int64]
    vehicle_type: NDArray[np.int64]
    capacity: NDArray[np.float64]
    fixed: NDArray[np.float64]
    per_km: NDArray[np.float64]
    available: NDArray[np.int64]
    line_end: NDArray[np.int64]
    stopovers: NDArray[np.bool_]
    allowed: NDArray[np.bool_]

    @property
    def size(self) -> int:
        return int(self.nodes.size)

    @property
    def options(self) -> int:
        return int(self.kind.size)


def subproblems(instance: LineHaulInstance) -> list[Subproblem]:
    """The subproblems with demand, by day, and within a day by falling demand."""
    primary = instance.primary_dimension
    result: list[Subproblem] = []
    for day in range(len(instance.days)):
        on_day = np.flatnonzero(instance.demand_day == day)
        origins = np.unique(instance.demand_origin[on_day])
        totals = [
            instance.demand_quantity[on_day[instance.demand_origin[on_day] == o], primary].sum()
            for o in origins
        ]
        for origin in origins[np.argsort(totals, kind="stable")[::-1]]:
            rows = on_day[instance.demand_origin[on_day] == origin]
            result.append(_subproblem(instance, day, int(origin), rows))
    return result


def _subproblem(
    instance: LineHaulInstance, day: int, origin: int, rows: NDArray[np.intp]
) -> Subproblem:
    rules = instance.rules
    primary = instance.primary_dimension
    destinations = np.unique(instance.demand_destination[rows])
    groups = np.flatnonzero(instance.owned_origin == origin)
    line_ends = np.setdiff1d(np.unique(instance.owned_destination[groups]), destinations)
    nodes = np.concatenate([destinations, line_ends]).astype(np.int64)
    index = {int(center): position for position, center in enumerate(nodes)}

    dimensions = instance.demand_quantity.shape[1]
    node_of_row = np.array([index[int(c)] for c in instance.demand_destination[rows]], np.int64)
    demand = np.zeros((nodes.size, dimensions))
    np.add.at(demand, node_of_row, instance.demand_quantity[rows])
    per_node = [rows[node_of_row == node] for node in range(nodes.size)]
    width = max((len(items) for items in per_node), default=0)
    item_row = np.full((nodes.size, width), -1, dtype=np.int64)
    item_quantity = np.zeros((nodes.size, width, dimensions))
    for node, items in enumerate(per_node):
        items = items[np.argsort(-instance.demand_quantity[items, primary], kind="stable")]
        item_row[node, : items.size] = items
        item_quantity[node, : items.size] = instance.demand_quantity[items]

    spot = np.flatnonzero(~instance.banned[origin, instance.spot_type])
    kind = np.concatenate([np.full(groups.size, OWNED), np.full(spot.size, SPOT)]).astype(np.int8)
    source = np.concatenate([groups, spot]).astype(np.int64)
    vehicle_type = np.concatenate([instance.owned_type[groups], instance.spot_type[spot]])
    owned_fixed = np.where(instance.owned_mandatory[groups], 0.0, instance.owned_fixed[groups])
    spot_available = instance.spot_available[spot]
    km = instance.distance_km
    return Subproblem(
        day=day,
        origin=origin,
        nodes=nodes,
        demand=demand,
        item_row=item_row,
        item_quantity=item_quantity,
        origin_km=km[origin, nodes],
        node_km=km[np.ix_(nodes, nodes)],
        kind=kind,
        source=source,
        vehicle_type=vehicle_type,
        capacity=instance.capacity[vehicle_type],
        fixed=np.concatenate([owned_fixed, instance.spot_fixed[spot]]),
        per_km=np.concatenate([instance.owned_per_km[groups], instance.spot_per_km[spot]]),
        available=np.concatenate(
            [
                instance.owned_count[groups],
                np.where(spot_available == UNLIMITED, UNBOUNDED, spot_available),
            ]
        ).astype(np.int64),
        line_end=np.concatenate(
            [[index[int(c)] for c in instance.owned_destination[groups]], np.full(spot.size, -1)]
        ).astype(np.int64),
        stopovers=np.concatenate(
            [instance.owned_stopovers[groups], np.ones(spot.size, dtype=np.bool_)]
        )
        & rules.stopover.enabled,
        allowed=~instance.banned[np.ix_(nodes, vehicle_type)].T,
    )


@dataclass(frozen=True, eq=False, slots=True)
class Routes:
    """Routes of one subproblem solution, in subproblem terms.

    Per route: the option and its stops (``stop_start`` as in :class:`~pirilib.Plan`).
    Per stop: the node and the quantity unloaded there. Per item load (only when
    demand may not be split): the stop and the item slot of the node.
    """

    option: NDArray[np.int64]
    stop_start: NDArray[np.int64]
    stop_node: NDArray[np.int64]
    stop_amount: NDArray[np.float64]
    item_stop: NDArray[np.int64]
    item_slot: NDArray[np.int64]


class PlanBuilder:
    """Collects the routes of subproblems and turns them into one :class:`~pirilib.Plan`."""

    def __init__(self, instance: LineHaulInstance) -> None:
        self.instance = instance
        self._parts: list[tuple[Subproblem, Routes]] = []

    def add(self, problem: Subproblem, routes: Routes) -> None:
        self._parts.append((problem, routes))

    def __iter__(self) -> Iterator[tuple[Subproblem, Routes]]:
        return iter(self._parts)

    def build(self) -> Plan:
        dimensions = self.instance.demand_quantity.shape[1]
        days, kinds, sources, origins, counts, centers = [], [], [], [], [], []
        load_dispatch, load_demand, load_quantity = [], [], []
        offset = 0
        for problem, routes in self._parts:
            size = routes.option.size
            days.append(np.full(size, problem.day, dtype=np.int64))
            kinds.append(problem.kind[routes.option])
            sources.append(problem.source[routes.option])
            origins.append(np.full(size, problem.origin, dtype=np.int64))
            counts.append(np.diff(routes.stop_start))
            centers.append(problem.nodes[routes.stop_node])
            stop_route = np.repeat(np.arange(size), np.diff(routes.stop_start))
            dispatch, demand, quantity = _loads(self.instance, problem, routes, stop_route)
            load_dispatch.append(dispatch + offset)
            load_demand.append(demand)
            load_quantity.append(quantity)
            offset += size
        stops = (
            np.concatenate([np.zeros(1, np.int64), *counts]) if counts else np.zeros(1, np.int64)
        )

        def join(arrays: Sequence[NDArray[Any]], dtype: type[np.generic]) -> NDArray[Any]:
            return np.concatenate(arrays).astype(dtype) if arrays else np.zeros(0, dtype=dtype)

        return Plan(
            day=join(days, np.int64),
            kind=join(kinds, np.int8),
            source=join(sources, np.int64),
            origin=join(origins, np.int64),
            stop_start=np.cumsum(stops),
            stop_center=join(centers, np.int64),
            load_dispatch=join(load_dispatch, np.int64),
            load_demand=join(load_demand, np.int64),
            load_quantity=(
                np.concatenate(load_quantity) if load_quantity else np.zeros((0, dimensions))
            ),
        )


def _loads(
    instance: LineHaulInstance, problem: Subproblem, routes: Routes, stop_route: NDArray[np.int64]
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Loads per demand row: whole items, or every stop's share of each row of its node."""
    if not instance.rules.split_demand:
        rows = problem.item_row[routes.stop_node[routes.item_stop], routes.item_slot]
        return stop_route[routes.item_stop], rows, instance.demand_quantity[rows]
    present = problem.item_row >= 0
    per_node = present.sum(axis=1)
    count = per_node[routes.stop_node]
    stop = np.repeat(np.arange(routes.stop_node.size), count)
    slot = np.arange(stop.size) - np.repeat(np.cumsum(count) - count, count)
    rows = problem.item_row[routes.stop_node[stop], slot]
    node_demand = problem.demand[routes.stop_node]
    safe = np.where(node_demand > 0, node_demand, 1.0)
    share = np.where(node_demand > 0, routes.stop_amount / safe, 0.0)
    quantity = instance.demand_quantity[rows] * share[stop]
    carried = quantity.max(axis=1, initial=0.0) > 0
    return stop_route[stop][carried], rows[carried], quantity[carried]

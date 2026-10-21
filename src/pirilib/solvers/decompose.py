"""Decomposition of a line-haul instance into (day, origin) subproblems.

Vehicles leave one origin on one day, so the dispatches of different (day,
origin) pairs only interact through the daily limits of spot offers and, on
scheduled instances, through the handling capacity and vehicle limits of the
centers they touch. Solvers work on one :class:`Subproblem` at a time and
:class:`PlanBuilder` puts the pieces back together. On scheduled instances a
:class:`Coordinator` hands the subproblems out in planning order and carries
demand left over on one day into the next.
"""

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from pirilib.evaluation import TOLERANCE
from pirilib.plan import OWNED, SPOT, Plan
from pirilib.spec.instance import UNLIMITED, LineHaulInstance

UNBOUNDED: Final = 1 << 40
"""Availability used for spot offers without a daily limit."""
MINUTES_PER_DAY: Final = 1440.0


@dataclass(frozen=True, eq=False, slots=True)
class Subproblem:
    """The demand leaving one origin on one day and the vehicles that may carry it.

    *Nodes* are the destinations with demand, followed by the line ends of owned
    groups that have no demand. *Options* are the owned groups of the origin and
    the spot offers allowed there. *Items* are the demand rows of each node,
    largest first, padded with -1. An item may be what is left of a demand that
    an earlier day did not carry; it keeps the demand's ready and due times.

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
        per_hour: (n,) charged per hour of use
        speed: (n,) km per hour, NaN where travel takes no time
        item_ready: (k, I) minutes, -inf where there is no item
        item_due: (k, I) minutes, inf where there is no due time
        item_final: (k, I) may not be left for the next day (True where there is no item)
        node_ready: (k,) latest ready time of the node's items, -inf without items
        defer_cost: (k, I) estimated cost per primary unit of leaving the item for
            the next day: the lateness if it leaves at 00:00 on a direct trip of its
            own, plus the cost of that trip, with the cheapest spot offer; 0 for
            final items
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
    per_hour: NDArray[np.float64]
    speed: NDArray[np.float64]
    item_ready: NDArray[np.float64]
    item_due: NDArray[np.float64]
    item_final: NDArray[np.bool_]
    node_ready: NDArray[np.float64]
    defer_cost: NDArray[np.float64]

    @property
    def size(self) -> int:
        return int(self.nodes.size)

    @property
    def options(self) -> int:
        return int(self.kind.size)

    @property
    def deferrable(self) -> bool:
        """True if some item may be left for the next day."""
        return bool(np.any((self.item_row >= 0) & ~self.item_final))

    def leftover(
        self, routes: "Routes", split: bool
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """What ``routes`` leave uncarried, per node and dimension, and per item."""
        present = self.item_row >= 0
        if split:
            delivered = np.zeros_like(self.demand)
            np.add.at(delivered, routes.stop_node, routes.stop_amount)
            remaining = np.clip(self.demand - delivered, 0.0, None)
            remaining[remaining <= TOLERANCE * self.demand] = 0.0
            return remaining, present & (remaining.max(axis=1, initial=0.0) > 0)[:, None]
        left = present.copy()
        left[routes.stop_node[routes.item_stop], routes.item_slot] = False
        remaining = np.einsum("ki,kid->kd", left.astype(np.float64), self.item_quantity)
        return np.asarray(remaining, dtype=np.float64), left

    def left_items(
        self, remaining: NDArray[np.float64], left: NDArray[np.bool_], split: bool
    ) -> NDArray[np.float64]:
        """(k, I, D): the uncarried quantity of every item, from :meth:`leftover`.

        Split demand is carried in proportion, so every item of a node keeps the
        same share of its quantity.
        """
        if not split:
            return np.asarray(self.item_quantity * left[:, :, None], dtype=np.float64)
        safe = np.where(self.demand > 0, self.demand, 1.0)
        share = np.where(self.demand > 0, remaining / safe, 0.0)
        return np.asarray(self.item_quantity * share[:, None, :], dtype=np.float64)


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


class Coordinator:
    """Hands out the subproblems of a scheduled instance in planning order.

    Days come in order. Within a day, subproblems come by least slack (the
    earliest due time of their items, minus 00:00 of the day), then by most
    primary demand, then by origin. The subproblems of a day are built when the
    first of them is asked for, so what :meth:`defer` hands back during the day
    before joins them.

    Attributes:
        waiting: Per (day, origin), demand rows carried over and their quantities
    """

    def __init__(self, instance: LineHaulInstance) -> None:
        self.instance = instance
        self.waiting: dict[tuple[int, int], list[tuple[int, NDArray[np.float64]]]] = {}

    def __iter__(self) -> Iterator[Subproblem]:
        for day in range(len(self.instance.days)):
            yield from self.day(day)

    def day(self, day: int) -> list[Subproblem]:
        """The subproblems of ``day``, with the demand carried over to it, in order."""
        instance = self.instance
        dimensions = instance.demand_quantity.shape[1]
        on_day = np.flatnonzero(instance.demand_day == day)
        carried_origins = {origin for when, origin in self.waiting if when == day}
        origins = sorted(set(instance.demand_origin[on_day].tolist()) | carried_origins)
        problems = []
        for origin in origins:
            fresh = on_day[instance.demand_origin[on_day] == origin]
            carried = self.waiting.pop((day, origin), [])
            rows = np.concatenate(
                [fresh, np.array([row for row, _ in carried], dtype=np.intp)]
            ).astype(np.intp)
            quantities = np.concatenate(
                [
                    instance.demand_quantity[fresh],
                    np.array([amount for _, amount in carried]).reshape(len(carried), dimensions),
                ]
            )
            problems.append(_subproblem(instance, day, origin, rows, quantities))
        return sorted(problems, key=self._priority)

    def _priority(self, problem: Subproblem) -> tuple[float, float, int]:
        present = problem.item_row >= 0
        due = np.min(np.where(present, problem.item_due, np.inf), initial=np.inf)
        slack = (due - problem.day * MINUTES_PER_DAY) / 60.0
        total = float(problem.demand[:, self.instance.primary_dimension].sum())
        return slack, -total, problem.origin

    def defer(
        self, problem: Subproblem, remaining: NDArray[np.float64], left: NDArray[np.bool_]
    ) -> None:
        """Carry the uncarried items of ``problem`` that are not final into the next day."""
        if problem.day + 1 >= len(self.instance.days):
            return
        items = problem.left_items(remaining, left, self.instance.rules.split_demand)
        keep = (problem.item_row >= 0) & ~problem.item_final & (items.max(axis=2) > 0)
        target = self.waiting.setdefault((problem.day + 1, problem.origin), [])
        for node, slot in zip(*np.nonzero(keep), strict=True):
            target.append((int(problem.item_row[node, slot]), items[node, slot].copy()))
        if not target:
            del self.waiting[(problem.day + 1, problem.origin)]


def _subproblem(
    instance: LineHaulInstance,
    day: int,
    origin: int,
    rows: NDArray[np.intp],
    quantities: NDArray[np.float64] | None = None,
) -> Subproblem:
    """The subproblem of ``rows``; ``quantities`` (rows, D) replaces their demand quantities."""
    rules = instance.rules
    primary = instance.primary_dimension
    amounts = instance.demand_quantity[rows] if quantities is None else quantities
    destinations = np.unique(instance.demand_destination[rows])
    groups = np.flatnonzero(instance.owned_origin == origin)
    line_ends = np.setdiff1d(np.unique(instance.owned_destination[groups]), destinations)
    nodes = np.concatenate([destinations, line_ends]).astype(np.int64)
    index = {int(center): position for position, center in enumerate(nodes)}

    dimensions = instance.demand_quantity.shape[1]
    node_of_row = np.array([index[int(c)] for c in instance.demand_destination[rows]], np.int64)
    demand = np.zeros((nodes.size, dimensions))
    np.add.at(demand, node_of_row, amounts)
    per_node = [np.flatnonzero(node_of_row == node) for node in range(nodes.size)]
    width = max((len(items) for items in per_node), default=0)
    item_row = np.full((nodes.size, width), -1, dtype=np.int64)
    item_quantity = np.zeros((nodes.size, width, dimensions))
    for node, items in enumerate(per_node):
        items = items[np.argsort(-amounts[items, primary], kind="stable")]
        item_row[node, : items.size] = rows[items]
        item_quantity[node, : items.size] = amounts[items]

    spot = np.flatnonzero(~instance.banned[origin, instance.spot_type])
    kind = np.concatenate([np.full(groups.size, OWNED), np.full(spot.size, SPOT)]).astype(np.int8)
    source = np.concatenate([groups, spot]).astype(np.int64)
    vehicle_type = np.concatenate([instance.owned_type[groups], instance.spot_type[spot]])
    owned_fixed = np.where(instance.owned_mandatory[groups], 0.0, instance.owned_fixed[groups])
    spot_available = instance.spot_available[spot]
    km = instance.distance_km
    origin_km = km[origin, nodes]
    allowed = ~instance.banned[np.ix_(nodes, vehicle_type)].T
    present = item_row >= 0
    safe_row = np.where(present, item_row, 0)
    item_ready = np.where(present, instance.demand_ready[safe_row], -np.inf)
    item_due = np.where(present, instance.demand_due[safe_row], np.inf)
    waited = day - instance.demand_day[safe_row]
    last = day >= len(instance.days) - 1
    item_final = ~present | last | (waited >= rules.service.max_deferral_days)
    defer_cost = np.zeros(item_row.shape)
    deferrable = present & ~item_final
    if deferrable.any():
        node_of_item = np.nonzero(deferrable)[0]
        defer_cost[deferrable] = _defer_costs(
            instance,
            day,
            item_quantity[deferrable, primary],
            item_ready[deferrable],
            item_due[deferrable],
            origin_km[node_of_item],
            allowed[groups.size :][:, node_of_item],
            spot,
        )
    return Subproblem(
        day=day,
        origin=origin,
        nodes=nodes,
        demand=demand,
        item_row=item_row,
        item_quantity=item_quantity,
        origin_km=origin_km,
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
        allowed=allowed,
        per_hour=np.concatenate([instance.owned_per_hour[groups], instance.spot_per_hour[spot]]),
        speed=instance.speed_kmh[vehicle_type],
        item_ready=item_ready,
        item_due=item_due,
        item_final=item_final,
        node_ready=np.max(item_ready, axis=1, initial=-np.inf),
        defer_cost=defer_cost,
    )


def _defer_costs(
    instance: LineHaulInstance,
    day: int,
    quantity: NDArray[np.float64],
    ready: NDArray[np.float64],
    due: NDArray[np.float64],
    km: NDArray[np.float64],
    allowed: NDArray[np.bool_],
    spot: NDArray[np.intp],
) -> NDArray[np.float64]:
    """Per item and primary unit, see :attr:`Subproblem.defer_cost`.

    ``allowed`` is (spot offers, items): the offer may stop at the item's node.
    """
    rules = instance.rules
    primary = instance.primary_dimension
    handling = np.ceil(np.round(rules.handling.minutes_per_unit * quantity, 9))
    start = np.maximum((day + 1) * MINUTES_PER_DAY, ready)
    best = np.full(quantity.size, np.inf)
    for position, offer in enumerate(spot):
        kind = instance.spot_type[offer]
        speed = float(instance.speed_kmh[kind])
        travel = np.ceil(np.round(km / speed * 60.0, 9)) if math.isfinite(speed) else 0.0 * km
        late = start + 2 * handling + travel - due
        hours = np.where(late > 1e-9, np.ceil(np.round(late / 60.0, 9)), 0.0)
        trip = (
            instance.spot_fixed[offer]
            + instance.spot_per_km[offer] * km
            + instance.spot_per_hour[offer] * (2 * handling + travel) / 60.0
        )
        trips = np.ceil(np.round(quantity / float(instance.capacity[kind, primary]), 9))
        value = rules.service.penalty * hours + trip * trips / quantity
        best = np.where(allowed[position], np.minimum(best, value), best)
    return np.asarray(np.where(np.isfinite(best), best, 0.0), dtype=np.float64)


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
    """Loads per demand row: whole items, or every stop's share of each item of its node."""
    if not instance.rules.split_demand:
        node, slot = routes.stop_node[routes.item_stop], routes.item_slot
        rows = problem.item_row[node, slot]
        return stop_route[routes.item_stop], rows, problem.item_quantity[node, slot]
    present = problem.item_row >= 0
    per_node = present.sum(axis=1)
    count = per_node[routes.stop_node]
    stop = np.repeat(np.arange(routes.stop_node.size), count)
    slot = np.arange(stop.size) - np.repeat(np.cumsum(count) - count, count)
    rows = problem.item_row[routes.stop_node[stop], slot]
    node_demand = problem.demand[routes.stop_node]
    safe = np.where(node_demand > 0, node_demand, 1.0)
    share = np.where(node_demand > 0, routes.stop_amount / safe, 0.0)
    quantity = problem.item_quantity[routes.stop_node[stop], slot] * share[stop]
    carried = quantity.max(axis=1, initial=0.0) > 0
    return stop_route[stop][carried], rows[carried], quantity[carried]

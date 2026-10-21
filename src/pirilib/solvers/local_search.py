"""Local search on the routes of one subproblem, with a linear objective.

Moves are applied until none of them improves the objective:

- *change vehicle*: carry a route with another vehicle option;
- *merge*: serve the stops of two routes with one vehicle;
- *relocate*: move the load of one stop onto another route;
- *rebalance*: shift load of a node between two routes that both serve it, to
  lift a route to its minimum fill (when demand may be split);
- *serve*: put demand that nobody carries onto a route with room (when demand
  may be split);
- *drop*: remove a route whose cost is higher than leaving its load uncarried.

Every route keeps the rules (capacity, bans, owned lines, availability,
stopovers, stops, stretch, forward-only); loads are never carried twice.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from pirilib.evaluation import TOLERANCE
from pirilib.plan import OWNED
from pirilib.solvers._colony import Shared
from pirilib.solvers._routing import fits, route_km, route_ok, route_value, unserved_value
from pirilib.solvers.decompose import Routes, Subproblem

_EPSILON = 1e-9


@dataclass(slots=True)
class _Stop:
    """A stop of a route being improved: its node and what is unloaded there.

    Attributes:
        items: Item slots, when demand may not be split
    """

    node: int
    amount: NDArray[np.float64]
    items: list[int] = field(default_factory=list)


@dataclass(slots=True)
class _Route:
    """A route being improved: its vehicle option and its stops in order."""

    option: int
    stops: list[_Stop]

    @property
    def nodes(self) -> list[int]:
        return [stop.node for stop in self.stops]

    @property
    def load(self) -> NDArray[np.float64]:
        return np.sum([stop.amount for stop in self.stops], axis=0)


class _Search:
    """Applies the local search moves to the routes of one subproblem."""

    def __init__(
        self,
        problem: Subproblem,
        coefficients: Mapping[str, float],
        shared: Shared,
        available: NDArray[np.int64],
        remaining: NDArray[np.float64],
    ) -> None:
        self.p, self.coefficients, self.shared = problem, coefficients, shared
        self.available = available
        self.remaining = remaining.copy()
        self.split = shared.rules.split_demand
        self.order = problem.origin_km

    def value(self, route: _Route) -> float:
        km = route_km(self.p, route.nodes)
        return route_value(self.p, self.coefficients, self.shared, route.option, km, route.load)

    def uncarried(self, remaining: NDArray[np.float64]) -> float:
        return unserved_value(self.p, self.coefficients, self.shared, remaining)

    def valid(self, route: _Route) -> bool:
        return route_ok(self.p, self.shared, route.option, route.nodes) and fits(
            self.p, route.option, route.load
        )

    def shortfall(self, route: _Route) -> float:
        """Primary quantity missing to the minimum fill (0 when reached)."""
        primary = self.shared.primary
        capacity = float(self.p.capacity[route.option, primary])
        share = float(self.shared.min_fill[self.p.kind[route.option]])
        return max(share * capacity - float(route.load[primary]), 0.0)

    def arranged(self, option: int, stops: list[_Stop]) -> list[list[_Stop]]:
        """Ways to visit ``stops`` with ``option``: empty stops dropped, owned line end last."""
        merged: dict[int, _Stop] = {}
        for stop in stops:
            if stop.node in merged:
                kept = merged[stop.node]
                merged[stop.node] = _Stop(
                    stop.node, kept.amount + stop.amount, kept.items + stop.items
                )
            else:
                merged[stop.node] = _Stop(stop.node, stop.amount.copy(), list(stop.items))
        owned = self.p.kind[option] == OWNED
        line = int(self.p.line_end[option])
        if owned and line not in merged:
            merged[line] = _Stop(line, np.zeros_like(stops[0].amount))
        kept_stops = [
            stop
            for stop in merged.values()
            if (owned and stop.node == line) or stop.amount.max() > 0
        ]
        if not kept_stops:
            return []
        if self.shared.rules.stopover.forward_only:
            return [sorted(kept_stops, key=lambda stop: float(self.order[stop.node]))]
        if owned:
            body = [stop for stop in kept_stops if stop.node != line]
            return [[*body, merged[line]]]
        return [kept_stops]

    def _usage(self, routes: list[_Route]) -> NDArray[np.int64]:
        return np.bincount([route.option for route in routes], minlength=self.p.options)

    def _better(self, new: float, old: float) -> bool:
        return new < old - _EPSILON * max(1.0, abs(old))

    def run(self, routes: list[_Route]) -> tuple[list[_Route], float]:
        values = [self.value(route) for route in routes]
        start = sum(values) + self.uncarried(self.remaining)
        moves = [
            self._change_vehicles,
            self._merge,
            self._relocate,
            self._rebalance,
            self._serve,
            self._drop,
        ]
        while any(move(routes, values) for move in moves):
            pass
        return routes, sum(values) + self.uncarried(self.remaining) - start

    def _change_vehicles(self, routes: list[_Route], values: list[float]) -> bool:
        usage = self._usage(routes)
        for index, route in enumerate(routes):
            for option in range(self.p.options):
                if option == route.option or usage[option] >= self.available[option]:
                    continue
                for stops in self.arranged(option, route.stops):
                    candidate = _Route(option, stops)
                    if not self.valid(candidate):
                        continue
                    value = self.value(candidate)
                    if self._better(value, values[index]):
                        routes[index], values[index] = candidate, value
                        return True
        return False

    def _merge(self, routes: list[_Route], values: list[float]) -> bool:
        usage = self._usage(routes)
        for first in range(len(routes)):
            for second in range(first + 1, len(routes)):
                old = values[first] + values[second]
                joined = routes[first].stops + routes[second].stops
                for option in range(self.p.options):
                    freed = (routes[first].option == option) + (routes[second].option == option)
                    if usage[option] - freed + 1 > self.available[option]:
                        continue
                    for stops in self.arranged(option, joined):
                        candidate = _Route(option, stops)
                        if not self.valid(candidate):
                            continue
                        value = self.value(candidate)
                        if self._better(value, old):
                            routes[first], values[first] = candidate, value
                            del routes[second], values[second]
                            return True
        return False

    def _relocate(self, routes: list[_Route], values: list[float]) -> bool:
        for source in range(len(routes)):
            for position, stop in enumerate(routes[source].stops):
                if stop.amount.max() <= 0:
                    continue
                rest = [other for at, other in enumerate(routes[source].stops) if at != position]
                left = self.arranged(routes[source].option, rest) if rest else []
                shrunk = _Route(routes[source].option, left[0]) if left else None
                if shrunk is not None and not any(s.amount.max() > 0 for s in shrunk.stops):
                    shrunk = None
                if shrunk is not None and not self.valid(shrunk):
                    continue
                shrunk_value = 0.0 if shrunk is None else self.value(shrunk)
                for target in range(len(routes)):
                    if target == source:
                        continue
                    receiver = routes[target]
                    for stops in self.arranged(receiver.option, [*receiver.stops, stop]):
                        candidate = _Route(receiver.option, stops)
                        if not self.valid(candidate):
                            continue
                        value = self.value(candidate)
                        old = values[source] + values[target]
                        if self._better(shrunk_value + value, old):
                            routes[target], values[target] = candidate, value
                            if shrunk is None:
                                del routes[source], values[source]
                            else:
                                routes[source], values[source] = shrunk, shrunk_value
                            return True
        return False

    def _rebalance(self, routes: list[_Route], values: list[float]) -> bool:
        if not self.split:
            return False
        primary = self.shared.primary
        for taker_index, taker in enumerate(routes):
            need = self.shortfall(taker)
            if need <= 0:
                continue
            capacity = self.p.capacity[taker.option]
            room = capacity - taker.load
            for stop in taker.stops:
                for giver_index, giver in enumerate(routes):
                    if giver_index == taker_index:
                        continue
                    given = next((s for s in giver.stops if s.node == stop.node), None)
                    if given is None or given.amount[primary] <= 0:
                        continue
                    giver_capacity = float(self.p.capacity[giver.option, primary])
                    giver_share = float(self.shared.min_fill[self.p.kind[giver.option]])
                    spare = float(giver.load[primary]) - giver_share * giver_capacity
                    moved = min(need, spare, float(given.amount[primary]))
                    if moved <= TOLERANCE * float(capacity[primary]):
                        continue
                    part = given.amount * (moved / float(given.amount[primary]))
                    scale = np.min(np.where(part > 0, room / np.where(part > 0, part, 1.0), np.inf))
                    part = part * min(1.0, float(scale))
                    if part[primary] <= TOLERANCE * float(capacity[primary]):
                        continue
                    old = values[taker_index] + values[giver_index]
                    given.amount = given.amount - part
                    stop.amount = stop.amount + part
                    new_taker, new_giver = self.value(taker), self.value(giver)
                    if self._better(new_taker + new_giver, old):
                        values[taker_index], values[giver_index] = new_taker, new_giver
                        return True
                    given.amount = given.amount + part
                    stop.amount = stop.amount - part
        return False

    def _serve(self, routes: list[_Route], values: list[float]) -> bool:
        if not self.split:
            return False
        for node in np.flatnonzero(self.remaining.max(axis=1) > 0):
            left = self.remaining[node]
            for index, route in enumerate(routes):
                if not self.p.allowed[route.option, node]:
                    continue
                capacity = self.p.capacity[route.option]
                room = capacity - route.load
                scale = np.min(np.where(left > 0, room / np.where(left > 0, left, 1.0), np.inf))
                part = left * min(1.0, max(float(scale), 0.0))
                if part.max() <= TOLERANCE * float(capacity.max()):
                    continue
                for stops in self.arranged(route.option, [*route.stops, _Stop(int(node), part)]):
                    candidate = _Route(route.option, stops)
                    if not self.valid(candidate):
                        continue
                    after = self.remaining.copy()
                    after[node] = np.where(left - part > TOLERANCE * left, left - part, 0.0)
                    value = self.value(candidate)
                    old = values[index] + self.uncarried(self.remaining)
                    if self._better(value + self.uncarried(after), old):
                        routes[index], values[index] = candidate, value
                        self.remaining = after
                        return True
        return False

    def _drop(self, routes: list[_Route], values: list[float]) -> bool:
        before = self.uncarried(self.remaining)
        for index, route in enumerate(routes):
            after = self.remaining.copy()
            for stop in route.stops:
                after[stop.node] += stop.amount
            if self._better(self.uncarried(after), values[index] + before):
                del routes[index], values[index]
                self.remaining = after
                return True
        return False


def improve(
    problem: Subproblem,
    routes: Routes,
    coefficients: Mapping[str, float],
    shared: Shared,
    available: NDArray[np.int64],
    remaining: NDArray[np.float64],
) -> tuple[Routes, float]:
    """Improve ``routes``; return them with the change of the objective (never positive).

    ``remaining`` is the demand per node that the routes leave uncarried.
    """
    current = _unpack(routes)
    search = _Search(problem, coefficients, shared, available, remaining)
    better, change = search.run(current)
    return _pack(better, problem.demand.shape[1]), change


def _unpack(routes: Routes) -> list[_Route]:
    items: dict[int, list[int]] = {}
    for stop, slot in zip(routes.item_stop.tolist(), routes.item_slot.tolist(), strict=True):
        items.setdefault(stop, []).append(slot)
    result = []
    for index, option in enumerate(routes.option.tolist()):
        start, end = routes.stop_start[index], routes.stop_start[index + 1]
        stops = [
            _Stop(int(routes.stop_node[s]), routes.stop_amount[s].copy(), items.get(int(s), []))
            for s in range(start, end)
        ]
        result.append(_Route(int(option), stops))
    return result


def _pack(routes: list[_Route], dimensions: int) -> Routes:
    nodes: list[int] = []
    amounts: list[NDArray[np.float64]] = []
    item_stop: list[int] = []
    item_slot: list[int] = []
    counts: list[int] = []
    for route in routes:
        counts.append(len(route.stops))
        for stop in route.stops:
            item_stop += [len(nodes)] * len(stop.items)
            item_slot += stop.items
            nodes.append(stop.node)
            amounts.append(stop.amount)
    return Routes(
        option=np.array([route.option for route in routes], dtype=np.int64),
        stop_start=np.concatenate([[0], np.cumsum(counts, dtype=np.int64)]).astype(np.int64),
        stop_node=np.array(nodes, dtype=np.int64),
        stop_amount=np.array(amounts, dtype=np.float64).reshape(len(nodes), dimensions),
        item_stop=np.array(item_stop, dtype=np.int64),
        item_slot=np.array(item_slot, dtype=np.int64),
    )

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
stopovers, stops, stretch, forward-only); loads are never carried twice. On
scheduled instances routes are timed, so hourly costs and lateness count, and a
move is only taken if the subproblem's :class:`~pirilib.solvers.ledger.Room`
still has the vehicle slots and handling capacity it needs on the day.

Change vehicle, merge and relocate remember the moves they tried that were no
better, by the stamps of the routes involved, and skip them while those routes
stay. Cheap checks of capacity, bans and stopovers on cached loads and nodes
skip most hopeless moves before a route is built. Both only save time: the
moves are tried in the same order and the first one taken is the same.
"""

import itertools
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from pirilib.evaluation import TOLERANCE
from pirilib.plan import OWNED
from pirilib.solvers._colony import Shared
from pirilib.solvers._routing import fits, route_km, route_ok, route_value, unserved_value
from pirilib.solvers._timing import route_lateness, route_timing
from pirilib.solvers.decompose import Routes, Subproblem
from pirilib.solvers.ledger import MIN_ROOM, Room

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
    """A route being improved: its vehicle option and its stops in order.

    The cached fields are set once, at construction. A route whose stops change
    is replaced by a new one with a new stamp, so what was remembered about the
    old stamp stays true.

    Attributes:
        stamp: Number of the route, unique within its search
        nodes_set: Nodes the route unloads something at
        load_tuple: :attr:`load` at construction, as floats
    """

    option: int
    stops: list[_Stop]
    stamp: int
    nodes_set: frozenset[int] = field(init=False)
    load_tuple: tuple[float, ...] = field(init=False)

    def __post_init__(self) -> None:
        self.nodes_set = frozenset(stop.node for stop in self.stops if stop.amount.max() > 0)
        self.load_tuple = tuple(np.atleast_1d(self.load).tolist())

    @property
    def nodes(self) -> list[int]:
        return [stop.node for stop in self.stops]

    @property
    def load(self) -> NDArray[np.float64]:
        return np.sum([stop.amount for stop in self.stops], axis=0)


class _Search:
    """Applies the local search moves to the routes of one subproblem.

    A move is remembered as rejected only when it failed for reasons that
    depend on its routes alone, never for vehicle availability or the room of
    the day. The capacity in :attr:`cap` has twice the tolerance of
    :func:`~pirilib.solvers._routing.fits`, so the cheap checks are looser
    than the rules and never skip a move the rules allow.

    Attributes:
        item_left: (k, I) items nobody carries, when demand may not be split
        stamps: Stamps for the routes of this search
        cap: Per option, the capacity per dimension with twice the tolerance
        allowed_nodes: Per option, the nodes it may stop at
        stopover_ok: Per option, whether it may serve several nodes
    """

    def __init__(
        self,
        problem: Subproblem,
        coefficients: Mapping[str, float],
        shared: Shared,
        available: NDArray[np.int64],
        remaining: NDArray[np.float64],
        room: Room | None = None,
        item_left: NDArray[np.bool_] | None = None,
    ) -> None:
        self.p, self.coefficients, self.shared = problem, coefficients, shared
        self.available = available
        self.remaining = remaining.copy()
        self.split = shared.rules.split_demand
        self.order = problem.origin_km
        self.room = room
        self.timed = shared.scheduled
        self.item_left = None if item_left is None else item_left.copy()
        self.stamps = itertools.count()
        self.cap: list[tuple[float, ...]] = [
            tuple(row) for row in (problem.capacity * (1 + 2 * TOLERANCE)).tolist()
        ]
        self.allowed_nodes = [frozenset(np.flatnonzero(row).tolist()) for row in problem.allowed]
        self.stopover_ok: list[bool] = problem.stopovers.tolist()
        self._merge_rejected: set[tuple[int, int, int]] = set()
        self._relocate_rejected: set[tuple[int, int, int]] = set()
        self._change_rejected: set[tuple[int, int]] = set()
        self._shrunk: dict[tuple[int, int], tuple[_Route | None, float] | None] = {}

    def route(self, option: int, stops: list[_Stop]) -> _Route:
        """A new route of this search, with a fresh stamp."""
        return _Route(option, stops, next(self.stamps))

    def admits(self, option: int, load: Sequence[float], nodes: frozenset[int]) -> bool:
        """False if no route of ``option`` may carry ``load`` to ``nodes``."""
        if any(amount > limit for amount, limit in zip(load, self.cap[option], strict=False)):
            return False
        if len(nodes) > 1 and not self.stopover_ok[option]:
            return False
        return nodes <= self.allowed_nodes[option]

    def value(self, route: _Route) -> float:
        km = route_km(self.p, route.nodes)
        if not self.timed:
            return route_value(self.p, self.coefficients, self.shared, route.option, km, route.load)
        primary = self.shared.primary
        nodes = route.nodes
        unloads = [float(stop.amount[primary]) for stop in route.stops]
        if self.split:
            ready = [self.p.node_ready[s.node] for s in route.stops if s.amount.max() > 0]
        else:
            ready = [self.p.item_ready[s.node, slot] for s in route.stops for slot in s.items]
        start, finishes = route_timing(
            self.p, self.shared, route.option, nodes, unloads, float(max(ready, default=-np.inf))
        )
        items = [stop.items for stop in route.stops]
        service = route_lateness(self.p, self.shared, nodes, unloads, items, finishes)
        return route_value(
            self.p,
            self.coefficients,
            self.shared,
            route.option,
            km,
            route.load,
            hours=(finishes[-1] - start) / 60.0,
            service=service,
        )

    def uncarried(
        self, remaining: NDArray[np.float64], item_left: NDArray[np.bool_] | None = None
    ) -> float:
        left = self.item_left if item_left is None else item_left
        return unserved_value(self.p, self.coefficients, self.shared, remaining, left)

    def roomy(self, routes: list[_Route]) -> bool:
        """True if ``routes`` fit the vehicle slots and handling room of the day."""
        if self.room is None:
            return True
        room, primary = self.room, self.shared.primary
        kinds = self.p.vehicle_type
        types = room.slots.shape[3]
        origin = np.bincount([kinds[route.option] for route in routes], minlength=types)
        if np.any(origin > room.origin_slots[0]):
            return False
        nodes = np.zeros((self.p.size, types), dtype=np.int64)
        handled = np.zeros(self.p.size)
        for route in routes:
            for stop in route.stops:
                nodes[stop.node, kinds[route.option]] += 1
                handled[stop.node] += float(stop.amount[primary])
        if np.any(nodes > room.node_slots[0]):
            return False
        limit = room.node_handling[0] * (1 + TOLERANCE) + MIN_ROOM
        if np.any(handled > limit):
            return False
        return bool(handled.sum() <= room.origin_handling[0] * (1 + TOLERANCE) + MIN_ROOM)

    def _with(self, routes: list[_Route], changes: dict[int, _Route | None]) -> list[_Route]:
        """``routes`` with some replaced (or removed, for None)."""
        trial = [changes.get(index, route) for index, route in enumerate(routes)]
        return [route for route in trial if route is not None]

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

    def _renew(self, routes: list[_Route], *indices: int) -> None:
        """Replace the routes at ``indices`` by new ones with the same stops."""
        for index in indices:
            routes[index] = self.route(routes[index].option, routes[index].stops)

    def _shrink(self, source: _Route, position: int) -> tuple[_Route | None, float] | None:
        """``source`` without its stop at ``position``, and the value of what is left.

        None when the stop has nothing to move or what is left breaks a rule;
        the route is None when nothing is left to carry. Remembered per stamp
        and position.
        """
        key = (source.stamp, position)
        if key in self._shrunk:
            return self._shrunk[key]
        result: tuple[_Route | None, float] | None = None
        if source.stops[position].amount.max() > 0:
            rest = [other for at, other in enumerate(source.stops) if at != position]
            left = self.arranged(source.option, rest) if rest else []
            shrunk = self.route(source.option, left[0]) if left else None
            if shrunk is not None and not any(s.amount.max() > 0 for s in shrunk.stops):
                shrunk = None
            if shrunk is None:
                result = (None, 0.0)
            elif self.valid(shrunk):
                result = (shrunk, self.value(shrunk))
        self._shrunk[key] = result
        return result

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
        rejected = self._change_rejected
        for index, route in enumerate(routes):
            for option in range(self.p.options):
                if option == route.option or usage[option] >= self.available[option]:
                    continue
                key = (route.stamp, option)
                if key in rejected:
                    continue
                if not self.admits(option, route.load_tuple, route.nodes_set):
                    rejected.add(key)
                    continue
                blocked = False
                for stops in self.arranged(option, route.stops):
                    candidate = self.route(option, stops)
                    if not self.valid(candidate):
                        continue
                    value = self.value(candidate)
                    if self._better(value, values[index]):
                        if self.roomy(self._with(routes, {index: candidate})):
                            routes[index], values[index] = candidate, value
                            return True
                        blocked = True
                if not blocked:
                    rejected.add(key)
        return False

    def _merge(self, routes: list[_Route], values: list[float]) -> bool:
        usage = self._usage(routes)
        rejected = self._merge_rejected
        for first in range(len(routes)):
            a = routes[first]
            for second in range(first + 1, len(routes)):
                b = routes[second]
                old = values[first] + values[second]
                load = [x + y for x, y in zip(a.load_tuple, b.load_tuple, strict=False)]
                nodes = a.nodes_set | b.nodes_set
                joined: list[_Stop] | None = None
                for option in range(self.p.options):
                    freed = (a.option == option) + (b.option == option)
                    if usage[option] - freed + 1 > self.available[option]:
                        continue
                    key = (a.stamp, b.stamp, option)
                    if key in rejected:
                        continue
                    if not self.admits(option, load, nodes):
                        rejected.add(key)
                        continue
                    if joined is None:
                        joined = a.stops + b.stops
                    blocked = False
                    for stops in self.arranged(option, joined):
                        candidate = self.route(option, stops)
                        if not self.valid(candidate):
                            continue
                        value = self.value(candidate)
                        if self._better(value, old):
                            if self.roomy(self._with(routes, {first: candidate, second: None})):
                                routes[first], values[first] = candidate, value
                                del routes[second], values[second]
                                return True
                            blocked = True
                    if not blocked:
                        rejected.add(key)
        return False

    def _relocate(self, routes: list[_Route], values: list[float]) -> bool:
        rejected = self._relocate_rejected
        for source in range(len(routes)):
            giver = routes[source]
            for position, stop in enumerate(giver.stops):
                shrink = self._shrink(giver, position)
                if shrink is None:
                    continue
                shrunk, shrunk_value = shrink
                amount = stop.amount.tolist()
                for target in range(len(routes)):
                    if target == source:
                        continue
                    receiver = routes[target]
                    key = (giver.stamp, position, receiver.stamp)
                    if key in rejected:
                        continue
                    option = receiver.option
                    load = [x + y for x, y in zip(receiver.load_tuple, amount, strict=False)]
                    if not self.admits(option, load, receiver.nodes_set | {stop.node}):
                        rejected.add(key)
                        continue
                    blocked = False
                    for stops in self.arranged(option, [*receiver.stops, stop]):
                        candidate = self.route(option, stops)
                        if not self.valid(candidate):
                            continue
                        value = self.value(candidate)
                        old = values[source] + values[target]
                        if self._better(shrunk_value + value, old):
                            if self.roomy(self._with(routes, {target: candidate, source: shrunk})):
                                routes[target], values[target] = candidate, value
                                if shrunk is None:
                                    del routes[source], values[source]
                                else:
                                    routes[source], values[source] = shrunk, shrunk_value
                                return True
                            blocked = True
                    if not blocked:
                        rejected.add(key)
        return False

    def _rebalance(self, routes: list[_Route], values: list[float]) -> bool:
        """Shift load between two routes; the stops change in place, so both are renewed.

        A shift that is undone may leave the amounts a rounding error away
        from what they were, so then both routes are renewed as well.
        """
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
                    given_before, taken_before = given.amount, stop.amount
                    given.amount = given.amount - part
                    stop.amount = stop.amount + part
                    new_taker, new_giver = self.value(taker), self.value(giver)
                    if self._better(new_taker + new_giver, old):
                        values[taker_index], values[giver_index] = new_taker, new_giver
                        self._renew(routes, taker_index, giver_index)
                        return True
                    given.amount = given.amount + part
                    stop.amount = stop.amount - part
                    if not (
                        np.array_equal(given.amount, given_before)
                        and np.array_equal(stop.amount, taken_before)
                    ):
                        self._renew(routes, taker_index, giver_index)
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
                    candidate = self.route(route.option, stops)
                    if not self.valid(candidate):
                        continue
                    after = self.remaining.copy()
                    after[node] = np.where(left - part > TOLERANCE * left, left - part, 0.0)
                    value = self.value(candidate)
                    old = values[index] + self.uncarried(self.remaining)
                    if self._better(value + self.uncarried(after), old) and self.roomy(
                        self._with(routes, {index: candidate})
                    ):
                        routes[index], values[index] = candidate, value
                        self.remaining = after
                        return True
        return False

    def _drop(self, routes: list[_Route], values: list[float]) -> bool:
        before = self.uncarried(self.remaining)
        for index, route in enumerate(routes):
            after = self.remaining.copy()
            left = None if self.item_left is None else self.item_left.copy()
            for stop in route.stops:
                after[stop.node] += stop.amount
                if left is not None:
                    left[stop.node, stop.items] = True
            if self._better(self.uncarried(after, left), values[index] + before):
                del routes[index], values[index]
                self.remaining = after
                self.item_left = left
                return True
        return False


def improve(
    problem: Subproblem,
    routes: Routes,
    coefficients: Mapping[str, float],
    shared: Shared,
    available: NDArray[np.int64],
    remaining: NDArray[np.float64],
    room: Room | None = None,
    item_left: NDArray[np.bool_] | None = None,
) -> tuple[Routes, float]:
    """Improve ``routes``; return them with the change of the objective (never positive).

    ``remaining`` is the demand per node that the routes leave uncarried, and
    ``item_left`` the items they leave (needed when demand may not be split and
    items may wait for the next day). ``room`` limits the moves on scheduled
    instances.
    """
    search = _Search(problem, coefficients, shared, available, remaining, room, item_left)
    better, change = search.run(_unpack(routes, search.stamps))
    return _pack(better, problem.demand.shape[1]), change


def _unpack(routes: Routes, stamps: Iterator[int]) -> list[_Route]:
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
        result.append(_Route(int(option), stops, next(stamps)))
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

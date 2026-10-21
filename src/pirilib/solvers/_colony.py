"""The inner loop of the ant colony: many colonies, on many subproblems, side by side.

Subproblems of similar size are stacked into padded arrays (:class:`Batch`).
Rows are (subproblem, ant) pairs, ``row = subproblem * ants + ant``, and every
construction step is a few numpy operations over all rows at once. Padded nodes
and options are never allowed, so they are never chosen.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from pirilib.evaluation import TOLERANCE
from pirilib.plan import OWNED, SPOT
from pirilib.solvers.decompose import Routes, Subproblem
from pirilib.spec.instance import LineHaulInstance
from pirilib.spec.models import Rules

NEED, ROUTE, DONE = 0, 1, 2
UNDERFILLED_END: Final = 0.1
MAX_STEPS: Final = 1_000_000


@dataclass(frozen=True, slots=True)
class Shared:
    """Instance-wide values the colonies need.

    Attributes:
        min_fill: Indexed by kind code
    """

    rules: Rules
    primary: int
    dimensions: tuple[str, ...]
    min_fill: NDArray[np.float64]

    @classmethod
    def of(cls, instance: LineHaulInstance) -> "Shared":
        min_fill = np.zeros(2)
        min_fill[OWNED] = instance.rules.min_fill.owned
        min_fill[SPOT] = instance.rules.min_fill.spot
        return cls(instance.rules, instance.primary_dimension, instance.dimensions, min_fill)


class Batch:
    """Subproblems stacked into padded arrays, indexed by subproblem, then node or option."""

    def __init__(self, problems: Sequence[Subproblem], shared: Shared) -> None:
        self.problems = list(problems)
        count = len(self.problems)
        k = max(problem.size for problem in self.problems)
        n = max(problem.options for problem in self.problems)
        items = max(problem.item_row.shape[1] for problem in self.problems)
        dimensions = len(shared.dimensions)
        self.count, self.k, self.n = count, k, n

        self.demand = np.zeros((count, k, dimensions))
        self.item_valid = np.zeros((count, k, items), dtype=np.bool_)
        self.item_quantity = np.zeros((count, k, items, dimensions))
        self.origin_km = np.zeros((count, k))
        self.node_km = np.zeros((count, k, k))
        self.kind = np.full((count, n), SPOT, dtype=np.int8)
        self.fixed = np.zeros((count, n))
        self.per_km = np.zeros((count, n))
        self.capacity = np.ones((count, n, dimensions))
        self.available = np.zeros((count, n), dtype=np.int64)
        self.line_end = np.full((count, n), -1, dtype=np.int64)
        self.stopovers = np.zeros((count, n), dtype=np.bool_)
        self.allowed = np.zeros((count, n, k), dtype=np.bool_)
        self.first_ok = np.zeros((count, n, k), dtype=np.bool_)
        self.first_cost = np.zeros((count, n, k))
        for index, problem in enumerate(self.problems):
            size, options, width = problem.size, problem.options, problem.item_row.shape[1]
            self.demand[index, :size] = problem.demand
            self.item_valid[index, :size, :width] = problem.item_row >= 0
            self.item_quantity[index, :size, :width] = problem.item_quantity
            self.origin_km[index, :size] = problem.origin_km
            self.node_km[index, :size, :size] = problem.node_km
            self.kind[index, :options] = problem.kind
            self.fixed[index, :options] = problem.fixed
            self.per_km[index, :options] = problem.per_km
            self.capacity[index, :options] = problem.capacity
            self.available[index, :options] = problem.available
            self.line_end[index, :options] = problem.line_end
            self.stopovers[index, :options] = problem.stopovers
            self.allowed[index, :options, :size] = problem.allowed
            ok, cost = _first_choices(problem, shared.rules)
            self.first_ok[index, :options, :size] = ok
            self.first_cost[index, :options, :size] = cost
        self.owned = self.kind == OWNED
        self.capacity_primary = self.capacity[:, :, shared.primary]
        self.min_share = shared.min_fill[self.kind]
        self.tiny = 1e-6 * (1.0 + self.first_cost.max(axis=(1, 2), initial=0.0))
        tiny = self.tiny[:, None, None]
        self.first_base = 1.0 / (self.first_cost + tiny)
        line0 = np.maximum(self.line_end, 0)
        self.line_km = self.node_km[np.arange(count)[:, None], :, line0]
        self.line_origin = np.take_along_axis(self.origin_km, line0, axis=1)
        owned = self.owned[:, :, None, None]
        detour = self.node_km[:, None, :, :] + np.where(
            owned, self.line_km[:, :, None, :] - self.line_km[:, :, :, None], 0.0
        )
        alternative = self.fixed[:, :, None, None] + (
            self.per_km[:, :, None, None] * self.origin_km[:, None, None, :]
        )
        marginal = self.per_km[:, :, None, None] * np.clip(detour, 0.0, None)
        self.step_base = alternative / (marginal + self.tiny[:, None, None, None])
        self.fits_new = self.item_valid[:, None] & np.all(
            self.item_quantity[:, None] <= self.capacity[:, :, None, None, :] * (1 + TOLERANCE),
            axis=-1,
        )


def _first_choices(
    problem: Subproblem, rules: Rules
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """(options, nodes): may the node be the first stop of the option, and the trip cost."""
    stopover = rules.stopover
    k = problem.size
    ok = problem.allowed.copy()
    detour = np.zeros((problem.options, k))
    for option in np.flatnonzero(problem.kind == OWNED):
        line = problem.line_end[option]
        detour[option] = np.where(np.arange(k) == line, 0.0, problem.node_km[:, line])
        allowed = np.zeros(k, dtype=np.bool_)
        allowed[line] = True
        if problem.stopovers[option] and (stopover.max_stops is None or stopover.max_stops >= 2):
            on_the_way = np.ones(k, dtype=np.bool_)
            on_the_way[line] = False
            if stopover.forward_only:
                on_the_way &= problem.origin_km < problem.origin_km[line]
            if stopover.max_stretch is not None:
                limit = stopover.max_stretch * problem.origin_km[line] * (1 + TOLERANCE)
                on_the_way &= problem.origin_km + problem.node_km[:, line] <= limit
            allowed |= on_the_way
        ok[option] &= allowed
    cost = problem.fixed[:, None] + problem.per_km[:, None] * (problem.origin_km[None, :] + detour)
    return ok, cost


@dataclass(slots=True)
class Pheromone:
    """Pheromone of every subproblem of a batch, in ``[tau_min, 1]``.

    Attributes:
        first: (subproblems, options, nodes): option and first stop
        step: (subproblems, nodes, nodes): from one stop to the next
        end: (subproblems, nodes): ending the route after a stop
    """

    first: NDArray[np.float64]
    step: NDArray[np.float64]
    end: NDArray[np.float64]

    @classmethod
    def filled(cls, batch: Batch, value: float) -> "Pheromone":
        count, n, k = batch.count, batch.n, batch.k
        return cls(
            np.full((count, n, k), value), np.full((count, k, k), value), np.full((count, k), value)
        )

    def update(
        self, used: "Pheromone", evaporation: float, tau_min: float, which: NDArray[np.bool_]
    ) -> None:
        """Hypercube update of the subproblems in ``which``."""
        for name in ("first", "step", "end"):
            tau, mark = getattr(self, name), getattr(used, name)
            tau[which] = np.clip(
                (1 - evaporation) * tau[which] + evaporation * mark[which], tau_min, 1.0
            )

    def copy_from(self, other: "Pheromone", which: NDArray[np.bool_]) -> None:
        for name in ("first", "step", "end"):
            getattr(self, name)[which] = getattr(other, name)[which]


@dataclass(frozen=True, eq=False, slots=True)
class Colony:
    """The solutions of all rows, grouped: routes by row, stops by route, items by stop.

    Attributes:
        route_offsets: (rows + 1,)
        stop_offsets: (routes + 1,)
        item_stop: Stop index, in the grouped order
        item_offsets: (stops + 1,)
        remaining: (rows, nodes, dimensions) not carried
        first: (choices, 3): row, option, node
        step: (choices, 3): row, from node, to node
        end: (choices, 2): row, node
    """

    ants: int
    route_row: NDArray[np.int64]
    route_option: NDArray[np.int64]
    route_km: NDArray[np.float64]
    route_load: NDArray[np.float64]
    route_offsets: NDArray[np.int64]
    stop_node: NDArray[np.int64]
    stop_amount: NDArray[np.float64]
    stop_offsets: NDArray[np.int64]
    item_stop: NDArray[np.int64]
    item_slot: NDArray[np.int64]
    item_offsets: NDArray[np.int64]
    remaining: NDArray[np.float64]
    first: NDArray[np.int64]
    step: NDArray[np.int64]
    end: NDArray[np.int64]

    def routes(self, row: int) -> Routes:
        """The solution of one row, in subproblem terms."""
        a, z = self.route_offsets[row], self.route_offsets[row + 1]
        s0, s1 = self.stop_offsets[a], self.stop_offsets[z]
        i0, i1 = self.item_offsets[s0], self.item_offsets[s1]
        return Routes(
            option=self.route_option[a:z],
            stop_start=self.stop_offsets[a : z + 1] - s0,
            stop_node=self.stop_node[s0:s1],
            stop_amount=self.stop_amount[s0:s1],
            item_stop=self.item_stop[i0:i1] - s0,
            item_slot=self.item_slot[i0:i1],
        )

    def usage(self, options: int) -> NDArray[np.int64]:
        """(rows, options): vehicles of each option used by each row."""
        rows = self.remaining.shape[0]
        counts = np.bincount(self.route_row * options + self.route_option, minlength=rows * options)
        return counts.reshape(rows, options)

    def used(self, rows: NDArray[np.int64], batch: Batch) -> Pheromone:
        """Indicators of the choices of one row per subproblem (``-1`` for none)."""
        used = Pheromone.filled(batch, 0.0)
        owner = np.full(self.remaining.shape[0], -1, dtype=np.int64)
        chosen = rows >= 0
        owner[rows[chosen]] = np.flatnonzero(chosen)
        first = self.first[owner[self.first[:, 0]] >= 0]
        used.first[owner[first[:, 0]], first[:, 1], first[:, 2]] = 1.0
        step = self.step[owner[self.step[:, 0]] >= 0]
        used.step[owner[step[:, 0]], step[:, 1], step[:, 2]] = 1.0
        end = self.end[owner[self.end[:, 0]] >= 0]
        used.end[owner[end[:, 0]], end[:, 1]] = 1.0
        return used


def sample(
    weights: NDArray[np.float64], rng: np.random.Generator, exploitation: float
) -> NDArray[np.int64]:
    """One column per row, with probability proportional to the weights; -1 for empty rows."""
    total = weights.sum(axis=1)
    cumulative = np.cumsum(weights, axis=1)
    threshold = rng.random(weights.shape[0]) * total
    choice = np.minimum((cumulative <= threshold[:, None]).sum(axis=1), weights.shape[1] - 1)
    if exploitation > 0:
        greedy = rng.random(weights.shape[0]) < exploitation
        choice[greedy] = weights[greedy].argmax(axis=1)
    choice[total <= 0] = -1
    return np.asarray(choice, dtype=np.int64)


class Build:
    """Builds one solution per row of a batch."""

    def __init__(
        self,
        batch: Batch,
        pheromone: Pheromone,
        shared: Shared,
        *,
        ants: int,
        alpha: float,
        beta: float,
        exploitation: float,
        rng: np.random.Generator,
        available: NDArray[np.int64],
        active: NDArray[np.bool_],
    ) -> None:
        self.batch, self.shared, self.rng = batch, shared, rng
        self.ants, self.beta, self.exploitation = ants, beta, exploitation
        stopover = shared.rules.stopover
        self.forward_only = stopover.forward_only
        self.max_stretch = stopover.max_stretch
        self.max_stops = stopover.max_stops
        self.split = shared.rules.split_demand
        rows = batch.count * ants
        self.of = np.repeat(np.arange(batch.count), ants)
        self.first_weight = np.where(
            batch.first_ok, pheromone.first**alpha * batch.first_base**beta, 0.0
        )
        self.step_weight = pheromone.step[:, None] ** alpha * batch.step_base**beta
        self.tau_end = pheromone.end**alpha

        self.available = available
        self.remaining = batch.demand[self.of]
        self.item_left = batch.item_valid[self.of]
        dimensions = batch.demand.shape[2]
        self.phase = np.where(active[self.of], NEED, DONE).astype(np.int8)
        self.option = np.zeros(rows, dtype=np.int64)
        self.last = np.zeros(rows, dtype=np.int64)
        self.route = np.full(rows, -1, dtype=np.int64)
        self.km = np.zeros(rows)
        self.load = np.zeros((rows, dimensions))
        self.cap = np.zeros((rows, dimensions))
        self.stops = np.zeros(rows, dtype=np.int64)
        self.visited = np.zeros((rows, batch.k), dtype=np.bool_)
        self.routes_started = 0
        self.stops_made = 0
        self.log: dict[str, list[NDArray[Any]]] = {}

    def record(self, name: str, values: NDArray[Any]) -> None:
        self.log.setdefault(name, []).append(values)

    def run(self) -> Colony:
        for _ in range(MAX_STEPS):
            need = np.flatnonzero(self.phase == NEED)
            if need.size:
                self._start(need)
            moving = np.flatnonzero(self.phase == ROUTE)
            if moving.size:
                self._extend(moving)
            if not need.size and not moving.size:
                return self._colony()
        raise RuntimeError("ant construction did not finish")

    def _start(self, rows: NDArray[np.intp]) -> None:
        batch, primary = self.batch, self.shared.primary
        of = self.of[rows]
        remaining = self.remaining[rows]
        if self.split:
            loadable = (remaining.max(axis=2, initial=0.0) > 0)[:, None, :]
        else:
            loadable = (self.item_left[rows][:, None] & batch.fits_new[of]).any(axis=-1)
        mask = loadable & (self.available[rows] > 0)[:, :, None]
        carry = np.minimum(remaining[:, None, :, primary], batch.capacity_primary[of][:, :, None])
        weights = np.where(mask, self.first_weight[of] * carry**self.beta, 0.0)
        choice = sample(weights.reshape(rows.size, -1), self.rng, self.exploitation)

        self.phase[rows[choice < 0]] = DONE
        chosen = choice >= 0
        go, of = rows[chosen], of[chosen]
        option, node = np.divmod(choice[chosen], batch.k)
        self.record("first", np.column_stack([go, option, node]))
        self.available[go, option] -= 1
        self.route[go] = self.routes_started + np.arange(go.size)
        self.routes_started += go.size
        self.record("route_row", go)
        self.record("route_option", option)
        self.option[go] = option
        self.cap[go] = batch.capacity[of, option]
        self.load[go] = 0.0
        self.km[go] = batch.origin_km[of, node]
        self.stops[go] = 0
        self.visited[go] = False
        self._visit(go, node)
        self._decide(go, batch.owned[of, option] & (node == batch.line_end[of, option]))

    def _extend(self, rows: NDArray[np.intp]) -> None:
        batch, primary, beta, k = self.batch, self.shared.primary, self.beta, self.batch.k
        of, option, here = self.of[rows], self.option[rows], self.last[rows]
        owned, line = batch.owned[of, option], batch.line_end[of, option]
        capacity = batch.capacity[of, option]
        cap, remaining = self.cap[rows], self.remaining[rows]
        if self.split:
            no_room = (remaining > 0) & (cap[:, None, :] <= TOLERANCE * capacity[:, None, :])
            loadable = (remaining.max(axis=2, initial=0.0) > 0) & ~no_room.any(axis=2)
        else:
            room = (cap + TOLERANCE * capacity)[:, None, None, :]
            fits = self.item_left[rows] & np.all(batch.item_quantity[of] <= room, axis=-1)
            loadable = fits.any(axis=-1)
        candidate = loadable & batch.allowed[of, option] & ~self.visited[rows]
        origin_km = batch.origin_km[of]
        position = np.arange(rows.size)
        if self.forward_only:
            candidate &= origin_km > origin_km[position, here][:, None]
        line_km = batch.line_km[of, option]
        line_origin = batch.line_origin[of, option]
        if self.max_stretch is not None:
            new_km = self.km[rows][:, None] + batch.node_km[of, here]
            spot_ok = new_km <= self.max_stretch * origin_km * (1 + TOLERANCE)
            owned_ok = (
                new_km + line_km <= (self.max_stretch * line_origin * (1 + TOLERANCE))[:, None]
            )
            candidate &= np.where(owned[:, None], owned_ok, spot_ok)
        if owned.any():
            blocked = np.arange(k)[None, :] == line[:, None]
            if self.forward_only:
                blocked |= origin_km >= line_origin[:, None]
            candidate &= ~(owned[:, None] & blocked)

        cap_primary = batch.capacity_primary[of, option]
        fill = np.minimum(remaining[:, :, primary], cap[:, primary][:, None]) / cap_primary[:, None]
        step_weight = np.where(candidate, self.step_weight[of, option, here] * fill**beta, 0.0)
        underfilled = self.load[rows, primary] < (
            batch.min_share[of, option] * cap_primary * (1 - TOLERANCE)
        )
        end_weight = self.tau_end[of, here] * np.where(underfilled, UNDERFILLED_END, 1.0) ** beta
        choice = sample(np.column_stack([step_weight, end_weight]), self.rng, self.exploitation)

        ending = choice == k
        self.record("end", np.column_stack([rows[ending], here[ending]]))
        self._finish(rows[ending])
        go, node = rows[~ending], choice[~ending]
        self.record("step", np.column_stack([go, self.last[go], node]))
        self.km[go] += batch.node_km[self.of[go], self.last[go], node]
        self._visit(go, node)
        self._decide(go, np.zeros(go.size, dtype=np.bool_))

    def _decide(self, rows: NDArray[np.intp], at_line: NDArray[np.bool_]) -> None:
        """After a visit: keep the route open, or finish it."""
        batch = self.batch
        of, option = self.of[rows], self.option[rows]
        full = np.all(self.cap[rows] <= TOLERANCE * batch.capacity[of, option], axis=1)
        keep = batch.stopovers[of, option] & ~full & ~at_line
        if self.max_stops is not None:
            reserved = batch.owned[of, option]
            keep &= self.stops[rows] + reserved < self.max_stops
        self.phase[rows[keep]] = ROUTE
        self._finish(rows[~keep])

    def _finish(self, rows: NDArray[np.intp]) -> None:
        """Close the routes of ``rows``; owned routes first go on to their line end."""
        batch = self.batch
        of, option = self.of[rows], self.option[rows]
        line = batch.line_end[of, option]
        detour = batch.owned[of, option] & (self.last[rows] != line)
        go, to = rows[detour], line[detour]
        self.km[go] += batch.node_km[self.of[go], self.last[go], to]
        self._visit(go, to)
        self.record("close_route", self.route[rows])
        self.record("close_km", self.km[rows])
        self.record("close_load", self.load[rows])
        self.phase[rows] = NEED

    def _visit(self, rows: NDArray[np.intp], nodes: NDArray[np.int64]) -> None:
        """Load at ``nodes`` what fits, and record the stops."""
        batch = self.batch
        of = self.of[rows]
        capacity = batch.capacity[of, self.option[rows]]
        demand = batch.demand[of, nodes]
        cap = self.cap[rows]
        left = self.remaining[rows, nodes]
        if self.split:
            ratio = np.where(left > 0, cap / np.where(left > 0, left, 1.0), np.inf)
            amount = np.clip(ratio.min(axis=1, initial=np.inf), 0.0, 1.0)[:, None] * left
        else:
            stop_ids = self.stops_made + np.arange(rows.size)
            amount = np.zeros_like(cap)
            for slot in range(batch.item_valid.shape[2]):
                quantity = batch.item_quantity[of, nodes, slot]
                fits = self.item_left[rows, nodes, slot] & np.all(
                    quantity <= cap - amount + TOLERANCE * capacity, axis=1
                )
                if fits.any():
                    self.item_left[rows[fits], nodes[fits], slot] = False
                    amount[fits] += quantity[fits]
                    self.record("item_stop", stop_ids[fits])
                    self.record("item_slot", np.full(int(fits.sum()), slot, dtype=np.int64))
        left = np.clip(left - amount, 0.0, None)
        left[left <= TOLERANCE * demand] = 0.0
        self.remaining[rows, nodes] = left
        cap = cap - amount
        cap[cap <= TOLERANCE * capacity] = 0.0
        self.cap[rows] = cap
        self.load[rows] += amount
        self.stops[rows] += 1
        self.last[rows] = nodes
        self.visited[rows, nodes] = True
        self.stops_made += rows.size
        self.record("stop_route", self.route[rows])
        self.record("stop_node", nodes)
        self.record("stop_amount", amount)

    def _colony(self) -> Colony:
        dimensions = self.batch.demand.shape[2]
        rows = self.of.size

        def get(name: str, empty: tuple[int, ...], dtype: type[np.generic]) -> NDArray[Any]:
            parts = self.log.get(name)
            return np.concatenate(parts).astype(dtype) if parts else np.zeros(empty, dtype=dtype)

        route_row = get("route_row", (0,), np.int64)
        closed = get("close_route", (0,), np.int64)
        route_km = np.zeros(self.routes_started)
        route_km[closed] = get("close_km", (0,), np.float64)
        route_load = np.zeros((self.routes_started, dimensions))
        route_load[closed] = get("close_load", (0, dimensions), np.float64)
        stop_route = get("stop_route", (0,), np.int64)
        item_stop = get("item_stop", (0,), np.int64)

        route_order = np.argsort(route_row, kind="stable")
        route_rank = np.empty_like(route_order)
        route_rank[route_order] = np.arange(route_order.size)
        stop_order = np.argsort(route_rank[stop_route], kind="stable")
        stop_rank = np.empty_like(stop_order)
        stop_rank[stop_order] = np.arange(stop_order.size)
        item_order = np.argsort(stop_rank[item_stop], kind="stable")

        def offsets(groups: NDArray[np.int64], size: int) -> NDArray[np.int64]:
            counts = np.bincount(groups, minlength=size)
            return np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

        return Colony(
            ants=self.ants,
            route_row=route_row[route_order],
            route_option=get("route_option", (0,), np.int64)[route_order],
            route_km=route_km[route_order],
            route_load=route_load[route_order],
            route_offsets=offsets(route_row, rows),
            stop_node=get("stop_node", (0,), np.int64)[stop_order],
            stop_amount=get("stop_amount", (0, dimensions), np.float64)[stop_order],
            stop_offsets=offsets(route_rank[stop_route], self.routes_started),
            item_stop=stop_rank[item_stop][item_order],
            item_slot=get("item_slot", (0,), np.int64)[item_order],
            item_offsets=offsets(stop_rank[item_stop], stop_route.size),
            remaining=self.remaining,
            first=get("first", (0, 3), np.int64),
            step=get("step", (0, 3), np.int64),
            end=get("end", (0, 2), np.int64),
        )


def scores(
    batch: Batch, colony: Colony, coefficients: Mapping[str, float], shared: Shared
) -> NDArray[np.float64]:
    """(subproblems, ants): the linear objective of every solution, without constants."""
    rows = colony.remaining.shape[0]
    of = colony.route_row // colony.ants
    option = colony.route_option
    kind = batch.kind[of, option]
    spot = kind == SPOT
    cost = batch.fixed[of, option] + batch.per_km[of, option] * colony.route_km
    capacity = batch.capacity_primary[of, option]
    load = colony.route_load[:, shared.primary]
    every = np.ones(option.size, dtype=np.bool_)

    def per_row(values: NDArray[np.float64], mask: NDArray[np.bool_]) -> NDArray[np.float64]:
        return np.bincount(colony.route_row[mask], weights=values[mask], minlength=rows)

    features: dict[str, NDArray[np.float64]] = {}
    for name, mask in (("owned", ~spot), ("spot", spot)):
        features[f"cost[{name}]"] = per_row(cost, mask)
        features[f"km[{name}]"] = per_row(colony.route_km, mask)
        features[f"dispatches[{name}]"] = per_row(np.ones(option.size), mask)
        features[f"unused[{name}]"] = per_row(np.clip(capacity - load, 0.0, None), mask)
    remaining = colony.remaining
    features["unserved"] = remaining.sum(axis=(1, 2))
    for column, name in enumerate(shared.dimensions):
        features[f"unserved[{name}]"] = remaining[:, :, column].sum(axis=1)
    shortfall = shared.min_fill[kind] * capacity - load
    features["violation[min_fill]"] = per_row(
        np.where(shortfall > TOLERANCE * capacity, shortfall, 0.0), every
    )
    demand = batch.demand[np.arange(rows) // colony.ants]
    features["violation[coverage]"] = np.where(
        remaining > TOLERANCE * np.maximum(demand, 1.0), remaining, 0.0
    ).sum(axis=(1, 2))

    total = np.zeros(rows)
    for feature, weight in coefficients.items():
        if feature in features:
            total += weight * features[feature]
    return total.reshape(batch.count, colony.ants)

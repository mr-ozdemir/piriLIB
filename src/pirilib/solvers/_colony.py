"""The inner loop of the ant colony: many colonies, on many subproblems, side by side.

Subproblems of similar size are stacked into padded arrays (:class:`Batch`).
Rows are (subproblem, ant) pairs, ``row = subproblem * ants + ant``, and every
construction step is a few numpy operations over all rows at once. Padded nodes
and options are never allowed, so they are never chosen.

On scheduled instances a batch holds one subproblem, and a
:class:`~pirilib.solvers.ledger.Room` says what its centers can still handle:
an ant then only takes a vehicle type that has a slot at the origin, stops only
where its type has a slot and the center has handling room, and loads no more
than that room. Construction charges everything to the subproblem's day; the
scores time the routes exactly.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from pirilib.evaluation import TOLERANCE
from pirilib.plan import OWNED, SPOT
from pirilib.solvers._timing import timeline
from pirilib.solvers.decompose import UNBOUNDED, Routes, Subproblem
from pirilib.solvers.ledger import MIN_ROOM, Room
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
        scheduled: Times, deferral or center limits matter (see LineHaulInstance)
        minutes_per_unit: Handling time per primary unit
        penalty: Lateness penalty per primary unit and started hour
    """

    rules: Rules
    primary: int
    dimensions: tuple[str, ...]
    min_fill: NDArray[np.float64]
    scheduled: bool = False
    minutes_per_unit: float = 0.0
    penalty: float = 0.0

    @classmethod
    def of(cls, instance: LineHaulInstance) -> "Shared":
        min_fill = np.zeros(2)
        min_fill[OWNED] = instance.rules.min_fill.owned
        min_fill[SPOT] = instance.rules.min_fill.spot
        return cls(
            instance.rules,
            instance.primary_dimension,
            instance.dimensions,
            min_fill,
            instance.scheduled,
            instance.rules.handling.minutes_per_unit,
            instance.rules.service.penalty,
        )


class Batch:
    """Subproblems stacked into padded arrays, indexed by subproblem, then node or option.

    Attributes:
        final_share: (subproblems, k, D) share of each node's demand that may not wait
        node_defer: (subproblems, k) cost per primary unit of the node's demand left
            for the next day, items in proportion (split demand)
    """

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
        self.day = np.array([problem.day for problem in self.problems], dtype=np.int64)
        self.vehicle_type = np.zeros((count, n), dtype=np.int64)
        self.per_hour = np.zeros((count, n))
        self.speed = np.full((count, n), np.nan)
        self.item_ready = np.full((count, k, items), -np.inf)
        self.item_due = np.full((count, k, items), np.inf)
        self.item_final = np.ones((count, k, items), dtype=np.bool_)
        self.defer_cost = np.zeros((count, k, items))
        self.node_ready = np.full((count, k), -np.inf)
        for index, problem in enumerate(self.problems):
            size, options, width = problem.size, problem.options, problem.item_row.shape[1]
            self.demand[index, :size] = problem.demand
            self.item_valid[index, :size, :width] = problem.item_row >= 0
            self.item_quantity[index, :size, :width] = problem.item_quantity
            self.vehicle_type[index, :options] = problem.vehicle_type
            self.per_hour[index, :options] = problem.per_hour
            self.speed[index, :options] = problem.speed
            self.item_ready[index, :size, :width] = problem.item_ready
            self.item_due[index, :size, :width] = problem.item_due
            self.item_final[index, :size, :width] = problem.item_final
            self.defer_cost[index, :size, :width] = problem.defer_cost
            self.node_ready[index, :size] = problem.node_ready
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
        primary = shared.primary
        safe = np.where(self.demand > 0, self.demand, 1.0)
        final = np.sum(self.item_quantity * self.item_final[..., None], axis=2)
        self.final_share = np.where(self.demand > 0, final / safe, 1.0)
        waiting = np.sum(
            self.item_quantity[..., primary] * ~self.item_final * self.defer_cost, axis=2
        )
        self.node_defer = np.where(self.demand[..., primary] > 0, waiting / safe[..., primary], 0.0)


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
    km = problem.origin_km[None, :] + detour
    cost = problem.fixed[:, None] + problem.per_km[:, None] * km
    if np.any(problem.per_hour > 0):
        speed = problem.speed[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            hours = np.where(np.isfinite(speed), km / speed, 0.0)
        cost = cost + problem.per_hour[:, None] * hours
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
        item_left: (rows, nodes, items) items not carried (whole items)
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
    item_left: NDArray[np.bool_]

    @classmethod
    def single(
        cls,
        problem: Subproblem,
        routes: Routes,
        remaining: NDArray[np.float64],
        item_left: NDArray[np.bool_],
    ) -> "Colony":
        """A colony of one row holding ``routes`` and what they leave uncarried."""
        size = routes.option.size
        counts = np.diff(routes.stop_start)
        stop_route = np.repeat(np.arange(size, dtype=np.int64), counts)
        node = routes.stop_node
        opening = np.zeros(node.size, dtype=np.bool_)
        opening[routes.stop_start[:-1][counts > 0]] = True
        previous = np.zeros_like(node)
        previous[1:] = node[:-1]
        previous[opening] = 0
        legs = np.where(opening, problem.origin_km[node], problem.node_km[previous, node])
        load = np.zeros((size, problem.demand.shape[1]))
        np.add.at(load, stop_route, routes.stop_amount)
        items = np.bincount(routes.item_stop, minlength=node.size)
        return cls(
            ants=1,
            route_row=np.zeros(size, dtype=np.int64),
            route_option=routes.option,
            route_km=np.bincount(stop_route, weights=legs, minlength=size),
            route_load=load,
            route_offsets=np.array([0, size], dtype=np.int64),
            stop_node=node,
            stop_amount=routes.stop_amount,
            stop_offsets=routes.stop_start,
            item_stop=routes.item_stop,
            item_slot=routes.item_slot,
            item_offsets=np.concatenate([[0], np.cumsum(items)]).astype(np.int64),
            remaining=remaining[None],
            first=np.zeros((0, 3), dtype=np.int64),
            step=np.zeros((0, 3), dtype=np.int64),
            end=np.zeros((0, 2), dtype=np.int64),
            item_left=item_left[None],
        )

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
    """One column per row, with probability proportional to the weights; -1 for empty rows.

    The chosen column is the first whose cumulative weight passes the threshold,
    the last when none does. Weights are not negative, so cumulative weights
    never fall, and this is the number of columns at or below the threshold.
    """
    total = weights.sum(axis=1)
    cumulative = np.cumsum(weights, axis=1)
    threshold = rng.random(weights.shape[0]) * total
    choice = (cumulative > threshold[:, None]).argmax(axis=1)
    choice[cumulative[:, -1] <= threshold] = weights.shape[1] - 1
    if exploitation > 0:
        greedy = rng.random(weights.shape[0]) < exploitation
        choice[greedy] = weights[greedy].argmax(axis=1)
    choice[total <= 0] = -1
    return np.asarray(choice, dtype=np.int64)


class Build:
    """Builds one solution per row of a batch.

    With a ``room`` (one subproblem per batch), every row keeps its own copy of
    the room on the subproblem's day and spends it as it builds.

    What depends only on the subproblem and the option is laid out once per
    build in the ``option_*`` tables, indexed by ``option_id = subproblem *
    options + option``; what depends on the subproblem and a node in the
    ``node_*`` tables, indexed by ``subproblem * k + node``; and the state of a
    row at a node in the ``*_cells`` arrays, indexed by ``cell = row * k + node``.
    A step then gathers with one index. ``batch`` keeps the same values in its
    own (subproblems, options, ...) layout under shorter names. Without a room,
    the weight of opening a route at a node changes only where the row loads or
    uses up a vehicle option, so it is kept in ``opening`` and recomputed only
    there. The tables and ``opening`` hold exactly the numbers a step would
    compute itself, with the same operations on the same values, so plans do
    not depend on this layout.

    Attributes:
        dimensions: number of capacity dimensions
        of: (rows,) subproblem of each row
        first_weight: (subproblems, options, k) pheromone and heuristic of an
            option and a first stop
        step_weight: (subproblems * options * k, k) pheromone and heuristic of the
            next stop, by ``option_id * k + node``
        tau_end: (subproblems * k,) pheromone of ending the route after a node
        end_factor: (2,) heuristic of ending: minimum fill reached, not reached
        option_capacity: (subproblems * options, dimensions) vehicle capacity
        option_slack: (subproblems * options, dimensions) free capacity at or
            below which a dimension counts as full
        option_primary: (subproblems * options,) primary capacity
        option_min_load: (subproblems * options,) primary load below which the
            route is underfilled
        option_owned: (subproblems * options,) the option is an owned line
        any_owned: the batch has an owned option
        option_line_end: (subproblems * options,) line end of an owned line, -1
            for spot options
        option_stopovers: (subproblems * options,) the route may make stopovers
        option_vehicle_type: (subproblems * options,) vehicle type
        option_allowed: (subproblems * options, k) nodes a route of the option may
            go on to: the subproblem's allowed nodes, and for owned lines neither
            the line end nor (forward only) beyond it
        option_line_stretch: (subproblems * options,) longest route of an owned
            line (with a maximum stretch)
        option_line_km: (subproblems * options, k) distance from each node to the
            option's line end (with a maximum stretch)
        node_legs: (subproblems * k, k) distance from a node to every node
        leg_km: ``node_legs`` as (subproblems * k * k,), by
            ``(subproblem * k + node) * k + next``
        node_ahead: (subproblems * k, k) nodes farther from the origin than a
            node (forward only)
        stretch: (subproblems, k) longest spot route to each node (with a maximum
            stretch)
        available: (rows, options) vehicles of each option a row may still use;
            ``available_cells`` is the same array as (rows * options,)
        remaining: (rows, k, dimensions) demand not carried yet; ``cells`` is the
            same array as (rows * k, dimensions)
        item_left: (rows, k, items) items not carried yet; ``item_cells`` is the
            same array as (rows * k, items)
        settled: (rows * k, dimensions) demand left at or below which a node
            counts as served
        origin_cells: (rows * k,) distance from the origin
        option_id: (rows,) ``option_id`` of the open route
        open: (rows, k) nodes the open route has not visited and may go on to;
            ``open_cells`` is the same array as (rows * k,)
        ratio: (rows, dimensions) space for the free capacity per unit left at a stop
        stops: (rows,) stops of the open route (with a maximum of stops)
        opening: (rows, options, k) weight of opening a route (without a room)
        pending: cells visited since ``opening`` was last brought up to date
    """

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
        room: Room | None = None,
    ) -> None:
        self.batch, self.shared, self.rng = batch, shared, rng
        self.ants, self.beta, self.exploitation = ants, beta, exploitation
        stopover = shared.rules.stopover
        self.forward_only = stopover.forward_only
        self.max_stretch = stopover.max_stretch
        self.max_stops = stopover.max_stops
        self.split = shared.rules.split_demand
        count, n, k = batch.count, batch.n, batch.k
        self.dimensions = dimensions = batch.demand.shape[2]
        rows = count * ants
        self.of = np.repeat(np.arange(count), ants)
        self.first_weight = np.where(
            batch.first_ok, pheromone.first**alpha * batch.first_base**beta, 0.0
        )
        step_weight = pheromone.step[:, None] ** alpha * batch.step_base**beta
        self.step_weight = step_weight.reshape(count * n * k, k)
        self.tau_end = (pheromone.end**alpha).reshape(count * k)
        self.end_factor = np.array([1.0, UNDERFILLED_END]) ** beta
        self._tables(batch)

        self.available = np.ascontiguousarray(available)
        self.available_cells = self.available.reshape(rows * n)
        self.remaining = batch.demand[self.of]
        self.cells = self.remaining.reshape(rows * k, dimensions)
        self.settled = TOLERANCE * self.cells
        self.origin_cells = batch.origin_km[self.of].reshape(rows * k)
        self.item_left = batch.item_valid[self.of]
        self.item_cells = self.item_left.reshape(rows * k, batch.item_valid.shape[2])
        self.phase = np.where(active[self.of], NEED, DONE).astype(np.int8)
        self.option_id = np.zeros(rows, dtype=np.int64)
        self.last = np.zeros(rows, dtype=np.int64)
        self.route = np.full(rows, -1, dtype=np.int64)
        self.km = np.zeros(rows)
        self.load = np.zeros((rows, dimensions))
        self.cap = np.zeros((rows, dimensions))
        self.stops = np.zeros(rows, dtype=np.int64)
        self.open = np.zeros((rows, k), dtype=np.bool_)
        self.open_cells = self.open.reshape(rows * k)
        self.ratio = np.empty((rows, dimensions))
        self.routes_started = 0
        self.stops_made = 0
        self.log: dict[str, list[NDArray[Any]]] = {}
        self.pending: list[NDArray[np.int64]] = []
        self.room = room is not None
        self.handled = room is not None and bool(np.isfinite(room.handling).any())
        self.slotted = room is not None and bool((room.slots < UNBOUNDED // 2).any())
        if room is not None:
            types = room.slots.shape[3]
            self.origin_room = np.broadcast_to(room.origin_handling, (rows,)).copy()
            self.node_room = np.broadcast_to(room.node_handling, (rows, batch.k)).copy()
            self.origin_slots = np.broadcast_to(room.origin_slots, (rows, types)).copy()
            self.node_slots = np.broadcast_to(room.node_slots, (rows, batch.k, types)).copy()
        else:
            need = (self.phase == NEED).nonzero()[0]
            if need.size == rows:
                self.opening = self._opening(need, self.of)
            else:
                self.opening = np.zeros((rows, n, k))
                self.opening[need] = self._opening(need, self.of[need])

    def _tables(self, batch: Batch) -> None:
        """Lay out the ``option_*`` and ``node_*`` tables."""
        count, n, k = batch.count, batch.n, batch.k
        self.option_capacity = batch.capacity.reshape(count * n, self.dimensions)
        self.option_slack = TOLERANCE * self.option_capacity
        self.option_primary = batch.capacity_primary.reshape(count * n)
        min_load = batch.min_share * batch.capacity_primary * (1 - TOLERANCE)
        self.option_min_load = min_load.reshape(count * n)
        self.option_owned = batch.owned.reshape(count * n)
        self.any_owned = bool(self.option_owned.any())
        self.option_line_end = batch.line_end.reshape(count * n)
        self.option_stopovers = batch.stopovers.reshape(count * n)
        self.option_vehicle_type = batch.vehicle_type.reshape(count * n)
        self.node_legs = batch.node_km.reshape(count * k, k)
        self.leg_km = batch.node_km.reshape(count * k * k)
        allowed = batch.allowed
        if self.any_owned:
            blocked = np.arange(k)[None, None, :] == batch.line_end[:, :, None]
            if self.forward_only:
                blocked |= batch.origin_km[:, None, :] >= batch.line_origin[:, :, None]
            allowed = allowed & ~(batch.owned[:, :, None] & blocked)
        self.option_allowed = allowed.reshape(count * n, k)
        if self.forward_only:
            ahead = batch.origin_km[:, None, :] > batch.origin_km[:, :, None]
            self.node_ahead = ahead.reshape(count * k, k)
        if self.max_stretch is not None:
            self.stretch = self.max_stretch * batch.origin_km * (1 + TOLERANCE)
            line_stretch = self.max_stretch * batch.line_origin * (1 + TOLERANCE)
            self.option_line_stretch = line_stretch.reshape(count * n)
            self.option_line_km = batch.line_km.reshape(count * n, k)

    def _limit(self, rows: NDArray[np.intp]) -> NDArray[np.float64]:
        """(rows, k): primary quantity that may still be loaded for each node."""
        return np.asarray(
            np.minimum(self.node_room[rows], self.origin_room[rows][:, None]), dtype=np.float64
        )

    def _open(self, rows: NDArray[np.intp], of: NDArray[np.int64]) -> NDArray[np.bool_]:
        """(rows, options, nodes): the room allows a new route of the option to the node.

        A room comes with one subproblem, whose options are the first ``n``
        entries of the ``option_*`` tables.
        """
        batch = self.batch
        result = np.ones((rows.size, batch.n, batch.k), dtype=np.bool_)
        if self.slotted:
            types = self.option_vehicle_type[: batch.n]
            origin_free = self.origin_slots[rows][:, types] > 0
            node_free = (self.node_slots[rows][:, :, types] > 0).transpose(0, 2, 1)
            line = self.option_line_end[: batch.n]
            line_free = node_free[:, np.arange(line.size), np.maximum(line, 0)] | (line < 0)
            result &= origin_free[:, :, None] & line_free[:, :, None] & node_free
        if self.handled:
            origin = (self.origin_room[rows] > MIN_ROOM)[:, None]
            result &= ((self.node_room[rows] > MIN_ROOM) & origin)[:, None, :]
        return result

    def record(self, name: str, values: NDArray[Any]) -> None:
        self.log.setdefault(name, []).append(values)

    def run(self) -> Colony:
        for _ in range(MAX_STEPS):
            need = (self.phase == NEED).nonzero()[0]
            if need.size:
                self._start(need)
            moving = (self.phase == ROUTE).nonzero()[0]
            if moving.size:
                self._extend(moving)
            if not need.size and not moving.size:
                return self._colony()
        raise RuntimeError("ant construction did not finish")

    def _opening(self, rows: NDArray[np.intp], of: NDArray[np.int64]) -> NDArray[np.float64]:
        """(rows, options, k): weight of opening a route of each option at each node.

        Without a room, ``_reopen`` is the same computation for single (row,
        node) cells and keeps ``opening`` equal to what this returns: a change to
        the mask or the carry here must be made there too.
        """
        batch, primary = self.batch, self.shared.primary
        remaining = self.remaining[rows]
        if self.split:
            loadable = (remaining.max(axis=2, initial=0.0) > 0)[:, None, :]
        else:
            items = self.item_left[rows]
            if self.handled:
                limit = self._limit(rows)[:, :, None] * (1 + TOLERANCE)
                items = items & (batch.item_quantity[of][..., primary] <= limit)
            loadable = (items[:, None] & batch.fits_new[of]).any(axis=-1)
        mask = loadable & (self.available[rows] > 0)[:, :, None]
        capacity = self.option_primary.reshape(batch.count, batch.n).take(of, axis=0)
        carry = np.minimum(remaining[:, None, :, primary], capacity[:, :, None])
        if self.room:
            mask &= self._open(rows, of)
        if self.handled:
            carry = np.minimum(carry, self._limit(rows)[:, None, :])
        return np.where(mask, self.first_weight[of] * carry**self.beta, 0.0)

    def _reopen(self) -> None:
        """Bring ``opening`` up to date at the cells visited since it last was.

        This is ``_opening`` without a room, cell by cell: the same mask (something
        left that a new vehicle can load, a vehicle of the option left) and the
        same carry, from the same values with the same operations, so that the
        weights it keeps are those ``_opening`` would compute. The two must
        change together.
        """
        cell = self.pending[0] if len(self.pending) == 1 else np.concatenate(self.pending)
        self.pending.clear()
        batch, primary = self.batch, self.shared.primary
        rows, nodes = np.divmod(cell, batch.k)
        of = self.of[rows]
        left = self.cells.take(cell, axis=0)
        if self.split and self.dimensions == 1:
            loadable = left > 0
        elif self.split:
            loadable = (left.max(axis=1, initial=0.0) > 0)[:, None]
        else:
            items = self.item_cells.take(cell, axis=0)[:, None]
            loadable = (items & batch.fits_new[of, :, nodes]).any(axis=-1)
        mask = loadable & (self.available.take(rows, axis=0) > 0)
        capacity = self.option_primary.reshape(batch.count, batch.n).take(of, axis=0)
        carry = np.minimum(left[:, primary][:, None], capacity)
        weight = self.first_weight[of, :, nodes] * carry**self.beta
        self.opening[rows, :, nodes] = np.where(mask, weight, 0.0)

    def _start(self, rows: NDArray[np.intp]) -> None:
        n, k = self.batch.n, self.batch.k
        of = self.of[rows]
        if self.room:
            weights = self._opening(rows, of)
        else:
            if self.pending:
                self._reopen()
            weights = self.opening.take(rows, axis=0)
        choice = sample(weights.reshape(rows.size, -1), self.rng, self.exploitation)

        done = choice < 0
        go = rows
        if done.any():
            self.phase[rows[done]] = DONE
            chosen = ~done
            go, of, choice = rows[chosen], of[chosen], choice[chosen]
            if not go.size:
                return
        option, node = np.divmod(choice, k)
        self.record("first_row", go)
        self.record("first_option", option)
        self.record("first_node", node)
        pick = go * n + option
        left = self.available_cells[pick] - 1
        self.available_cells[pick] = left
        if not self.room:
            spent = left <= 0
            if spent.any():
                self.opening[go[spent], option[spent]] = 0.0
        option_id = of * n + option
        if self.slotted:
            kind = self.option_vehicle_type[option_id]
            self.origin_slots[go, kind] -= 1
            owned = self.option_owned[option_id]
            line = self.option_line_end[option_id]
            self.node_slots[go[owned], line[owned], kind[owned]] -= 1
        self.route[go] = self.routes_started + np.arange(go.size)
        self.routes_started += go.size
        self.record("route_row", go)
        self.record("route_option", option)
        self.option_id[go] = option_id
        self.cap[go] = self.option_capacity.take(option_id, axis=0)
        self.load[go] = 0.0
        cell = go * k + node
        self.km[go] = self.origin_cells[cell]
        if self.max_stops is not None:
            self.stops[go] = 0
        self.open[go] = self.option_allowed.take(option_id, axis=0)
        full = self._visit(go, node, of, option_id, cell)
        at_line = (
            self.option_owned[option_id] & (node == self.option_line_end[option_id])
            if self.any_owned
            else None
        )
        self._finish(self._decide(go, option_id, full, at_line))

    def _extend(self, rows: NDArray[np.intp]) -> None:
        primary, beta, k = self.shared.primary, self.beta, self.batch.k
        of, option_id, here = self.of[rows], self.option_id[rows], self.last[rows]
        at = of * k + here
        cap, remaining = self.cap.take(rows, axis=0), self.remaining.take(rows, axis=0)
        if self.split:
            if self.dimensions == 1:
                loadable = remaining[:, :, 0] > 0
            else:
                loadable = remaining.max(axis=2, initial=0.0) > 0
            full = cap <= self.option_slack.take(option_id, axis=0)
            if full.any():
                loadable &= ~((remaining > 0) & full[:, None, :]).any(axis=2)
        else:
            room = (cap + self.option_slack.take(option_id, axis=0))[:, None, None, :]
            quantity = self.batch.item_quantity.take(of, axis=0)
            fits = self.item_left.take(rows, axis=0) & np.all(quantity <= room, axis=-1)
            if self.handled:
                limit = self._limit(rows)[:, :, None] * (1 + TOLERANCE)
                fits &= quantity[..., primary] <= limit
            loadable = fits.any(axis=-1)
        candidate = loadable & self.open.take(rows, axis=0)
        if self.slotted:
            kind = self.option_vehicle_type[option_id]
            candidate &= self.node_slots[rows[:, None], np.arange(k)[None, :], kind[:, None]] > 0
        if self.handled:
            handled = self.node_room[rows] > MIN_ROOM
            candidate &= handled & (self.origin_room[rows] > MIN_ROOM)[:, None]
        if self.forward_only:
            candidate &= self.node_ahead.take(at, axis=0)
        if self.max_stretch is not None:
            new_km = self.km[rows][:, None] + self.node_legs.take(at, axis=0)
            within = new_km <= self.stretch.take(of, axis=0)
            if self.any_owned:
                owned = self.option_owned[option_id].nonzero()[0]
                if owned.size:
                    line = option_id[owned]
                    within[owned] = (
                        new_km[owned] + self.option_line_km[line]
                        <= self.option_line_stretch[line][:, None]
                    )
            candidate &= within

        reach = np.minimum(remaining[:, :, primary], cap[:, primary][:, None])
        if self.handled:
            reach = np.minimum(reach, self._limit(rows))
        fill = reach / self.option_primary[option_id][:, None]
        step_weight = self.step_weight.take(option_id * k + here, axis=0)
        weights = np.empty((rows.size, k + 1))
        weights[:, :k] = np.where(candidate, step_weight * fill**beta, 0.0)
        underfilled = self.load.take(rows, axis=0)[:, primary] < self.option_min_load[option_id]
        weights[:, k] = self.tau_end[at] * self.end_factor[underfilled.view(np.int8)]
        choice = sample(weights, self.rng, self.exploitation)

        ending = choice == k
        go, node, ended = rows, choice, None
        if ending.any():
            ended = rows[ending]
            self.record("end_row", ended)
            self.record("end_node", here[ending])
            going = ~ending
            go, node = rows[going], choice[going]
            if not go.size:
                self._finish(ended)
                return
            of, option_id, here, at = of[going], option_id[going], here[going], at[going]
        self.record("step_row", go)
        self.record("step_from", here)
        self.record("step_to", node)
        self.km[go] += self.leg_km[at * k + node]
        full = self._visit(go, node, of, option_id, go * k + node)
        closing = self._decide(go, option_id, full)
        self._finish(closing if ended is None else np.concatenate([ended, closing]))

    def _decide(
        self,
        rows: NDArray[np.intp],
        option_id: NDArray[np.int64],
        full: NDArray[np.bool_],
        at_line: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.intp]:
        """After a visit: keep the route open, or return the rows whose route ends."""
        keep = self.option_stopovers[option_id] & ~full
        if at_line is not None:
            keep &= ~at_line
        if self.max_stops is not None:
            keep &= self.stops[rows] + self.option_owned[option_id] < self.max_stops
        self.phase[rows[keep]] = ROUTE
        return rows[~keep]

    def _finish(self, rows: NDArray[np.intp]) -> None:
        """Close the routes of ``rows``; owned routes first go on to their line end."""
        if not rows.size:
            return
        if self.any_owned:
            k = self.batch.k
            option_id, last = self.option_id[rows], self.last[rows]
            line = self.option_line_end[option_id]
            detour = self.option_owned[option_id] & (last != line)
            if detour.any():
                go, to = rows[detour], line[detour]
                of = self.of[go]
                self.km[go] += self.leg_km[(of * k + last[detour]) * k + to]
                self._visit(go, to, of, option_id[detour], go * k + to)
        self.record("close_route", self.route[rows])
        self.record("close_km", self.km[rows])
        self.record("close_load", self.load.take(rows, axis=0))
        self.phase[rows] = NEED

    def _visit(
        self,
        rows: NDArray[np.intp],
        nodes: NDArray[np.int64],
        of: NDArray[np.int64],
        option_id: NDArray[np.int64],
        cell: NDArray[np.int64],
    ) -> NDArray[np.bool_]:
        """Load at ``nodes`` what fits, and record the stops; return where the vehicle is full."""
        batch, primary = self.batch, self.shared.primary
        slack = self.option_slack.take(option_id, axis=0)
        cap = self.cap.take(rows, axis=0)
        left = self.cells.take(cell, axis=0)
        if self.handled:
            limit = np.minimum(self.node_room[rows, nodes], self.origin_room[rows])
        if self.split:
            ratio = self.ratio[: rows.size]
            ratio.fill(np.inf)
            np.divide(cap, left, out=ratio, where=left > 0)
            if self.dimensions == 1:
                share = np.minimum(ratio[:, 0], 1.0)
            else:
                share = ratio.min(axis=1, initial=1.0)
            if self.handled:
                own = left[:, primary]
                room = np.maximum(limit, 0.0) / np.where(own > 0, own, 1.0)
                share = np.minimum(share, np.where(own > 0, room, np.inf))
            amount = share[:, None] * left
        else:
            stop_ids = self.stops_made + np.arange(rows.size)
            amount = np.zeros_like(cap)
            for item in range(batch.item_valid.shape[2]):
                quantity = batch.item_quantity[of, nodes, item]
                fits = self.item_cells[cell, item] & np.all(
                    quantity <= cap - amount + slack, axis=1
                )
                if self.handled:
                    fits &= quantity[:, primary] <= (limit - amount[:, primary]) + (
                        TOLERANCE * quantity[:, primary]
                    )
                if fits.any():
                    self.item_cells[cell[fits], item] = False
                    amount[fits] += quantity[fits]
                    self.record("item_stop", stop_ids[fits])
                    self.record("item_slot", np.full(int(fits.sum()), item, dtype=np.int64))
        left = np.maximum(left - amount, 0.0)
        left[left <= self.settled.take(cell, axis=0)] = 0.0
        self.cells[cell] = left
        cap = cap - amount
        full = cap <= slack
        cap[full] = 0.0
        self.cap[rows] = cap
        self.load[rows] = self.load.take(rows, axis=0) + amount
        if self.slotted:
            kind = self.option_vehicle_type[option_id]
            reserved = self.option_owned[option_id] & (nodes == self.option_line_end[option_id])
            self.node_slots[rows[~reserved], nodes[~reserved], kind[~reserved]] -= 1
        if self.handled:
            self.node_room[rows, nodes] -= amount[:, primary]
            self.origin_room[rows] -= amount[:, primary]
        if not self.room:
            self.pending.append(cell)
        if self.max_stops is not None:
            self.stops[rows] += 1
        self.last[rows] = nodes
        self.open_cells[cell] = False
        self.stops_made += rows.size
        self.record("stop_route", self.route[rows])
        self.record("stop_node", nodes)
        self.record("stop_amount", amount)
        return full[:, 0] if self.dimensions == 1 else full.all(axis=1)

    def _colony(self) -> Colony:
        dimensions = self.dimensions
        rows = self.of.size

        def get(name: str, empty: tuple[int, ...], dtype: type[np.generic]) -> NDArray[Any]:
            parts = self.log.get(name)
            return np.concatenate(parts).astype(dtype) if parts else np.zeros(empty, dtype=dtype)

        def choices(*names: str) -> NDArray[np.int64]:
            return np.column_stack([get(name, (0,), np.int64) for name in names])

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
            first=choices("first_row", "first_option", "first_node"),
            step=choices("step_row", "step_from", "step_to"),
            end=choices("end_row", "end_node"),
            item_left=self.item_left,
        )


def scores(
    batch: Batch,
    colony: Colony,
    coefficients: Mapping[str, float],
    shared: Shared,
    room: Room | None = None,
    *,
    waiting: bool = True,
) -> NDArray[np.float64]:
    """(subproblems, ants): the linear objective of every solution, without constants.

    On scheduled instances the routes are timed as the evaluator times them:
    hourly costs join the dispatch costs, lateness joins ``service``, and only
    demand that may not wait for the next day counts as unserved. Demand that
    may wait costs its :attr:`~pirilib.solvers.decompose.Subproblem.defer_cost`
    under ``service``, unless ``waiting`` is False (it is then planned, and paid
    for, the next day). With a ``room`` (one subproblem), handling and vehicles
    beyond it are violations.
    """
    rows = colony.remaining.shape[0]
    of = colony.route_row // colony.ants
    option = colony.route_option
    kind = batch.kind[of, option]
    spot = kind == SPOT
    cost = batch.fixed[of, option] + batch.per_km[of, option] * colony.route_km
    timing = timeline(batch, colony, shared) if shared.scheduled else None
    if timing is not None:
        cost = cost + batch.per_hour[of, option] * timing.hours
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
    if timing is not None:
        remaining, deferred = _waiting(batch, colony, shared)
        service = per_row(timing.service, every)
        features["service"] = service + deferred if waiting else service
    if room is not None and timing is not None:
        handling, vehicles = room.excess(timing, rows)
        features["violation[handling]"] = handling
        features["violation[vehicle_limit]"] = vehicles
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


def _waiting(
    batch: Batch, colony: Colony, shared: Shared
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per row: the uncarried demand that may not wait, and the cost of the rest waiting."""
    primary = shared.primary
    of = np.arange(colony.remaining.shape[0]) // colony.ants
    if shared.rules.split_demand:
        final = colony.remaining * batch.final_share[of]
        cost = np.sum(colony.remaining[..., primary] * batch.node_defer[of], axis=1)
        return final, cost
    waits = colony.item_left & ~batch.item_final[of]
    quantity = batch.item_quantity[of]
    final = np.clip(colony.remaining - np.sum(quantity * waits[..., None], axis=2), 0.0, None)
    cost = np.sum(quantity[..., primary] * waits * batch.defer_cost[of], axis=(1, 2))
    return final, cost

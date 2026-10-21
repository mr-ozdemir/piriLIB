"""The reduction of a subproblem to an open vehicle routing problem.

External routing solvers do not split a destination's demand over vehicles,
have no minimum load and know nothing of pirilib's rules, so a subproblem is
reduced to a problem they do know before they see it:

- **Full vehicles first.** While the demand of a destination does not fit into
  any vehicle left that may serve it, a full vehicle goes straight there, as in
  :class:`~pirilib.solvers.direct.DirectShipping`. The rest becomes one client
  that is visited once. Splits are therefore limited to these full trips.
- **One-way routes.** Every vehicle option is a vehicle type that starts at
  the origin. Spot types end at a depot of their own that every client reaches
  at no cost; owned types end at their line's destination.
- **Rules as arcs.** A banned stop, a step backwards (``forward_only``), a stop
  after an owned line's destination and a second stop where stopovers are not
  allowed are arcs left out of the type's matrix (they get
  :data:`FORBIDDEN` distance and duration).
- **Stretch as duration.** The duration of an arc is its length. Owned types
  may last ``max_stretch`` times the direct distance to the line's
  destination. For spot types the arc from the last stop to the end lasts
  :data:`HORIZON` minus ``max_stretch`` times the direct distance to that
  stop, and routes may last :data:`HORIZON`, which is the stretch rule up to
  rounding to a millimetre.
- **Stops as a load.** With ``max_stops``, every client loads one unit of an
  extra dimension. Owned types get ``max_stops - 1`` of it, because their
  line's destination is a stop even when it is not a client; this is stricter
  than the rule when the destination is a client.
- **Objective.** Fixed cost and cost per km of a type come from the linear
  form of the objective (cost, km and dispatch terms). A client is optional
  unless an unlimited spot offer can carry it straight. The prize of an
  optional client is the objective's value of leaving its demand uncarried,
  at most :data:`PRIZE_TRIPS` times its dearest direct trip, so that the
  external solver's penalties stay in scale. Routes the external solver
  reports as infeasible are dropped; their clients stay uncarried.

Quantities, distances and costs are scaled to integers; capacities are
rounded down and loads up, so a route the external solver keeps within
capacity also fits in pirilib. The plan is built from the routes and checked
by :func:`~pirilib.evaluate` like any other.

Not supported (the adapters raise :class:`ValueError`): demand that may not be
split, minimum fill, times and the other scheduling rules, objectives without
a linear form or with unused-capacity terms, and constraints of
your own.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from pirilib.constraints import Constraint
from pirilib.objective import Objective, linear_form
from pirilib.plan import OWNED, SPOT
from pirilib.solvers._colony import Shared
from pirilib.solvers._routing import route_value, unserved_value
from pirilib.solvers._sequential import RouteList, SequentialSolver
from pirilib.solvers.base import linear_coefficients
from pirilib.solvers.decompose import UNBOUNDED, Routes, Subproblem
from pirilib.solvers.direct import DirectPlanner
from pirilib.spec.instance import LineHaulInstance

QUANTITY_SCALE: Final = 1000
"""Integer load units per quantity unit."""
DISTANCE_SCALE: Final = 100
"""Integer distance units per km, for costs."""
DURATION_SCALE: Final = 1_000_000
"""Integer duration units per km, for the stretch rule."""
COST_SCALE: Final = 10_000
"""Integer cost units per unit of the objective."""
HORIZON: Final = 1 << 36
"""Longest route duration, in duration units (about 68 000 km)."""
FORBIDDEN: Final = 1 << 37
"""Distance and duration of arcs a vehicle type may not use."""
MAX_PRIZE: Final = 1 << 50
"""Upper bound on the prize of one client."""
PRIZE_TRIPS: Final = 100
"""A prize is at most this many times the dearest direct trip to the client."""
_EPSILON: Final = 1e-12
_UNSUPPORTED_FEATURES: Final = ("unused[owned]", "unused[spot]")


@dataclass(frozen=True, eq=False, slots=True)
class RoutingModel:
    """An open vehicle routing problem with a heterogeneous fleet, all in integers.

    Locations are numbered: 0 is the origin, ``1 .. clients`` are the
    clients, and ``clients + 1 + t`` is where vehicle type ``t`` ends.

    Attributes:
        client_node: (c,) subproblem node of each client
        client_share: (c,) share of the node's demand the client carries
        delivery: (c, L) load per client and load dimension
        prize: (c,) value of serving an optional client
        required: (c,) the client must be served
        option: (T,) subproblem option of each vehicle type
        count: (T,) vehicles of the type
        capacity: (T, L)
        fixed_cost: (T,) per vehicle used
        unit_cost: (T,) per distance unit
        max_duration: (T,) longest route, in duration units
        distance: (T, N, N) distance units, FORBIDDEN where the type may not go
        duration: (T, N, N) duration units, FORBIDDEN where the type may not go
    """

    client_node: NDArray[np.int64]
    client_share: NDArray[np.float64]
    delivery: NDArray[np.int64]
    prize: NDArray[np.int64]
    required: NDArray[np.bool_]
    option: NDArray[np.int64]
    count: NDArray[np.int64]
    capacity: NDArray[np.int64]
    fixed_cost: NDArray[np.int64]
    unit_cost: NDArray[np.int64]
    max_duration: NDArray[np.int64]
    distance: NDArray[np.int64]
    duration: NDArray[np.int64]

    @property
    def clients(self) -> int:
        return int(self.client_node.size)

    @property
    def types(self) -> int:
        return int(self.option.size)

    @property
    def locations(self) -> int:
        return 1 + self.clients + self.types

    def end(self, vehicle_type: int) -> int:
        """Location where vehicles of ``vehicle_type`` end."""
        return 1 + self.clients + vehicle_type

    def visitable(self, vehicle_type: int) -> NDArray[np.bool_]:
        """(c,) clients that vehicles of the type can reach at all."""
        inbound = self.distance[vehicle_type, :, 1 : 1 + self.clients] < FORBIDDEN
        return np.asarray(inbound.any(axis=0), dtype=np.bool_)


@dataclass(frozen=True, slots=True)
class RoutingSolution:
    """Routes found by an external solver.

    Attributes:
        routes: (vehicle type, clients in visiting order) per vehicle used
        feasible: The solver reports its solution as feasible
    """

    routes: tuple[tuple[int, tuple[int, ...]], ...]
    feasible: bool


class Reduction:
    """Builds the :class:`RoutingModel` of one subproblem; see the module documentation."""

    def __init__(
        self,
        problem: Subproblem,
        shared: Shared,
        coefficients: Mapping[str, float],
        available: NDArray[np.int64],
    ) -> None:
        self.problem = problem
        self.shared = shared
        self.coefficients = coefficients
        self.planner = DirectPlanner(problem, shared, available)

    def single(self, option: int) -> bool:
        """Vehicles of ``option`` serve one destination per route."""
        limit = self.shared.rules.stopover.max_stops
        return not bool(self.problem.stopovers[option]) or limit == 1

    def reaches(self, option: int, node: int) -> bool:
        """``option`` may serve ``node`` on some route, ignoring distances."""
        problem = self.problem
        if not problem.allowed[option, node]:
            return False
        if problem.kind[option] == OWNED and self.single(option):
            return bool(problem.line_end[option] == node)
        return True

    def routable(self, node: int, load: NDArray[np.float64]) -> bool:
        """Some vehicle left may carry ``load`` to ``node`` on a route."""
        return any(
            self.planner.available[option] > 0
            and self.reaches(option, node)
            and self.planner.fits(option, load)
            for option in range(self.problem.options)
        )

    def guaranteed(self, node: int, load: NDArray[np.float64]) -> bool:
        """An unlimited spot offer may carry ``load`` straight to ``node``."""
        problem = self.problem
        return any(
            problem.kind[option] == SPOT
            and self.planner.available[option] >= UNBOUNDED
            and problem.allowed[option, node]
            and self.planner.fits(option, load)
            for option in range(problem.options)
        )

    def build(self, routes: RouteList) -> RoutingModel | None:
        """Send the full vehicles to ``routes``; return the model of the rest, if any."""
        problem = self.problem
        nodes: list[int] = []
        shares: list[float] = []
        for node in self.planner.nodes():
            total = problem.demand[node]
            remaining = 1.0
            while not self.routable(node, total * remaining):
                before = remaining
                remaining = self.planner.full_truck(node, remaining, routes)
                if remaining <= _EPSILON or remaining >= before:
                    break
            if remaining > _EPSILON and self.routable(node, total * remaining):
                nodes.append(node)
                shares.append(remaining)
        types = [option for option in range(problem.options) if self.planner.available[option] > 0]
        if not nodes or not types:
            return None
        return self._model(np.array(nodes, dtype=np.int64), np.array(shares), types)

    def _model(
        self, nodes: NDArray[np.int64], shares: NDArray[np.float64], types: Sequence[int]
    ) -> RoutingModel:
        problem, shared = self.problem, self.shared
        rules = shared.rules.stopover
        longest = float(problem.origin_km.max()) * (rules.max_stretch or 1.0) * 4
        if longest * DURATION_SCALE >= HORIZON:
            raise ValueError("distances are too long for the integer routing model")
        clients = nodes.size
        loads = problem.demand[nodes] * shares[:, None]
        counting = rules.max_stops is not None and rules.max_stops > 1
        delivery = np.ceil(np.round(loads * QUANTITY_SCALE, 6)).astype(np.int64)
        if counting:
            delivery = np.column_stack([delivery, np.ones(clients, dtype=np.int64)])
        required = np.array(
            [self.guaranteed(int(node), load) for node, load in zip(nodes, loads, strict=True)],
            dtype=np.bool_,
        )
        option = np.array(types, dtype=np.int64)
        count = np.array(
            [
                clients
                if self.planner.available[o] >= UNBOUNDED
                else min(int(self.planner.available[o]), clients)
                for o in types
            ],
            dtype=np.int64,
        )
        capacity = np.floor(np.round(problem.capacity[option] * QUANTITY_SCALE, 6)).astype(np.int64)
        if counting:
            assert rules.max_stops is not None
            stops = [
                rules.max_stops - 1 if problem.kind[o] == OWNED else rules.max_stops for o in types
            ]
            capacity = np.column_stack([capacity, np.array(stops, dtype=np.int64)])
        dimensions = problem.demand.shape[1]
        empty = np.zeros(dimensions)
        fixed = np.array(
            [route_value(problem, self.coefficients, shared, o, 0.0, empty) for o in types]
        )
        per_km = (
            np.array(
                [route_value(problem, self.coefficients, shared, o, 1.0, empty) for o in types]
            )
            - fixed
        )
        size = 1 + clients + len(types)
        distance = np.full((len(types), size, size), FORBIDDEN, dtype=np.int64)
        duration = np.full((len(types), size, size), FORBIDDEN, dtype=np.int64)
        max_duration = np.full(len(types), HORIZON, dtype=np.int64)
        for t, o in enumerate(types):
            end = 1 + clients + t
            self._arcs(o, nodes, distance[t], duration[t], end)
            if problem.kind[o] == OWNED and rules.max_stretch is not None:
                line = int(problem.line_end[o])
                max_duration[t] = round(
                    rules.max_stretch * float(problem.origin_km[line]) * DURATION_SCALE
                )
        fixed_cost = np.array([round(value * COST_SCALE) for value in fixed], dtype=np.int64)
        unit_cost = np.array(
            [round(value * COST_SCALE / DISTANCE_SCALE) for value in per_km], dtype=np.int64
        )
        trips = fixed_cost[:, None] + unit_cost[:, None] * distance[:, 0, 1 : 1 + clients]
        trips = np.where(distance[:, 0, 1 : 1 + clients] < FORBIDDEN, trips, 0)
        prize = np.zeros(clients, dtype=np.int64)
        for position, node in enumerate(nodes):
            if required[position]:
                continue
            remaining = np.zeros_like(problem.demand)
            remaining[node] = loads[position]
            value = unserved_value(problem, self.coefficients, shared, remaining) * COST_SCALE
            cap = PRIZE_TRIPS * int(trips[:, position].max(initial=0))
            prize[position] = max(0, min(round(value), cap or MAX_PRIZE, MAX_PRIZE))
        return RoutingModel(
            client_node=nodes,
            client_share=shares,
            delivery=delivery,
            prize=prize,
            required=required,
            option=option,
            count=count,
            capacity=capacity,
            fixed_cost=fixed_cost,
            unit_cost=unit_cost,
            max_duration=max_duration,
            distance=distance,
            duration=duration,
        )

    def _arcs(
        self,
        option: int,
        nodes: NDArray[np.int64],
        distance: NDArray[np.int64],
        duration: NDArray[np.int64],
        end: int,
    ) -> None:
        """Fill the matrices of one vehicle type with the arcs it may use."""
        problem, rules = self.problem, self.shared.rules.stopover
        owned = problem.kind[option] == OWNED
        line = int(problem.line_end[option])
        single = self.single(option)
        origin_km = problem.origin_km

        def put(start: int, stop: int, km: float, minutes: float | None = None) -> None:
            distance[start, stop] = round(km * DISTANCE_SCALE)
            duration[start, stop] = round((km if minutes is None else minutes) * DURATION_SCALE)

        for location in range(distance.shape[0]):
            distance[location, location] = 0
            duration[location, location] = 0
        put(0, end, 0.0)
        for i, node in enumerate(nodes):
            node = int(node)
            if not self.reaches(option, node):
                continue
            put(0, 1 + i, float(origin_km[node]))
            if owned:
                if node == line:
                    put(1 + i, end, 0.0)
                elif not rules.forward_only or origin_km[line] > origin_km[node]:
                    put(1 + i, end, float(problem.node_km[node, line]))
            elif rules.max_stretch is None:
                put(1 + i, end, 0.0, 0.0)
            else:
                distance[1 + i, end] = 0
                duration[1 + i, end] = HORIZON - round(
                    rules.max_stretch * float(origin_km[node]) * DURATION_SCALE
                )
            if single or (owned and node == line):
                continue
            for j, after in enumerate(nodes):
                after = int(after)
                if after == node or not self.reaches(option, after):
                    continue
                if rules.forward_only and origin_km[after] <= origin_km[node]:
                    continue
                put(1 + i, 1 + j, float(problem.node_km[node, after]))


class RoutingAdapter(SequentialSolver):
    """Base of the adapters to external routing solvers.

    Subclasses load their package in :meth:`load` and solve a
    :class:`RoutingModel` in :meth:`route`. Everything else, from the
    reduction to the plan, is shared; see the module documentation.
    """

    mode = "routing"
    package = ""
    extra = ""

    def load(self) -> None:
        """Import the external package; raise :class:`ImportError` with a hint if it is missing."""
        raise NotImplementedError

    def route(self, model: RoutingModel, seed: int) -> RoutingSolution:
        raise NotImplementedError

    def missing(self, error: ImportError) -> ImportError:
        return ImportError(
            f"{type(self).__name__} needs the {self.package} package: "
            f'pip install "pirilib[{self.extra}]" ({error})'
        )

    def prepare(
        self, instance: LineHaulInstance, objective: Objective, constraints: Sequence[Constraint]
    ) -> Mapping[str, float]:
        self.load()
        name = type(self).__name__
        form = linear_form(objective)
        problems: list[str] = []
        if form is None:
            problems.append("objectives without a linear form")
        if constraints:
            problems.append("constraints of your own")
        if instance.scheduled:
            problems.append("times, deferral, handling capacity or vehicle limits")
        if not instance.rules.split_demand:
            problems.append("rules.split_demand = false")
        if instance.rules.min_fill.spot > 0 or instance.rules.min_fill.owned > 0:
            problems.append("minimum fill (rules.min_fill)")
        coefficients = linear_coefficients(instance, form) if form is not None else {}
        used = [feature for feature in _UNSUPPORTED_FEATURES if coefficients.get(feature)]
        if used:
            problems.append(f"objective terms {used}")
        negative = [
            feature
            for feature, value in coefficients.items()
            if value < 0 and feature.split("[")[0] in ("cost", "km", "dispatches")
        ]
        if negative:
            problems.append(f"negative weights of {negative}")
        if problems:
            raise ValueError(f"{name} does not support " + "; ".join(problems))
        return coefficients

    def plan_subproblem(
        self,
        problem: Subproblem,
        shared: Shared,
        coefficients: Mapping[str, float],
        available: NDArray[np.int64],
        seed: int,
        stats: dict[str, float | int | str],
    ) -> Routes:
        routes = RouteList(problem)
        model = Reduction(problem, shared, coefficients, available).build(routes)
        if model is None:
            return routes.build()
        solution = self.route(model, seed)
        if not solution.feasible:
            stats["infeasible"] = int(stats.get("infeasible", 0)) + 1
        dimensions = problem.demand.shape[1]
        for vehicle_type, visits in solution.routes:
            if not visits:
                continue
            option = int(model.option[vehicle_type])
            stops = [int(model.client_node[client]) for client in visits]
            amounts = [
                problem.demand[model.client_node[client]] * model.client_share[client]
                for client in visits
            ]
            if problem.kind[option] == OWNED and stops[-1] != problem.line_end[option]:
                stops.append(int(problem.line_end[option]))
                amounts.append(np.zeros(dimensions))
            routes.add(option, stops, amounts)
        return routes.build()

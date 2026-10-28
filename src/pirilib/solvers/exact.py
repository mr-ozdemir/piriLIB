"""Exact optimization of small subproblems by mixed-integer programming.

Needs scipy (``pip install "pirilib[exact]"``), whose HiGHS solver does the work.
Every feasible route of a subproblem becomes a column: an integer number of
vehicles on the route, and the continuous share of each stop's demand they
deliver. Identical vehicles on one route share its loads evenly, so the program
is exact for the linear objective, minimum fill included.

On scheduled instances the subproblems come in the order of the
:class:`~pirilib.solvers.decompose.Coordinator`, against the room the
:class:`~pirilib.solvers.ledger.Ledger` leaves them. Every route is timed as the
evaluator times it, so its hourly cost, the lateness of each share it delivers
and the days it uses each center are fixed per column. Handling and vehicles
beyond the room are slack columns at their violation weight, and demand that may
wait costs its deferral estimate. This is exact only while loading and unloading
take no time (``rules.handling.minutes_per_unit = 0``); with handling time, the
times of a route depend on its load.

The result is optimal per subproblem. Subproblems share limited spot offers,
and on scheduled instances the handling capacity and vehicle slots of centers,
one after another, so with such limits the whole plan may not be optimal.
Demand that may not be split is not supported.
"""

import ctypes
import ctypes.util
import itertools
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType

import numpy as np
from numpy.typing import NDArray

from pirilib.constraints import Constraint
from pirilib.evaluation import TOLERANCE, evaluate
from pirilib.objective import Objective, linear_form
from pirilib.plan import OWNED, SPOT
from pirilib.solvers._colony import Batch, Colony, Shared, scores
from pirilib.solvers._routing import route_km, route_ok, route_value
from pirilib.solvers._timing import late_hours, route_timing, timeline
from pirilib.solvers.base import SolveResult, linear_coefficients, require_objective
from pirilib.solvers.decompose import (
    MINUTES_PER_DAY,
    UNBOUNDED,
    Coordinator,
    PlanBuilder,
    Routes,
    Subproblem,
    subproblems,
)
from pirilib.solvers.ledger import Ledger, Room
from pirilib.spec.instance import UNLIMITED, LineHaulInstance


def _flush_c_streams() -> None:
    """Flush the C runtime's buffered output, so it lands where it was written."""
    for name in ("ucrtbase", "msvcrt", None):
        try:
            library = ctypes.CDLL(name) if name else ctypes.CDLL(ctypes.util.find_library("c"))
            library.fflush(None)
        except (OSError, AttributeError, TypeError):
            continue


class QuietNative:
    """Sends what native code writes to standard output to the null device while active.

    HiGHS prints diagnostic lines from C++ in some versions; they bypass Python's
    ``sys.stdout``, so the file descriptor itself is redirected.
    """

    def __init__(self) -> None:
        self._saved: int | None = None
        self._descriptor = 1

    def __enter__(self) -> "QuietNative":
        try:
            sys.stdout.flush()
            descriptor = sys.stdout.fileno()
        except (AttributeError, OSError, ValueError):
            return self
        self._saved = os.dup(descriptor)
        sink = os.open(os.devnull, os.O_WRONLY)
        os.dup2(sink, descriptor)
        os.close(sink)
        self._descriptor = descriptor
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        if self._saved is not None:
            _flush_c_streams()
            os.dup2(self._saved, self._descriptor)
            os.close(self._saved)
            self._saved = None


@dataclass(frozen=True, slots=True)
class SubproblemResult:
    """The optimal routes of one subproblem.

    Attributes:
        value: Objective without constants
    """

    routes: Routes
    value: float
    optimal: bool
    columns: int


class ExactSolver:
    """Solve every subproblem to optimality; for small subproblems and for benchmarks."""

    def __init__(self, max_nodes: int = 9, time_limit: float | None = None) -> None:
        self.max_nodes = max_nodes
        self.time_limit = time_limit

    def solve(
        self,
        instance: LineHaulInstance,
        objective: Objective,
        constraints: Sequence[Constraint] = (),
        *,
        seed: int = 0,
    ) -> SolveResult:
        goal = require_objective(objective, "ExactSolver.solve")
        form = linear_form(goal)
        if form is None or constraints:
            raise ValueError(
                "ExactSolver needs an objective with a linear form and no own constraints"
            )
        if instance.scheduled and instance.rules.handling.minutes_per_unit > 0:
            raise ValueError(
                "ExactSolver plans scheduled instances only when handling takes no time "
                "(rules.handling.minutes_per_unit = 0); use ACOSolver"
            )
        if not instance.rules.split_demand:
            raise ValueError("ExactSolver does not support rules.split_demand = false")
        coefficients = linear_coefficients(instance, form)
        started = time.perf_counter()
        shared = Shared.of(instance)
        limits = np.where(
            instance.spot_available == UNLIMITED, UNBOUNDED, instance.spot_available
        ).astype(np.int64)
        spot_left = {day: limits.copy() for day in range(len(instance.days))}
        builder = PlanBuilder(instance)
        optimal, columns, score = True, 0, 0.0
        ledger = Ledger(instance) if instance.scheduled else None
        coordinator = Coordinator(instance) if instance.scheduled else None
        for problem in coordinator or subproblems(instance):
            available = problem.available.copy()
            spot = problem.kind == SPOT
            left = spot_left[problem.day]
            available[spot] = np.minimum(available[spot], left[problem.source[spot]])
            room = ledger.room(problem) if ledger is not None else None
            result = solve_subproblem(
                problem, coefficients, shared, available, self.max_nodes, self.time_limit, room
            )
            builder.add(problem, result.routes)
            used = np.bincount(result.routes.option, minlength=problem.options)
            left[problem.source[spot]] -= used[spot]
            optimal &= result.optimal
            columns += result.columns
            if ledger is not None and coordinator is not None:
                score += _settle(problem, result.routes, coefficients, shared, room, ledger)
                coordinator.defer(problem, *problem.leftover(result.routes, True))
            else:
                score += result.value
        plan = builder.build()
        evaluation = evaluate(instance, plan)
        return SolveResult(
            plan=plan,
            evaluation=evaluation,
            objective=float(goal(evaluation)),
            seconds=time.perf_counter() - started,
            stats={
                "mode": "exact",
                "optimal": str(optimal),
                "columns": columns,
                "score": score + coefficients.get("owned_fixed", 0.0) * evaluation.owned_fixed_cost,
            },
        )


def candidate_routes(
    problem: Subproblem, shared: Shared, max_nodes: int
) -> list[tuple[int, tuple[int, ...]]]:
    """Every (option, stops) route that keeps the rules, except capacity and minimum fill."""
    loaded = [node for node in range(problem.size) if problem.demand[node].max() > 0]
    if len(loaded) > max_nodes:
        raise ValueError(
            f"subproblem (day {problem.day}, origin {problem.origin}) has {len(loaded)} "
            f"destinations, more than max_nodes={max_nodes}"
        )
    rules = shared.rules.stopover
    routes: list[tuple[int, tuple[int, ...]]] = []
    for option in range(problem.options):
        owned = problem.kind[option] == OWNED
        line = int(problem.line_end[option])
        body = [node for node in loaded if problem.allowed[option, node] and node != line]
        longest = problem.size if problem.stopovers[option] else 1
        if rules.max_stops is not None:
            longest = min(longest, rules.max_stops)
        sizes = range(longest) if owned else range(1, longest + 1)
        for count in sizes:
            for chosen in itertools.combinations(body, count):
                if rules.forward_only:
                    orders = [
                        tuple(sorted(chosen, key=lambda node: float(problem.origin_km[node])))
                    ]
                else:
                    orders = list(itertools.permutations(chosen))
                for order in orders:
                    stops = (*order, line) if owned else order
                    if route_ok(problem, shared, option, stops):
                        routes.append((option, stops))
    return routes


Cell = tuple[float, list[tuple[int, float]]]
"""The room left in one center cell and the (column, amount) entries that use it."""


@dataclass(frozen=True, eq=False, slots=True)
class ColumnTimes:
    """When the vehicles of every candidate route run, if handling takes no time.

    Days count from the first day of the room and stop at its last one.

    Attributes:
        hours: (routes,) from the start of loading to the end of the last unloading
        late: Per (route, node), the lateness penalty of the route carrying all of
            the node's demand
        start: (routes,) day loading starts at the origin
        arrival: Per (route, node), day the route reaches the stop
    """

    hours: NDArray[np.float64]
    late: dict[tuple[int, int], float]
    start: NDArray[np.int64]
    arrival: dict[tuple[int, int], int]

    @classmethod
    def of(
        cls,
        problem: Subproblem,
        shared: Shared,
        routes: Sequence[tuple[int, tuple[int, ...]]],
        room: Room | None,
    ) -> "ColumnTimes":
        """Untimed instances get zero hours and no lateness."""
        hours = np.zeros(len(routes))
        start_day = np.zeros(len(routes), dtype=np.int64)
        late: dict[tuple[int, int], float] = {}
        arrival: dict[tuple[int, int], int] = {}
        if not shared.scheduled:
            return cls(hours, late, start_day, arrival)
        first = problem.day if room is None else room.day
        last = 0 if room is None else room.handling.shape[2] - 1

        def day_of(minute: float) -> int:
            return min(max(math.floor(minute / MINUTES_PER_DAY) - first, 0), last)

        primary = shared.primary
        for r, (option, stops) in enumerate(routes):
            carried = [stop for stop in stops if problem.demand[stop].max() > 0]
            ready = max((float(problem.node_ready[stop]) for stop in carried), default=-math.inf)
            unloads = [0.0] * len(stops)
            start, finishes = route_timing(problem, shared, option, stops, unloads, ready)
            hours[r] = (finishes[-1] - start) / 60.0
            start_day[r] = day_of(start)
            for stop, finish in zip(stops, finishes, strict=True):
                arrival[(r, stop)] = day_of(finish)
                if stop in carried:
                    overdue = late_hours(finish - problem.item_due[stop])
                    late[(r, stop)] = shared.penalty * float(
                        np.sum(problem.item_quantity[stop, :, primary] * overdue)
                    )
        return cls(hours, late, start_day, arrival)

    def cells(
        self,
        problem: Subproblem,
        routes: Sequence[tuple[int, tuple[int, ...]]],
        shares: Sequence[tuple[int, int]],
        coefficients: Mapping[str, float],
        room: Room | None,
        primary: int,
    ) -> tuple[list[Cell], list[Cell]]:
        """The limited (center, day) handling cells and (center, day, type) vehicle cells.

        Share entries are numbered from 0 after the vehicle counts, as in
        :func:`solve_subproblem`. Cells whose violation has no weight are left out.
        """
        if room is None:
            return [], []
        handling: dict[tuple[int, int], list[tuple[int, float]]] = {}
        if coefficients.get("violation[handling]", 0.0) and np.isfinite(room.limit).any():
            for index, (r, node) in enumerate(shares):
                amount = float(problem.demand[node, primary])
                for place, day in ((0, int(self.start[r])), (1 + node, self.arrival[(r, node)])):
                    if np.isfinite(room.limit[place]):
                        handling.setdefault((place, day), []).append((len(routes) + index, amount))
        vehicles: dict[tuple[int, int, int], list[tuple[int, float]]] = {}
        if (
            coefficients.get("violation[vehicle_limit]", 0.0)
            and (room.slots < UNBOUNDED // 2).any()
        ):
            for r, (option, stops) in enumerate(routes):
                kind = int(problem.vehicle_type[option])
                places = [(0, int(self.start[r]))]
                places += [(1 + stop, self.arrival[(r, stop)]) for stop in stops]
                for place, day in places:
                    if room.slots[0, place, day, kind] < UNBOUNDED // 2:
                        vehicles.setdefault((place, day, kind), []).append((r, 1.0))
        return (
            [
                (max(float(room.handling[0, place, day]), 0.0), entries)
                for (place, day), entries in handling.items()
            ],
            [
                (float(max(int(room.slots[0, place, day, kind]), 0)), entries)
                for (place, day, kind), entries in vehicles.items()
            ],
        )


def _settle(
    problem: Subproblem,
    routes: Routes,
    coefficients: Mapping[str, float],
    shared: Shared,
    room: Room | None,
    ledger: Ledger,
) -> float:
    """Charge the routes of a scheduled subproblem to the ledger; return their value.

    The value leaves out demand handed to the next day, which is valued there.
    """
    remaining, item_left = problem.leftover(routes, True)
    alone = Batch([problem], shared)
    final = Colony.single(problem, routes, remaining, item_left)
    value = float(scores(alone, final, coefficients, shared, room, waiting=False)[0, 0])
    ledger.charge(problem, timeline(alone, final, shared))
    return value


def solve_subproblem(
    problem: Subproblem,
    coefficients: Mapping[str, float],
    shared: Shared,
    available: NDArray[np.int64],
    max_nodes: int = 9,
    time_limit: float | None = None,
    room: Room | None = None,
) -> SubproblemResult:
    """The optimal routes of one subproblem for a linear objective.

    On scheduled instances routes are timed (handling must take no time), and
    handling and vehicles beyond ``room`` cost their violation weights.
    """
    from scipy.optimize import Bounds, LinearConstraint, linprog, milp
    from scipy.sparse import coo_array

    routes = candidate_routes(problem, shared, max_nodes)
    demand = problem.demand
    loaded = [node for node in range(problem.size) if demand[node].max() > 0]
    dimensions = demand.shape[1]
    primary = shared.primary

    x_count = len(routes)
    shares = [(r, node) for r, (_, stops) in enumerate(routes) for node in stops if node in loaded]
    f_base, u_base = x_count, x_count + len(shares)
    s_base = u_base + len(loaded)
    times = ColumnTimes.of(problem, shared, routes, room)
    handling, slots = times.cells(problem, routes, shares, coefficients, room, primary)
    h_base = s_base + x_count
    v_base = h_base + len(handling)
    size = v_base + len(slots)
    cost = np.zeros(size)
    upper = np.full(size, np.inf)
    integrality = np.zeros(size)
    integrality[:x_count] = 1
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []

    def add_row(entries: list[tuple[int, float]], low: float, high: float) -> None:
        row = len(lower_bounds)
        for col, value in entries:
            rows.append(row)
            cols.append(col)
            vals.append(value)
        lower_bounds.append(low)
        upper_bounds.append(high)

    total = demand.sum(axis=0)
    for r, (option, stops) in enumerate(routes):
        km = route_km(problem, stops)
        empty = np.zeros(dimensions)
        cost[r] = route_value(
            problem, coefficients, shared, option, km, empty, hours=float(times.hours[r])
        ) - _shortfall_part(problem, coefficients, shared, option, empty)
        bound = available[option]
        if bound >= UNBOUNDED:
            bound = math.ceil(float(np.max(total / problem.capacity[option]))) + 1
        upper[r] = bound
    kind_of = ["owned" if problem.kind[option] == OWNED else "spot" for option, _ in routes]
    service = coefficients.get("service", 0.0)
    for index, (r, node) in enumerate(shares):
        cost[f_base + index] = -coefficients.get(f"unused[{kind_of[r]}]", 0.0) * demand[
            node, primary
        ] + service * times.late.get((r, node), 0.0)
    for position, node in enumerate(loaded):
        waits = ~problem.item_final[node]
        quantity = demand[node] - problem.item_quantity[node][waits].sum(axis=0)
        waiting = problem.item_quantity[node, waits, primary]
        deferred = float(np.sum(waiting * problem.defer_cost[node, waits]))
        cost[u_base + position] = (
            coefficients.get("unserved", 0.0) * quantity.sum()
            + sum(
                coefficients.get(f"unserved[{name}]", 0.0) * quantity[column]
                for column, name in enumerate(shared.dimensions)
            )
            + coefficients.get("violation[coverage]", 0.0) * quantity.sum()
            + service * deferred
        )
        upper[u_base + position] = 1.0
    cost[s_base:h_base] = coefficients.get("violation[min_fill]", 0.0)
    cost[h_base:v_base] = coefficients.get("violation[handling]", 0.0)
    cost[v_base:] = coefficients.get("violation[vehicle_limit]", 0.0)

    by_route: dict[int, list[int]] = {}
    by_node: dict[int, list[int]] = {}
    for index, (r, node) in enumerate(shares):
        by_route.setdefault(r, []).append(index)
        by_node.setdefault(node, []).append(index)
    for position, node in enumerate(loaded):
        entries = [(f_base + index, 1.0) for index in by_node.get(node, [])]
        add_row([*entries, (u_base + position, 1.0)], 1.0, 1.0)
    for r, (option, _) in enumerate(routes):
        indices = by_route.get(r, [])
        for column in range(dimensions):
            entries = [(f_base + index, demand[shares[index][1], column]) for index in indices]
            add_row([*entries, (r, -float(problem.capacity[option, column]))], -np.inf, 0.0)
        share = float(shared.min_fill[problem.kind[option]])
        if share > 0 and coefficients.get("violation[min_fill]", 0.0):
            capacity = float(problem.capacity[option, primary])
            entries = [(f_base + index, -demand[shares[index][1], primary]) for index in indices]
            add_row([*entries, (r, share * capacity), (s_base + r, -1.0)], -np.inf, 0.0)
        else:
            upper[s_base + r] = 0.0
    for option in range(problem.options):
        entries = [(r, 1.0) for r, (o, _) in enumerate(routes) if o == option]
        if entries and available[option] < UNBOUNDED:
            add_row(entries, -np.inf, float(available[option]))
    for position, (left, entries) in enumerate(handling):
        add_row([*entries, (h_base + position, -1.0)], -np.inf, left)
    for position, (left, entries) in enumerate(slots):
        add_row([*entries, (v_base + position, -1.0)], -np.inf, left)

    matrix = coo_array((vals, (rows, cols)), shape=(len(lower_bounds), size)).tocsr()
    options: dict[str, float] = {"mip_rel_gap": 1e-9}
    if time_limit is not None:
        options["time_limit"] = time_limit
    with QuietNative():
        result = milp(
            cost,
            constraints=LinearConstraint(matrix, lower_bounds, upper_bounds),
            integrality=integrality,
            bounds=Bounds(np.zeros(size), upper),
            options=options,
        )
    if result.x is None:
        raise RuntimeError(f"no solution for subproblem (day {problem.day}): {result.message}")
    vehicles = np.rint(result.x[:x_count]).astype(np.int64)

    low, high = np.array(lower_bounds), np.array(upper_bounds)
    equal = low == high
    lower = np.zeros(size)
    lower[:x_count] = vehicles
    fixed_upper = upper.copy()
    fixed_upper[:x_count] = vehicles
    with QuietNative():
        polished = linprog(
            cost,
            A_ub=matrix[~equal],
            b_ub=high[~equal],
            A_eq=matrix[equal],
            b_eq=high[equal],
            bounds=np.column_stack([lower, fixed_upper]),
            method="highs",
            options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10},
        )
    solution = polished.x if polished.status == 0 else result.x
    option_list, counts, nodes, amounts = [], [], [], []
    for r, (option, stops) in enumerate(routes):
        if vehicles[r] <= 0:
            continue
        delivered = {shares[i][1]: solution[f_base + i] for i in by_route.get(r, [])}
        for _ in range(vehicles[r]):
            option_list.append(option)
            counts.append(len(stops))
            for node in stops:
                amounts.append(
                    np.clip(demand[node] * delivered.get(node, 0.0) / vehicles[r], 0, None)
                )
                nodes.append(node)
    routes_out = Routes(
        option=np.array(option_list, dtype=np.int64),
        stop_start=np.concatenate([[0], np.cumsum(counts, dtype=np.int64)]).astype(np.int64),
        stop_node=np.array(nodes, dtype=np.int64),
        stop_amount=np.array(amounts, dtype=np.float64).reshape(len(nodes), dimensions),
        item_stop=np.zeros(0, dtype=np.int64),
        item_slot=np.zeros(0, dtype=np.int64),
    )
    return SubproblemResult(
        routes=routes_out,
        value=float(polished.fun if polished.status == 0 else result.fun),
        optimal=bool(result.status == 0),
        columns=x_count,
    )


def _shortfall_part(
    problem: Subproblem,
    coefficients: Mapping[str, float],
    shared: Shared,
    option: int,
    load: NDArray[np.float64],
) -> float:
    """The minimum-fill part of :func:`route_value` (handled by a variable instead)."""
    capacity = float(problem.capacity[option, shared.primary])
    shortfall = float(shared.min_fill[problem.kind[option]]) * capacity - float(
        load[shared.primary]
    )
    if shortfall > TOLERANCE * capacity:
        return coefficients.get("violation[min_fill]", 0.0) * shortfall
    return 0.0

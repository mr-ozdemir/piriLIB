"""Route-level rules and objective of a subproblem, shared by local search and the exact solver.

They mirror :func:`pirilib.solvers._colony.scores` and the evaluator: a route is
an option and its stops in visiting order; its value is the part of a linear
objective that the route contributes. On scheduled instances the caller times
the route (see :mod:`pirilib.solvers._timing`) and passes its hours and lateness.
Lateness counts twice in an objective: as the cost ``rules.service.penalty`` under
``service``, and as the amount of the ``lateness`` rule.
"""

import itertools
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from pirilib.evaluation import TOLERANCE
from pirilib.plan import OWNED
from pirilib.solvers._colony import Shared
from pirilib.solvers.decompose import Subproblem


def route_km(problem: Subproblem, stops: Sequence[int]) -> float:
    """Length of the route from the origin through ``stops``."""
    km = float(problem.origin_km[stops[0]])
    for before, after in itertools.pairwise(stops):
        km += float(problem.node_km[before, after])
    return km


def route_ok(problem: Subproblem, shared: Shared, option: int, stops: Sequence[int]) -> bool:
    """True if the route keeps every rule except capacity and minimum fill."""
    if not stops or len(set(stops)) != len(stops):
        return False
    if not all(problem.allowed[option, stop] for stop in stops):
        return False
    rules = shared.rules.stopover
    if len(stops) > 1 and not problem.stopovers[option]:
        return False
    if rules.max_stops is not None and len(stops) > rules.max_stops:
        return False
    if problem.kind[option] == OWNED and stops[-1] != problem.line_end[option]:
        return False
    if rules.forward_only and any(
        problem.origin_km[after] <= problem.origin_km[before]
        for before, after in itertools.pairwise(stops)
    ):
        return False
    if rules.max_stretch is not None:
        limit = rules.max_stretch * float(problem.origin_km[stops[-1]]) * (1 + TOLERANCE)
        if route_km(problem, stops) > limit:
            return False
    return True


def fits(problem: Subproblem, option: int, load: NDArray[np.float64]) -> bool:
    capacity = problem.capacity[option]
    return bool(np.all(load <= capacity + TOLERANCE * capacity))


def route_gap(
    shared: Shared, kind: int, capacity: NDArray[np.float64], load: NDArray[np.float64]
) -> tuple[float, int]:
    """The load one vehicle misses to its minimum fill, and the dimension it is measured in.

    ``(0.0, -1)`` when the vehicle reaches the minimum of any dimension or has none;
    otherwise the smallest of its gaps, as :func:`pirilib.evaluation.fill_shortfall`.
    """
    least, nearest = 0.0, -1
    for column, share in shared.thresholds[kind]:
        room = float(capacity[column])
        gap = share * room - float(load[column])
        if gap <= TOLERANCE * room:
            return 0.0, -1
        if nearest < 0 or gap < least:
            least, nearest = gap, column
    return least, nearest


def route_value(
    problem: Subproblem,
    coefficients: Mapping[str, float],
    shared: Shared,
    option: int,
    km: float,
    load: NDArray[np.float64],
    *,
    hours: float = 0.0,
    late: float = 0.0,
) -> float:
    """The route's share of the linear objective.

    ``late`` is the primary quantity of its loads times the started hours they are
    late: it costs ``rules.service.penalty`` under ``service`` and is the amount of
    the ``lateness`` rule.
    """
    kind = "owned" if problem.kind[option] == OWNED else "spot"
    cost = float(problem.fixed[option] + problem.per_km[option] * km)
    if hours:
        cost += float(problem.per_hour[option]) * hours
    if problem.unit_priced:
        cost += float(problem.per_unit[option] @ load)
    capacity = float(problem.capacity[option, shared.primary])
    carried = float(load[shared.primary])
    shortfall = route_gap(shared, int(problem.kind[option]), problem.capacity[option], load)[0]
    value = (
        coefficients.get(f"cost[{kind}]", 0.0) * cost
        + coefficients.get(f"km[{kind}]", 0.0) * km
        + coefficients.get(f"dispatches[{kind}]", 0.0)
        + coefficients.get(f"unused[{kind}]", 0.0) * max(capacity - carried, 0.0)
    )
    if shortfall:
        value += coefficients.get("violation[min_fill]", 0.0) * shortfall
    if late:
        value += coefficients.get("service", 0.0) * (shared.penalty * late)
        value += coefficients.get("violation[lateness]", 0.0) * late
    return value


def unserved_value(
    problem: Subproblem,
    coefficients: Mapping[str, float],
    shared: Shared,
    remaining: NDArray[np.float64],
    item_left: NDArray[np.bool_] | None = None,
) -> float:
    """The objective's share for demand left uncarried, ``remaining`` per node and dimension.

    Items that may wait for the next day (whole items: those in ``item_left``)
    cost their :attr:`~pirilib.solvers.decompose.Subproblem.defer_cost` under
    ``service``, and their estimated lateness under the ``lateness`` rule, instead
    of counting as unserved.
    """
    value = 0.0
    if problem.deferrable:
        primary = shared.primary
        waits = ~problem.item_final
        quantity = problem.item_quantity
        if shared.rules.split_demand:
            safe = np.where(problem.demand > 0, problem.demand, 1.0)
            share = np.where(problem.demand > 0, remaining / safe, 0.0)
            waiting = quantity * share[:, None, :] * waits[:, :, None]
        else:
            assert item_left is not None
            waiting = quantity * (waits & item_left)[:, :, None]
        deferred = float(np.sum(waiting[:, :, primary] * problem.defer_cost))
        value += coefficients.get("service", 0.0) * deferred
        weight = coefficients.get("violation[lateness]", 0.0)
        if weight:
            value += weight * float(np.sum(waiting[:, :, primary] * problem.defer_late))
        remaining = np.clip(remaining - waiting.sum(axis=1), 0.0, None)
    value += coefficients.get("unserved", 0.0) * float(remaining.sum())
    for column, name in enumerate(shared.dimensions):
        value += coefficients.get(f"unserved[{name}]", 0.0) * float(remaining[:, column].sum())
    threshold = TOLERANCE * np.maximum(problem.demand, 1.0)
    covered = np.where(remaining > threshold, remaining, 0.0)
    return value + coefficients.get("violation[coverage]", 0.0) * float(covered.sum())

"""Route-level rules and objective of a subproblem, shared by local search and the exact solver.

They mirror :func:`pirilib.solvers._colony.scores` and the evaluator: a route is
an option and its stops in visiting order; its value is the part of a linear
objective that the route contributes.
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


def route_value(
    problem: Subproblem,
    coefficients: Mapping[str, float],
    shared: Shared,
    option: int,
    km: float,
    load: NDArray[np.float64],
) -> float:
    """The route's share of the linear objective."""
    kind = "owned" if problem.kind[option] == OWNED else "spot"
    cost = float(problem.fixed[option] + problem.per_km[option] * km)
    capacity = float(problem.capacity[option, shared.primary])
    carried = float(load[shared.primary])
    shortfall = float(shared.min_fill[problem.kind[option]]) * capacity - carried
    value = (
        coefficients.get(f"cost[{kind}]", 0.0) * cost
        + coefficients.get(f"km[{kind}]", 0.0) * km
        + coefficients.get(f"dispatches[{kind}]", 0.0)
        + coefficients.get(f"unused[{kind}]", 0.0) * max(capacity - carried, 0.0)
    )
    if shortfall > TOLERANCE * capacity:
        value += coefficients.get("violation[min_fill]", 0.0) * shortfall
    return value


def unserved_value(
    problem: Subproblem,
    coefficients: Mapping[str, float],
    shared: Shared,
    remaining: NDArray[np.float64],
) -> float:
    """The objective's share for demand left uncarried, ``remaining`` per node and dimension."""
    value = coefficients.get("unserved", 0.0) * float(remaining.sum())
    for column, name in enumerate(shared.dimensions):
        value += coefficients.get(f"unserved[{name}]", 0.0) * float(remaining[:, column].sum())
    threshold = TOLERANCE * np.maximum(problem.demand, 1.0)
    covered = np.where(remaining > threshold, remaining, 0.0)
    return value + coefficients.get("violation[coverage]", 0.0) * float(covered.sum())

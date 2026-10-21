"""Adapter to PyVRP, a hybrid genetic search for vehicle routing (extra ``pirilib[pyvrp]``)."""

import importlib
from typing import Any

import numpy as np

from pirilib.solvers.adapters.base import FORBIDDEN, RoutingAdapter, RoutingModel, RoutingSolution

_FAR = 10**9


class PyVRPSolver(RoutingAdapter):
    """Plan every subproblem with PyVRP; see :mod:`pirilib.solvers.adapters.base`.

    Every vehicle type gets a routing profile of its own, so that bans, owned
    lines, forward-only stops and the stretch rule are part of its distance
    and duration matrices. The search is seeded and, without a time limit,
    reproducible.

    Attributes:
        iterations: Iterations of the genetic search per subproblem
        time_limit: Seconds per subproblem; None for no limit
    """

    mode = "pyvrp"
    package = "pyvrp"
    extra = "pyvrp"

    def __init__(self, iterations: int = 1000, time_limit: float | None = None) -> None:
        if iterations < 1:
            raise ValueError("iterations must be at least 1")
        if time_limit is not None and time_limit <= 0:
            raise ValueError("time_limit must be positive")
        self.iterations = iterations
        self.time_limit = time_limit

    def load(self) -> None:
        try:
            importlib.import_module("pyvrp")
        except ImportError as error:
            raise self.missing(error) from error

    def route(self, model: RoutingModel, seed: int) -> RoutingSolution:
        from pyvrp import Model
        from pyvrp.stop import MaxIterations, MaxRuntime, MultipleCriteria

        vrp = Model()
        reach = np.minimum(model.distance[:, 0, :].min(axis=0), _FAR)
        locations = [
            vrp.add_location(x=float(reach[place]), y=0.0) for place in range(model.locations)
        ]
        start = vrp.add_depot(locations[0])
        ends = [vrp.add_depot(locations[model.end(t)]) for t in range(model.types)]
        for client in range(model.clients):
            vrp.add_client(
                locations[1 + client],
                delivery=[int(value) for value in model.delivery[client]],
                prize=int(model.prize[client]),
                required=bool(model.required[client]),
            )
        for t in range(model.types):
            profile = vrp.add_profile()
            allowed = model.distance[t] < FORBIDDEN
            for first, second in zip(*allowed.nonzero(), strict=True):
                vrp.add_edge(
                    locations[first],
                    locations[second],
                    distance=int(model.distance[t, first, second]),
                    duration=int(model.duration[t, first, second]),
                    profile=profile,
                )
            vrp.add_vehicle_type(
                num_available=int(model.count[t]),
                capacity=[int(value) for value in model.capacity[t]],
                start_depot=start,
                end_depot=ends[t],
                fixed_cost=int(model.fixed_cost[t]),
                unit_distance_cost=int(model.unit_cost[t]),
                shift_duration=int(model.max_duration[t]),
                profile=profile,
            )
        stop: Any = MaxIterations(self.iterations)
        if self.time_limit is not None:
            stop = MultipleCriteria([stop, MaxRuntime(self.time_limit)])
        result = vrp.solve(
            stop, seed=seed, collect_stats=False, display=False, missing_value=FORBIDDEN
        )
        routes = tuple(
            (
                int(route.vehicle_type()),
                tuple(int(visit.idx) for visit in route if visit.is_client()),
            )
            for route in result.best.routes()
            if route.is_feasible()
        )
        return RoutingSolution(routes=routes, feasible=bool(result.is_feasible()))

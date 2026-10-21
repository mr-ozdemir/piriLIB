"""Adapter to the routing library of Google OR-Tools (extra ``pirilib[ortools]``)."""

import warnings
from collections.abc import Callable
from typing import Any, Final

from pirilib.solvers.adapters.base import RoutingAdapter, RoutingModel, RoutingSolution

_STOPPING: Final = ("GREEDY_DESCENT", "AUTOMATIC")


class ORToolsSolver(RoutingAdapter):
    """Plan every subproblem with the OR-Tools routing solver.

    See :mod:`pirilib.solvers.adapters.base`. Every vehicle is one OR-Tools
    vehicle with the arc costs of its type; the stretch rule and the arcs a
    type may not use are a duration dimension per vehicle, loads are capacity
    dimensions, and optional clients are disjunctions with their prize as
    penalty. Unlimited spot offers get one vehicle per client.

    The default search (cheapest arc, then greedy descent) stops by itself and
    is deterministic; the seed is not used. Metaheuristics such as
    ``GUIDED_LOCAL_SEARCH`` need a ``time_limit`` and depend on the machine.

    The package is imported with its deprecation warnings silenced: on Python
    3.14, turning them into errors makes the import crash.

    Attributes:
        first_solution: Name of an OR-Tools ``FirstSolutionStrategy``
        metaheuristic: Name of an OR-Tools ``LocalSearchMetaheuristic``
        time_limit: Seconds per subproblem; None for no limit
    """

    mode = "ortools"
    package = "ortools"
    extra = "ortools"

    def __init__(
        self,
        first_solution: str = "PATH_CHEAPEST_ARC",
        metaheuristic: str = "GREEDY_DESCENT",
        time_limit: float | None = None,
    ) -> None:
        if time_limit is not None and time_limit <= 0:
            raise ValueError("time_limit must be positive")
        if metaheuristic not in _STOPPING and time_limit is None:
            raise ValueError(f"metaheuristic {metaheuristic} needs a time_limit")
        self.first_solution = first_solution
        self.metaheuristic = metaheuristic
        self.time_limit = time_limit

    def _modules(self) -> tuple[Any, Any]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                from ortools.constraint_solver import pywrapcp, routing_enums_pb2
        except ImportError as error:
            raise self.missing(error) from error
        return pywrapcp, routing_enums_pb2

    def load(self) -> None:
        _, enums = self._modules()
        for kind, name in (
            (enums.FirstSolutionStrategy, self.first_solution),
            (enums.LocalSearchMetaheuristic, self.metaheuristic),
        ):
            if not hasattr(kind, name):
                raise ValueError(f"unknown OR-Tools search setting {name!r}")

    def route(self, model: RoutingModel, seed: int) -> RoutingSolution:
        pywrapcp, enums = self._modules()
        kinds = [t for t in range(model.types) for _ in range(int(model.count[t]))]
        size = model.locations
        manager = pywrapcp.RoutingIndexManager(
            size, len(kinds), [0] * len(kinds), [model.end(t) for t in kinds]
        )
        routing = pywrapcp.RoutingModel(manager)
        node = manager.IndexToNode

        def arcs(matrix: list[list[int]]) -> Callable[[int, int], int]:
            return lambda first, second: matrix[node(first)][node(second)]

        cost_callbacks, duration_callbacks = [], []
        for t in range(model.types):
            costs = (model.distance[t] * int(model.unit_cost[t])).tolist()
            cost_callbacks.append(routing.RegisterTransitCallback(arcs(costs)))
            duration_callbacks.append(
                routing.RegisterTransitCallback(arcs(model.duration[t].tolist()))
            )
        for vehicle, t in enumerate(kinds):
            routing.SetArcCostEvaluatorOfVehicle(cost_callbacks[t], vehicle)
            routing.SetFixedCostOfVehicle(int(model.fixed_cost[t]), vehicle)
        routing.AddDimensionWithVehicleTransitAndCapacity(
            [duration_callbacks[t] for t in kinds],
            0,
            [int(model.max_duration[t]) for t in kinds],
            True,
            "stretch",
        )
        for column in range(model.delivery.shape[1]):
            loads = [0] * size
            for client in range(model.clients):
                loads[1 + client] = int(model.delivery[client, column])
            callback = routing.RegisterUnaryTransitCallback(
                lambda index, loads=loads: loads[node(index)]
            )
            routing.AddDimensionWithVehicleCapacity(
                callback, 0, [int(model.capacity[t, column]) for t in kinds], True, f"load{column}"
            )
        reachable = [model.visitable(t) for t in range(model.types)]
        for client in range(model.clients):
            index = manager.NodeToIndex(1 + client)
            allowed = [vehicle for vehicle, t in enumerate(kinds) if reachable[t][client]]
            if allowed:
                routing.VehicleVar(index).SetValues([-1, *allowed])
            if not model.required[client]:
                routing.AddDisjunction([index], int(model.prize[client]))
        parameters = pywrapcp.DefaultRoutingSearchParameters()
        parameters.first_solution_strategy = getattr(
            enums.FirstSolutionStrategy, self.first_solution
        )
        parameters.local_search_metaheuristic = getattr(
            enums.LocalSearchMetaheuristic, self.metaheuristic
        )
        if self.time_limit is not None:
            parameters.time_limit.FromMilliseconds(max(1, round(self.time_limit * 1000)))
        solution = routing.SolveWithParameters(parameters)
        if solution is None:
            return RoutingSolution(routes=(), feasible=False)
        routes = []
        for vehicle, t in enumerate(kinds):
            index = solution.Value(routing.NextVar(routing.Start(vehicle)))
            visits = []
            while not routing.IsEnd(index):
                visits.append(node(index) - 1)
                index = solution.Value(routing.NextVar(index))
            if visits:
                routes.append((t, tuple(visits)))
        return RoutingSolution(routes=tuple(routes), feasible=True)

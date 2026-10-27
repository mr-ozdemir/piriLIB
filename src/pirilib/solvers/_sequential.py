"""Solvers that plan one (day, origin) subproblem after another.

:class:`SequentialSolver` holds what such solvers share: the subproblems in the
order of :func:`~pirilib.solvers.decompose.subproblems`, the daily limits of
spot offers taken by the origins of a day one after another, and the final
evaluation. :class:`RouteList` collects the routes of one subproblem.
"""

import time
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from pirilib.constraints import Constraint
from pirilib.evaluation import evaluate
from pirilib.objective import Objective
from pirilib.plan import SPOT
from pirilib.solvers._colony import Shared
from pirilib.solvers.base import SolveResult, require_objective
from pirilib.solvers.decompose import UNBOUNDED, PlanBuilder, Routes, Subproblem, subproblems
from pirilib.spec.instance import UNLIMITED, LineHaulInstance


class RouteList:
    """The routes of one subproblem, collected one by one and turned into :class:`Routes`."""

    def __init__(self, problem: Subproblem) -> None:
        self.problem = problem
        self._option: list[int] = []
        self._counts: list[int] = []
        self._nodes: list[int] = []
        self._amounts: list[NDArray[np.float64]] = []
        self._item_stop: list[int] = []
        self._item_slot: list[int] = []

    def add(
        self,
        option: int,
        stops: Sequence[int],
        amounts: Sequence[NDArray[np.float64]],
        items: Sequence[tuple[int, int]] = (),
    ) -> None:
        """Add a route; ``items`` are (position of the stop in ``stops``, item slot) pairs."""
        first = len(self._nodes)
        self._option.append(option)
        self._counts.append(len(stops))
        self._nodes.extend(stops)
        self._amounts.extend(np.asarray(amount, dtype=np.float64) for amount in amounts)
        for position, slot in items:
            self._item_stop.append(first + position)
            self._item_slot.append(slot)

    def __len__(self) -> int:
        return len(self._option)

    def build(self) -> Routes:
        dimensions = self.problem.demand.shape[1]
        return Routes(
            option=np.array(self._option, dtype=np.int64),
            stop_start=np.concatenate([[0], np.cumsum(self._counts, dtype=np.int64)]).astype(
                np.int64
            ),
            stop_node=np.array(self._nodes, dtype=np.int64),
            stop_amount=np.array(self._amounts, dtype=np.float64).reshape(
                len(self._nodes), dimensions
            ),
            item_stop=np.array(self._item_stop, dtype=np.int64),
            item_slot=np.array(self._item_slot, dtype=np.int64),
        )


class SequentialSolver:
    """Base of solvers that plan subproblems one after another.

    Subclasses implement :meth:`plan_subproblem` and may refuse instances or
    objectives in :meth:`prepare`. Spot offers with a daily limit are shared by
    the origins of a day in the order of the subproblems, so with such limits
    the whole plan is greedy across origins.
    """

    mode = "sequential"

    def solve(
        self,
        instance: LineHaulInstance,
        objective: Objective,
        constraints: Sequence[Constraint] = (),
        *,
        seed: int = 0,
    ) -> SolveResult:
        goal = require_objective(objective, f"{type(self).__name__}.solve")
        coefficients = self.prepare(instance, goal, constraints)
        started = time.perf_counter()
        shared = Shared.of(instance)
        limits = np.where(
            instance.spot_available == UNLIMITED, UNBOUNDED, instance.spot_available
        ).astype(np.int64)
        spot_left = {day: limits.copy() for day in range(len(instance.days))}
        builder = PlanBuilder(instance)
        stats: dict[str, float | int | str] = {"mode": self.mode}
        count = 0
        for index, problem in enumerate(subproblems(instance)):
            available = problem.available.copy()
            spot = problem.kind == SPOT
            left = spot_left[problem.day]
            available[spot] = np.minimum(available[spot], left[problem.source[spot]])
            routes = self.plan_subproblem(
                problem, shared, coefficients, available, seed + index, stats
            )
            builder.add(problem, routes)
            used = np.bincount(routes.option, minlength=problem.options)
            left[problem.source[spot]] -= used[spot]
            count += 1
        stats["subproblems"] = count
        plan = builder.build()
        evaluation = evaluate(instance, plan, constraints)
        return SolveResult(
            plan=plan,
            evaluation=evaluation,
            objective=float(goal(evaluation)),
            seconds=time.perf_counter() - started,
            stats=stats,
        )

    def prepare(
        self, instance: LineHaulInstance, objective: Objective, constraints: Sequence[Constraint]
    ) -> Mapping[str, float]:
        """Check that the solver supports the problem; return the coefficients it plans with."""
        return {"cost[owned]": 1.0, "cost[spot]": 1.0}

    def plan_subproblem(
        self,
        problem: Subproblem,
        shared: Shared,
        coefficients: Mapping[str, float],
        available: NDArray[np.int64],
        seed: int,
        stats: dict[str, float | int | str],
    ) -> Routes:
        """The routes of one subproblem; ``available`` are the vehicles left per option."""
        raise NotImplementedError

import math
from typing import Any

import numpy as np
import pytest

from pirilib import ACOSolver, synthetic
from pirilib import objective as obj
from pirilib.solvers._colony import Shared
from pirilib.solvers.base import linear_coefficients
from pirilib.solvers.decompose import Routes, Subproblem, subproblems
from pirilib.solvers.local_search import improve
from pirilib.spec import LineHaulInstance, parse_spec

ONE = 6371.0 * math.pi / 180.0
B, C = 0, 1


def setup(
    document: dict[str, Any], compile_document: Any, demand: dict[str, float], **rules: Any
) -> tuple[LineHaulInstance, Subproblem]:
    document["fleet"] = {"spot": [{"type": "small", "cost": {"fixed": 150.0, "per_km": 1.5}}]}
    document["demand"] = [
        {"origin": "A", "destination": to, "date": "2026-01-05", "quantity": {"desi": amount}}
        for to, amount in demand.items()
    ]
    document["rules"] = rules
    instance = compile_document(document)
    [problem] = subproblems(instance)
    return instance, problem


def routes(*stops: list[tuple[int, float]]) -> Routes:
    """Routes of the single spot option; each argument is one route's (node, amount) stops."""
    flat = [stop for route in stops for stop in route]
    return Routes(
        option=np.zeros(len(stops), dtype=np.int64),
        stop_start=np.cumsum([0, *(len(route) for route in stops)]).astype(np.int64),
        stop_node=np.array([node for node, _ in flat], dtype=np.int64),
        stop_amount=np.array([[amount] for _, amount in flat], dtype=np.float64).reshape(-1, 1),
        item_stop=np.zeros(0, dtype=np.int64),
        item_slot=np.zeros(0, dtype=np.int64),
    )


def run(
    instance: LineHaulInstance,
    problem: Subproblem,
    start: Routes,
    goal: obj.Sum,
    remaining: list[float] | None = None,
) -> tuple[Routes, float]:
    left = np.array([[value] for value in remaining or [0.0] * problem.size])
    return improve(
        problem,
        start,
        linear_coefficients(instance, goal.linear_form() or {}),
        Shared.of(instance),
        problem.available,
        left,
    )


def described(result: Routes) -> list[list[tuple[int, float]]]:
    return [
        [
            (int(result.stop_node[s]), round(float(result.stop_amount[s, 0]), 6))
            for s in range(result.stop_start[r], result.stop_start[r + 1])
        ]
        for r in range(result.option.size)
    ]


GOAL = obj.total_cost() + 1e6 * obj.violations()


def test_merge(document: dict[str, Any], compile_document: Any) -> None:
    instance, problem = setup(document, compile_document, {"B": 100.0, "C": 100.0})
    result, change = run(instance, problem, routes([(B, 100.0)], [(C, 100.0)]), GOAL)
    assert described(result) == [[(B, 100.0), (C, 100.0)]]
    assert change == pytest.approx(-(150 + 1.5 * ONE))


def test_relocate(document: dict[str, Any], compile_document: Any) -> None:
    document["centers"][2].update(lat=1.0, lon=0.0)
    instance, problem = setup(
        document, compile_document, {"B": 380.0, "C": 300.0}, stopover={"forward_only": False}
    )
    start = routes([(B, 300.0)], [(B, 80.0), (C, 300.0)])
    result, change = run(instance, problem, start, GOAL)
    assert sorted(described(result)) == [[(B, 380.0)], [(C, 300.0)]]
    detour = ONE + math.dist((0, 1), (1, 0)) * ONE - ONE
    assert change == pytest.approx(-1.5 * detour, rel=1e-3)


def test_rebalance_lifts_minimum_fill(document: dict[str, Any], compile_document: Any) -> None:
    instance, problem = setup(document, compile_document, {"B": 500.0}, min_fill={"spot": 0.5})
    result, change = run(instance, problem, routes([(B, 400.0)], [(B, 100.0)]), GOAL)
    loads = sorted(amount for route in described(result) for _, amount in route)
    assert loads == [200.0, 300.0]
    assert change == pytest.approx(-1e6 * 100.0)


def test_drop_when_leaving_demand_is_cheaper(
    document: dict[str, Any], compile_document: Any
) -> None:
    instance, problem = setup(document, compile_document, {"B": 100.0})
    cheap_coverage = obj.total_cost() + 0.01 * obj.violation("coverage")
    result, change = run(instance, problem, routes([(B, 100.0)]), cheap_coverage)
    assert result.option.size == 0
    assert change == pytest.approx(-(150 + 1.5 * ONE) + 0.01 * 100.0)


def test_serve_carries_what_was_left(document: dict[str, Any], compile_document: Any) -> None:
    instance, problem = setup(document, compile_document, {"B": 150.0})
    result, change = run(instance, problem, routes([(B, 100.0)]), GOAL, remaining=[50.0])
    assert described(result) == [[(B, 150.0)]]
    assert change == pytest.approx(-1e6 * 50.0)


def test_nothing_to_improve(document: dict[str, Any], compile_document: Any) -> None:
    instance, problem = setup(document, compile_document, {"B": 400.0})
    start = routes([(B, 400.0)])
    result, change = run(instance, problem, start, GOAL)
    assert described(result) == described(start)
    assert change == 0.0


def test_colony_plans_with_local_search_did_not_change() -> None:
    """Pinned with the local search before its moves were remembered."""
    instance = parse_spec(
        synthetic.generate(centers=20, days=2, demand_per_day=150, seed=7)
    ).compile()
    result = ACOSolver().solve(instance, seed=0)
    assert result.stats["score"] == pytest.approx(313495540.81793475, rel=1e-12)
    assert len(result.plan) == 145

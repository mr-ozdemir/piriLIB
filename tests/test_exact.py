import math
from typing import Any

import numpy as np
import pytest

from pirilib import ACOConfig, ACOSolver, check_plan
from pirilib import objective as obj
from pirilib.solvers._colony import Shared
from pirilib.solvers.decompose import subproblems
from pirilib.solvers.exact import ExactSolver, candidate_routes
from pirilib.spec import LineHaulInstance, parse_spec

ONE = 6371.0 * math.pi / 180.0
GOAL = obj.total_cost() + 1e6 * obj.violations()


def small_spot_only(document: dict[str, Any]) -> dict[str, Any]:
    document["fleet"] = {"spot": [{"type": "small", "cost": {"fixed": 150.0, "per_km": 1.5}}]}
    document["demand"] = [
        {"origin": "A", "destination": "B", "date": "2026-01-05", "quantity": {"desi": 100.0}},
        {"origin": "A", "destination": "C", "date": "2026-01-05", "quantity": {"desi": 100.0}},
    ]
    return document


def test_known_optimum(document: dict[str, Any], compile_document: Any) -> None:
    instance = compile_document(small_spot_only(document))
    result = ExactSolver().solve(instance, GOAL)
    assert result.evaluation.feasible
    assert len(result.plan) == 1
    assert result.plan.stops(0).tolist() == [1, 2]
    assert result.evaluation.total_cost == pytest.approx(150 + 1.5 * 2 * ONE)
    assert result.stats["optimal"] == "True"
    assert result.stats["score"] == pytest.approx(result.objective)


def test_routes_follow_the_rules(document: dict[str, Any], compile_document: Any) -> None:
    document = small_spot_only(document)
    document["rules"] = {"stopover": {"max_stops": 1}}
    instance = compile_document(document)
    [problem] = subproblems(instance)
    routes = candidate_routes(problem, Shared.of(instance), max_nodes=9)
    assert sorted(stops for _, stops in routes) == [(0,), (1,)]


@pytest.mark.parametrize("seed", range(12))
def test_never_worse_than_the_colony(seed: int, random_document: Any) -> None:
    document = random_document(seed)
    document["rules"]["split_demand"] = True
    document["fleet"]["spot"][1].pop("available", None)
    instance = parse_spec(document).compile()
    exact = ExactSolver().solve(instance, GOAL)
    colony = ACOSolver(ACOConfig(ants=6, iterations=8, patience=4)).solve(instance, GOAL, seed=seed)
    assert exact.objective <= colony.objective * (1 + 1e-9)
    report = check_plan(instance, exact.plan)
    assert report.total_cost == pytest.approx(exact.evaluation.total_cost)
    for name in ("route", "load_route", "ban", "owned_line", "capacity", "overdelivery"):
        assert exact.evaluation.violations[name] == 0, name


def test_hybrid_colony_solves_small_subproblems_exactly(instance: LineHaulInstance) -> None:
    hybrid = ACOSolver(ACOConfig(exact_nodes=9)).solve(instance, GOAL)
    exact = ExactSolver().solve(instance, GOAL)
    assert hybrid.stats["exact"] == len(subproblems(instance))
    assert hybrid.objective == pytest.approx(exact.objective)


def test_requirements(
    instance: LineHaulInstance, document: dict[str, Any], compile_document: Any
) -> None:
    with pytest.raises(ValueError, match="linear form"):
        ExactSolver().solve(instance, lambda e: e.total_cost)
    document["rules"] = {"split_demand": False}
    with pytest.raises(ValueError, match="split_demand"):
        ExactSolver().solve(compile_document(document), GOAL)
    document["rules"] = {}
    with pytest.raises(ValueError, match="more than max_nodes=1"):
        ExactSolver(max_nodes=1).solve(compile_document(small_spot_only(document)), GOAL)


def test_unservable_demand_is_left_when_cheaper(
    document: dict[str, Any], compile_document: Any
) -> None:
    document = small_spot_only(document)
    document["rules"] = {"min_fill": {"spot": 0.9}}
    instance = compile_document(document)
    goal = obj.total_cost() + 10.0 * obj.violation("min_fill") + 1.0 * obj.violation("coverage")
    result = ExactSolver().solve(instance, goal)
    assert len(result.plan) == 0
    assert np.isclose(result.evaluation.violations["coverage"], 200.0)

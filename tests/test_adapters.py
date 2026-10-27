import importlib.util
import sys
from collections.abc import Callable
from typing import Any

import pytest

from pirilib import check_plan
from pirilib import objective as obj
from pirilib.constraints import MaxRouteKm
from pirilib.solvers.adapters import ORToolsSolver, PyVRPSolver, RoutingAdapter
from pirilib.spec import LineHaulInstance, parse_spec

type RandomDocument = Callable[[int], dict[str, Any]]

GOAL = obj.total_cost() + 1e6 * obj.violations()
HAS_PYVRP = importlib.util.find_spec("pyvrp") is not None
HAS_ORTOOLS = importlib.util.find_spec("ortools") is not None
BY_CONSTRUCTION = (
    "route",
    "load_route",
    "ban",
    "owned_line",
    "owned_availability",
    "spot_availability",
    "capacity",
    "stopover",
    "max_stops",
    "stretch",
    "forward_only",
    "split",
    "overdelivery",
)

ADAPTERS = [
    pytest.param(
        lambda: PyVRPSolver(iterations=100),
        id="pyvrp",
        marks=pytest.mark.skipif(not HAS_PYVRP, reason="pyvrp is not installed"),
    ),
    pytest.param(
        ORToolsSolver,
        id="ortools",
        marks=pytest.mark.skipif(not HAS_ORTOOLS, reason="ortools is not installed"),
    ),
]


def routable(document: dict[str, Any]) -> dict[str, Any]:
    document["rules"]["min_fill"] = {"spot": 0.0}
    document["rules"]["split_demand"] = True
    return document


@pytest.mark.parametrize("make", ADAPTERS)
@pytest.mark.parametrize("seed", range(12))
def test_plans_keep_the_rules(
    make: Callable[[], RoutingAdapter], seed: int, random_document: RandomDocument
) -> None:
    instance = parse_spec(routable(random_document(seed))).compile()
    result = make().solve(instance, GOAL, seed=seed)
    evaluation = result.evaluation
    broken = {name: evaluation.violations[name] for name in BY_CONSTRUCTION}
    assert not any(broken.values()), broken
    if (instance.spot_available == -1).all():
        assert evaluation.feasible, evaluation.violated()
    report = check_plan(instance, result.plan)
    assert report.total_cost == pytest.approx(evaluation.total_cost, rel=1e-12)
    for name, amount in report.totals().items():
        assert evaluation.violations[name] == pytest.approx(amount, rel=1e-9, abs=1e-6)


@pytest.mark.parametrize("make", ADAPTERS)
def test_not_better_than_the_exact_solver(
    make: Callable[[], RoutingAdapter], random_document: RandomDocument
) -> None:
    from pirilib.solvers.exact import ExactSolver

    for seed in range(4):
        document = routable(random_document(seed))
        document["fleet"]["spot"][1].pop("available", None)
        instance = parse_spec(document).compile()
        exact = ExactSolver().solve(instance, GOAL)
        routed = make().solve(instance, GOAL, seed=seed)
        assert routed.evaluation.feasible
        assert routed.objective >= exact.objective * (1 - 1e-9)


@pytest.mark.parametrize("make", ADAPTERS)
def test_unsupported_problems_are_refused(
    make: Callable[[], RoutingAdapter],
    document: dict[str, Any],
    compile_document: Callable[[dict[str, Any]], LineHaulInstance],
) -> None:
    solver = make()
    document["rules"] = {"min_fill": {"spot": 0.1}, "split_demand": False}
    instance = compile_document(document)
    with pytest.raises(ValueError, match=r"split_demand = false; minimum fill"):
        solver.solve(instance, GOAL)
    document["rules"] = {}
    instance = compile_document(document)
    with pytest.raises(ValueError, match="objectives without a linear form"):
        solver.solve(instance, objective=lambda evaluation: evaluation.total_cost)
    with pytest.raises(ValueError, match="constraints of your own"):
        solver.solve(instance, GOAL, constraints=[MaxRouteKm(100)])
    with pytest.raises(ValueError, match=r"objective terms \['unused\[spot\]'\]"):
        solver.solve(instance, objective=obj.total_cost() + obj.unused_capacity("spot"))
    assert solver.solve(instance, GOAL).evaluation.feasible


def test_missing_package_is_explained(
    monkeypatch: pytest.MonkeyPatch, instance: LineHaulInstance
) -> None:
    monkeypatch.setitem(sys.modules, "pyvrp", None)
    with pytest.raises(ImportError, match=r'PyVRPSolver needs .* pip install "pirilib\[pyvrp\]"'):
        PyVRPSolver().solve(instance, GOAL)
    monkeypatch.setitem(sys.modules, "ortools.constraint_solver", None)
    with pytest.raises(ImportError, match=r'pip install "pirilib\[ortools\]"'):
        ORToolsSolver().solve(instance, GOAL)


def test_settings_are_checked() -> None:
    with pytest.raises(ValueError, match="iterations must be at least 1"):
        PyVRPSolver(iterations=0)
    with pytest.raises(ValueError, match="needs a time_limit"):
        ORToolsSolver(metaheuristic="GUIDED_LOCAL_SEARCH")

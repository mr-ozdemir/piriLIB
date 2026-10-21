from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import pirilib
from pirilib import ACOConfig, ACOSolver, check_plan, evaluate
from pirilib import objective as obj
from pirilib.constraints import CATALOG, MaxDispatchesPerDay
from pirilib.solvers import default_objective
from pirilib.spec import LineHaulInstance, parse_spec

type RandomDocument = Callable[[int], dict[str, Any]]

SMALL = ACOConfig(ants=6, iterations=8, patience=4)
BY_CONSTRUCTION = [name for name in CATALOG if name not in ("min_fill", "coverage")]


def solve(instance: LineHaulInstance, **options: Any) -> pirilib.SolveResult:
    return ACOSolver(options.pop("config", SMALL)).solve(instance, **options)


def test_example_problem(example_path: Path) -> None:
    instance = pirilib.load_spec(example_path).compile()
    result = solve(instance, config=ACOConfig(), seed=1)
    assert result.evaluation.feasible
    assert result.stats["mode"] == "linear"
    assert result.objective == pytest.approx(result.evaluation.total_cost)
    assert result.stats["score"] == pytest.approx(result.objective, rel=1e-9)
    report = check_plan(instance, result.plan)
    assert report.feasible
    assert report.total_cost == pytest.approx(result.evaluation.total_cost, rel=1e-12)


@pytest.mark.parametrize("seed", range(40))
def test_rules_hold_by_construction(seed: int, random_document: RandomDocument) -> None:
    instance = parse_spec(random_document(seed)).compile()
    result = solve(instance, seed=seed)
    evaluation = result.evaluation
    broken = {name: evaluation.violations[name] for name in BY_CONSTRUCTION}
    assert not any(broken.values()), broken
    if np.all(instance.spot_available == -1):
        plain = solve(instance, config=replace(SMALL, local_search=False), seed=seed)
        assert plain.evaluation.violations["coverage"] == 0.0
        assert result.objective <= plain.objective * (1 + 1e-12)
    assert result.objective == pytest.approx(default_objective()(evaluation))
    assert result.stats["score"] == pytest.approx(result.objective, rel=1e-9)
    report = check_plan(instance, result.plan)
    assert report.total_cost == pytest.approx(evaluation.total_cost, rel=1e-12)
    for name, amount in report.totals().items():
        assert evaluation.violations[name] == pytest.approx(amount, rel=1e-9, abs=1e-6)


def test_whole_items_when_demand_may_not_be_split(random_document: RandomDocument) -> None:
    for seed in range(60):
        document = random_document(seed)
        if not document["rules"]["split_demand"]:
            break
    instance = parse_spec(document).compile()
    plan = solve(instance, seed=0).plan
    assert np.array_equal(plan.load_quantity, instance.demand_quantity[plan.load_demand])
    assert len(set(plan.load_demand.tolist())) == plan.load_demand.size


def test_same_seed_same_plan(example_path: Path) -> None:
    instance = pirilib.load_spec(example_path).compile()
    first, second = solve(instance, seed=3), solve(instance, seed=3)
    for name in ("day", "kind", "source", "origin", "stop_start", "stop_center", "load_quantity"):
        assert np.array_equal(getattr(first.plan, name), getattr(second.plan, name)), name


def test_objective_with_a_linear_form_steers_the_search(instance: LineHaulInstance) -> None:
    cheap = solve(instance, objective=obj.total_cost() + 1e6 * obj.violations(), seed=0)
    few_spot = solve(
        instance,
        objective=obj.total_cost() + 1e6 * obj.violations() + 1e5 * obj.dispatches("spot"),
        seed=0,
    )

    def spot(result: pirilib.SolveResult) -> int:
        return int(np.count_nonzero(result.plan.kind == 1))

    assert spot(few_spot) <= spot(cheap)
    assert few_spot.evaluation.feasible


def test_plain_function_objectives_use_whole_plans(instance: LineHaulInstance) -> None:
    def goal(evaluation: pirilib.Evaluation) -> float:
        return evaluation.total_cost + 1e6 * sum(evaluation.violations.values())

    result = solve(instance, objective=goal, seed=0)
    assert result.stats["mode"] == "general"
    assert result.evaluation.feasible
    assert result.objective == pytest.approx(goal(result.evaluation))


def test_constraints_of_your_own_use_whole_plans(instance: LineHaulInstance) -> None:
    constraint = MaxDispatchesPerDay(5)
    result = solve(instance, constraints=[constraint], seed=0)
    assert result.stats["mode"] == "general"
    assert "max_dispatches_per_day" in result.evaluation.violations
    assert result.evaluation.feasible


def test_spot_limits_are_shared_by_the_origins_of_a_day(
    document: dict[str, Any], compile_document: Any
) -> None:
    document["fleet"] = {"spot": [{"type": "small", "cost": {"fixed": 1.0}, "available": 2}]}
    document["demand"] = [
        {"origin": "A", "destination": "C", "date": "2026-01-05", "quantity": {"desi": 300.0}},
        {"origin": "B", "destination": "C", "date": "2026-01-05", "quantity": {"desi": 300.0}},
        {"origin": "C", "destination": "A", "date": "2026-01-05", "quantity": {"desi": 300.0}},
    ]
    instance = compile_document(document)
    for result in (solve(instance, seed=0), solve(instance, objective=lambda e: e.total_cost)):
        assert result.evaluation.violations["spot_availability"] == 0
        assert result.evaluation.violations["coverage"] == 300.0


def test_unknown_linear_features_are_rejected(instance: LineHaulInstance) -> None:
    odd = obj.Term("odd", lambda e: 0.0, linear={"tolls": 1.0})
    with pytest.raises(ValueError, match=r"unknown features \['tolls'\]"):
        solve(instance, objective=obj.total_cost() + odd)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"ants": 0}, "ants must be at least 1"),
        ({"evaporation": 0.0}, "evaporation must be in"),
        ({"tau_min": 2.0}, "tau_min must be in"),
        ({"update": "elitist"}, "update must be"),
    ],
)
def test_config_is_checked(change: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ACOConfig(**change)


def test_solve_shortcut(instance: LineHaulInstance) -> None:
    result = pirilib.solve(instance, solver=ACOSolver(SMALL))
    assert result.evaluation.feasible
    assert evaluate(instance, result.plan).total_cost == pytest.approx(result.evaluation.total_cost)
    assert repr(result).startswith("SolveResult(objective=")


def test_scheduled_problems_are_planned_piece_by_piece(
    document: dict[str, Any], compile_document: Any
) -> None:
    document["schema"] = "pirilib/linehaul@0.2"
    for vehicle in document["vehicle_types"]:
        vehicle["speed_kmh"] = 60.0
    document["demand"][0]["ready"] = "09:00"
    document["rules"] = {"service": {"due_hours": 6, "penalty": 1.0}}
    instance = compile_document(document)
    assert instance.scheduled
    result = solve(instance, seed=0)
    assert result.stats["mode"] == "linear"
    for name in ("route", "load_route", "ready", "deferral", "ban", "owned_line", "capacity"):
        assert result.evaluation.violations[name] == 0, name
    assert result.objective == pytest.approx(default_objective()(result.evaluation))
    assert result.stats["score"] == pytest.approx(result.objective, rel=1e-9)

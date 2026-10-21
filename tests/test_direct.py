from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from pirilib import check_plan
from pirilib.plan import OWNED, SPOT
from pirilib.solvers.direct import DirectShipping
from pirilib.spec import LineHaulInstance, parse_spec

type RandomDocument = Callable[[int], dict[str, Any]]


def without_fill_and_limits(document: dict[str, Any]) -> dict[str, Any]:
    document["rules"]["min_fill"] = {"spot": 0.0}
    for offer in document["fleet"]["spot"]:
        offer.pop("available", None)
    return document


@pytest.mark.parametrize("seed", range(30))
def test_feasible_without_min_fill_and_spot_limits(
    seed: int, random_document: RandomDocument
) -> None:
    instance = parse_spec(without_fill_and_limits(random_document(seed))).compile()
    result = DirectShipping().solve(instance, seed=seed)
    assert result.evaluation.feasible, result.evaluation.violated()
    report = check_plan(instance, result.plan)
    assert report.feasible, str(report)
    assert report.total_cost == pytest.approx(result.evaluation.total_cost, rel=1e-12)
    assert np.all(np.diff(result.plan.stop_start) == 1)
    assert result.stats["mode"] == "direct"


def test_full_vehicles_then_the_cheapest_that_fits(
    document: dict[str, Any], compile_document: Callable[[dict[str, Any]], LineHaulInstance]
) -> None:
    document["demand"] = [
        {"origin": "A", "destination": "B", "date": "2026-01-05", "quantity": {"desi": 2300.0}}
    ]
    instance = compile_document(document)
    plan = DirectShipping().solve(instance).plan
    used = sorted(zip(plan.kind.tolist(), plan.source.tolist(), strict=True))
    assert used == [(OWNED, 0), (SPOT, 0), (SPOT, 1)]
    loads = {
        (int(plan.kind[d]), int(plan.source[d])): float(q[0])
        for d, q in zip(plan.load_dispatch, plan.load_quantity, strict=True)
    }
    assert loads[(OWNED, 0)] == pytest.approx(1000.0)
    assert loads[(SPOT, 0)] == pytest.approx(1000.0)
    assert loads[(SPOT, 1)] == pytest.approx(300.0)
    assert check_plan(instance, plan).feasible


def test_bans_and_owned_lines_are_kept(instance: LineHaulInstance) -> None:
    result = DirectShipping().solve(instance)
    assert result.evaluation.feasible
    for dispatch in result.plan.dispatches():
        if dispatch.stops == (2,):
            assert dispatch.kind == "spot"
            assert dispatch.source == 1
        if dispatch.kind == "owned":
            assert (dispatch.origin, dispatch.stops) == (0, (1,))


def test_spot_limits_leave_demand_uncarried(
    document: dict[str, Any], compile_document: Callable[[dict[str, Any]], LineHaulInstance]
) -> None:
    document["fleet"] = {"spot": [{"type": "small", "cost": {"fixed": 1.0}, "available": 2}]}
    document["demand"] = [
        {"origin": "A", "destination": "C", "date": "2026-01-05", "quantity": {"desi": 300.0}},
        {"origin": "B", "destination": "C", "date": "2026-01-05", "quantity": {"desi": 300.0}},
        {"origin": "C", "destination": "A", "date": "2026-01-05", "quantity": {"desi": 300.0}},
    ]
    evaluation = DirectShipping().solve(compile_document(document)).evaluation
    assert evaluation.violations["spot_availability"] == 0
    assert evaluation.violations["coverage"] == 300.0


def test_prefers_a_vehicle_that_reaches_its_minimum_fill(
    document: dict[str, Any], compile_document: Callable[[dict[str, Any]], LineHaulInstance]
) -> None:
    document["fleet"] = {
        "spot": [
            {"type": "big", "cost": {"fixed": 100.0}},
            {"type": "small", "cost": {"fixed": 150.0}},
        ]
    }
    document["demand"] = [
        {"origin": "A", "destination": "B", "date": "2026-01-05", "quantity": {"desi": 350.0}}
    ]
    document["rules"] = {"min_fill": {"spot": 0.5}}
    result = DirectShipping().solve(compile_document(document))
    assert result.plan.source.tolist() == [1]
    assert result.evaluation.feasible
    document["rules"] = {"min_fill": {"spot": 0.95}}
    result = DirectShipping().solve(compile_document(document))
    assert result.plan.source.tolist() == [0]
    assert result.evaluation.violated() == {"min_fill": pytest.approx(600.0)}


def test_whole_demands_when_they_may_not_be_split(random_document: RandomDocument) -> None:
    checked = 0
    for seed in range(60):
        document = random_document(seed)
        if document["rules"]["split_demand"]:
            continue
        instance = parse_spec(without_fill_and_limits(document)).compile()
        result = DirectShipping().solve(instance)
        plan = result.plan
        assert np.array_equal(plan.load_quantity, instance.demand_quantity[plan.load_demand])
        assert len(set(plan.load_demand.tolist())) == plan.load_demand.size
        assert result.evaluation.feasible, result.evaluation.violated()
        assert check_plan(instance, plan).feasible
        checked += 1
    assert checked >= 5

"""Times, lateness, deferral, handling capacity and vehicle limits (schema 0.2)."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from pirilib import (
    ACOConfig,
    ACOSolver,
    Dispatch,
    Load,
    Plan,
    SpecError,
    check_plan,
    evaluate,
    synthetic,
)
from pirilib.solvers import default_objective
from pirilib.solvers.exact import ExactSolver
from pirilib.spec import LineHaulInstance, parse_spec

A, B, C = 0, 1, 2
BIG, SMALL = 0, 1


def timed(document: dict[str, Any]) -> dict[str, Any]:
    """The test document with speeds, hourly costs, ready times and a 3-hour service."""
    document["schema"] = "pirilib/linehaul@0.2"
    for vehicle in document["vehicle_types"]:
        vehicle["speed_kmh"] = 60.0
    document["fleet"]["owned"][0]["cost"]["per_hour"] = 10.0
    document["demand"][0]["ready"] = "09:00"
    document["rules"] = {
        "service": {"due_hours": 3, "penalty": 0.4},
        "handling": {"minutes_per_unit": 0.1},
    }
    return document


def plan(*dispatches: Dispatch) -> Plan:
    return Plan.from_dispatches(dispatches, 1)


OWNED_AB = Dispatch(day=0, kind="owned", source=0, origin=A, stops=(B,), loads=(Load(0, (900.0,)),))
SMALL_AC = Dispatch(
    day=0, kind="spot", source=SMALL, origin=A, stops=(C,), loads=(Load(1, (250.0,)),)
)
SMALL_BA = Dispatch(
    day=1, kind="spot", source=SMALL, origin=B, stops=(A,), loads=(Load(2, (120.0,)),)
)


def agree(instance: LineHaulInstance, chosen: Plan) -> Any:
    evaluation = evaluate(instance, chosen)
    report = check_plan(instance, chosen)
    assert report.total_cost == pytest.approx(evaluation.total_cost, rel=1e-12)
    for name, amount in report.totals().items():
        assert evaluation.violations[name] == pytest.approx(amount), name
    return evaluation


def test_timeline_and_lateness(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    instance = compile_document(timed(document))
    evaluation = agree(instance, plan(OWNED_AB, SMALL_AC, SMALL_BA))
    assert evaluation.start_minute[0] == 540.0
    assert evaluation.end_minute[0] == 540 + 90 + 112 + 90
    assert evaluation.hours[0] == pytest.approx((832 - 540) / 60)
    assert evaluation.load_late_hours.tolist()[0] == 2.0
    assert evaluation.end_minute[1] == 273.0
    assert evaluation.load_late_hours.tolist()[1] == 2.0
    assert evaluation.load_late_hours.tolist()[2] == 0.0
    assert evaluation.service_cost == pytest.approx(0.4 * (900 * 2 + 250 * 2))
    hourly = 10.0 * (832 - 540) / 60
    assert evaluation.dispatch_cost[0] == pytest.approx(
        1.0 * 6371.0 * 3.141592653589793 / 180 + hourly
    )
    assert evaluation.feasible


def test_loading_before_ready(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    instance = compile_document(timed(document))
    early = Dispatch(
        day=0, kind="owned", source=0, origin=A, stops=(B,), loads=OWNED_AB.loads, start=480.0
    )
    evaluation = agree(instance, plan(early, SMALL_AC, SMALL_BA))
    assert evaluation.violated() == {"ready": 1}
    late_start = Dispatch(
        day=0, kind="owned", source=0, origin=A, stops=(B,), loads=OWNED_AB.loads, start=600.0
    )
    evaluation = agree(instance, plan(late_start, SMALL_AC, SMALL_BA))
    assert evaluation.feasible
    assert evaluation.start_minute[0] == 600.0


def test_deferral(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    document = timed(document)
    later = Dispatch(day=1, kind="owned", source=0, origin=A, stops=(B,), loads=OWNED_AB.loads)
    instance = compile_document(document)
    assert agree(instance, plan(later, SMALL_AC, SMALL_BA)).violated() == {"deferral": 1}

    document["rules"]["service"]["max_deferral_days"] = 1
    instance = compile_document(document)
    evaluation = agree(instance, plan(later, SMALL_AC, SMALL_BA))
    assert evaluation.feasible
    assert evaluation.start_minute[0] == 1440.0
    assert evaluation.load_late_hours.tolist()[0] == 17.0


def test_handling_capacity_and_vehicle_limits(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    document = timed(document)
    document["centers"][0]["handling_capacity"] = 1000.0
    document["centers"][0]["vehicle_limits"] = {"small": 0}
    instance = compile_document(document)
    evaluation = agree(instance, plan(OWNED_AB, SMALL_AC, SMALL_BA))
    assert evaluation.violated() == {"handling": 150.0, "vehicle_limit": 2}


def test_untimed_documents_behave_as_before(instance: LineHaulInstance) -> None:
    evaluation = evaluate(instance, plan(OWNED_AB, SMALL_AC, SMALL_BA))
    assert evaluation.service_cost == 0.0
    assert not instance.timed
    assert not instance.scheduled
    assert evaluation.start_minute.tolist() == [0.0, 0.0, 1440.0]


def test_times_need_speeds(document: dict[str, Any]) -> None:
    document["rules"] = {"service": {"due_hours": 3}}
    document["demand"][0]["due"] = "2026-01-04T12:00:00"
    with pytest.raises(SpecError) as info:
        parse_spec(document)
    lines = str(info.value).splitlines()[1:]
    assert (
        "- vehicle_types[0].speed_kmh: is required, because hourly costs or due times are used"
        in lines
    )
    assert any(
        line.startswith("- demand[0].due: 2026-01-04 12:00:00 is before it is ready")
        for line in lines
    )


def test_unknown_vehicle_limit_type(document: dict[str, Any]) -> None:
    document["centers"][1]["vehicle_limits"] = {"huge": 1}
    with pytest.raises(
        SpecError, match=r"centers\[1\].vehicle_limits: unknown vehicle type 'huge'"
    ):
        parse_spec(document)


KEPT = ("route", "load_route", "ready", "deferral", "ban", "owned_line", "capacity", "split")
QUICK = ACOConfig(ants=6, iterations=8, patience=4)


def synthetic_case(capacity: float | None = 60000.0, deferral: int = 0) -> LineHaulInstance:
    """The synthetic case of the v0.3 measurements: 10 centers, 2 days, 80 demands."""
    document = synthetic.generate(centers=10, days=2, demand_per_day=40, seed=3)
    document["schema"] = "pirilib/linehaul@0.2"
    for vehicle in document["vehicle_types"]:
        vehicle["speed_kmh"] = 70.0
    for entry in [*document["fleet"]["owned"], *document["fleet"]["spot"]]:
        entry["cost"]["per_hour"] = 320.0
    for position, row in enumerate(document["demand"]):
        row["ready"] = "17:00" if position % 2 == 0 else "09:00"
    document["rules"]["service"] = {
        "due_hours": 24,
        "penalty": 0.5,
        "max_deferral_days": deferral,
    }
    document["rules"]["handling"] = {"minutes_per_unit": 0.01}
    if capacity is not None:
        document["centers"][0]["handling_capacity"] = capacity
    return parse_spec(document).compile()


def scheduled_document(seed: int, base: dict[str, Any]) -> dict[str, Any]:
    """A random problem with times, deferral, handling capacities and vehicle limits."""
    rng = np.random.default_rng(1000 + seed)
    base["schema"] = "pirilib/linehaul@0.2"
    for vehicle in base["vehicle_types"]:
        vehicle["speed_kmh"] = float(rng.choice([50.0, 70.0, 90.0]))
    for entry in [*base["fleet"]["owned"], *base["fleet"]["spot"]]:
        entry["cost"]["per_hour"] = float(rng.uniform(0, 60))
    for row in base["demand"]:
        row["ready"] = str(rng.choice(["00:00", "09:00", "17:00", "21:30"]))
    base["rules"]["service"] = {
        "due_hours": float(rng.choice([4, 8, 24])),
        "penalty": float(rng.choice([0.0, 0.2, 1.0])),
        "max_deferral_days": int(rng.integers(0, 3)),
    }
    base["rules"]["handling"] = {"minutes_per_unit": float(rng.choice([0.0, 0.01, 0.1]))}
    for center in base["centers"]:
        if rng.random() < 0.4:
            center["handling_capacity"] = float(rng.uniform(300, 3000))
        if rng.random() < 0.3:
            center["vehicle_limits"] = {str(rng.choice(["big", "small"])): int(rng.integers(0, 3))}
    return base


def test_the_colony_plans_with_handling_capacity() -> None:
    instance = synthetic_case()
    result = ACOSolver().solve(instance, seed=0)
    violations = result.evaluation.violations
    assert result.stats["mode"] == "linear"
    for name in ("handling", "vehicle_limit", "ready", "deferral"):
        assert violations[name] == 0, name
    assert result.stats["score"] == pytest.approx(result.objective, rel=1e-6)

    roomy = ACOSolver().solve(synthetic_case(250000.0), seed=0)
    assert roomy.evaluation.feasible
    assert roomy.evaluation.total_cost < 1_059_052
    assert roomy.stats["score"] == pytest.approx(roomy.evaluation.total_cost, rel=1e-6)


def test_whole_plans_keep_handling_capacity() -> None:
    result = ACOSolver(QUICK).solve(
        synthetic_case(), objective=lambda evaluation: default_objective()(evaluation), seed=0
    )
    assert result.stats["mode"] == "general"
    for name in ("handling", "vehicle_limit", "ready", "deferral"):
        assert result.evaluation.violations[name] == 0, name


@pytest.mark.parametrize("seed", range(16))
def test_the_score_is_the_evaluated_objective(
    seed: int, random_document: Callable[[int], dict[str, Any]]
) -> None:
    instance = parse_spec(scheduled_document(seed, random_document(seed))).compile()
    assert instance.scheduled
    result = ACOSolver(QUICK).solve(instance, seed=seed)
    evaluation = result.evaluation
    assert result.stats["mode"] == "linear"
    assert result.objective == pytest.approx(default_objective()(evaluation))
    assert result.stats["score"] == pytest.approx(result.objective, rel=1e-6)
    if evaluation.feasible:
        assert result.stats["score"] == pytest.approx(evaluation.total_cost, rel=1e-6)
    for name in KEPT:
        assert evaluation.violations[name] == 0, name


def deferral_document(document: dict[str, Any], days: int) -> dict[str, Any]:
    """900 desi from A to B, but B handles only 500 a day."""
    document = timed(document)
    document["fleet"] = {"spot": [{"type": "big", "cost": {"fixed": 300.0, "per_km": 2.0}}]}
    document["centers"][1]["handling_capacity"] = 500.0
    document["demand"] = [
        {
            "origin": "A",
            "destination": "B",
            "date": "2026-01-05",
            "ready": "09:00",
            "quantity": {"desi": 900.0},
        }
    ]
    document["rules"] = {
        "service": {"due_hours": 6, "penalty": 1.0, "max_deferral_days": days},
        "handling": {"minutes_per_unit": 0.1},
    }
    return document


def test_what_does_not_fit_waits_for_the_next_day(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    instance = compile_document(deferral_document(document, 1))
    result = ACOSolver(QUICK).solve(instance, seed=0)
    evaluation = result.evaluation
    plan = evaluation.plan
    assert evaluation.feasible
    assert plan.day.tolist() == [0, 1]
    assert plan.load_demand.tolist() == [0, 0]
    assert plan.load_quantity[:, 0].tolist() == [500.0, 400.0]
    assert evaluation.start_minute.tolist() == [540.0, 1440.0]
    assert instance.demand_due[0] == 900.0
    assert evaluation.load_late_hours.tolist() == [0.0, 13.0]
    assert evaluation.service_cost == pytest.approx(400.0 * 13.0)
    assert result.stats["score"] == pytest.approx(evaluation.total_cost, rel=1e-9)


def test_without_deferral_what_does_not_fit_is_not_carried(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    instance = compile_document(deferral_document(document, 0))
    result = ACOSolver(QUICK).solve(instance, seed=0)
    assert result.evaluation.violated() == {"coverage": 400.0}
    assert result.evaluation.plan.day.tolist() == [0]
    assert result.stats["score"] == pytest.approx(result.objective, rel=1e-9)


def test_vehicle_limits_are_kept(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    document = timed(document)
    document["centers"][1]["vehicle_limits"] = {"big": 0}
    instance = compile_document(document)
    result = ACOSolver(QUICK).solve(instance, seed=0)
    evaluation = result.evaluation
    assert evaluation.violations["vehicle_limit"] == 0
    assert evaluation.violations["ban"] == 0
    plan = evaluation.plan
    stop_type = evaluation.vehicle_type[np.repeat(np.arange(len(plan)), np.diff(plan.stop_start))]
    assert not np.any((plan.stop_center == B) & (stop_type == BIG))


def test_untimed_plans_did_not_change() -> None:
    instance = parse_spec(
        synthetic.generate(centers=10, days=2, demand_per_day=40, seed=3)
    ).compile()
    result = ACOSolver().solve(instance, seed=0)
    assert not instance.scheduled
    assert result.stats["score"] == pytest.approx(855347.2161499044, rel=1e-12)
    assert len(result.plan) == 40


def test_the_exact_solver_does_not_plan_times_yet(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    instance = compile_document(timed(document))
    with pytest.raises(ValueError, match="not yet supported"):
        ExactSolver().solve(instance)

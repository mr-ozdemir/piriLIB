"""Times, lateness, deferral, handling capacity and vehicle limits (schema 0.2)."""

from collections.abc import Callable
from typing import Any

import pytest

from pirilib import Dispatch, Load, Plan, SpecError, check_plan, evaluate
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

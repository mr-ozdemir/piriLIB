"""The plan report: one JSON document with the plan and everything computed about it.

The plan and the problem are those of ``tests/test_plan_output.py``: the timed fixture,
the owned ``big`` vehicle with 900 desi from ``A`` to ``B`` (09:00 to 13:52, due 12:00) and
a ``small`` spot vehicle with 250 desi from ``A`` to ``C``; the 120 desi from ``B`` to ``A``
on the second day stay behind. The reasons of uncarried demand are tested on problems of
their own: one where the vehicles run out, one where a center is full and one where
nothing is in the way.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from pirilib import Dispatch, Evaluation, Load, Plan, evaluate
from pirilib import objective as obj
from pirilib.report import FORMAT, Reason, plan_report, unserved, write_report
from pirilib.spec import LineHaulInstance

type CompileDocument = Callable[[dict[str, Any]], LineHaulInstance]

A, B, C = 0, 1, 2
BIG, SMALL = 0, 1

HOURLY = 10.0
OWNED_HOURS = (90 + 112 + 90) / 60

GOAL = obj.total_cost() + 1000 * obj.violation("coverage")

OWNED_AB = Dispatch(day=0, kind="owned", source=0, origin=A, stops=(B,), loads=(Load(0, (900.0,)),))
SMALL_AC = Dispatch(
    day=0, kind="spot", source=SMALL, origin=A, stops=(C,), loads=(Load(1, (250.0,)),)
)
SMALL_BA = Dispatch(
    day=1, kind="spot", source=SMALL, origin=B, stops=(A,), loads=(Load(2, (120.0,)),)
)
BIG_AB_THEN_C = Dispatch(
    day=0,
    kind="spot",
    source=BIG,
    origin=A,
    stops=(B, C),
    loads=(Load(0, (600.0,)), Load(1, (250.0,))),
)
"""One vehicle that leaves 600 desi at ``B`` and 250 at ``C`` (``C`` must allow its type)."""


def timed(document: dict[str, Any]) -> dict[str, Any]:
    """The fixture with speeds, handling time, an hourly cost and due times."""
    document["vehicle_types"][0]["speed_kmh"] = 60.0
    document["vehicle_types"][1]["speed_kmh"] = 60.0
    document["fleet"]["owned"][0]["cost"]["per_hour"] = HOURLY
    document["demand"][0]["ready"] = "09:00"
    document["rules"] = {
        "service": {"due_hours": 3, "penalty": 0.4},
        "handling": {"minutes_per_unit": 0.1},
    }
    return document


def plan(*dispatches: Dispatch) -> Plan:
    return Plan.from_dispatches(dispatches, 1)


@pytest.fixture
def evaluation(document: dict[str, Any], compile_document: CompileDocument) -> Evaluation:
    """The module's plan on the timed fixture: two trips on the first day, none on the second."""
    return evaluate(compile_document(timed(document)), plan(OWNED_AB, SMALL_AC))


def reasons(evaluation: Evaluation) -> dict[str, str]:
    """The reason of every demand the report lists as not carried, by demand id."""
    return {row["demand"]: row["reason"] for row in plan_report(evaluation)["unserved"]}


def test_the_report_names_its_format(evaluation: Evaluation) -> None:
    report = plan_report(evaluation)

    assert report["format"] == FORMAT == "pirilib/plan-report@1"


def test_the_report_describes_the_problem(evaluation: Evaluation) -> None:
    report = plan_report(evaluation)

    assert report["problem"] == {
        "centers": 3,
        "demands": 3,
        "first_day": "2026-01-05",
        "last_day": "2026-01-06",
        "dimensions": ["desi"],
        "primary_dimension": "desi",
    }


def test_the_report_lists_every_dispatch(evaluation: Evaluation) -> None:
    report = plan_report(evaluation)

    assert [row["source"] for row in report["dispatches"]] == ["fleet.owned[0]", "fleet.spot[1]"]


def test_a_dispatch_has_its_times_as_local_dates_and_times(evaluation: Evaluation) -> None:
    first = plan_report(evaluation)["dispatches"][0]

    assert (first["start"], first["end"]) == ("2026-01-05T09:00", "2026-01-05T13:52")


def test_a_dispatch_has_its_cost_and_the_parts_of_it(evaluation: Evaluation) -> None:
    first = plan_report(evaluation)["dispatches"][0]

    assert first["cost"] == {
        "total": pytest.approx(111.19492664455873 + HOURLY * OWNED_HOURS),
        "fixed": 0.0,
        "km": pytest.approx(111.19492664455873),
        "hours": pytest.approx(HOURLY * OWNED_HOURS),
        "units": 0.0,
    }


def test_a_dispatch_says_how_full_it_is(evaluation: Evaluation) -> None:
    first = plan_report(evaluation)["dispatches"][0]

    assert (first["load"], first["capacity"], first["fill"]) == (
        {"desi": 900.0},
        {"desi": 1000.0},
        0.9,
    )


def test_a_load_says_when_it_is_due_and_how_late_it_is(evaluation: Evaluation) -> None:
    load = plan_report(evaluation)["dispatches"][0]["loads"][0]

    assert load == {
        "demand": "D000001",
        "destination": "B",
        "quantity": {"desi": 900.0},
        "unloaded": "2026-01-05T13:52",
        "due": "2026-01-05T12:00",
        "late_hours": 2,
    }


def test_a_stop_says_what_is_unloaded_there_and_what_stays_onboard(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    """850 desi leave ``A``: 600 come off at ``B``, the other 250 at ``C``."""
    document["centers"][C]["banned_vehicle_types"] = []
    evaluation = evaluate(compile_document(document), plan(BIG_AB_THEN_C))

    stops = plan_report(evaluation)["dispatches"][0]["stops"]

    assert [(stop["center"], stop["unloaded"], stop["onboard"]) for stop in stops] == [
        ("B", {"desi": 600.0}, {"desi": 250.0}),
        ("C", {"desi": 250.0}, {"desi": 0.0}),
    ]


def test_a_stop_says_when_the_vehicle_arrives_and_when_it_is_done(evaluation: Evaluation) -> None:
    stop = plan_report(evaluation)["dispatches"][0]["stops"][0]

    assert (stop["arrival"], stop["finish"]) == ("2026-01-05T12:22", "2026-01-05T13:52")


def test_a_spot_dispatch_carries_the_name_of_its_offer(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    document["fleet"]["spot"][SMALL]["name"] = "market"
    evaluation = evaluate(compile_document(document), plan(OWNED_AB, SMALL_AC))

    offers = [row["offer"] for row in plan_report(evaluation)["dispatches"]]

    assert offers == [None, "market"]


def test_the_report_holds_the_summary(evaluation: Evaluation) -> None:
    report = plan_report(evaluation)

    assert report["summary"]["served"] == pytest.approx(1150.0 / 1270.0)


def test_the_report_holds_the_objective_by_term(evaluation: Evaluation) -> None:
    report = plan_report(evaluation, objective=GOAL)

    assert report["objective"] == {
        "value": pytest.approx(evaluation.total_cost + 1000 * 120.0),
        "terms": {
            "total_cost": pytest.approx(evaluation.total_cost),
            "violation[coverage]": 120000.0,
        },
    }


def test_an_objective_that_is_a_function_has_a_value_and_no_terms(evaluation: Evaluation) -> None:
    report = plan_report(evaluation, objective=lambda evaluation: 7.0)

    assert report["objective"] == {"value": 7.0}


def test_the_report_holds_the_stats_it_is_given(evaluation: Evaluation) -> None:
    report = plan_report(evaluation, stats={"mode": "linear"})

    assert report["stats"] == {"mode": "linear"}


def test_without_an_objective_and_stats_the_report_leaves_them_out(evaluation: Evaluation) -> None:
    report = plan_report(evaluation)

    assert ("objective" in report, "stats" in report) == (False, False)


def test_the_report_lists_what_is_not_carried(evaluation: Evaluation) -> None:
    report = plan_report(evaluation)

    assert report["unserved"] == [
        {
            "demand": "D000003",
            "origin": "B",
            "destination": "A",
            "date": "2026-01-06",
            "quantity": {"desi": 120.0},
            "reason": "by_choice",
        }
    ]


def test_a_plan_that_carries_everything_lists_nothing(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    evaluation = evaluate(compile_document(document), plan(OWNED_AB, SMALL_AC, SMALL_BA))

    assert unserved(evaluation) == []


def test_a_demand_left_while_a_vehicle_and_room_were_free_is_left_by_choice(
    evaluation: Evaluation,
) -> None:
    found = unserved(evaluation)

    assert [(item.demand, item.quantity, item.reason) for item in found] == [
        (2, (120.0,), Reason.BY_CHOICE)
    ]


def test_a_demand_left_when_every_vehicle_is_in_use_has_no_vehicle(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    """One ``small`` vehicle a day is the whole fleet, and ``A`` to ``C`` uses it."""
    document["fleet"] = {"spot": [{"type": "small", "cost": {"fixed": 1.0}, "available": 1}]}
    document["demand"][0]["quantity"]["desi"] = 300.0
    trip = Dispatch(day=0, kind="spot", source=0, origin=A, stops=(C,), loads=(Load(1, (250.0,)),))
    evaluation = evaluate(compile_document(document), plan(trip))

    found = reasons(evaluation)

    assert found == {"D000001": "no_vehicle", "D000003": "by_choice"}


def test_a_vehicle_free_on_a_later_day_counts_when_the_demand_may_wait(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    """The same fleet, but demand may leave a day late, and the vehicle is free on day two."""
    document["fleet"] = {"spot": [{"type": "small", "cost": {"fixed": 1.0}, "available": 1}]}
    document["demand"][0]["quantity"]["desi"] = 300.0
    document["rules"] = {"service": {"max_deferral_days": 1}}
    trip = Dispatch(day=0, kind="spot", source=0, origin=A, stops=(C,), loads=(Load(1, (250.0,)),))
    evaluation = evaluate(compile_document(document), plan(trip))

    found = reasons(evaluation)

    assert found["D000001"] == "by_choice"


def test_a_demand_left_at_a_center_without_handling_capacity_has_a_center_limit(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    """``A`` handles 250 desi a day, which the trip to ``C`` uses up."""
    document["centers"][A]["handling_capacity"] = 250.0
    evaluation = evaluate(compile_document(document), plan(SMALL_AC, SMALL_BA))

    found = reasons(evaluation)

    assert found == {"D000001": "center_limit"}


def test_a_demand_left_at_a_center_without_a_vehicle_slot_has_a_center_limit(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    """Only ``big`` vehicles can go from ``A`` to ``B`` here, and ``B`` takes none of them."""
    document["fleet"]["spot"][SMALL]["available"] = 1
    document["centers"][B]["vehicle_limits"] = {"big": 0}
    evaluation = evaluate(compile_document(document), plan(SMALL_AC, SMALL_BA))

    found = reasons(evaluation)

    assert found == {"D000001": "center_limit"}


def test_the_report_is_written_as_json(evaluation: Evaluation, tmp_path: Path) -> None:
    path = write_report(evaluation, tmp_path / "report.json", objective=GOAL)

    document = json.loads(path.read_text(encoding="utf-8"))

    assert (document["format"], len(document["dispatches"])) == (FORMAT, 2)


def test_a_report_without_dispatches_is_valid_json(
    document: dict[str, Any], compile_document: CompileDocument, tmp_path: Path
) -> None:
    """The indicators that have nothing to divide by are null, not NaN."""
    evaluation = evaluate(compile_document(document), plan())

    path = write_report(evaluation, tmp_path / "report.json")

    assert json.loads(path.read_text(encoding="utf-8"))["summary"]["fill"] is None

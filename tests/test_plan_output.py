"""What a written plan says beyond the plan itself: cost parts, times and indicators.

The plan is hand-made for the shared fixture, timed as in ``tests/test_schedule.py``: both
vehicle types drive 60 km/h, loading and unloading take 0.1 minute per desi, the owned
vehicle costs 10 an hour and the spot ``small`` offer 0.5 per desi carried. The owned ``big``
vehicle takes 900 desi from ``A`` to ``B``, starting when the load is ready at 09:00:
90 minutes of loading, 112 minutes on the road and 90 minutes of unloading end at 13:52,
against a due time of 12:00. A ``small`` vehicle takes 250 desi from ``A`` to ``C``. The
120 desi from ``B`` to ``A`` on the second day stay behind.
"""

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from pirilib import Dispatch, Evaluation, Load, Plan, evaluate
from pirilib.planio import columns, read_plan, rows, summary, write_plan
from pirilib.spec import LineHaulInstance

type CompileDocument = Callable[[dict[str, Any]], LineHaulInstance]

A, B, C = 0, 1, 2
SMALL = 1

KM_PER_DEGREE = 111.19492664455873
"""One degree of the equator, the distance from ``A`` to ``B``."""

HOURLY = 10.0
PER_DESI = 0.5

OWNED_HOURS = (90 + 112 + 90) / 60
"""Loading, driving and unloading of the owned dispatch."""

DEMANDED = 900.0 + 250.0 + 120.0
CARRIED = 900.0 + 250.0
CAPACITY = 1000.0 + 400.0
"""The ``big`` and the ``small`` vehicle of the plan."""

OWNED_AB = Dispatch(day=0, kind="owned", source=0, origin=A, stops=(B,), loads=(Load(0, (900.0,)),))
SMALL_AC = Dispatch(
    day=0, kind="spot", source=SMALL, origin=A, stops=(C,), loads=(Load(1, (250.0,)),)
)
EMPTY = Dispatch(day=1, kind="spot", source=SMALL, origin=B, stops=(A,))
"""A dispatch without loads: it has costs and times, but nothing to unload."""


def timed(document: dict[str, Any]) -> dict[str, Any]:
    """The fixture with speeds, handling time, an hourly and a per unit cost and due times."""
    document["vehicle_types"][0]["speed_kmh"] = 60.0
    document["vehicle_types"][1]["speed_kmh"] = 60.0
    document["fleet"]["owned"][0]["cost"]["per_hour"] = HOURLY
    document["fleet"]["spot"][SMALL]["cost"]["per_unit"] = {"desi": PER_DESI}
    document["demand"][0]["ready"] = "09:00"
    document["rules"] = {
        "service": {"due_hours": 3, "penalty": 0.4},
        "handling": {"minutes_per_unit": 0.1},
    }
    return document


@pytest.fixture
def evaluation(document: dict[str, Any], compile_document: CompileDocument) -> Evaluation:
    """The evaluation of the module's plan on the timed fixture."""
    instance = compile_document(timed(document))
    return evaluate(instance, Plan.from_dispatches((OWNED_AB, SMALL_AC), 1))


def written(evaluation: Evaluation, folder: Path) -> list[dict[str, str]]:
    """The rows of the plan file of ``evaluation``, as the text a reader finds in it."""
    path = folder / "plan.csv"
    write_plan(evaluation, path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def test_the_cost_parts_add_up_to_the_dispatch_cost(evaluation: Evaluation) -> None:
    parts = (
        evaluation.cost_fixed + evaluation.cost_km + evaluation.cost_hours + evaluation.cost_units
    )

    assert parts.tolist() == evaluation.dispatch_cost.tolist()


def test_a_mandatory_owned_dispatch_has_no_fixed_part(evaluation: Evaluation) -> None:
    """Its fixed cost is paid per day, used or not, and is not part of the dispatch."""
    assert evaluation.cost_fixed.tolist() == [0.0, 150.0]


def test_the_km_part_is_the_price_per_km_times_the_route(evaluation: Evaluation) -> None:
    assert evaluation.cost_km.tolist() == pytest.approx([KM_PER_DEGREE, 1.5 * 2 * KM_PER_DEGREE])


def test_the_hour_part_is_the_price_per_hour_times_the_hours(evaluation: Evaluation) -> None:
    assert evaluation.cost_hours.tolist() == pytest.approx([HOURLY * OWNED_HOURS, 0.0])


def test_the_unit_part_is_the_price_per_unit_times_the_load(evaluation: Evaluation) -> None:
    assert evaluation.cost_units.tolist() == [0.0, PER_DESI * 250.0]


def test_a_stop_is_reached_after_loading_and_driving(evaluation: Evaluation) -> None:
    """09:00 plus 90 minutes of loading and 112 on the road; the other trip starts at 00:00."""
    assert evaluation.stop_arrival.tolist() == [540.0 + 90 + 112, 25.0 + 223]


def test_a_stop_is_finished_when_its_unloading_ends(evaluation: Evaluation) -> None:
    assert evaluation.stop_finish.tolist() == [540.0 + 90 + 112 + 90, 25.0 + 223 + 25]


def test_the_plan_table_ends_with_the_timing_of_the_load(evaluation: Evaluation) -> None:
    names = columns(evaluation.instance)

    assert names[-3:] == ("desi", "unloaded", "late_hours")


def test_a_row_carries_the_cost_parts_of_its_dispatch(evaluation: Evaluation) -> None:
    first = next(rows(evaluation))

    assert (first["cost_fixed"], first["cost_hours"], first["cost_units"]) == (
        0.0,
        pytest.approx(HOURLY * OWNED_HOURS),
        0.0,
    )


def test_a_row_says_when_its_dispatch_ends(evaluation: Evaluation) -> None:
    first = next(rows(evaluation))

    assert (first["start"], first["end"]) == ("09:00", "13:52")


def test_a_row_says_when_its_load_is_unloaded_and_how_late(evaluation: Evaluation) -> None:
    """Due at 12:00 and unloaded at 13:52: two started hours late."""
    first = next(rows(evaluation))

    assert (first["unloaded"], first["late_hours"]) == ("13:52", 2)


def test_a_row_is_timed_by_its_own_load_whatever_the_order_of_the_loads(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    """The loads are listed last dispatch first; the first row is still the owned trip."""
    instance = compile_document(timed(document))
    plan = Plan(
        day=[0, 0],
        kind=[0, 1],
        source=[0, SMALL],
        origin=[A, A],
        stop_start=[0, 1, 2],
        stop_center=[B, C],
        load_dispatch=[1, 0],
        load_demand=[1, 0],
        load_quantity=[[250.0], [900.0]],
    )

    first = next(rows(evaluate(instance, plan)))

    assert (first["demand"], first["unloaded"]) == ("D000001", "13:52")


def test_the_plan_file_holds_the_new_columns(evaluation: Evaluation, tmp_path: Path) -> None:
    first = written(evaluation, tmp_path)[0]

    assert (first["end"], first["unloaded"], first["late_hours"]) == ("13:52", "13:52", "2")


def test_a_dispatch_without_loads_leaves_the_load_timing_empty(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    instance = compile_document(timed(document))
    evaluation = evaluate(instance, Plan.from_dispatches((EMPTY,), 1))

    row = next(rows(evaluation))

    assert (row["demand"], row["unloaded"], row["late_hours"]) == ("", "", "")


def test_a_written_plan_reads_back_with_its_new_columns(
    evaluation: Evaluation, tmp_path: Path
) -> None:
    path = tmp_path / "plan.csv"
    write_plan(evaluation, path)

    plan = read_plan(evaluation.instance, path)

    assert evaluate(evaluation.instance, plan).total_cost == evaluation.total_cost


def test_the_summary_says_how_much_of_the_demand_is_carried(evaluation: Evaluation) -> None:
    numbers = summary(evaluation)

    assert numbers["served"] == pytest.approx(CARRIED / DEMANDED)


def test_the_summary_says_how_full_the_vehicles_are(evaluation: Evaluation) -> None:
    numbers = summary(evaluation)

    assert numbers["fill"] == pytest.approx(CARRIED / CAPACITY)


def test_the_summary_says_how_much_arrives_in_time(evaluation: Evaluation) -> None:
    """Both loads are late: the 900 desi by two hours, the 250 desi by two as well."""
    numbers = summary(evaluation)

    assert numbers["on_time"] == 0.0


def test_the_summary_gives_the_cost_per_unit_carried(evaluation: Evaluation) -> None:
    numbers = summary(evaluation)

    assert numbers["cost_per_unit"] == pytest.approx(evaluation.total_cost / CARRIED)


def test_an_empty_plan_has_no_indicators_but_served(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    instance = compile_document(timed(document))
    evaluation = evaluate(instance, Plan.from_dispatches((), 1))

    numbers = summary(evaluation)

    assert (numbers["served"], numbers["fill"], numbers["on_time"], numbers["cost_per_unit"]) == (
        0.0,
        None,
        None,
        None,
    )

"""Travel times given per line (``matrix.durations``) instead of distance over speed.

The problem is the shared fixture: ``A``, ``B`` and ``C`` lie one degree apart on the
equator, 111.19 km a degree, so at 60 km/h a leg of one degree takes 112 whole minutes.
A duration replaces that time for one direction of one pair, for every vehicle type or
for one. The evaluator, the independent checker and the solvers must time a leg the same
way; the solvers are also covered by the ``durations`` scenarios of
``tests/test_rule_support.py``.
"""

from collections.abc import Callable
from typing import Any

import pytest

from pirilib import ACOConfig, ACOSolver, Dispatch, Load, Plan, SpecError, check_plan, evaluate
from pirilib import objective as obj
from pirilib.solvers.exact import ExactSolver
from pirilib.spec import LineHaulInstance, parse_spec

type CompileDocument = Callable[[dict[str, Any]], LineHaulInstance]

A, B, C = 0, 1, 2
BIG, SMALL = 0, 1

SPEED_KMH = 60.0
"""Speed of both vehicle types: one degree of the equator takes 112 whole minutes."""

DEGREE_MINUTES = 112.0
"""Whole minutes of a leg of one degree at :data:`SPEED_KMH`."""

LINE_HOURS = 3.0
"""The duration given for ``A`` to ``B``: 180 minutes, longer than the speed gives."""

SMALL_HOURS = 4.0
"""The duration given for ``A`` to ``B`` for the ``small`` type alone."""

HOURLY = 10.0
"""Cost per hour of the owned vehicle, so that the cost of a plan depends on its times."""

GOAL = obj.total_cost() + 1e6 * obj.violations()

QUICK = ACOConfig(ants=6, iterations=8, patience=4)

EVERY_PAIR = [
    {"origin": origin, "destination": destination, "hours": 2.0}
    for origin in "ABC"
    for destination in "ABC"
    if origin != destination
]
"""A duration for every ordered pair of the three centers."""

OWNED_AB = Dispatch(day=0, kind="owned", source=0, origin=A, stops=(B,), loads=(Load(0, (900.0,)),))
SMALL_AC = Dispatch(
    day=0, kind="spot", source=SMALL, origin=A, stops=(C,), loads=(Load(1, (250.0,)),)
)
SMALL_BA = Dispatch(
    day=1, kind="spot", source=SMALL, origin=B, stops=(A,), loads=(Load(2, (120.0,)),)
)
PLAN = Plan.from_dispatches((OWNED_AB, SMALL_AC, SMALL_BA), 1)
"""Every demand of the fixture on a direct trip of its own."""


def timed(document: dict[str, Any], *durations: dict[str, Any]) -> dict[str, Any]:
    """The fixture with speeds, an hourly cost, a due time 3 hours after 00:00 and ``durations``.

    Loading and unloading take no time, so a dispatch lasts as long as its legs.
    """
    for vehicle in document["vehicle_types"]:
        vehicle["speed_kmh"] = SPEED_KMH
    document["fleet"]["owned"][0]["cost"]["per_hour"] = HOURLY
    document["rules"] = {"service": {"due_hours": 3, "penalty": 0.4}}
    document["matrix"]["durations"] = list(durations)
    return document


def line(hours: float = LINE_HOURS, **more: Any) -> dict[str, Any]:
    """A duration row for ``A`` to ``B``."""
    return {"origin": "A", "destination": "B", "hours": hours, **more}


def problems(document: dict[str, Any]) -> list[str]:
    """The problems :func:`parse_spec` reports for ``document``, without the leading dashes."""
    with pytest.raises(SpecError) as info:
        parse_spec(document)
    return [text.removeprefix("- ") for text in str(info.value).splitlines()[1:]]


def test_a_duration_replaces_distance_over_speed(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    instance = compile_document(timed(document, line()))

    minutes = instance.travel_minutes(BIG, A, B)

    assert minutes == LINE_HOURS * 60


def test_a_duration_is_for_one_direction(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    instance = compile_document(timed(document, line()))

    minutes = instance.travel_minutes(BIG, B, A)

    assert minutes == DEGREE_MINUTES


def test_a_leg_without_a_duration_is_timed_from_the_speed(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    instance = compile_document(timed(document, line()))

    minutes = instance.travel_minutes(BIG, B, C)

    assert minutes == DEGREE_MINUTES


@pytest.mark.parametrize(
    ("vehicle_type", "minutes"),
    [
        pytest.param(BIG, LINE_HOURS * 60, id="type-without-a-row-of-its-own"),
        pytest.param(SMALL, SMALL_HOURS * 60, id="type-with-a-row-of-its-own"),
    ],
)
def test_a_duration_for_one_vehicle_type_wins_over_the_one_for_every_type(
    vehicle_type: int,
    minutes: float,
    document: dict[str, Any],
    compile_document: CompileDocument,
) -> None:
    rows = (line(SMALL_HOURS, vehicle_type="small"), line())
    instance = compile_document(timed(document, *rows))

    found = instance.travel_minutes(vehicle_type, A, B)

    assert found == minutes


def test_a_duration_is_rounded_up_to_whole_minutes(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    """1.501 hours are 90.06 minutes."""
    instance = compile_document(timed(document, line(1.501)))

    minutes = instance.travel_minutes(BIG, A, B)

    assert minutes == 91.0


def test_a_document_without_durations_holds_no_duration_table(
    instance: LineHaulInstance,
) -> None:
    assert instance.duration_minutes is None


def test_without_a_speed_and_a_duration_a_leg_takes_no_time(instance: LineHaulInstance) -> None:
    minutes = instance.travel_minutes(BIG, A, B)

    assert minutes == 0.0


@pytest.mark.parametrize(
    ("rows", "problem"),
    [
        pytest.param(
            [{"origin": "A", "destination": "Z", "hours": 1.0}],
            "matrix.durations[0].destination: unknown center 'Z'",
            id="unknown-center",
        ),
        pytest.param(
            [line(vehicle_type="van")],
            "matrix.durations[0].vehicle_type: unknown vehicle type 'van'",
            id="unknown-vehicle-type",
        ),
        pytest.param(
            [{"origin": "A", "destination": "A", "hours": 1.0}],
            "matrix.durations[0]: origin and destination are both 'A'",
            id="loop",
        ),
        pytest.param(
            [line(), line(2.0)], "matrix.durations[1]: duplicate duration A -> B", id="given-twice"
        ),
        pytest.param(
            [line(vehicle_type="big"), line(2.0, vehicle_type="big")],
            "matrix.durations[1]: duplicate duration A -> B for big",
            id="given-twice-for-a-type",
        ),
        pytest.param(
            [line(-1.0)],
            "matrix.durations[0].hours: Input should be greater than or equal to 0",
            id="negative-hours",
        ),
    ],
)
def test_a_wrong_duration_row_is_reported(
    rows: list[dict[str, Any]], problem: str, document: dict[str, Any]
) -> None:
    document["matrix"]["durations"] = rows

    found = problems(document)

    assert found == [problem]


def test_durations_for_every_pair_make_the_speeds_optional(document: dict[str, Any]) -> None:
    """Due times need travel times; with a duration for every pair no speed is missing."""
    document["rules"] = {"service": {"due_hours": 3}}
    document["matrix"]["durations"] = EVERY_PAIR

    spec = parse_spec(document)

    assert [vehicle.speed_kmh for vehicle in spec.vehicle_types] == [None, None]


def test_durations_for_some_pairs_leave_the_speeds_required(document: dict[str, Any]) -> None:
    document["rules"] = {"service": {"due_hours": 3}}
    document["matrix"]["durations"] = EVERY_PAIR[:-1]

    found = problems(document)

    assert found == [
        f"vehicle_types[{position}].speed_kmh: is required, because hourly costs or due times "
        "are used and matrix.durations does not time every pair of centers"
        for position in (0, 1)
    ]


def test_durations_for_one_type_leave_the_speed_of_the_other_required(
    document: dict[str, Any],
) -> None:
    document["rules"] = {"service": {"due_hours": 3}}
    document["matrix"]["durations"] = [row | {"vehicle_type": "big"} for row in EVERY_PAIR]

    found = problems(document)

    assert found == [
        "vehicle_types[1].speed_kmh: is required, because hourly costs or due times are used "
        "and matrix.durations does not time every pair of centers"
    ]


def test_the_evaluator_times_a_leg_with_its_duration(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    instance = compile_document(timed(document, line()))

    evaluation = evaluate(instance, PLAN)

    assert evaluation.end_minute[0] == LINE_HOURS * 60


def test_the_evaluator_times_the_other_legs_from_the_speed(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    """``A`` to ``C`` is two degrees: 222.4 km at 60 km/h take 223 whole minutes."""
    instance = compile_document(timed(document, line()))

    evaluation = evaluate(instance, PLAN)

    assert evaluation.end_minute[1] == 223.0


def test_a_longer_duration_makes_a_load_late(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    """Due 3 hours after 00:00 and unloaded after 5 hours: 2 started hours late."""
    instance = compile_document(timed(document, line(5.0)))

    evaluation = evaluate(instance, PLAN)

    assert evaluation.load_late_hours[0] == 2.0


def test_the_checker_finds_the_evaluated_cost_with_durations(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    """The hourly cost and the lateness both follow the times of the legs."""
    rows = (line(5.0), line(SMALL_HOURS, vehicle_type="small"))
    instance = compile_document(timed(document, *rows))

    report = check_plan(instance, PLAN)

    assert report.total_cost == pytest.approx(evaluate(instance, PLAN).total_cost, rel=1e-12)


@pytest.mark.parametrize("hours", [0.5, LINE_HOURS, 9.0])
def test_the_colony_score_is_the_objective_with_durations(
    hours: float, document: dict[str, Any], compile_document: CompileDocument
) -> None:
    instance = compile_document(timed(document, line(hours), line(2 * hours, vehicle_type="small")))

    result = ACOSolver(QUICK).solve(instance, GOAL)

    assert result.stats["score"] == pytest.approx(result.objective, rel=1e-9)


@pytest.mark.parametrize("hours", [0.5, LINE_HOURS, 9.0])
def test_the_exact_score_is_the_objective_with_durations(
    hours: float, document: dict[str, Any], compile_document: CompileDocument
) -> None:
    pytest.importorskip("scipy")
    instance = compile_document(timed(document, line(hours), line(2 * hours, vehicle_type="small")))

    result = ExactSolver().solve(instance, GOAL)

    assert result.stats["score"] == pytest.approx(result.objective, rel=1e-9)


def test_the_colony_plans_a_problem_timed_by_durations_alone(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    """No vehicle type has a speed; every leg has a duration of 2 hours, within the 3 allowed."""
    document["rules"] = {"service": {"due_hours": 3, "penalty": 0.4}}
    document["matrix"]["durations"] = EVERY_PAIR
    instance = compile_document(document)

    result = ACOSolver(QUICK).solve(instance, GOAL)

    assert (result.evaluation.feasible, result.evaluation.service_cost) == (True, 0.0)

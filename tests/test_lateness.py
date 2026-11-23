"""Lateness as a rule of :data:`pirilib.constraints.CATALOG`.

A load unloaded after the due time of its demand breaks ``lateness`` by its primary
quantity times the started hours it is late. ``rules.service.penalty`` prices that amount
as a cost; ``objective.penalties`` weighs it like every other rule, or makes it hard.

The problem is one demand of 300 desi from ``A`` to ``B``, one degree of the equator
(111.19 km), ready at 00:00 and due two hours later. The ``small`` vehicle is cheap and
slow: at 30 km/h it needs 223 minutes and is two started hours late. The ``big`` vehicle is
dear and fast: at 120 km/h it needs 56 minutes.
"""

from collections.abc import Callable
from typing import Any

import pytest

from pirilib import ACOConfig, ACOSolver, Dispatch, Load, Plan, SpecError, check_plan, evaluate
from pirilib.solvers.base import Solver
from pirilib.solvers.exact import ExactSolver
from pirilib.spec import LineHaulInstance, parse_spec

type CompileDocument = Callable[[dict[str, Any]], LineHaulInstance]

A, B = 0, 1
FAST, SLOW = 0, 1
"""The spot offers: ``big`` at 120 km/h for 500, ``small`` at 30 km/h for 100."""

DESI = 300.0
LATE_HOURS = 2.0
"""223 minutes on the way against the 120 allowed: two started hours."""

AMOUNT = DESI * LATE_HOURS
"""The amount of the rule when the slow vehicle carries the demand."""

SERVICE_PENALTY = 0.1
"""Cost per desi and started hour: 60 for the late load, less than the fast vehicle costs."""

SLOW_COST, FAST_COST = 100.0, 500.0

OTHER_WEIGHT = 1e6

SOLVERS: dict[str, Solver] = {
    "aco": ACOSolver(ACOConfig(ants=6, iterations=8, patience=4)),
    "exact": ExactSolver(),
}

OLDER_SCHEMAS = ("pirilib/linehaul@0.1", "pirilib/linehaul@0.2")


def late_or_dear(document: dict[str, Any], lateness: float | str | None = 0.0) -> dict[str, Any]:
    """The problem of this module; ``lateness`` is its penalty, None to leave it to ``other``."""
    document["vehicle_types"][0]["speed_kmh"] = 120.0
    document["vehicle_types"][1]["speed_kmh"] = 30.0
    document["fleet"] = {
        "spot": [
            {"type": "big", "cost": {"fixed": FAST_COST}},
            {"type": "small", "cost": {"fixed": SLOW_COST}},
        ]
    }
    document["demand"] = [
        {"origin": "A", "destination": "B", "date": "2026-01-05", "quantity": {"desi": DESI}}
    ]
    document["rules"] = {"service": {"due_hours": 2, "penalty": SERVICE_PENALTY}}
    penalties: dict[str, Any] = {"coverage": 1e7, "other": OTHER_WEIGHT}
    if lateness is not None:
        penalties["lateness"] = lateness
    document["objective"] = {"cost": 1, "penalties": penalties}
    return document


def trip(offer: int) -> Plan:
    """The demand on a direct trip of the spot offer ``offer``."""
    dispatch = Dispatch(
        day=0, kind="spot", source=offer, origin=A, stops=(B,), loads=(Load(0, (DESI,)),)
    )
    return Plan.from_dispatches((dispatch,), 1)


def every_rule_but_lateness(weight: float) -> dict[str, float]:
    """A penalty for every built-in rule except ``lateness``, without ``other``."""
    from pirilib.constraints import CATALOG

    return {name: weight for name in CATALOG if name != "lateness"}


def test_the_helper_names_every_rule_but_lateness() -> None:
    penalties = every_rule_but_lateness(1.0)

    assert (len(penalties), "lateness" in penalties, penalties["coverage"]) == (19, False, 1.0)


def test_a_load_unloaded_after_its_due_time_breaks_the_rule(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    instance = compile_document(late_or_dear(document))

    evaluation = evaluate(instance, trip(SLOW))

    assert evaluation.violations["lateness"] == AMOUNT


def test_a_load_unloaded_in_time_keeps_the_rule(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    instance = compile_document(late_or_dear(document))

    evaluation = evaluate(instance, trip(FAST))

    assert evaluation.violations["lateness"] == 0.0


def test_the_lateness_cost_is_the_service_penalty_times_the_amount(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    instance = compile_document(late_or_dear(document))

    evaluation = evaluate(instance, trip(SLOW))

    assert evaluation.service_cost == pytest.approx(SERVICE_PENALTY * AMOUNT)


def test_the_checker_finds_the_same_amount(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    instance = compile_document(late_or_dear(document))

    report = check_plan(instance, trip(SLOW))

    assert report.totals()["lateness"] == AMOUNT


def test_the_checker_says_which_demand_is_late(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    instance = compile_document(late_or_dear(document))

    report = check_plan(instance, trip(SLOW))

    assert [item.message for item in report.violations] == [
        "dispatch 0 (2026-01-05, spot small from A): demand D000001 is unloaded 2 hour(s) "
        "after it is due"
    ]


def test_a_late_plan_is_not_feasible_when_no_rule_is_hard(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    instance = compile_document(late_or_dear(document))

    evaluation = evaluate(instance, trip(SLOW))

    assert not evaluation.feasible


def test_a_late_plan_is_feasible_when_lateness_is_not_among_the_hard_rules(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    instance = compile_document(late_or_dear(document))

    evaluation = evaluate(instance, trip(SLOW), hard={"coverage", "capacity"})

    assert evaluation.feasible


@pytest.mark.parametrize("weight", [0.0, 0.5, 40.0])
def test_the_objective_adds_the_weight_times_the_amount(
    weight: float, document: dict[str, Any]
) -> None:
    spec = parse_spec(late_or_dear(document, weight))
    evaluation = evaluate(spec.compile(), trip(SLOW))

    value = spec.build_objective()(evaluation)

    assert value == pytest.approx(SLOW_COST + SERVICE_PENALTY * AMOUNT + weight * AMOUNT)


@pytest.mark.parametrize("name", SOLVERS)
def test_with_weight_zero_the_cheap_late_vehicle_goes(name: str, document: dict[str, Any]) -> None:
    """Late costs 100 + 60, in time costs 500."""
    spec = parse_spec(late_or_dear(document, 0.0))

    result = SOLVERS[name].solve(spec.compile(), spec.build_objective())

    assert result.plan.source.tolist() == [SLOW]


@pytest.mark.parametrize("name", SOLVERS)
@pytest.mark.parametrize("lateness", [1.0, "hard", None], ids=["weight", "hard", "other"])
def test_a_weight_on_lateness_sends_the_vehicle_that_is_in_time(
    lateness: float | str | None, name: str, document: dict[str, Any]
) -> None:
    """A weight of 1 adds 600 to the late trip; ``hard`` and ``other`` add far more."""
    spec = parse_spec(late_or_dear(document, lateness))

    result = SOLVERS[name].solve(spec.compile(), spec.build_objective())

    assert result.plan.source.tolist() == [FAST]


@pytest.mark.parametrize("name", SOLVERS)
def test_the_score_is_the_objective_of_a_late_plan(name: str, document: dict[str, Any]) -> None:
    """Only the slow vehicle is offered, so the plan is late whatever the weight."""
    document = late_or_dear(document, 7.0)
    del document["fleet"]["spot"][FAST]
    spec = parse_spec(document)

    result = SOLVERS[name].solve(spec.compile(), spec.build_objective())

    assert result.stats["score"] == pytest.approx(result.objective, rel=1e-9)


@pytest.mark.parametrize("name", SOLVERS)
def test_a_load_that_can_only_arrive_late_stays_behind_when_lateness_is_hard(
    name: str, document: dict[str, Any]
) -> None:
    """Only the slow vehicle is offered. A hard rule outweighs every weight, also the one
    of ``coverage``, so the solver leaves the 300 desi rather than unload them late."""
    document = late_or_dear(document, "hard")
    del document["fleet"]["spot"][FAST]
    spec = parse_spec(document)

    result = SOLVERS[name].solve(spec.compile(), spec.build_objective())

    assert result.evaluation.violated() == {"coverage": DESI}


def test_a_late_plan_breaks_hard_lateness(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    instance = compile_document(late_or_dear(document))

    evaluation = evaluate(instance, trip(SLOW), hard={"lateness"})

    assert evaluation.broken() == {"lateness": AMOUNT}


@pytest.mark.parametrize("schema", OLDER_SCHEMAS)
def test_an_older_document_gives_lateness_no_weight(schema: str, document: dict[str, Any]) -> None:
    """Lateness was a cost and not a rule before 0.3, so ``other`` does not reach it."""
    document = late_or_dear(document, None) | {"schema": schema}

    spec = parse_spec(document)

    assert spec.objective is not None
    assert spec.objective.penalty("lateness") == 0.0


@pytest.mark.parametrize("schema", OLDER_SCHEMAS)
def test_an_older_document_that_names_lateness_keeps_its_weight(
    schema: str, document: dict[str, Any]
) -> None:
    document = late_or_dear(document, 3.0) | {"schema": schema}

    spec = parse_spec(document)

    assert spec.objective is not None
    assert spec.objective.penalty("lateness") == 3.0


def test_a_current_document_gives_lateness_the_weight_of_other(document: dict[str, Any]) -> None:
    spec = parse_spec(late_or_dear(document, None))

    assert spec.objective is not None
    assert spec.objective.penalty("lateness") == OTHER_WEIGHT


@pytest.mark.parametrize("schema", OLDER_SCHEMAS)
def test_an_older_document_may_name_every_rule_it_knew(
    schema: str, document: dict[str, Any]
) -> None:
    document = late_or_dear(document) | {"schema": schema}
    document["objective"]["penalties"] = every_rule_but_lateness(5.0)

    spec = parse_spec(document)

    assert spec.objective is not None
    assert spec.objective.penalty("lateness") == 0.0


def test_a_current_document_without_other_has_to_name_lateness(document: dict[str, Any]) -> None:
    document = late_or_dear(document)
    document["objective"]["penalties"] = every_rule_but_lateness(5.0)

    with pytest.raises(SpecError, match="no penalty for lateness; every rule needs a weight"):
        parse_spec(document)

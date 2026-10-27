from typing import Any

import pytest
from pydantic import ValidationError

from pirilib import Dispatch, Load, Plan, SpecError, evaluate
from pirilib import objective as obj
from pirilib.constraints import CATALOG
from pirilib.spec import ObjectiveSpec, parse_spec

A, B, C = 0, 1, 2
SMALL = 1
SECTION: dict[str, Any] = {"cost": 1, "penalties": {"coverage": 1e7, "other": 1e6}}


def problems(document: dict[str, Any]) -> list[str]:
    """Return the problems listed by the SpecError raised for ``document``."""
    with pytest.raises(SpecError) as info:
        parse_spec(document)
    _, *lines = str(info.value).splitlines()
    return [line.removeprefix("- ") for line in lines]


def broken_plan() -> Plan:
    """An overloaded small vehicle, a half-empty one and a demand left behind."""
    return Plan.from_dispatches(
        [
            Dispatch(0, "spot", SMALL, A, (B,), (Load(0, (900.0,)),)),
            Dispatch(0, "spot", SMALL, A, (C,), (Load(1, (250.0,)),)),
        ],
        dimensions=1,
    )


def test_the_section_is_optional(document: dict[str, Any]) -> None:
    assert parse_spec(document).objective is None
    document["objective"] = SECTION
    spec = parse_spec(document)
    assert spec.objective == ObjectiveSpec(cost=1.0, penalties={"coverage": 1e7, "other": 1e6})


def test_unknown_rules_are_rejected(document: dict[str, Any]) -> None:
    document["objective"] = {"cost": 1, "penalties": {"tolls": 5, "other": 1}}
    found = problems(document)
    assert len(found) == 1
    assert found[0].startswith("objective.penalties.tolls: unknown rule 'tolls'; known: route,")
    assert found[0].endswith(", overdelivery, other")


def test_every_rule_needs_a_penalty_without_other(document: dict[str, Any]) -> None:
    named = {name: 1.0 for name in CATALOG if name not in ("split", "coverage")}
    document["objective"] = {"cost": 1, "penalties": named}
    assert problems(document) == [
        "objective.penalties: no penalty for split, coverage; every rule needs a weight: "
        "add them, or 'other' for every rule not named"
    ]
    document["objective"]["penalties"] = {**named, "split": 0.0, "coverage": 1.0}
    assert parse_spec(document).objective is not None


def test_objective_problems_are_reported_with_the_others(document: dict[str, Any]) -> None:
    document["calendar"]["end"] = "soon"
    document["objective"] = {"cost": 0, "penalties": {"tolls": 1}}
    found = problems(document)
    assert found[0].startswith("calendar.end: Input should be a valid date")
    assert found[1] == "objective.cost: Input should be greater than 0"
    assert found[2].startswith("objective.penalties.tolls: unknown rule 'tolls'")
    assert found[3].startswith("objective.penalties: no penalty for route, load_route,")
    assert len(found) == 4


def test_cost_and_penalties_are_required() -> None:
    with pytest.raises(ValidationError, match="cost\n  Field required"):
        ObjectiveSpec.model_validate({"penalties": {"other": 1}})
    with pytest.raises(ValidationError, match="penalties\n  Field required"):
        ObjectiveSpec.model_validate({"cost": 1})


def test_other_covers_the_rules_not_named() -> None:
    section = ObjectiveSpec.model_validate(SECTION)
    assert section.penalty("coverage") == 1e7
    assert {name: section.penalty(name) for name in CATALOG if name != "coverage"} == dict.fromkeys(
        (name for name in CATALOG if name != "coverage"), 1e6
    )
    with pytest.raises(KeyError, match="no built-in rule named 'tolls'"):
        section.penalty("tolls")


def test_build_equals_the_objective_written_by_hand(document: dict[str, Any]) -> None:
    document["objective"] = {
        "cost": 2,
        "distance_km": 3,
        "dispatches": 50,
        "unused_capacity": 0.5,
        "penalties": {"capacity": 7, "coverage": 1e4, "other": 0},
    }
    spec = parse_spec(document)
    evaluation = evaluate(spec.compile(), broken_plan())
    assert evaluation.violations["capacity"] == 500.0
    assert evaluation.violations["coverage"] == 120.0
    by_hand = (
        2 * obj.total_cost()
        + 3 * obj.distance_km()
        + 50 * obj.dispatches()
        + 0.5 * obj.unused_capacity()
        + 7 * obj.violation("capacity")
        + 1e4 * obj.violation("coverage")
    )
    goal = spec.build_objective()
    assert goal(evaluation) == pytest.approx(by_hand(evaluation), rel=1e-12)
    assert goal.breakdown(evaluation)["violation[capacity]"] == 3500.0


def test_build_has_a_linear_form() -> None:
    section = ObjectiveSpec.model_validate({**SECTION, "dispatches": 10})
    form = section.build().linear_form()
    assert form is not None
    assert form["cost[spot]"] == form["owned_fixed"] == form["service"] == 1.0
    assert form["dispatches[owned]"] == form["dispatches[spot]"] == 10.0
    assert "km[spot]" not in form
    assert form["violation[coverage]"] == 1e7
    assert {f"violation[{name}]" for name in CATALOG} <= set(form)


def test_zero_penalties_are_kept_and_zero_extras_left_out() -> None:
    section = ObjectiveSpec.model_validate({"cost": 1, "penalties": {"other": 0}})
    names = [term.name for term in section.build().terms]
    assert names == ["total_cost", *(f"violation[{name}]" for name in CATALOG)]


def test_build_objective_needs_the_section(document: dict[str, Any]) -> None:
    spec = parse_spec(document)
    with pytest.raises(SpecError, match="no objective: section") as info:
        spec.build_objective()
    message = str(info.value)
    assert "objective:\n  cost: 1\n  penalties:\n    coverage: 10000000" in message
    assert "'other'" in message

"""Tests of the ``objective:`` section of a problem document (:class:`ObjectiveSpec`).

The requirement is the docstring of :class:`pirilib.spec.ObjectiveSpec`: ``cost`` is a positive
weight of the total cost, the extra terms default to 0 and are left out at 0, every built-in
rule of :data:`pirilib.constraints.CATALOG` needs a penalty by name or through ``other``, and
:meth:`LineHaulSpec.build_objective` fails with an example when the section is missing.

The plans are written on the ``document`` fixture of ``tests/conftest.py``: centers ``A``, ``B``
and ``C`` are indices 0, 1 and 2, vehicle type ``small`` (400 desi) is index 1.
"""

import re
from typing import Any

import pytest
from pydantic import ValidationError

from pirilib import Dispatch, Load, Plan, SpecError, evaluate
from pirilib import objective as obj
from pirilib.spec import ObjectiveSpec, parse_spec

A, B, C = 0, 1, 2
SMALL = 1

COVERAGE_WEIGHT = 1e7
"""The penalty of ``coverage`` in :data:`SECTION`, ten times :data:`OTHER_WEIGHT` so the two
weights can be told apart in every result."""

OTHER_WEIGHT = 1e6
"""The ``other`` penalty of :data:`SECTION`: every rule but ``coverage``."""

SECTION: dict[str, Any] = {
    "cost": 1,
    "penalties": {"coverage": COVERAGE_WEIGHT, "other": OTHER_WEIGHT},
}
"""The smallest complete section: unit cost weight, one named rule and ``other``."""

RULES = (
    "route",
    "load_route",
    "ready",
    "deferral",
    "ban",
    "owned_line",
    "owned_availability",
    "spot_availability",
    "handling",
    "vehicle_limit",
    "capacity",
    "min_fill",
    "stopover",
    "max_stops",
    "stretch",
    "forward_only",
    "split",
    "coverage",
    "overdelivery",
)
"""The built-in rules in catalog order, as the messages and the objective list them."""

PENALTY_TERMS = (
    "violation[route]",
    "violation[load_route]",
    "violation[ready]",
    "violation[deferral]",
    "violation[ban]",
    "violation[owned_line]",
    "violation[owned_availability]",
    "violation[spot_availability]",
    "violation[handling]",
    "violation[vehicle_limit]",
    "violation[capacity]",
    "violation[min_fill]",
    "violation[stopover]",
    "violation[max_stops]",
    "violation[stretch]",
    "violation[forward_only]",
    "violation[split]",
    "violation[coverage]",
    "violation[overdelivery]",
)
"""The penalty term of every rule of :data:`RULES`, in the same order."""

RULE_LIST = ", ".join(RULES)
"""The rule names as the messages write them."""

TOTAL_COST_FORM = {"cost[owned]": 1.0, "cost[spot]": 1.0, "owned_fixed": 1.0, "service": 1.0}
"""The linear form of ``total_cost()`` with weight 1: dispatch costs, owned fixed cost and
lateness."""

UNKNOWN_TOLLS = f"- objective.penalties.tolls: unknown rule 'tolls'; known: {RULE_LIST}, other"
"""The problem line of an unknown ``tolls`` penalty: it lists every name that is allowed."""

MISSING_RULES = "; every rule needs a weight: add them, or 'other' for every rule not named"
"""The end of the problem line of rules without a penalty, after their names."""

OVERLOAD = 500.0
"""Desi over capacity in :func:`broken_plan`: 900 desi on a 400 desi ``small`` vehicle."""

LEFT_BEHIND = 120.0
"""Desi not served by :func:`broken_plan`: the 120 desi from ``B`` to ``A``."""

BROKEN_PLAN_VIOLATIONS = {
    **dict.fromkeys(RULES, 0.0),
    "capacity": OVERLOAD,
    "coverage": LEFT_BEHIND,
}
"""Every built-in violation of :func:`broken_plan`: only ``capacity`` and ``coverage``."""

TINY = 1e-9
"""A cost weight just above the bound 0."""


def problems(document: dict[str, Any]) -> list[str]:
    """Return the problem lines (``- <location>: <message>``) of the SpecError for ``document``."""
    with pytest.raises(SpecError) as info:
        parse_spec(document)
    _, *lines = str(info.value).splitlines()
    return lines


def broken_plan() -> Plan:
    """An overloaded small vehicle, a half-empty one and a demand left behind.

    Day one: 900 desi from ``A`` to ``B`` and 250 desi from ``A`` to ``C``, each on a spot
    ``small`` vehicle; the 120 desi from ``B`` to ``A`` on day two is not served. The expected
    violations are :data:`BROKEN_PLAN_VIOLATIONS`.
    """
    return Plan.from_dispatches(
        [
            Dispatch(0, "spot", SMALL, A, (B,), (Load(0, (900.0,)),)),
            Dispatch(0, "spot", SMALL, A, (C,), (Load(1, (250.0,)),)),
        ],
        dimensions=1,
    )


def test_broken_plan_breaks_only_capacity_and_coverage(document: dict[str, Any]) -> None:
    instance = parse_spec(document).compile()

    evaluation = evaluate(instance, broken_plan())

    assert dict(evaluation.violations) == BROKEN_PLAN_VIOLATIONS


def test_a_document_without_the_section_has_no_objective(document: dict[str, Any]) -> None:
    spec = parse_spec(document)

    assert spec.objective is None


def test_the_section_is_read_into_an_objective_spec(document: dict[str, Any]) -> None:
    document["objective"] = SECTION

    spec = parse_spec(document)

    assert spec.objective == ObjectiveSpec(
        cost=1.0, penalties={"coverage": COVERAGE_WEIGHT, "other": OTHER_WEIGHT}
    )


def test_weights_at_their_lower_bound_are_accepted(document: dict[str, Any]) -> None:
    document["objective"] = {
        "cost": TINY,
        "distance_km": 0,
        "dispatches": 0,
        "unused_capacity": 0,
        "penalties": {"other": 0},
    }

    spec = parse_spec(document)

    assert spec.objective == ObjectiveSpec(cost=TINY, penalties={"other": 0.0})


@pytest.mark.parametrize(
    ("section", "problem"),
    [
        pytest.param(
            {"cost": 0, "penalties": {"other": 1}},
            "- objective.cost: Input should be greater than 0",
            id="zero-cost",
        ),
        pytest.param(
            {"cost": -1, "penalties": {"other": 1}},
            "- objective.cost: Input should be greater than 0",
            id="negative-cost",
        ),
        pytest.param(
            {"cost": float("inf"), "penalties": {"other": 1}},
            "- objective.cost: Input should be a finite number",
            id="infinite-cost",
        ),
        pytest.param(
            {"cost": 1, "distance_km": -1, "penalties": {"other": 1}},
            "- objective.distance_km: Input should be greater than or equal to 0",
            id="negative-distance-weight",
        ),
        pytest.param(
            {"cost": 1, "dispatches": -1, "penalties": {"other": 1}},
            "- objective.dispatches: Input should be greater than or equal to 0",
            id="negative-dispatch-weight",
        ),
        pytest.param(
            {"cost": 1, "unused_capacity": -1, "penalties": {"other": 1}},
            "- objective.unused_capacity: Input should be greater than or equal to 0",
            id="negative-unused-capacity-weight",
        ),
        pytest.param(
            {"cost": 1, "penalties": {"other": -1}},
            "- objective.penalties.other: Input should be greater than or equal to 0",
            id="negative-penalty",
        ),
        pytest.param(
            {"cost": 1, "penalties": {"coverage": float("nan"), "other": 1}},
            "- objective.penalties.coverage: Input should be a finite number",
            id="nan-penalty",
        ),
        pytest.param(
            {"cost": 1, "penalties": {}},
            f"- objective.penalties: no penalty for {RULE_LIST}{MISSING_RULES}",
            id="no-penalties",
        ),
    ],
)
def test_a_weight_out_of_range_is_rejected(
    document: dict[str, Any], section: dict[str, Any], problem: str
) -> None:
    document["objective"] = section

    found = problems(document)

    assert found == [problem]


def test_unknown_rules_are_rejected(document: dict[str, Any]) -> None:
    document["objective"] = {"cost": 1, "penalties": {"tolls": 5, "other": 1}}

    found = problems(document)

    assert found == [UNKNOWN_TOLLS]


def test_a_rule_without_a_penalty_is_rejected_without_other(document: dict[str, Any]) -> None:
    named = dict.fromkeys(set(RULES) - {"split", "coverage"}, 1.0)
    document["objective"] = {"cost": 1, "penalties": named}

    found = problems(document)

    assert found == [f"- objective.penalties: no penalty for split, coverage{MISSING_RULES}"]


def test_naming_every_rule_without_other_is_accepted(document: dict[str, Any]) -> None:
    """A weight of 0 still names the rule; ``split`` is named with 0."""
    penalties = {**dict.fromkeys(RULES, 1.0), "split": 0.0}
    document["objective"] = {"cost": 1, "penalties": penalties}

    spec = parse_spec(document)

    assert spec.objective == ObjectiveSpec(cost=1.0, penalties=penalties)


def test_objective_problems_are_reported_with_the_others(document: dict[str, Any]) -> None:
    """The date message is pydantic's own wording, so only its location and start are fixed."""
    document["calendar"]["end"] = "soon"
    document["objective"] = {"cost": 0, "penalties": {"tolls": 1}}

    first, *others = problems(document)

    assert first.startswith("- calendar.end: Input should be a valid date")
    assert others == [
        "- objective.cost: Input should be greater than 0",
        UNKNOWN_TOLLS,
        f"- objective.penalties: no penalty for {RULE_LIST}{MISSING_RULES}",
    ]


@pytest.mark.parametrize(
    ("section", "missing"),
    [
        pytest.param({"penalties": {"other": 1}}, "cost", id="cost"),
        pytest.param({"cost": 1}, "penalties", id="penalties"),
    ],
)
def test_cost_and_penalties_are_required(section: dict[str, Any], missing: str) -> None:
    with pytest.raises(ValidationError, match=f"{missing}\n  Field required"):
        ObjectiveSpec.model_validate(section)


def test_a_named_rule_has_its_own_penalty() -> None:
    section = ObjectiveSpec.model_validate(SECTION)

    weight = section.penalty("coverage")

    assert weight == COVERAGE_WEIGHT


@pytest.mark.parametrize("name", sorted(set(RULES) - {"coverage"}))
def test_other_is_the_penalty_of_every_rule_not_named(name: str) -> None:
    section = ObjectiveSpec.model_validate(SECTION)

    weight = section.penalty(name)

    assert weight == OTHER_WEIGHT


def test_the_penalty_of_an_unknown_rule_is_a_key_error() -> None:
    section = ObjectiveSpec.model_validate(SECTION)

    with pytest.raises(
        KeyError, match=re.escape(f"no built-in rule named 'tolls'; known: {RULE_LIST}")
    ):
        section.penalty("tolls")


def test_build_equals_the_objective_written_by_hand(document: dict[str, Any]) -> None:
    """Every weight is different, so a weight on the wrong term changes the total."""
    document["objective"] = {
        "cost": 2,
        "distance_km": 3,
        "dispatches": 50,
        "unused_capacity": 0.5,
        "penalties": {"capacity": 7, "coverage": 1e4, "other": 0},
    }
    spec = parse_spec(document)
    evaluation = evaluate(spec.compile(), broken_plan())
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


def test_each_violation_is_weighted_by_its_penalty(document: dict[str, Any]) -> None:
    """Capacity 7 x :data:`OVERLOAD`, coverage 10000 x :data:`LEFT_BEHIND`, route 0 x 0."""
    document["objective"] = {"cost": 1, "penalties": {"capacity": 7, "coverage": 1e4, "other": 0}}
    spec = parse_spec(document)
    evaluation = evaluate(spec.compile(), broken_plan())

    breakdown = spec.build_objective().breakdown(evaluation)

    assert (
        breakdown.items()
        >= {
            "violation[capacity]": 3500.0,
            "violation[coverage]": 1.2e6,
            "violation[route]": 0.0,
        }.items()
    )


def test_build_has_a_linear_form() -> None:
    """``dispatches: 10`` weighs owned and spot dispatches; ``distance_km`` 0 adds no km."""
    section = ObjectiveSpec.model_validate({**SECTION, "dispatches": 10})

    form = section.build().linear_form()

    assert form == {
        **TOTAL_COST_FORM,
        "dispatches[owned]": 10.0,
        "dispatches[spot]": 10.0,
        **dict.fromkeys(PENALTY_TERMS, OTHER_WEIGHT),
        "violation[coverage]": COVERAGE_WEIGHT,
    }


def test_zero_penalties_are_kept_and_zero_extras_left_out(document: dict[str, Any]) -> None:
    document["objective"] = {"cost": 1, "penalties": {"other": 0}}
    spec = parse_spec(document)
    evaluation = evaluate(spec.compile(), broken_plan())

    breakdown = spec.build_objective().breakdown(evaluation)

    assert list(breakdown) == ["total_cost", *PENALTY_TERMS]


def test_build_objective_needs_the_section(document: dict[str, Any]) -> None:
    spec = parse_spec(document)

    with pytest.raises(SpecError, match="has no objective: section; solving and tuning need one"):
        spec.build_objective()


def test_the_missing_section_error_shows_an_example(document: dict[str, Any]) -> None:
    spec = parse_spec(document)

    with pytest.raises(SpecError) as info:
        spec.build_objective()

    message = str(info.value)
    assert (
        "objective:\n  cost: 1\n  penalties:\n    coverage: 10000000\n    other: 1000000" in message
    )
    assert "by name or through 'other'" in message

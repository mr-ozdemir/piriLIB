"""Tests of the objective building blocks in :mod:`pirilib.objective`.

Requirement source: the docstrings of :mod:`pirilib.objective` and of each term factory
(the value a term reads, its name, its linear form, weighting, sums and breakdowns, and the
errors for an unknown constraint, dimension or vehicle kind). The risk these tests reduce:
a term that reads the wrong quantity, drops a weight or loses a term in a sum silently makes
every solver minimize the wrong number.

Every value test uses one fixed plan on the conftest ``document`` problem: the owned ``big``
vehicle carries the 900 desi from ``A`` to ``B`` and a spot ``small`` vehicle carries the
250 desi from ``A`` to ``C``; the 120 desi from ``B`` to ``A`` on the second day is left
behind. Expected values are worked out by hand from the document, not read back from
:func:`pirilib.evaluate`.

Blind spots, left out on purpose: user-defined constraints in :func:`obj.violation`, the
linear form of :func:`obj.violations` over the whole catalog (covered through
``ObjectiveSpec`` in ``tests/spec/test_objective_spec.py``), and plans with lateness.
"""

import math
import re
from collections.abc import Callable
from typing import Any

import pytest

from pirilib import Dispatch, Evaluation, Load, Plan, evaluate
from pirilib import objective as obj
from pirilib.spec import LineHaulInstance

TOLERANCE = 1e-12
"""Relative tolerance of float comparisons; only the order of float additions may differ."""

KM_PER_DEGREE = 6371.0 * math.pi / 180.0
"""One degree along the equator on the 6371 km sphere of the haversine matrix."""

OWNED_KM = KM_PER_DEGREE
"""Route of the owned dispatch, ``A`` to ``B``."""

SPOT_KM = 2 * KM_PER_DEGREE
"""Route of the spot dispatch, ``A`` to ``C``."""

OWNED_FIXED = 200.0
"""Fixed cost 100 of the mandatory owned vehicle over the two calendar days."""

OWNED_DISPATCH_COST = 1.0 * OWNED_KM
"""1 per km for the owned ``big`` vehicle; its fixed cost is in :data:`OWNED_FIXED`."""

SPOT_DISPATCH_COST = 150.0 + 1.5 * SPOT_KM
"""Fixed 150 and 1.5 per km for the spot ``small`` vehicle."""

TOTAL_COST = OWNED_FIXED + OWNED_DISPATCH_COST + SPOT_DISPATCH_COST
"""Dispatch costs and the owned fixed cost; the plan is never late, so service cost is 0."""

OWNED_EMPTY = 1000.0 - 900.0
"""Empty desi on the owned ``big`` vehicle."""

SPOT_EMPTY = 400.0 - 250.0
"""Empty desi on the spot ``small`` vehicle."""

LEFT_BEHIND = 120.0
"""Desi of the second-day demand that no dispatch carries."""

COVERAGE_WEIGHT = 1000.0
"""Weight of the coverage violation in :data:`GOAL`."""

SPOT_WEIGHT = 50.0
"""Weight of the number of spot dispatches in :data:`GOAL`."""

GOAL = (
    obj.total_cost()
    + COVERAGE_WEIGHT * obj.violation("coverage")
    + obj.dispatches("spot") * SPOT_WEIGHT
)
"""A weighted sum built with both ``weight * term`` and ``term * weight``."""

GOAL_VALUE = TOTAL_COST + COVERAGE_WEIGHT * LEFT_BEHIND + SPOT_WEIGHT * 1
"""The value of :data:`GOAL`: the plan has one spot dispatch."""

DIMENSIONS = 1
"""The document has one quantity dimension, ``desi``."""

DISPATCHES = [
    Dispatch(day=0, kind="owned", source=0, origin=0, stops=(1,), loads=(Load(0, (900.0,)),)),
    Dispatch(day=0, kind="spot", source=1, origin=0, stops=(2,), loads=(Load(1, (250.0,)),)),
]
"""The owned ``big`` vehicle to ``B`` and spot offer 1 (``small``) to ``C`` on the first day."""

UNKNOWN_CONSTRAINT_ERROR = re.escape("no constraint named 'capacty'; known: ") + ".*coverage"
"""The misspelt ``capacty`` is named and the known names, ``coverage`` among them, listed."""

UNKNOWN_DIMENSION_ERROR = re.escape("no quantity dimension 'kg'; known: desi")
"""The unknown ``kg`` is named and the only known dimension listed."""

UNKNOWN_KIND_ERROR = "^" + re.escape("kind must be 'owned', 'spot' or None, not 'rented'") + "$"
"""The whole message: the allowed kinds and the rejected one."""


def cost_plus_twice_unserved(evaluation: Evaluation) -> float:
    """A plain function objective: total cost plus 2 per unit of unserved demand."""
    return evaluation.total_cost + 2.0 * float(evaluation.unserved.sum())


@pytest.fixture
def evaluation(instance: LineHaulInstance) -> Evaluation:
    """The evaluation of :data:`DISPATCHES` on the conftest problem."""
    return evaluate(instance, Plan.from_dispatches(DISPATCHES, DIMENSIONS))


@pytest.mark.parametrize(
    ("term", "expected"),
    [
        pytest.param(obj.total_cost(), TOTAL_COST, id="total_cost"),
        pytest.param(obj.owned_fixed_cost(), OWNED_FIXED, id="owned_fixed_cost"),
        pytest.param(obj.service_cost(), 0.0, id="service_cost-never-late"),
        pytest.param(
            obj.dispatch_cost(), OWNED_DISPATCH_COST + SPOT_DISPATCH_COST, id="dispatch_cost"
        ),
        pytest.param(obj.dispatch_cost("owned"), OWNED_DISPATCH_COST, id="dispatch_cost-owned"),
        pytest.param(obj.dispatch_cost("spot"), SPOT_DISPATCH_COST, id="dispatch_cost-spot"),
        pytest.param(obj.distance_km(), OWNED_KM + SPOT_KM, id="distance_km"),
        pytest.param(obj.distance_km("owned"), OWNED_KM, id="distance_km-owned"),
        pytest.param(obj.distance_km("spot"), SPOT_KM, id="distance_km-spot"),
        pytest.param(obj.dispatches(), 2.0, id="dispatches"),
        pytest.param(obj.dispatches("owned"), 1.0, id="dispatches-owned"),
        pytest.param(obj.dispatches("spot"), 1.0, id="dispatches-spot"),
        pytest.param(obj.unused_capacity(), OWNED_EMPTY + SPOT_EMPTY, id="unused_capacity"),
        pytest.param(obj.unused_capacity("owned"), OWNED_EMPTY, id="unused_capacity-owned"),
        pytest.param(obj.unused_capacity("spot"), SPOT_EMPTY, id="unused_capacity-spot"),
        pytest.param(obj.unserved(), LEFT_BEHIND, id="unserved"),
        pytest.param(obj.unserved("desi"), LEFT_BEHIND, id="unserved-desi"),
        pytest.param(obj.violation("coverage"), LEFT_BEHIND, id="violation-coverage"),
        pytest.param(obj.violation("capacity"), 0.0, id="violation-capacity-kept"),
        pytest.param(obj.violations(), LEFT_BEHIND, id="violations-all"),
        pytest.param(obj.violations(["capacity", "coverage"]), LEFT_BEHIND, id="violations-named"),
    ],
)
def test_a_term_is_the_quantity_it_names(
    evaluation: Evaluation, term: obj.Term, expected: float
) -> None:
    value = term(evaluation)

    assert value == pytest.approx(expected, rel=TOLERANCE)


def test_a_weighted_sum_is_the_sum_of_its_weighted_terms(evaluation: Evaluation) -> None:
    value = GOAL(evaluation)

    assert value == pytest.approx(GOAL_VALUE, rel=TOLERANCE)


def test_a_breakdown_gives_every_weighted_term_by_name(evaluation: Evaluation) -> None:
    breakdown = GOAL.breakdown(evaluation)

    assert breakdown == pytest.approx(
        {
            "total_cost": TOTAL_COST,
            "violation[coverage]": COVERAGE_WEIGHT * LEFT_BEHIND,
            "dispatches[spot]": SPOT_WEIGHT,
        },
        rel=TOLERANCE,
    )


def test_a_single_term_breaks_down_to_its_weighted_value(evaluation: Evaluation) -> None:
    term = 2 * obj.dispatches("spot")

    breakdown = term.breakdown(evaluation)

    assert breakdown == {"dispatches[spot]": 2.0}


def test_scaling_a_sum_scales_its_value(evaluation: Evaluation) -> None:
    doubled = 2 * GOAL

    value = doubled(evaluation)

    assert value == pytest.approx(2 * GOAL_VALUE, rel=TOLERANCE)


def test_adding_a_term_to_a_sum_appends_it() -> None:
    distance = obj.distance_km()

    combined = GOAL + distance

    assert combined.terms == (*GOAL.terms, distance)


def test_adding_two_sums_joins_their_terms() -> None:
    left = obj.total_cost() + obj.unserved()
    right = obj.dispatches() + obj.distance_km()

    combined = left + right

    assert combined.terms == (*left.terms, *right.terms)


def test_terms_with_the_same_name_are_added_in_breakdowns(evaluation: Evaluation) -> None:
    """Two dispatches counted once and once with weight 2 give one entry of 2 + 4."""
    goal = obj.dispatches() + 2 * obj.dispatches()

    breakdown = goal.breakdown(evaluation)

    assert breakdown == {"dispatches": 6.0}


def test_plain_functions_and_terms_are_both_objectives(evaluation: Evaluation) -> None:
    """Both fit the ``Objective`` type; the function adds 2 per unserved desi."""
    plain: obj.Objective = cost_plus_twice_unserved
    term: obj.Objective = obj.total_cost()

    values = (plain(evaluation), term(evaluation))

    assert values == pytest.approx((TOTAL_COST + 2.0 * LEFT_BEHIND, TOTAL_COST), rel=TOLERANCE)


@pytest.mark.parametrize(
    ("term", "expected"),
    [
        pytest.param(
            obj.total_cost(),
            {"cost[owned]": 1.0, "cost[spot]": 1.0, "owned_fixed": 1.0, "service": 1.0},
            id="total_cost",
        ),
        pytest.param(obj.service_cost(), {"service": 1.0}, id="service_cost"),
        pytest.param(obj.owned_fixed_cost(), {"owned_fixed": 1.0}, id="owned_fixed_cost"),
        pytest.param(obj.dispatch_cost("spot"), {"cost[spot]": 1.0}, id="dispatch_cost-spot"),
        pytest.param(obj.distance_km(), {"km[owned]": 1.0, "km[spot]": 1.0}, id="distance_km"),
        pytest.param(obj.dispatches("owned"), {"dispatches[owned]": 1.0}, id="dispatches-owned"),
        pytest.param(
            obj.unused_capacity(), {"unused[owned]": 1.0, "unused[spot]": 1.0}, id="unused"
        ),
        pytest.param(obj.unserved(), {"unserved": 1.0}, id="unserved"),
        pytest.param(obj.unserved("desi"), {"unserved[desi]": 1.0}, id="unserved-desi"),
        pytest.param(
            obj.violation("coverage"), {"violation[coverage]": 1.0}, id="violation-coverage"
        ),
        pytest.param(
            obj.violations(["capacity", "coverage"]),
            {"violation[capacity]": 1.0, "violation[coverage]": 1.0},
            id="violations-named",
        ),
    ],
)
def test_a_term_has_the_linear_form_of_its_features(
    term: obj.Term, expected: dict[str, float]
) -> None:
    form = obj.linear_form(term)

    assert form == expected


def test_a_weight_scales_the_linear_form() -> None:
    term = 3 * obj.dispatches("spot")

    form = obj.linear_form(term)

    assert form == {"dispatches[spot]": 3.0}


def test_a_sum_adds_the_linear_forms_of_its_terms() -> None:
    """``dispatches()`` counts both kinds; the weighted spot term adds 2 more to spot."""
    goal = obj.dispatches() + 2 * obj.dispatches("spot")

    form = obj.linear_form(goal)

    assert form == {"dispatches[owned]": 1.0, "dispatches[spot]": 3.0}


@pytest.mark.parametrize(
    "objective",
    [
        pytest.param(cost_plus_twice_unserved, id="plain-function"),
        pytest.param(obj.violation("my_rule"), id="violation-not-built-in"),
        pytest.param(obj.violations(["coverage", "my_rule"]), id="violations-not-built-in"),
        pytest.param(obj.total_cost() + obj.violation("my_rule"), id="sum-with-such-a-term"),
    ],
)
def test_an_objective_outside_the_built_in_features_has_no_linear_form(
    objective: obj.Objective,
) -> None:
    form = obj.linear_form(objective)

    assert form is None


def test_the_violation_of_an_unknown_constraint_is_a_key_error_naming_the_known(
    evaluation: Evaluation,
) -> None:
    with pytest.raises(KeyError, match=UNKNOWN_CONSTRAINT_ERROR):
        obj.violation("capacty")(evaluation)


def test_an_unknown_quantity_dimension_is_a_key_error_naming_the_known(
    evaluation: Evaluation,
) -> None:
    with pytest.raises(KeyError, match=UNKNOWN_DIMENSION_ERROR):
        obj.unserved("kg")(evaluation)


@pytest.mark.parametrize(
    "factory",
    [obj.dispatch_cost, obj.distance_km, obj.dispatches, obj.unused_capacity],
    ids=["dispatch_cost", "distance_km", "dispatches", "unused_capacity"],
)
def test_an_unknown_vehicle_kind_is_a_value_error(
    evaluation: Evaluation, factory: Callable[[Any], obj.Term]
) -> None:
    """Building and reading the term is one step: the docs do not say which of them checks."""
    with pytest.raises(ValueError, match=UNKNOWN_KIND_ERROR):
        factory("rented")(evaluation)

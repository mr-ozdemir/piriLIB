"""Tests of the minimum fill with a share per quantity dimension.

The requirement is the documentation of :class:`pirilib.spec.MinFill` and the ``min_fill`` entry
of :data:`pirilib.constraints.CATALOG`: a number is the share of the primary dimension, a
mapping gives a share per dimension, a vehicle that reaches the share of any one dimension is
filled, and a vehicle that reaches none misses the smallest of its gaps.

Oracles: the amounts are computed by hand from the fixture below, and
:func:`~pirilib.check_plan` computes them independently of the evaluator. The solvers are
checked against the evaluator in ``tests/test_rule_support.py``.

The fixture is the shared ``document`` with a second dimension. ``big`` holds 1000 desi and
500 kg, ``small`` 400 desi and 100 kg. The plan carries every demand whole: the owned ``big``
vehicle takes 900 desi and 300 kg, one spot ``small`` vehicle 250 desi and 40 kg (62.5 % and
40 % of its capacity) and another 120 desi and 60 kg (30 % and 60 %).
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from pirilib import Dispatch, Load, Plan, SpecError, check_plan, evaluate
from pirilib.evaluation import fill_shortfall
from pirilib.spec import LineHaulInstance, MinFill, parse_spec
from pirilib.validate import Violation

SMALL = 1
"""Position of the ``small`` spot offer."""

PLAN = Plan.from_dispatches(
    [
        Dispatch(
            day=0, kind="owned", source=0, origin=0, stops=(1,), loads=(Load(0, (900.0, 300.0)),)
        ),
        Dispatch(
            day=0, kind="spot", source=SMALL, origin=0, stops=(2,), loads=(Load(1, (250.0, 40.0)),)
        ),
        Dispatch(
            day=1, kind="spot", source=SMALL, origin=1, stops=(0,), loads=(Load(2, (120.0, 60.0)),)
        ),
    ],
    dimensions=2,
)
"""Every demand carried whole; see the module documentation for the loads."""

CASES = [
    pytest.param({"desi": 0.5, "kg": 0.5}, None, 0.0, id="each-reaches-one"),
    pytest.param({"desi": 0.75, "kg": 0.5}, None, 10.0, id="one-reaches-none"),
    pytest.param({"kg": 0.5}, None, 10.0, id="second-dimension-only"),
    pytest.param({"desi": 0.75}, None, 230.0, id="first-dimension-only"),
    pytest.param({"desi": 0.75, "kg": 0.0}, None, 230.0, id="zero-is-no-minimum"),
    pytest.param(0.75, None, 230.0, id="number-is-the-primary"),
    pytest.param(0.5, "kg", 10.0, id="number-with-another-primary"),
]
"""The spot minimum fill, the primary dimension (None: the first) and the amount by hand.

- 50 % of either: the first ``small`` reaches it in desi (250 of 200), the second in kg.
- 75 % desi or 50 % kg: the first ``small`` misses 50 desi and 10 kg, so 10; the second
  reaches its 50 kg.
- 50 % kg alone: the same 10; desi is not looked at.
- 75 % desi alone: 300 - 250 = 50 and 300 - 120 = 180, so 230.
"""


@pytest.fixture
def weighed(document: dict[str, Any]) -> dict[str, Any]:
    """The shared document with a ``kg`` dimension; see the module documentation."""
    capacities = {"big": 500.0, "small": 100.0}
    for vehicle in document["vehicle_types"]:
        vehicle["capacity"]["kg"] = capacities[vehicle["name"]]
    for row, kg in zip(document["demand"], (300.0, 40.0, 60.0), strict=True):
        row["quantity"]["kg"] = kg
    return document


def with_min_fill(document: dict[str, Any], spot: object, primary: str | None) -> LineHaulInstance:
    """The compiled document with ``spot`` as the spot minimum fill and ``primary`` as primary."""
    document["rules"] = {"min_fill": {"spot": spot}, "primary_dimension": primary}
    return parse_spec(document).compile()


def test_the_helper_sets_the_spot_minimum_fill(weighed: dict[str, Any]) -> None:
    """The helper's own test: the mapping reaches the instance as one share per dimension."""
    instance = with_min_fill(weighed, {"kg": 0.5}, None)

    assert instance.min_fill_spot.tolist() == [0.0, 0.5]


@pytest.mark.parametrize(("spot", "primary", "amount"), CASES)
def test_the_evaluator_counts_the_smallest_gap_of_a_vehicle_that_reaches_no_minimum(
    weighed: dict[str, Any], spot: object, primary: str | None, amount: float
) -> None:
    instance = with_min_fill(weighed, spot, primary)

    evaluation = evaluate(instance, PLAN)

    assert evaluation.violations["min_fill"] == amount


@pytest.mark.parametrize(("spot", "primary", "amount"), CASES)
def test_the_checker_counts_the_same_amount(
    weighed: dict[str, Any], spot: object, primary: str | None, amount: float
) -> None:
    instance = with_min_fill(weighed, spot, primary)

    report = check_plan(instance, PLAN)

    assert report.totals().get("min_fill", 0.0) == amount


def test_the_checker_names_the_nearest_minimum(weighed: dict[str, Any]) -> None:
    """The first ``small`` misses 50 desi and 10 kg: the message is about the 40 of 50 kg."""
    instance = with_min_fill(weighed, {"desi": 0.75, "kg": 0.5}, None)

    report = check_plan(instance, PLAN)

    assert (
        Violation(
            "min_fill",
            10.0,
            "dispatch 1 (2026-01-05, spot small from A) carries 40 kg, the minimum fill is "
            "0.5 x 100; no other minimum fill is reached either",
        )
        in report.violations
    )


@pytest.mark.parametrize(
    ("share", "load", "missing"),
    [
        ([0.5, 0.5], [120.0, 60.0], 0.0),
        ([0.5, 0.5], [120.0, 40.0], 10.0),
        ([0.75, 0.5], [290.0, 40.0], 10.0),
        ([0.0, 0.5], [0.0, 40.0], 10.0),
        ([0.0, 0.0], [0.0, 0.0], 0.0),
        ([0.5, 0.0], [200.0 - 1e-8, 0.0], 0.0),
    ],
    ids=["one-reached", "nearest-is-kg", "tie", "one-minimum", "no-minimum", "within-tolerance"],
)
def test_fill_shortfall_of_one_vehicle(
    share: list[float], load: list[float], missing: float
) -> None:
    """A vehicle of 400 desi and 100 kg: the smallest gap, 0 once any minimum is reached.

    Both gaps are 10 in the tie; a load 1e-8 below 200 desi is within the relative tolerance
    of 1e-9 times the capacity of 400.
    """
    capacity = np.array([[400.0, 100.0]])

    shortfall = fill_shortfall(np.array([share]), capacity, np.array([load]))

    assert shortfall.tolist() == [missing]


def test_fill_shortfall_is_given_per_vehicle() -> None:
    """Two vehicles with the same minimums: the first reaches desi, the second misses 10 kg."""
    share = np.array([[0.5, 0.5], [0.5, 0.5]])
    capacity = np.array([[400.0, 100.0], [400.0, 100.0]])
    load = np.array([[250.0, 40.0], [120.0, 40.0]])

    shortfall = fill_shortfall(share, capacity, load)

    assert shortfall.tolist() == [0.0, 10.0]


def test_fill_shortfall_of_no_vehicles_is_empty() -> None:
    empty = np.zeros((0, 2))

    shortfall = fill_shortfall(empty, empty, empty)

    assert shortfall.shape == (0,)


@pytest.mark.parametrize(
    ("given", "primary", "shares"),
    [
        (0.3, "desi", (0.3, 0.0)),
        (0.3, "kg", (0.0, 0.3)),
        ({"kg": 0.5}, "desi", (0.0, 0.5)),
        ({"desi": 0.4, "kg": 0.5}, "desi", (0.4, 0.5)),
    ],
    ids=["number", "number-other-primary", "one-dimension", "both-dimensions"],
)
def test_min_fill_gives_a_share_per_dimension(
    given: float | dict[str, float], primary: str, shares: tuple[float, ...]
) -> None:
    """A number belongs to the primary dimension; a dimension that is left out has no minimum."""
    min_fill = MinFill(spot=given)

    found = min_fill.shares("spot", ("desi", "kg"), primary)

    assert found == shares


def test_owned_and_spot_vehicles_have_their_own_minimum_fill(weighed: dict[str, Any]) -> None:
    weighed["rules"] = {"min_fill": {"spot": {"kg": 0.5}, "owned": 0.2}}

    instance = parse_spec(weighed).compile()

    assert (instance.min_fill_owned.tolist(), instance.min_fill_spot.tolist()) == (
        [0.2, 0.0],
        [0.0, 0.5],
    )


@pytest.mark.parametrize(
    ("spot", "problem"),
    [
        (
            {"ton": 0.5},
            "rules.min_fill.spot: unknown dimension(s) ['ton']; the dimensions are ['desi', 'kg']",
        ),
        ({"kg": 1.0}, "rules.min_fill.spot.kg: Input should be less than 1"),
        ({"kg": -0.1}, "rules.min_fill.spot.kg: Input should be greater than or equal to 0"),
    ],
    ids=["unknown-dimension", "share-of-one", "negative-share"],
)
def test_a_minimum_fill_that_cannot_be_used_is_reported(
    weighed: dict[str, Any], spot: dict[str, float], problem: str
) -> None:
    weighed["rules"] = {"min_fill": {"spot": spot}}

    with pytest.raises(SpecError) as info:
        parse_spec(weighed)

    assert str(info.value).splitlines() == ["<data>: invalid problem specification", f"- {problem}"]


def test_a_vehicle_filled_by_weight_is_not_underfilled(
    weighed: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    """The reason for the rule: 120 desi is 30 % of the vehicle, but its 60 kg are 60 %."""
    weighed["rules"] = {"min_fill": {"spot": {"desi": 0.5, "kg": 0.5}}}
    instance = compile_document(weighed)

    evaluation = evaluate(instance, PLAN)

    assert evaluation.violated() == {}

"""Tests of the cost per unit carried (``cost.per_unit``).

The requirement is the documentation of :class:`pirilib.spec.Cost` and "How the total cost is
computed" in ``docs/objective.md``: a dispatch pays, besides its fixed, km and hour parts, the
price per unit of every quantity it carries, whatever the distance, and a dimension without a
price costs nothing.

Oracles: the costs are computed by hand from the shared ``document`` fixture,
:func:`~pirilib.check_plan` computes them independently of the evaluator, and the solvers are
compared with the evaluator in ``tests/test_rule_support.py``.

The plan carries every demand whole: the owned ``big`` vehicle 900 desi from ``A`` to ``B``,
one spot ``small`` vehicle 250 desi from ``A`` to ``C`` and another 120 desi from ``B`` to
``A``.
"""

from typing import Any

import pytest

from pirilib import ACOConfig, ACOSolver, Dispatch, Load, Plan, SpecError, check_plan, evaluate
from pirilib import objective as obj
from pirilib.solvers.base import Solver
from pirilib.solvers.direct import DirectShipping
from pirilib.solvers.exact import ExactSolver
from pirilib.spec import parse_spec

pytest.importorskip("scipy", reason="the exact solver needs scipy (pip install 'pirilib[exact]')")

BIG, SMALL = 0, 1
"""Positions of the spot offers."""

PLAN = Plan.from_dispatches(
    [
        Dispatch(day=0, kind="owned", source=0, origin=0, stops=(1,), loads=(Load(0, (900.0,)),)),
        Dispatch(
            day=0, kind="spot", source=SMALL, origin=0, stops=(2,), loads=(Load(1, (250.0,)),)
        ),
        Dispatch(
            day=1, kind="spot", source=SMALL, origin=1, stops=(0,), loads=(Load(2, (120.0,)),)
        ),
    ],
    dimensions=1,
)
"""Every demand carried whole; see the module documentation."""

GOAL = obj.total_cost() + 1e6 * obj.violations()
"""Costs first; any broken rule is far dearer than a plan of this size."""

RELATIVE_TOLERANCE = 1e-12
"""Rounding of a sum of a few costs."""

SOLVERS: dict[str, Solver] = {
    "direct": DirectShipping(),
    "aco": ACOSolver(ACOConfig(ants=6, iterations=10, patience=5)),
    "exact": ExactSolver(),
}


def test_a_dispatch_pays_for_every_unit_it_carries(document: dict[str, Any]) -> None:
    """Owned at 0.1 a desi: 90 for 900. Spot ``small`` at 0.5: 125 for 250 and 60 for 120."""
    free = evaluate(parse_spec(document).compile(), PLAN)
    document["fleet"]["owned"][0]["cost"]["per_unit"] = {"desi": 0.1}
    document["fleet"]["spot"][SMALL]["cost"]["per_unit"] = {"desi": 0.5}

    priced = evaluate(parse_spec(document).compile(), PLAN)

    assert (priced.dispatch_cost - free.dispatch_cost).tolist() == pytest.approx(
        [90.0, 125.0, 60.0], rel=RELATIVE_TOLERANCE
    )


def test_units_are_part_of_the_total_cost(document: dict[str, Any]) -> None:
    """90 + 125 + 60 = 275 on top of the plan's cost without prices per unit."""
    free = evaluate(parse_spec(document).compile(), PLAN)
    document["fleet"]["owned"][0]["cost"]["per_unit"] = {"desi": 0.1}
    document["fleet"]["spot"][SMALL]["cost"]["per_unit"] = {"desi": 0.5}

    priced = evaluate(parse_spec(document).compile(), PLAN)

    assert priced.total_cost - free.total_cost == pytest.approx(275.0, rel=RELATIVE_TOLERANCE)


def test_an_offer_that_is_not_used_charges_nothing(document: dict[str, Any]) -> None:
    """No dispatch of the plan is a spot ``big`` vehicle, so its price changes no cost."""
    free = evaluate(parse_spec(document).compile(), PLAN)
    document["fleet"]["spot"][BIG]["cost"]["per_unit"] = {"desi": 9.0}

    priced = evaluate(parse_spec(document).compile(), PLAN)

    assert priced.total_cost == free.total_cost


def test_every_dimension_has_its_own_price(document: dict[str, Any]) -> None:
    """The second ``small`` carries 120 desi and 60 kg: 120 x 0.5 + 60 x 2 = 180; kg alone 120."""
    for vehicle in document["vehicle_types"]:
        vehicle["capacity"]["kg"] = 500.0
    for row, kg in zip(document["demand"], (300.0, 40.0, 60.0), strict=True):
        row["quantity"]["kg"] = kg
    plan = Plan.from_dispatches(
        [
            Dispatch(
                day=1,
                kind="spot",
                source=SMALL,
                origin=1,
                stops=(0,),
                loads=(Load(2, (120.0, 60.0)),),
            )
        ],
        dimensions=2,
    )
    free = evaluate(parse_spec(document).compile(), plan)
    document["fleet"]["spot"][SMALL]["cost"]["per_unit"] = {"desi": 0.5, "kg": 2.0}

    priced = evaluate(parse_spec(document).compile(), plan)

    assert priced.total_cost - free.total_cost == pytest.approx(180.0, rel=RELATIVE_TOLERANCE)


def test_the_checker_computes_the_same_cost(document: dict[str, Any]) -> None:
    document["fleet"]["owned"][0]["cost"]["per_unit"] = {"desi": 0.1}
    document["fleet"]["spot"][SMALL]["cost"]["per_unit"] = {"desi": 0.5}
    instance = parse_spec(document).compile()

    report = check_plan(instance, PLAN)

    assert report.total_cost == pytest.approx(
        evaluate(instance, PLAN).total_cost, rel=RELATIVE_TOLERANCE
    )


def test_prices_per_unit_reach_the_instance_by_dimension(document: dict[str, Any]) -> None:
    """One row per offer or group and one column per dimension; no price is 0."""
    document["fleet"]["spot"][SMALL]["cost"]["per_unit"] = {"desi": 0.5}

    instance = parse_spec(document).compile()

    assert instance.spot_per_unit.tolist() == [[0.0], [0.5]]
    assert instance.owned_per_unit.tolist() == [[0.0]]


@pytest.mark.parametrize(
    ("per_unit", "problem"),
    [
        (
            {"ton": 1.0},
            "fleet.spot[1].cost.per_unit: unknown dimension(s) ['ton']; "
            "the dimensions are ['desi']",
        ),
        (
            {"desi": -0.5},
            "fleet.spot[1].cost.per_unit.desi: Input should be greater than or equal to 0",
        ),
    ],
    ids=["unknown-dimension", "negative-price"],
)
def test_a_price_per_unit_that_cannot_be_used_is_reported(
    document: dict[str, Any], per_unit: dict[str, float], problem: str
) -> None:
    document["fleet"]["spot"][SMALL]["cost"]["per_unit"] = per_unit

    with pytest.raises(SpecError) as info:
        parse_spec(document)

    assert str(info.value).splitlines() == ["<data>: invalid problem specification", f"- {problem}"]


@pytest.mark.parametrize("name", SOLVERS)
def test_a_solver_counts_the_units_when_it_picks_a_vehicle(
    document: dict[str, Any], name: str
) -> None:
    """The 120 desi from ``B`` to ``A`` go by ``small`` (150 + 1.5 per km against 300 + 2).

    At 5 a desi the ``small`` vehicle costs 600 more, far beyond the difference to ``big``, so
    the load goes by ``big``. Only this demand leaves from ``B``, so its dispatch is found by
    the origin.
    """
    document["fleet"]["spot"][SMALL]["cost"]["per_unit"] = {"desi": 5.0}
    instance = parse_spec(document).compile()

    result = SOLVERS[name].solve(instance, GOAL, seed=0)

    assert [dispatch.source for dispatch in result.plan.dispatches() if dispatch.origin == 1] == [
        BIG
    ]


@pytest.mark.parametrize("name", SOLVERS)
def test_a_solver_takes_the_small_vehicle_without_a_price_per_unit(
    document: dict[str, Any], name: str
) -> None:
    """The other half of the test above: without the price the same load goes by ``small``."""
    instance = parse_spec(document).compile()

    result = SOLVERS[name].solve(instance, GOAL, seed=0)

    assert [dispatch.source for dispatch in result.plan.dispatches() if dispatch.origin == 1] == [
        SMALL
    ]

"""Several spot offers for one vehicle type: price tiers and carriers.

The problem is the shared fixture with three loads of 300 desi from ``A`` to ``B`` on one
day: 900 desi that need three ``small`` vehicles (400 desi each). Two offers hire ``small``
vehicles: the contract gives two a day for 100 each, the market any number for 180 each.
The cheapest plan takes both contract vehicles and one from the market, for 380.
"""

from collections.abc import Callable
from typing import Any

import pytest

from pirilib import ACOConfig, ACOSolver, Dispatch, Load, Plan, check_plan
from pirilib import objective as obj
from pirilib.solvers.base import Solver
from pirilib.solvers.direct import DirectShipping
from pirilib.solvers.exact import ExactSolver
from pirilib.spec import LineHaulInstance, parse_spec

type CompileDocument = Callable[[dict[str, Any]], LineHaulInstance]

A, B = 0, 1
CONTRACT, MARKET = 0, 1
"""Positions of the two offers under ``fleet.spot``."""

CONTRACT_PRICE, MARKET_PRICE = 100.0, 180.0
CONTRACT_VEHICLES = 2
LOADS = 3
DESI = 300.0

CHEAPEST = CONTRACT_VEHICLES * CONTRACT_PRICE + MARKET_PRICE
"""Both contract vehicles and one market vehicle."""

GOAL = obj.total_cost() + 1e6 * obj.violations()

SOLVERS: dict[str, Solver] = {
    "aco": ACOSolver(ACOConfig(ants=6, iterations=8, patience=4)),
    "exact": ExactSolver(),
    "direct": DirectShipping(),
}


def tiers(document: dict[str, Any]) -> dict[str, Any]:
    """The fixture with the two ``small`` offers and the three loads of this module."""
    document["fleet"] = {
        "spot": [
            {
                "type": "small",
                "name": "contract",
                "cost": {"fixed": CONTRACT_PRICE},
                "available": CONTRACT_VEHICLES,
            },
            {"type": "small", "name": "market", "cost": {"fixed": MARKET_PRICE}},
        ]
    }
    document["demand"] = [
        {"origin": "A", "destination": "B", "date": "2026-01-05", "quantity": {"desi": DESI}}
        for _ in range(LOADS)
    ]
    return document


def contract_only() -> Plan:
    """Every load on a contract vehicle: one more than the contract gives a day."""
    dispatches = [
        Dispatch(
            day=0, kind="spot", source=CONTRACT, origin=A, stops=(B,), loads=(Load(row, (DESI,)),)
        )
        for row in range(LOADS)
    ]
    return Plan.from_dispatches(dispatches, 1)


def test_two_offers_may_have_the_same_vehicle_type(document: dict[str, Any]) -> None:
    spec = parse_spec(tiers(document))

    assert [offer.type for offer in spec.fleet.spot] == ["small", "small"]


def test_the_instance_keeps_the_names_of_the_offers(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    instance = compile_document(tiers(document))

    assert instance.spot_names == ("contract", "market")


def test_an_offer_without_a_name_has_an_empty_one(instance: LineHaulInstance) -> None:
    assert instance.spot_names == ("", "")


@pytest.mark.parametrize("name", SOLVERS)
def test_the_cheaper_offer_is_used_up_before_the_dearer_one(
    name: str, document: dict[str, Any], compile_document: CompileDocument
) -> None:
    instance = compile_document(tiers(document))

    result = SOLVERS[name].solve(instance, GOAL)

    assert sorted(result.plan.source.tolist()) == [CONTRACT, CONTRACT, MARKET]


@pytest.mark.parametrize("name", SOLVERS)
def test_the_plan_costs_the_cheapest_mix_of_the_offers(
    name: str, document: dict[str, Any], compile_document: CompileDocument
) -> None:
    instance = compile_document(tiers(document))

    result = SOLVERS[name].solve(instance, GOAL)

    assert result.evaluation.total_cost == pytest.approx(CHEAPEST)


@pytest.mark.parametrize("name", SOLVERS)
def test_each_offer_keeps_its_own_daily_limit(
    name: str, document: dict[str, Any], compile_document: CompileDocument
) -> None:
    instance = compile_document(tiers(document))

    result = SOLVERS[name].solve(instance, GOAL)

    assert result.evaluation.feasible


def test_the_limit_of_one_offer_does_not_cover_the_other(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    instance = compile_document(tiers(document))

    report = check_plan(instance, contract_only())

    assert [(item.constraint, item.amount) for item in report.violations] == [
        ("spot_availability", 1.0)
    ]


def test_the_checker_names_the_offer_whose_limit_is_broken(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    instance = compile_document(tiers(document))

    report = check_plan(instance, contract_only())

    assert [item.message for item in report.violations] == [
        "spot offer 0 (contract) is used 3 times on 2026-01-05, 2 available"
    ]

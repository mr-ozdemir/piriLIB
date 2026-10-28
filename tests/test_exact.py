"""Tests of the exact solver, :class:`~pirilib.solvers.exact.ExactSolver`.

The requirement is the documentation of :mod:`pirilib.solvers.exact`: every subproblem is solved
to optimality by mixed-integer programming over every route that keeps the rules, only linear
objectives without own constraints are accepted, demand that may not be split is not supported,
and a subproblem with more destinations than ``max_nodes`` is refused. :class:`~pirilib.ACOConfig`
documents ``exact_nodes``: the colony then solves such small subproblems exactly. These are
integration tests (``tests/conftest.py`` marks them so): they run the real HiGHS solver of scipy,
which is the risk they reduce, namely a plan said to be optimal that is not, that breaks a rule or
that costs other than pirilib says. Without scipy the module is skipped with the reason.

Oracles: hand-computed costs on the three equator centers of the ``document`` fixture,
:func:`~pirilib.check_plan` as an independent checker of the cost, and the ant colony as a bound
the optimum may not exceed.

Blind spots, left out on purpose: scheduled instances (``tests/test_schedule.py``), time limits
(they depend on the machine) and problems larger than the random ones of ``tests/conftest.py``.
"""

import math
import re
from collections.abc import Callable
from typing import Any

import pytest

from pirilib import ACOConfig, ACOSolver, check_plan
from pirilib import objective as obj
from pirilib.constraints import MaxRouteKm
from pirilib.solvers.exact import ExactSolver
from pirilib.spec import LineHaulInstance, parse_spec

pytest.importorskip("scipy", reason="the exact solver needs scipy (pip install 'pirilib[exact]')")

type CompileDocument = Callable[[dict[str, Any]], LineHaulInstance]
type RandomDocument = Callable[[int], dict[str, Any]]

KM_PER_DEGREE = 6371.0 * math.pi / 180.0
"""Haversine distance of one degree along the equator, on the 6371 km Earth pirilib uses.

The ``document`` fixture puts ``A``, ``B`` and ``C`` on the equator one degree apart, so ``A`` to
``B`` is one degree and ``A`` to ``C`` two.
"""

SMALL_FIXED = 150.0
"""Fixed cost of a small spot vehicle in :func:`small_spot_problem`."""

SMALL_PER_KM = 1.5
"""Cost per km of a small spot vehicle in :func:`small_spot_problem`."""

TWO_STOP_ROUTE_COST = SMALL_FIXED + SMALL_PER_KM * 2 * KM_PER_DEGREE
"""One small vehicle from ``A`` by ``B`` to ``C``: one fixed cost over two degrees (483.58)."""

ONE_STOP_ROUTES_COST = 2 * SMALL_FIXED + SMALL_PER_KM * (1 + 2) * KM_PER_DEGREE
"""Two small vehicles, ``A`` to ``B`` (one degree) and ``A`` to ``C`` (two degrees) (800.38)."""

VIOLATION_WEIGHT = 1e6
"""Objective weight of one unit of violation, far above the cost of any of these small plans."""

GOAL = obj.total_cost() + VIOLATION_WEIGHT * obj.violations()

RELATIVE_TOLERANCE = 1e-9
"""Relative slack of an objective comparison, for the rounding of floating-point sums only."""

RANDOM_PROBLEMS = 12
"""Seeds of ``random_document`` tried; they give 4 to 11 subproblems of varied rules."""

SHORT_COLONY = ACOConfig(ants=6, iterations=8, patience=4)
"""A short colony run: the bound holds for any run, and a short one keeps the test fast."""

SUBPROBLEMS_OF_DOCUMENT = 2
"""Subproblems of the ``document`` fixture: one per day and origin (day 1 from A, day 2 from B)."""

HARD_RULES = (
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
    "stopover",
    "max_stops",
    "stretch",
    "forward_only",
    "split",
    "overdelivery",
)
"""Every built-in constraint except ``min_fill`` and ``coverage``.

The candidate routes keep the rules and the program keeps capacity, availability and the demand,
so none of these is ever broken. A minimum fill and the demand carried are prices in the
objective instead: an optimal plan may run a vehicle below its minimum fill or leave demand
when that is cheaper.
"""


def small_spot_problem(document: dict[str, Any]) -> dict[str, Any]:
    """``document`` with one small spot offer and 100 desi from ``A`` to each of ``B`` and ``C``.

    The offer is unlimited (fixed 150, 1.5 per km); a small vehicle holds 400 desi, so both loads
    fit one vehicle, whose cheapest route is ``A`` by ``B`` to ``C``.
    """
    document["fleet"] = {
        "spot": [{"type": "small", "cost": {"fixed": SMALL_FIXED, "per_km": SMALL_PER_KM}}]
    }
    document["demand"] = [
        {"origin": "A", "destination": "B", "date": "2026-01-05", "quantity": {"desi": 100.0}},
        {"origin": "A", "destination": "C", "date": "2026-01-05", "quantity": {"desi": 100.0}},
    ]
    return document


def exactly_solvable(document: dict[str, Any]) -> LineHaulInstance:
    """A random problem compiled with split demand and an unlimited small spot offer.

    The exact solver does not support demand that may not be split. Subproblems share limited
    spot offers, so with a limit the whole plan may not be optimal; without one, the optimum of
    every subproblem is the optimum of the whole plan.
    """
    document["rules"]["split_demand"] = True
    document["fleet"]["spot"][1].pop("available", None)
    return parse_spec(document).compile()


def test_two_loads_from_one_origin_share_one_vehicle_on_the_cheapest_route(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    """Both loads ride one small vehicle ``A`` by ``B`` (center 1) to ``C`` (center 2)."""
    instance = compile_document(small_spot_problem(document))

    result = ExactSolver().solve(instance, GOAL)

    assert result.evaluation.feasible
    assert len(result.plan) == 1
    assert result.plan.stops(0).tolist() == [1, 2]
    assert result.evaluation.total_cost == pytest.approx(TWO_STOP_ROUTE_COST)


def test_stats_mark_the_plan_optimal_and_score_it_at_the_optimum(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    """The score is the objective without constants; this goal has none, so it is the optimum."""
    instance = compile_document(small_spot_problem(document))

    result = ExactSolver().solve(instance, GOAL)

    assert result.stats["optimal"] == "True"
    assert result.stats["score"] == pytest.approx(TWO_STOP_ROUTE_COST)
    assert result.objective == pytest.approx(TWO_STOP_ROUTE_COST)


def test_routes_keep_the_stop_limit_even_when_a_longer_route_is_cheaper(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    """With ``max_stops = 1`` the cheaper two-stop route is no candidate: two vehicles run."""
    document = small_spot_problem(document)
    document["rules"] = {"stopover": {"max_stops": 1}}
    instance = compile_document(document)

    result = ExactSolver().solve(instance, GOAL)

    assert len(result.plan) == 2
    assert sorted(result.plan.stop_center.tolist()) == [1, 2]
    assert result.evaluation.total_cost == pytest.approx(ONE_STOP_ROUTES_COST)


def test_subproblems_of_one_day_share_a_limited_spot_offer(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    """One small vehicle a day serves only one of two origins; the other origin's 100 desi wait.

    ``available`` limits the vehicles of a spot offer per day (``docs/guide.md``). The loads from
    ``A`` and from ``B`` are separate subproblems of the same day, so together they may hire one
    vehicle only.
    """
    document = small_spot_problem(document)
    document["fleet"]["spot"][0]["available"] = 1
    document["demand"] = [
        {"origin": "A", "destination": "B", "date": "2026-01-05", "quantity": {"desi": 100.0}},
        {"origin": "B", "destination": "A", "date": "2026-01-05", "quantity": {"desi": 100.0}},
    ]
    instance = compile_document(document)

    result = ExactSolver().solve(instance, GOAL)

    assert len(result.plan) == 1
    assert result.evaluation.violations["spot_availability"] == 0.0
    assert result.evaluation.violations["coverage"] == pytest.approx(100.0)


@pytest.mark.parametrize("seed", range(RANDOM_PROBLEMS))
def test_never_worse_than_the_colony(seed: int, random_document: RandomDocument) -> None:
    """No plan of the colony beats the optimum."""
    instance = exactly_solvable(random_document(seed))
    colony = ACOSolver(SHORT_COLONY).solve(instance, GOAL, seed=seed)

    exact = ExactSolver().solve(instance, GOAL)

    assert exact.objective <= colony.objective * (1 + RELATIVE_TOLERANCE)


@pytest.mark.parametrize("seed", range(RANDOM_PROBLEMS))
def test_plan_cost_agrees_with_the_independent_checker(
    seed: int, random_document: RandomDocument
) -> None:
    """:func:`~pirilib.check_plan` prices the plan as the evaluator does."""
    instance = exactly_solvable(random_document(seed))

    exact = ExactSolver().solve(instance, GOAL)

    assert check_plan(instance, exact.plan).total_cost == pytest.approx(exact.evaluation.total_cost)


@pytest.mark.parametrize("seed", range(RANDOM_PROBLEMS))
def test_plan_breaks_no_hard_rule(seed: int, random_document: RandomDocument) -> None:
    """Every constraint of :data:`HARD_RULES` measures zero."""
    instance = exactly_solvable(random_document(seed))

    exact = ExactSolver().solve(instance, GOAL)

    assert exact.evaluation.violations.items() >= dict.fromkeys(HARD_RULES, 0.0).items()


def test_hybrid_colony_solves_small_subproblems_exactly(instance: LineHaulInstance) -> None:
    """With ``exact_nodes`` above every subproblem's size, the colony reaches the optimum."""
    exact = ExactSolver().solve(instance, GOAL)

    hybrid = ACOSolver(ACOConfig(exact_nodes=9)).solve(instance, GOAL)

    assert hybrid.stats["exact"] == SUBPROBLEMS_OF_DOCUMENT
    assert hybrid.objective == pytest.approx(exact.objective)


def test_objective_without_a_linear_form_is_rejected(instance: LineHaulInstance) -> None:
    """A plain function of the evaluation has no linear form."""
    solver = ExactSolver()

    with pytest.raises(ValueError, match=re.escape("needs an objective with a linear form")):
        solver.solve(instance, lambda evaluation: evaluation.total_cost)


def test_own_constraints_are_rejected(instance: LineHaulInstance) -> None:
    """Constraints of one's own are refused even with a linear objective."""
    solver = ExactSolver()

    with pytest.raises(ValueError, match=re.escape("and no own constraints")):
        solver.solve(instance, GOAL, [MaxRouteKm(1000.0)])


def test_demand_that_may_not_be_split_is_rejected(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    """``rules.split_demand = false`` is not supported."""
    document["rules"] = {"split_demand": False}
    instance = compile_document(document)

    with pytest.raises(ValueError, match=re.escape("does not support rules.split_demand = false")):
        ExactSolver().solve(instance, GOAL)


def test_subproblem_with_more_destinations_than_max_nodes_is_rejected(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    """The subproblem from ``A`` on the first day has two destinations, ``B`` and ``C``."""
    instance = compile_document(small_spot_problem(document))

    with pytest.raises(ValueError, match=re.escape("has 2 destinations, more than max_nodes=1")):
        ExactSolver(max_nodes=1).solve(instance, GOAL)


def test_demand_is_left_when_serving_it_costs_more(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    """Leaving the 200 desi costs 200; serving them costs more than the route alone.

    A minimum fill of 0.9 asks 360 of the 400 desi of a small vehicle. Serving both loads costs
    at least :data:`TWO_STOP_ROUTE_COST` (483.58) plus 10 per desi missing from the minimum fill;
    leaving them costs 1 per desi of demand not carried.
    """
    document = small_spot_problem(document)
    document["rules"] = {"min_fill": {"spot": 0.9}}
    instance = compile_document(document)
    goal = obj.total_cost() + 10.0 * obj.violation("min_fill") + 1.0 * obj.violation("coverage")

    result = ExactSolver().solve(instance, goal)

    assert len(result.plan) == 0
    assert result.evaluation.violations["coverage"] == pytest.approx(200.0)

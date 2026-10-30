"""Unit tests of :func:`pirilib.solvers.local_search.improve`, the local search of one subproblem.

Requirement sources: the docstrings of :mod:`pirilib.solvers.local_search` (the moves and
the rules every route keeps) and of :func:`~pirilib.solvers.local_search.improve` (the
returned change is never positive, ``remaining`` is what the routes leave uncarried,
``room`` limits the moves): six moves, drop follows the objective, moves check the vehicle
slots of the day, and the cheap checks and remembered rejections leave plans and scores
unchanged.

Risk reduced: the colony hands its best routes to this search, so a move that breaks a rule,
misses a saving or reports a wrong change of the objective reaches every linear-objective
plan. Each move test starts from routes written by hand, so the move under test is the only
improvement there is, and checks the routes and the change against values worked out by hand
from the scenario (haversine kilometres on the equator, the costs of the fleet below).

Equivalence classes, per move: the move improves the objective and is taken, or it does not
and the routes stay. Boundaries tested on both sides: the merged load at the 400 desi of a
``small`` vehicle (400 merged, 401 not), the cost of leaving a load against the cost of its
route (drop below, keep above) and the vehicle slots of the day (0 blocks, 1 allows).

Blind spots, on purpose: demand that may not be split (``item_left``, item slots), timed
routes of scheduled instances (hours, lateness), owned lines and ``forward_only`` ordering are
tested through the solvers in ``test_aco.py`` and ``test_schedule.py``, not here. Speed (the
point of the cheap checks) is measured in ``tests/performance``. The search is single threaded,
so there is no parallel behaviour. Floating point: costs are compared with a stated
relative tolerance and loads rounded to :data:`AMOUNT_DECIMALS`; the relocate saving uses a
planar distance and a looser tolerance. Whole plans are only checked by the regression pin
at the end; passing these tests does not show that every move is right on every subproblem.
"""

import copy
import math
from typing import Any

import numpy as np
import pytest

from pirilib import ACOSolver, synthetic
from pirilib import objective as obj
from pirilib.solvers.base import linear_coefficients
from pirilib.solvers.decompose import UNBOUNDED, Routes, Subproblem, subproblems
from pirilib.solvers.ledger import Room
from pirilib.solvers.local_search import Shared, improve
from pirilib.spec import LineHaulInstance, parse_spec

DEGREE_KM = 6371.0 * math.pi / 180.0
"""Haversine length of one degree on the equator, with the 6371 km Earth radius of pirilib.geo.

The centers of the ``document`` fixture lie on the equator one degree apart: ``A`` to ``B``
is one degree and ``A`` to ``C`` two.
"""

NODE_B, NODE_C = 0, 1
"""Subproblem nodes of ``B`` and ``C``: the destinations of the demand, in document order."""

SMALL_FIXED, SMALL_PER_KM = 150.0, 1.5
"""Cost of a ``small`` spot vehicle (400 desi): fixed per dispatch and per km."""

BIG_FIXED, BIG_PER_KM = 300.0, 2.0
"""Cost of a ``big`` spot vehicle (1000 desi): fixed per dispatch and per km."""

SMALL_ONLY = {"spot": [{"type": "small", "cost": {"fixed": SMALL_FIXED, "per_km": SMALL_PER_KM}}]}
"""A fleet of unlimited ``small`` spot vehicles only: :data:`ONLY` is its one option."""

BIG_AND_SMALL = {
    "spot": [
        {"type": "big", "cost": {"fixed": BIG_FIXED, "per_km": BIG_PER_KM}},
        {"type": "small", "cost": {"fixed": SMALL_FIXED, "per_km": SMALL_PER_KM}},
    ]
}
"""Unlimited ``big`` and ``small`` spot vehicles: options :data:`BIG` and :data:`SMALL`."""

BIG, SMALL = 0, 1
"""Options of :data:`BIG_AND_SMALL`, in fleet order; also the indices of the vehicle types."""

ONLY = 0
"""The one option of :data:`SMALL_ONLY`."""

SMALL_TO_B = SMALL_FIXED + SMALL_PER_KM * DEGREE_KM
"""Cost of one ``small`` vehicle from ``A`` to ``B`` (about 316.8)."""

BIG_TO_SMALL_SAVING = (BIG_FIXED - SMALL_FIXED) + (BIG_PER_KM - SMALL_PER_KM) * DEGREE_KM
"""What carrying ``A`` to ``B`` with ``small`` instead of ``big`` saves: 150 fixed, 0.5 per km."""

VIOLATION_WEIGHT = 1e6
"""Weight of every violation in :data:`GOAL`, far above any cost of the scenarios."""

GOAL = obj.total_cost() + VIOLATION_WEIGHT * obj.violations()
"""The objective of most tests: cost, with violations all but forbidden."""

COST_TOLERANCE = 1e-9
"""Relative tolerance of a change of the objective worked out by hand."""

PLANAR_TOLERANCE = 1e-3
"""Relative tolerance where a planar distance stands for the haversine one, near the equator."""

AMOUNT_DECIMALS = 6
"""Decimals loads are rounded to by :func:`described`, so a shifted load compares exactly."""

PINNED_TOLERANCE = 1e-12
"""Relative tolerance of the regression pin; scores stay the same to the bit."""

Stops = list[tuple[int, float]]
"""The stops of one route: (node, desi unloaded there)."""


def scenario(
    document: dict[str, Any],
    compile_document: Any,
    demand: dict[str, float],
    fleet: dict[str, Any] = SMALL_ONLY,
    **rules: Any,
) -> tuple[LineHaulInstance, Subproblem]:
    """The fixture document with ``fleet``, ``rules`` and demand from ``A`` on the first day.

    ``demand`` maps a destination to its desi. All of it leaves ``A`` on 2026-01-05, so the
    instance has exactly one subproblem, which is returned with it. ``fleet`` is copied, so
    the module constants stay as they are.
    """
    document["fleet"] = copy.deepcopy(fleet)
    document["demand"] = [
        {"origin": "A", "destination": to, "date": "2026-01-05", "quantity": {"desi": amount}}
        for to, amount in demand.items()
    ]
    document["rules"] = rules
    instance = compile_document(document)
    [problem] = subproblems(instance)
    return instance, problem


def routes(*plan: tuple[int, Stops]) -> Routes:
    """Routes of split demand; each argument is one route's (option, stops)."""
    flat = [stop for _, stops in plan for stop in stops]
    return Routes(
        option=np.array([option for option, _ in plan], dtype=np.int64),
        stop_start=np.cumsum([0, *(len(stops) for _, stops in plan)]).astype(np.int64),
        stop_node=np.array([node for node, _ in flat], dtype=np.int64),
        stop_amount=np.array([amount for _, amount in flat], dtype=np.float64).reshape(-1, 1),
        item_stop=np.zeros(0, dtype=np.int64),
        item_slot=np.zeros(0, dtype=np.int64),
    )


def described(result: Routes) -> list[tuple[int, Stops]]:
    """``result`` as :func:`routes` takes it: (option, stops) per route, loads rounded."""
    return [
        (
            int(result.option[r]),
            [
                (int(result.stop_node[s]), round(float(result.stop_amount[s, 0]), AMOUNT_DECIMALS))
                for s in range(result.stop_start[r], result.stop_start[r + 1])
            ],
        )
        for r in range(result.option.size)
    ]


def improve_routes(
    instance: LineHaulInstance,
    problem: Subproblem,
    start: Routes,
    goal: obj.Sum,
    remaining: list[float],
    room: Room | None = None,
) -> tuple[Routes, float]:
    """:func:`improve` on ``start`` for ``goal``; ``remaining`` is the desi left per node."""
    return improve(
        problem,
        start,
        linear_coefficients(instance, goal.linear_form()),
        Shared.of(instance),
        problem.available,
        np.array(remaining, dtype=np.float64).reshape(-1, 1),
        room,
    )


def small_slots_at_the_origin(slots: int) -> Room:
    """Room of a one-node subproblem: unlimited, except ``slots`` ``small`` vehicles at ``A``."""
    return Room(
        handling=np.full((1, 2, 1), np.inf),
        slots=np.array([[[[UNBOUNDED, slots]], [[UNBOUNDED, UNBOUNDED]]]], dtype=np.int64),
        limit=np.full(2, np.inf),
        day=0,
    )


def test_scenario_has_one_subproblem_from_a_with_the_given_demand(
    document: dict[str, Any], compile_document: Any
) -> None:
    """The input the move tests rely on: nodes ``B`` (center 1), ``C`` (center 2), ``small``."""
    _, problem = scenario(document, compile_document, {"B": 100.0, "C": 150.0})

    np.testing.assert_array_equal(problem.nodes, [1, 2])
    np.testing.assert_array_equal(problem.demand, [[100.0], [150.0]])
    np.testing.assert_array_equal(problem.vehicle_type, [SMALL])


def test_scenario_with_both_types_offers_big_then_small(
    document: dict[str, Any], compile_document: Any
) -> None:
    """The options :data:`BIG` and :data:`SMALL` stand for, at the one node ``B``."""
    _, problem = scenario(document, compile_document, {"B": 100.0}, BIG_AND_SMALL)

    np.testing.assert_array_equal(problem.vehicle_type, [BIG, SMALL])


def test_small_slots_helper_limits_only_small_vehicles_at_the_origin() -> None:
    room = small_slots_at_the_origin(1)

    np.testing.assert_array_equal(room.origin_slots, [[UNBOUNDED, 1]])
    np.testing.assert_array_equal(room.node_slots, [[[UNBOUNDED, UNBOUNDED]]])
    np.testing.assert_array_equal(room.origin_handling, [np.inf])


def test_routes_helper_lays_out_the_stops_route_by_route() -> None:
    result = routes((BIG, [(NODE_B, 300.0)]), (SMALL, [(NODE_B, 80.0), (NODE_C, 300.0)]))

    np.testing.assert_array_equal(result.option, [BIG, SMALL])
    np.testing.assert_array_equal(result.stop_start, [0, 1, 3])
    np.testing.assert_array_equal(result.stop_node, [NODE_B, NODE_B, NODE_C])
    np.testing.assert_array_equal(result.stop_amount, [[300.0], [80.0], [300.0]])
    np.testing.assert_array_equal(result.item_stop, np.zeros(0))


def test_described_helper_reads_option_and_rounded_stops_per_route() -> None:
    result = Routes(
        option=np.array([SMALL, BIG]),
        stop_start=np.array([0, 2, 3]),
        stop_node=np.array([NODE_B, NODE_C, NODE_B]),
        stop_amount=np.array([[100.0000001], [50.0], [25.0]]),
        item_stop=np.zeros(0, dtype=np.int64),
        item_slot=np.zeros(0, dtype=np.int64),
    )

    description = described(result)

    assert description == [(SMALL, [(NODE_B, 100.0), (NODE_C, 50.0)]), (BIG, [(NODE_B, 25.0)])]


def test_merge_serves_two_routes_with_one_vehicle(
    document: dict[str, Any], compile_document: Any
) -> None:
    """``A``-``B``, ``A``-``C`` become ``A``-``B``-``C``: one fixed cost and one degree less."""
    instance, problem = scenario(document, compile_document, {"B": 100.0, "C": 100.0})
    start = routes((ONLY, [(NODE_B, 100.0)]), (ONLY, [(NODE_C, 100.0)]))

    result, change = improve_routes(instance, problem, start, GOAL, [0.0, 0.0])

    assert described(result) == [(ONLY, [(NODE_B, 100.0), (NODE_C, 100.0)])]
    assert change == pytest.approx(-SMALL_TO_B, rel=COST_TOLERANCE)


def test_merge_may_carry_two_routes_with_a_larger_vehicle(
    document: dict[str, Any], compile_document: Any
) -> None:
    """Two ``small`` routes of 300 desi to ``B`` become one ``big`` route of 600.

    Only merge can do this: relocating a stop keeps the receiving ``small`` vehicle, which
    holds 400. Two ``small`` dispatches cost 300 fixed and 3 per km, one ``big`` 300 fixed
    and 2 per km, so the saving is one degree at 1 per km.
    """
    instance, problem = scenario(document, compile_document, {"B": 600.0}, BIG_AND_SMALL)
    start = routes((SMALL, [(NODE_B, 300.0)]), (SMALL, [(NODE_B, 300.0)]))

    result, change = improve_routes(instance, problem, start, GOAL, [0.0])

    assert described(result) == [(BIG, [(NODE_B, 600.0)])]
    assert change == pytest.approx(-(2 * SMALL_PER_KM - BIG_PER_KM) * DEGREE_KM, rel=COST_TOLERANCE)


def test_merge_up_to_the_capacity_fills_one_vehicle(
    document: dict[str, Any], compile_document: Any
) -> None:
    """200 + 200 desi is exactly the 400 a ``small`` vehicle holds."""
    instance, problem = scenario(document, compile_document, {"B": 200.0, "C": 200.0})
    start = routes((ONLY, [(NODE_B, 200.0)]), (ONLY, [(NODE_C, 200.0)]))

    result, change = improve_routes(instance, problem, start, GOAL, [0.0, 0.0])

    assert described(result) == [(ONLY, [(NODE_B, 200.0), (NODE_C, 200.0)])]
    assert change == pytest.approx(-SMALL_TO_B, rel=COST_TOLERANCE)


def test_merge_over_the_capacity_keeps_both_routes(
    document: dict[str, Any], compile_document: Any
) -> None:
    """201 + 200 desi is one more than a ``small`` vehicle holds, and no other move helps."""
    instance, problem = scenario(document, compile_document, {"B": 201.0, "C": 200.0})
    start = routes((ONLY, [(NODE_B, 201.0)]), (ONLY, [(NODE_C, 200.0)]))

    result, change = improve_routes(instance, problem, start, GOAL, [0.0, 0.0])

    assert described(result) == [(ONLY, [(NODE_B, 201.0)]), (ONLY, [(NODE_C, 200.0)])]
    assert change == 0.0


def test_relocate_moves_a_stop_onto_the_route_already_serving_its_node(
    document: dict[str, Any], compile_document: Any
) -> None:
    """``C`` moves to one degree north of ``A``, so ``A``-``B``-``C`` is a detour.

    Moving the 80 desi for ``B`` onto the ``A``-``B`` route (380 of 400 desi) leaves
    ``A``-``C`` direct. Both routes stay, so the saving is the detour ``A``-``B``-``C``
    against ``A``-``C``: one degree plus the diagonal ``B``-``C`` minus one degree, the
    diagonal being about the square root of two degrees this close to the equator.
    """
    document["centers"][2].update(lat=1.0, lon=0.0)
    instance, problem = scenario(
        document, compile_document, {"B": 380.0, "C": 300.0}, stopover={"forward_only": False}
    )
    start = routes((ONLY, [(NODE_B, 300.0)]), (ONLY, [(NODE_B, 80.0), (NODE_C, 300.0)]))

    result, change = improve_routes(instance, problem, start, GOAL, [0.0, 0.0])

    assert sorted(described(result)) == [(ONLY, [(NODE_B, 380.0)]), (ONLY, [(NODE_C, 300.0)])]
    assert change == pytest.approx(-SMALL_PER_KM * math.sqrt(2.0) * DEGREE_KM, rel=PLANAR_TOLERANCE)


def test_rebalance_shifts_load_to_lift_a_route_to_its_minimum_fill(
    document: dict[str, Any], compile_document: Any
) -> None:
    """Spot vehicles must be half full (200 of 400 desi); the 100 desi route lacks 100.

    The 400 desi route gives 100 and keeps 300, above its own minimum, so the 100 desi of
    minimum fill violation go away.
    """
    instance, problem = scenario(document, compile_document, {"B": 500.0}, min_fill={"spot": 0.5})
    start = routes((ONLY, [(NODE_B, 400.0)]), (ONLY, [(NODE_B, 100.0)]))

    result, change = improve_routes(instance, problem, start, GOAL, [0.0])

    assert sorted(described(result)) == [(ONLY, [(NODE_B, 200.0)]), (ONLY, [(NODE_B, 300.0)])]
    assert change == pytest.approx(-VIOLATION_WEIGHT * 100.0, rel=COST_TOLERANCE)


@pytest.mark.parametrize(
    ("coverage_weight", "expected_change"),
    [
        pytest.param(0.01, -SMALL_TO_B + 0.01 * 100.0, id="far-cheaper"),
        pytest.param(3.1, -SMALL_TO_B + 3.1 * 100.0, id="just-cheaper"),
    ],
)
def test_drop_removes_a_route_when_leaving_its_demand_is_cheaper(
    document: dict[str, Any], compile_document: Any, coverage_weight: float, expected_change: float
) -> None:
    """The route costs about 316.8; leaving its 100 desi costs 100 times the weight (1 or 310)."""
    instance, problem = scenario(document, compile_document, {"B": 100.0})
    goal = obj.total_cost() + coverage_weight * obj.violation("coverage")
    start = routes((ONLY, [(NODE_B, 100.0)]))

    result, change = improve_routes(instance, problem, start, goal, [0.0])

    assert described(result) == []
    assert change == pytest.approx(expected_change, rel=COST_TOLERANCE)


def test_drop_keeps_a_route_when_leaving_its_demand_is_dearer(
    document: dict[str, Any], compile_document: Any
) -> None:
    """Leaving the 100 desi at weight 3.2 costs 320, more than the route's 316.8."""
    instance, problem = scenario(document, compile_document, {"B": 100.0})
    goal = obj.total_cost() + 3.2 * obj.violation("coverage")
    start = routes((ONLY, [(NODE_B, 100.0)]))

    result, change = improve_routes(instance, problem, start, goal, [0.0])

    assert described(result) == [(ONLY, [(NODE_B, 100.0)])]
    assert change == 0.0


def test_serve_puts_what_was_left_onto_a_route_with_room(
    document: dict[str, Any], compile_document: Any
) -> None:
    """50 of 150 desi are left uncarried; the 100 desi route has room for them."""
    instance, problem = scenario(document, compile_document, {"B": 150.0})
    start = routes((ONLY, [(NODE_B, 100.0)]))

    result, change = improve_routes(instance, problem, start, GOAL, [50.0])

    assert described(result) == [(ONLY, [(NODE_B, 150.0)])]
    assert change == pytest.approx(-VIOLATION_WEIGHT * 50.0, rel=COST_TOLERANCE)


def test_change_vehicle_carries_a_route_with_a_cheaper_option(
    document: dict[str, Any], compile_document: Any
) -> None:
    """100 desi to ``B`` fit both types; ``small`` is the cheaper one."""
    instance, problem = scenario(document, compile_document, {"B": 100.0}, BIG_AND_SMALL)
    start = routes((BIG, [(NODE_B, 100.0)]))

    result, change = improve_routes(instance, problem, start, GOAL, [0.0])

    assert described(result) == [(SMALL, [(NODE_B, 100.0)])]
    assert change == pytest.approx(-BIG_TO_SMALL_SAVING, rel=COST_TOLERANCE)


@pytest.mark.parametrize(
    ("slots", "expected_option", "expected_change"),
    [
        pytest.param(0, BIG, 0.0, id="no-small-slot"),
        pytest.param(1, SMALL, -BIG_TO_SMALL_SAVING, id="one-small-slot"),
    ],
)
def test_change_vehicle_is_taken_only_with_a_vehicle_slot_left_on_the_day(
    document: dict[str, Any],
    compile_document: Any,
    slots: int,
    expected_option: int,
    expected_change: float,
) -> None:
    """The cheaper ``small`` vehicle needs one of the ``small`` slots left at ``A``."""
    instance, problem = scenario(document, compile_document, {"B": 100.0}, BIG_AND_SMALL)
    start = routes((BIG, [(NODE_B, 100.0)]))
    room = small_slots_at_the_origin(slots)

    result, change = improve_routes(instance, problem, start, GOAL, [0.0], room)

    assert described(result) == [(expected_option, [(NODE_B, 100.0)])]
    assert change == pytest.approx(expected_change, rel=COST_TOLERANCE)


def test_routes_that_no_move_improves_are_returned_unchanged(
    document: dict[str, Any], compile_document: Any
) -> None:
    """One full ``small`` vehicle (400 of 400 desi) on the only route there is."""
    instance, problem = scenario(document, compile_document, {"B": 400.0})
    start = routes((ONLY, [(NODE_B, 400.0)]))

    result, change = improve_routes(instance, problem, start, GOAL, [0.0])

    assert described(result) == [(ONLY, [(NODE_B, 400.0)])]
    assert change == 0.0


def test_no_routes_and_nothing_left_stay_empty(
    document: dict[str, Any], compile_document: Any
) -> None:
    """The empty input: no move has a route to work on."""
    instance, problem = scenario(document, compile_document, {"B": 100.0})

    result, change = improve_routes(instance, problem, routes(), GOAL, [0.0])

    assert described(result) == []
    assert change == 0.0


def test_colony_plan_with_local_search_matches_the_recorded_regression_plan() -> None:
    """Regression pin: the cheap checks and remembered rejections change nothing.

    The score and the number of dispatches were recorded with the local search before it
    remembered rejected moves; that search tried every move afresh, so the recording is the
    reference the faster search must reproduce. The synthetic problem (20 centers, 2 days,
    150 demand rows a day, generator seed 7, solver seed 0) has many subproblems with
    several routes each and solves in about a second.
    """
    instance = parse_spec(
        synthetic.generate(centers=20, days=2, demand_per_day=150, seed=7)
    ).compile()

    result = ACOSolver().solve(instance, GOAL, seed=0)

    assert result.stats["score"] == pytest.approx(313495540.81793475, rel=PINNED_TOLERANCE)
    assert len(result.plan) == 145

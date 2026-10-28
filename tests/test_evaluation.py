"""Tests of evaluating a plan: its costs, its per dispatch arrays and its violations.

Every test evaluates a plan for the ``document`` fixture of ``tests/conftest.py``, whose
docstring gives every value it holds. Its names become positions in the order the document
lists them: centers ``A``, ``B``, ``C`` are 0, 1, 2 (on the equator, one degree of longitude
apart, so ``A`` to ``C`` is two degrees); the spot offers ``big`` and ``small`` are 0 and 1,
and so are the vehicle types; the only owned group (one ``big`` vehicle on the line ``A`` to
``B``) is 0; the calendar days are 0 and 1; and the demands are 0 (``A`` to ``B``, 900 desi,
day 0), 1 (``A`` to ``C``, 250 desi, day 0) and 2 (``B`` to ``A``, 120 desi, day 1).

Each plan below is :data:`FEASIBLE` with the fewest changes that break one rule, so the
expected violations are worked out by hand from the rule's entry in
:data:`pirilib.constraints.CATALOG`. :func:`pirilib.check_plan` is an independent checker
written as plain Python loops; the agreement tests at the end compare the evaluator with it
on every plan of this module.
"""

import math
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pytest

from pirilib import Dispatch, Load, Plan, PlanError, check_plan, evaluate
from pirilib.constraints import CATALOG, Constraint, MaxDispatchesPerDay, MaxRouteKm, constraint
from pirilib.spec import LineHaulInstance

ONE_DEGREE_KM = 6371.0 * math.pi / 180.0
"""The great circle length of one degree on a sphere of the mean Earth radius (6371 km)."""

A, B, C = 0, 1, 2
"""The fixture's centers by position."""

BIG, SMALL = 0, 1
"""The fixture's spot offers, and its vehicle types, by position."""

OWNED_GROUP = 0
"""The fixture's only owned vehicle group."""

FIRST_DAY, SECOND_DAY = 0, 1
"""The fixture's calendar days by position."""

A_TO_B, A_TO_C, B_TO_A = 0, 1, 2
"""The fixture's demands by position."""

DIMENSIONS = 1
"""The fixture measures quantities in one dimension (desi)."""

RELATIVE_TOLERANCE = 1e-12
"""Computed costs and distances agree with the hand worked values up to float rounding."""

AGREEMENT_TOLERANCE = 1e-9
"""Relative and absolute tolerance of the agreement with :func:`pirilib.check_plan`.

It is the relative tolerance the evaluator documents for its capacity, fill, stretch and
coverage comparisons; the two implementations add the same amounts in different orders.
"""

ROUNDING_NOISE = 1e-10
"""An error far below the evaluator's relative tolerance of 1e-9, as float sums leave."""

OWNED_A_TO_B = Dispatch(
    day=FIRST_DAY,
    kind="owned",
    source=OWNED_GROUP,
    origin=A,
    stops=(B,),
    loads=(Load(A_TO_B, (900.0,)),),
)
"""The owned ``big`` vehicle on its line, carrying all 900 desi of demand 0."""

SMALL_A_TO_C = Dispatch(
    day=FIRST_DAY, kind="spot", source=SMALL, origin=A, stops=(C,), loads=(Load(A_TO_C, (250.0,)),)
)
"""A spot ``small`` vehicle carrying all 250 desi of demand 1 to ``C``, which bans ``big``."""

SMALL_B_TO_A = Dispatch(
    day=SECOND_DAY,
    kind="spot",
    source=SMALL,
    origin=B,
    stops=(A,),
    loads=(Load(B_TO_A, (120.0,)),),
)
"""A spot ``small`` vehicle carrying all 120 desi of demand 2 on its day."""

SMALL_EMPTY_A_TO_B = Dispatch(day=SECOND_DAY, kind="spot", source=SMALL, origin=A, stops=(B,))
"""An empty spot ``small`` dispatch that breaks no rule unless there are too many of them."""

SMALL_B_TO_C_TO_A = Dispatch(
    day=SECOND_DAY,
    kind="spot",
    source=SMALL,
    origin=B,
    stops=(C, A),
    loads=(Load(B_TO_A, (120.0,)),),
)
"""Demand 2 by way of ``C``: two stops, 3 degrees of route against 1 degree direct."""


@dataclass(frozen=True)
class Case:
    """A plan for the fixture's problem, and the changes to the fixture it is evaluated under.

    Attributes:
        dispatches: The plan
        rules: The document's ``rules`` section; empty for the schema defaults
        mandatory: Whether the owned group's fixed cost is paid every day (the default)
    """

    dispatches: tuple[Dispatch, ...]
    rules: dict[str, Any] = field(default_factory=dict)
    mandatory: bool = True

    @property
    def plan(self) -> Plan:
        """The dispatches in array form."""
        return Plan.from_dispatches(self.dispatches, dimensions=DIMENSIONS)


def compile_case(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance], case: Case
) -> LineHaulInstance:
    """The fixture's problem with the rules and the owned group setting of ``case``."""
    document["rules"] = case.rules
    document["fleet"]["owned"][OWNED_GROUP]["mandatory"] = case.mandatory
    return compile_document(document)


FEASIBLE = Case((OWNED_A_TO_B, SMALL_A_TO_C, SMALL_B_TO_A))
"""Every demand carried whole on its day by a vehicle allowed on its route."""

EMPTY = Case(())
"""No dispatches at all."""

PER_DISPATCH = Case(FEASIBLE.dispatches, mandatory=False)
"""The feasible plan, with an owned group whose fixed cost is paid only when it runs."""

MALFORMED_ROUTES = Case(
    (
        *FEASIBLE.dispatches,
        Dispatch(day=SECOND_DAY, kind="spot", source=SMALL, origin=A, stops=()),
        Dispatch(day=SECOND_DAY, kind="spot", source=SMALL, origin=A, stops=(A, B, B)),
    )
)
"""Two extra routes: one without stops; one that stops at its origin and visits ``B`` twice."""

LOAD_OFF_ROUTE = Case(
    (
        OWNED_A_TO_B,
        Dispatch(
            day=FIRST_DAY,
            kind="spot",
            source=SMALL,
            origin=A,
            stops=(B,),
            loads=(Load(A_TO_C, (250.0,)),),
        ),
        SMALL_B_TO_A,
    )
)
"""Demand 1 (bound for ``C``) loaded on a vehicle that only stops at ``B``."""

EARLY_LOAD = Case(
    (
        OWNED_A_TO_B,
        SMALL_A_TO_C,
        Dispatch(
            day=FIRST_DAY,
            kind="spot",
            source=SMALL,
            origin=B,
            stops=(A,),
            loads=(Load(B_TO_A, (120.0,)),),
        ),
    )
)
"""Demand 2 (due on the second day) carried on the first day."""

BANNED = Case(
    (
        OWNED_A_TO_B,
        Dispatch(
            day=FIRST_DAY,
            kind="spot",
            source=BIG,
            origin=A,
            stops=(C,),
            loads=(Load(A_TO_C, (250.0,)),),
        ),
        Dispatch(
            day=SECOND_DAY,
            kind="spot",
            source=BIG,
            origin=C,
            stops=(B, A),
            loads=(Load(B_TO_A, (120.0,)),),
        ),
    )
)
"""Spot ``big`` vehicles that stop at ``C`` and leave from ``C``, where ``big`` is banned.

The second one also carries demand 2, which leaves from ``B``, from ``C``.
"""

OFF_LINE = Case(
    (
        Dispatch(day=FIRST_DAY, kind="owned", source=OWNED_GROUP, origin=B, stops=(C,)),
        SMALL_A_TO_C,
        SMALL_B_TO_A,
    )
)
"""The owned vehicle runs ``B`` to ``C`` instead of its line ``A`` to ``B``, empty.

So it also enters ``C``, which bans its type, and demand 0 is left behind.
"""

EXTRA_OWNED = Case(
    (
        *FEASIBLE.dispatches,
        Dispatch(day=FIRST_DAY, kind="owned", source=OWNED_GROUP, origin=A, stops=(B,)),
    )
)
"""A second dispatch of the one owned vehicle on the first day."""

EXTRA_SPOT = Case((*FEASIBLE.dispatches, *(SMALL_EMPTY_A_TO_B,) * 3))
"""Four spot ``small`` dispatches on the second day, of the 3 a day the offer has."""

OVERLOADED = Case(
    (
        Dispatch(
            day=FIRST_DAY,
            kind="spot",
            source=SMALL,
            origin=A,
            stops=(B,),
            loads=(Load(A_TO_B, (900.0,)),),
        ),
        SMALL_A_TO_C,
        SMALL_B_TO_A,
    )
)
"""The 900 desi of demand 0 in a ``small`` vehicle, which holds 400."""

MIN_FILL = Case(FEASIBLE.dispatches, rules={"min_fill": {"spot": 0.5, "owned": 0.95}})
"""The feasible plan, with owned vehicles to be 95% full and spot vehicles 50% full."""

NO_STOPOVERS = Case(
    (
        OWNED_A_TO_B,
        Dispatch(
            day=FIRST_DAY,
            kind="spot",
            source=SMALL,
            origin=A,
            stops=(B, C),
            loads=(Load(A_TO_C, (250.0,)),),
        ),
        SMALL_B_TO_A,
    ),
    rules={"stopover": {"enabled": False}},
)
"""Demand 1 by way of ``B`` while the rules disable stopovers."""

OWNED_DETOUR = Case(
    (
        Dispatch(
            day=FIRST_DAY,
            kind="owned",
            source=OWNED_GROUP,
            origin=A,
            stops=(C, B),
            loads=(Load(A_TO_B, (900.0,)),),
        ),
        SMALL_A_TO_C,
        SMALL_B_TO_A,
    )
)
"""The owned vehicle to ``B`` by way of ``C``, though its group may not stop over.

``B`` is nearer to ``A`` than ``C``, so the route also turns back, and ``C`` bans ``big``.
"""

STOP_LIMIT = Case(
    (OWNED_A_TO_B, SMALL_A_TO_C, SMALL_B_TO_C_TO_A),
    rules={"stopover": {"max_stops": 1, "forward_only": False}},
)
"""Two stops where the rules allow one (turning back is allowed, so only the count counts)."""

STRETCH_LIMIT = Case(
    (OWNED_A_TO_B, SMALL_A_TO_C, SMALL_B_TO_C_TO_A),
    rules={"stopover": {"max_stretch": 1.5, "forward_only": False}},
)
"""A route of 3 degrees where the rules allow 1.5 times the direct 1 degree."""

BOTH_LIMITS = Case(
    (OWNED_A_TO_B, SMALL_A_TO_C, SMALL_B_TO_C_TO_A),
    rules={"stopover": {"max_stops": 1, "max_stretch": 1.5, "forward_only": False}},
)
"""The same route under both the stop limit and the stretch limit."""

SPLIT = Case(
    (
        Dispatch(
            day=FIRST_DAY,
            kind="owned",
            source=OWNED_GROUP,
            origin=A,
            stops=(B,),
            loads=(Load(A_TO_B, (500.0,)),),
        ),
        Dispatch(
            day=FIRST_DAY,
            kind="spot",
            source=BIG,
            origin=A,
            stops=(B,),
            loads=(Load(A_TO_B, (400.0,)),),
        ),
        SMALL_A_TO_C,
        SMALL_B_TO_A,
    ),
    rules={"split_demand": False},
)
"""Demand 0 in two loads of 500 and 400 desi while the rules forbid splitting a demand."""

UNCOVERED = Case((OWNED_A_TO_B, SMALL_A_TO_C))
"""Demand 2 (120 desi) is never carried."""

OVERDELIVERED = Case(
    (
        OWNED_A_TO_B,
        Dispatch(
            day=FIRST_DAY,
            kind="spot",
            source=SMALL,
            origin=A,
            stops=(C,),
            loads=(Load(A_TO_C, (300.0,)),),
        ),
        SMALL_B_TO_A,
    )
)
"""300 desi carried for demand 1, which is 250 desi."""

ROUNDED = Case(
    (
        Dispatch(
            day=FIRST_DAY,
            kind="owned",
            source=OWNED_GROUP,
            origin=A,
            stops=(B,),
            loads=(Load(A_TO_B, (300.0 + ROUNDING_NOISE,)),) * 3,
        ),
        SMALL_A_TO_C,
        SMALL_B_TO_A,
    )
)
"""Demand 0 in three thirds of 300 desi, each a little too large by rounding noise."""

CASES = {
    "feasible": FEASIBLE,
    "empty": EMPTY,
    "per_dispatch": PER_DISPATCH,
    "malformed_routes": MALFORMED_ROUTES,
    "load_off_route": LOAD_OFF_ROUTE,
    "early_load": EARLY_LOAD,
    "banned": BANNED,
    "off_line": OFF_LINE,
    "extra_owned": EXTRA_OWNED,
    "extra_spot": EXTRA_SPOT,
    "overloaded": OVERLOADED,
    "min_fill": MIN_FILL,
    "no_stopovers": NO_STOPOVERS,
    "owned_detour": OWNED_DETOUR,
    "stop_limit": STOP_LIMIT,
    "stretch_limit": STRETCH_LIMIT,
    "both_limits": BOTH_LIMITS,
    "split": SPLIT,
    "uncovered": UNCOVERED,
    "overdelivered": OVERDELIVERED,
    "rounded": ROUNDED,
}
"""Every plan of this module, for the agreement tests."""

FEASIBLE_EXACT_ARRAYS = {
    "vehicle_type": [BIG, SMALL, SMALL],
    "load": [[900.0], [250.0], [120.0]],
    "capacity": [[1000.0], [400.0], [400.0]],
}
"""The feasible plan's per dispatch arrays that are copied, not computed: the vehicle type of
the owned group and of the spot offers, the desi loaded and the type's capacity."""

FEASIBLE_COMPUTED_ARRAYS = {
    "route_km": [ONE_DEGREE_KM, 2 * ONE_DEGREE_KM, ONE_DEGREE_KM],
    "direct_km": [ONE_DEGREE_KM, 2 * ONE_DEGREE_KM, ONE_DEGREE_KM],
    "fill": [900.0 / 1000.0, 250.0 / 400.0, 120.0 / 400.0],
    "dispatch_cost": [
        1.0 * ONE_DEGREE_KM,
        150.0 + 1.5 * 2 * ONE_DEGREE_KM,
        150.0 + 1.5 * ONE_DEGREE_KM,
    ],
}
"""The feasible plan's computed per dispatch arrays. Every route has one stop, so its length is
the direct distance. Fill is load over capacity. The owned dispatch pays only its 1 per km (its
group is mandatory, so the fixed 100 is paid per day instead); a spot ``small`` dispatch pays
150 fixed and 1.5 per km. No speed is given, so no dispatch has an hourly cost."""

MANDATORY_FIXED_COST = 1 * 100.0 * 2
"""One mandatory owned vehicle at a fixed 100 a day over the two calendar days."""

FEASIBLE_TOTAL_COST = 500.0 + 5.5 * ONE_DEGREE_KM
"""The feasible plan's dispatch costs (300 fixed and 5.5 degrees of per km cost) plus the
mandatory fixed cost of 200; 1111.57 when rounded to cents."""

FEASIBLE_REPR = "Evaluation(total_cost=1111.57, dispatches=3, total_km=444.8, feasible)"
"""The feasible plan's summary: :data:`FEASIBLE_TOTAL_COST` to cents and its 4 degrees of
route to a tenth of a km."""

EMPTY_REPR = (
    "Evaluation(total_cost=200.00, dispatches=0, total_km=0.0, violated={'coverage': 1270.0})"
)
"""The empty plan's summary: only the mandatory fixed cost, and all 1270 desi of demand
(900 + 250 + 120) not carried."""

ARRAY_FIELDS = (
    "vehicle_type",
    "route_km",
    "direct_km",
    "load",
    "capacity",
    "fill",
    "dispatch_cost",
    "start_minute",
    "end_minute",
    "load_finish",
    "load_late_hours",
    "served",
    "unserved",
)
"""Every array field of :class:`pirilib.Evaluation`."""


def test_feasible_plan_violates_nothing(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    instance = compile_case(document, compile_document, FEASIBLE)

    evaluation = evaluate(instance, FEASIBLE.plan)

    assert evaluation.violated() == {}
    assert evaluation.feasible is True


def test_violations_name_every_built_in_constraint_in_catalog_order(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    instance = compile_case(document, compile_document, FEASIBLE)

    evaluation = evaluate(instance, FEASIBLE.plan)

    assert tuple(evaluation.violations) == tuple(CATALOG)


@pytest.mark.parametrize("name", FEASIBLE_EXACT_ARRAYS)
def test_feasible_plan_copies_types_loads_and_capacities_per_dispatch(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance], name: str
) -> None:
    instance = compile_case(document, compile_document, FEASIBLE)

    evaluation = evaluate(instance, FEASIBLE.plan)

    np.testing.assert_array_equal(getattr(evaluation, name), FEASIBLE_EXACT_ARRAYS[name])


@pytest.mark.parametrize("name", FEASIBLE_COMPUTED_ARRAYS)
def test_feasible_plan_computes_distances_fill_and_cost_per_dispatch(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance], name: str
) -> None:
    instance = compile_case(document, compile_document, FEASIBLE)

    evaluation = evaluate(instance, FEASIBLE.plan)

    np.testing.assert_allclose(
        getattr(evaluation, name),
        FEASIBLE_COMPUTED_ARRAYS[name],
        rtol=RELATIVE_TOLERANCE,
        atol=0.0,
    )


def test_mandatory_owned_vehicles_pay_their_fixed_cost_every_day(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    instance = compile_case(document, compile_document, FEASIBLE)

    evaluation = evaluate(instance, FEASIBLE.plan)

    assert evaluation.owned_fixed_cost == MANDATORY_FIXED_COST


def test_total_cost_adds_dispatch_costs_and_the_mandatory_fixed_cost(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    instance = compile_case(document, compile_document, FEASIBLE)

    evaluation = evaluate(instance, FEASIBLE.plan)

    assert evaluation.total_cost == pytest.approx(FEASIBLE_TOTAL_COST, rel=RELATIVE_TOLERANCE)


def test_total_km_adds_every_route(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    """The routes are 1, 2 and 1 degrees long."""
    instance = compile_case(document, compile_document, FEASIBLE)

    evaluation = evaluate(instance, FEASIBLE.plan)

    assert evaluation.total_km == pytest.approx(4 * ONE_DEGREE_KM, rel=RELATIVE_TOLERANCE)


def test_feasible_plan_leaves_no_demand_unserved(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    instance = compile_case(document, compile_document, FEASIBLE)

    evaluation = evaluate(instance, FEASIBLE.plan)

    np.testing.assert_array_equal(evaluation.unserved, [[0.0], [0.0], [0.0]])


def test_repr_of_a_feasible_plan_sums_up_cost_dispatches_and_distance(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    instance = compile_case(document, compile_document, FEASIBLE)

    evaluation = evaluate(instance, FEASIBLE.plan)

    assert repr(evaluation) == FEASIBLE_REPR


def test_repr_of_an_infeasible_plan_names_its_violations(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    instance = compile_case(document, compile_document, EMPTY)

    evaluation = evaluate(instance, EMPTY.plan)

    assert repr(evaluation) == EMPTY_REPR


def test_empty_plan_still_pays_the_mandatory_fixed_cost(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    instance = compile_case(document, compile_document, EMPTY)

    evaluation = evaluate(instance, EMPTY.plan)

    assert evaluation.total_cost == MANDATORY_FIXED_COST


def test_empty_plan_violates_only_coverage_by_all_the_demand(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    """All 1270 desi of demand (900 + 250 + 120) is missing."""
    instance = compile_case(document, compile_document, EMPTY)

    evaluation = evaluate(instance, EMPTY.plan)

    assert evaluation.violated() == {"coverage": 1270.0}


def test_empty_plan_leaves_every_demand_unserved(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    instance = compile_case(document, compile_document, EMPTY)

    evaluation = evaluate(instance, EMPTY.plan)

    np.testing.assert_array_equal(evaluation.unserved, [[900.0], [250.0], [120.0]])


def test_non_mandatory_owned_vehicles_pay_no_fixed_cost_per_day(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    instance = compile_case(document, compile_document, PER_DISPATCH)

    evaluation = evaluate(instance, PER_DISPATCH.plan)

    assert evaluation.owned_fixed_cost == 0.0


def test_non_mandatory_owned_vehicles_pay_their_fixed_cost_per_dispatch(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    """The owned dispatch pays the group's fixed 100 and 1 per km over its 1 degree."""
    instance = compile_case(document, compile_document, PER_DISPATCH)

    evaluation = evaluate(instance, PER_DISPATCH.plan)

    assert evaluation.dispatch_cost[0] == pytest.approx(
        100.0 + 1.0 * ONE_DEGREE_KM, rel=RELATIVE_TOLERANCE
    )


def test_malformed_routes_count_each_missing_stop_origin_stop_and_repeated_stop(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    """One route has no stops (1); the other stops at its origin (1) and at ``B`` twice (1).

    Its second ``B`` is no farther from ``A`` than the first, so it also turns back once.
    """
    instance = compile_case(document, compile_document, MALFORMED_ROUTES)

    evaluation = evaluate(instance, MALFORMED_ROUTES.plan)

    assert evaluation.violated() == {"route": 3, "forward_only": 1}


def test_load_whose_destination_is_not_a_stop_violates_load_route(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    instance = compile_case(document, compile_document, LOAD_OFF_ROUTE)

    evaluation = evaluate(instance, LOAD_OFF_ROUTE.plan)

    assert evaluation.violated() == {"load_route": 1}


def test_load_leaving_before_its_demand_date_violates_deferral(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    instance = compile_case(document, compile_document, EARLY_LOAD)

    evaluation = evaluate(instance, EARLY_LOAD.plan)

    assert evaluation.violated() == {"deferral": 1}


def test_bans_count_every_banned_stop_and_origin(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    """One ``big`` vehicle stops at ``C`` (1), the other leaves from ``C`` (1).

    The second also carries demand 2 from ``C`` instead of ``B``, one load off its route.
    """
    instance = compile_case(document, compile_document, BANNED)

    evaluation = evaluate(instance, BANNED.plan)

    assert evaluation.violated() == {"ban": 2, "load_route": 1}


def test_owned_vehicles_count_a_wrong_origin_and_a_wrong_end_of_their_line(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    """``B`` is not the line's origin (1) and ``C`` not its destination (1).

    The ``big`` vehicle at ``C`` is also banned there, and demand 0 (900 desi) stays behind.
    """
    instance = compile_case(document, compile_document, OFF_LINE)

    evaluation = evaluate(instance, OFF_LINE.plan)

    assert evaluation.violated() == {"owned_line": 2, "ban": 1, "coverage": 900.0}


def test_owned_dispatches_beyond_the_group_count_violate_owned_availability(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    instance = compile_case(document, compile_document, EXTRA_OWNED)

    evaluation = evaluate(instance, EXTRA_OWNED.plan)

    assert evaluation.violated() == {"owned_availability": 1}


def test_spot_dispatches_beyond_the_daily_limit_violate_spot_availability(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    instance = compile_case(document, compile_document, EXTRA_SPOT)

    evaluation = evaluate(instance, EXTRA_SPOT.plan)

    assert evaluation.violated() == {"spot_availability": 1}


def test_capacity_counts_the_load_above_the_vehicle_capacity(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    """900 desi in a vehicle that holds 400 is 500 desi too many."""
    instance = compile_case(document, compile_document, OVERLOADED)

    evaluation = evaluate(instance, OVERLOADED.plan)

    assert evaluation.violated() == {"capacity": 500.0}


def test_min_fill_counts_the_load_missing_to_the_minimum_fill(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    """The owned 900 of 1000 desi is 50 short of 95%; the spot 250 of 400 reaches 50%; the spot
    120 of 400 is 80 short of 50%: 130 desi in all."""
    instance = compile_case(document, compile_document, MIN_FILL)

    evaluation = evaluate(instance, MIN_FILL.plan)

    assert evaluation.violated() == {"min_fill": pytest.approx(130.0, rel=RELATIVE_TOLERANCE)}


def test_disabled_stopovers_forbid_a_second_stop(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    instance = compile_case(document, compile_document, NO_STOPOVERS)

    evaluation = evaluate(instance, NO_STOPOVERS.plan)

    assert evaluation.violated() == {"stopover": 1}


def test_owned_group_without_stopovers_may_not_stop_twice(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    """The detour also turns back towards ``A`` (1) and enters ``C``, which bans ``big`` (1)."""
    instance = compile_case(document, compile_document, OWNED_DETOUR)

    evaluation = evaluate(instance, OWNED_DETOUR.plan)

    assert evaluation.violated() == {"stopover": 1, "forward_only": 1, "ban": 1}


def test_max_stops_counts_the_stops_beyond_the_limit(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    instance = compile_case(document, compile_document, STOP_LIMIT)

    evaluation = evaluate(instance, STOP_LIMIT.plan)

    assert evaluation.violated() == {"max_stops": 1}


def test_stretch_counts_the_route_length_beyond_the_stretched_direct_distance(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    """3 degrees of route against 1.5 times the direct 1 degree is 1.5 degrees too long."""
    instance = compile_case(document, compile_document, STRETCH_LIMIT)

    evaluation = evaluate(instance, STRETCH_LIMIT.plan)

    assert evaluation.violated() == {
        "stretch": pytest.approx(1.5 * ONE_DEGREE_KM, rel=RELATIVE_TOLERANCE)
    }


def test_stop_and_stretch_limits_are_measured_independently(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    """Under both limits the route breaks each by the amount it breaks it alone."""
    instance = compile_case(document, compile_document, BOTH_LIMITS)

    evaluation = evaluate(instance, BOTH_LIMITS.plan)

    assert evaluation.violated() == {
        "max_stops": 1,
        "stretch": pytest.approx(1.5 * ONE_DEGREE_KM, rel=RELATIVE_TOLERANCE),
    }


def test_unsplittable_demand_counts_every_extra_load(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    instance = compile_case(document, compile_document, SPLIT)

    evaluation = evaluate(instance, SPLIT.plan)

    assert evaluation.violated() == {"split": 1}


def test_coverage_counts_the_demand_not_carried(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    instance = compile_case(document, compile_document, UNCOVERED)

    evaluation = evaluate(instance, UNCOVERED.plan)

    assert evaluation.violated() == {"coverage": 120.0}


def test_overdelivery_counts_the_quantity_carried_beyond_the_demand(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    """300 desi carried for a demand of 250 is 50 desi too many."""
    instance = compile_case(document, compile_document, OVERDELIVERED)

    evaluation = evaluate(instance, OVERDELIVERED.plan)

    assert evaluation.violated() == {"overdelivery": 50.0}


def test_rounding_noise_within_the_tolerance_is_not_a_violation(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    instance = compile_case(document, compile_document, ROUNDED)

    evaluation = evaluate(instance, ROUNDED.plan)

    assert evaluation.feasible is True


def test_own_constraints_are_reported_after_the_built_in_ones(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    """The 2 degree route exceeds a limit of 1.5 degrees by half a degree; there is one spot
    dispatch a day, within a limit of one; and two dispatches use the ``small`` type."""
    instance = compile_case(document, compile_document, FEASIBLE)
    own: list[Constraint] = [
        MaxRouteKm(1.5 * ONE_DEGREE_KM),
        MaxDispatchesPerDay(1, kind="spot"),
        constraint("no_small", lambda evaluation: float((evaluation.vehicle_type == SMALL).sum())),
    ]

    evaluation = evaluate(instance, FEASIBLE.plan, constraints=own)

    assert list(evaluation.violations.items()) == [
        *dict.fromkeys(CATALOG, 0.0).items(),
        ("max_route_km", pytest.approx(0.5 * ONE_DEGREE_KM, rel=RELATIVE_TOLERANCE)),
        ("max_dispatches_per_day", 0.0),
        ("no_small", 2.0),
    ]


def test_a_violated_own_constraint_makes_the_plan_infeasible(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    instance = compile_case(document, compile_document, FEASIBLE)
    own = [constraint("always", lambda evaluation: 1.0)]

    evaluation = evaluate(instance, FEASIBLE.plan, constraints=own)

    assert evaluation.feasible is False


def test_own_constraint_names_must_not_reuse_a_built_in_name(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    instance = compile_case(document, compile_document, FEASIBLE)
    own = [constraint("capacity", lambda evaluation: 0.0)]

    with pytest.raises(ValueError, match="'capacity' is already in use"):
        evaluate(instance, FEASIBLE.plan, own)


@pytest.mark.parametrize(
    "amount", [-1.0, math.nan, math.inf], ids=["negative", "not_a_number", "infinite"]
)
def test_own_constraint_amounts_must_be_finite_and_not_negative(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance], amount: float
) -> None:
    instance = compile_case(document, compile_document, FEASIBLE)
    own = [constraint("odd", lambda evaluation: amount)]

    with pytest.raises(ValueError, match="expected a number >= 0"):
        evaluate(instance, FEASIBLE.plan, own)


def test_plans_must_fit_the_instance(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    """Day 3 is outside the fixture's two day calendar."""
    instance = compile_case(document, compile_document, FEASIBLE)
    plan = Case((Dispatch(day=3, kind="spot", source=BIG, origin=A, stops=(B,)),)).plan

    with pytest.raises(PlanError, match=r"day index outside 0..1"):
        evaluate(instance, plan)


@pytest.mark.parametrize("name", ARRAY_FIELDS)
def test_result_arrays_are_read_only(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance], name: str
) -> None:
    instance = compile_case(document, compile_document, FEASIBLE)
    array = getattr(evaluate(instance, FEASIBLE.plan), name)

    with pytest.raises(ValueError, match="read-only"):
        array[0] = 0


def test_result_violations_are_read_only(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    """The mapping is typed read-only; the test writes to it as a careless caller would."""
    instance = compile_case(document, compile_document, FEASIBLE)
    violations = cast("MutableMapping[str, float]", evaluate(instance, FEASIBLE.plan).violations)

    with pytest.raises(TypeError):
        violations["capacity"] = 1.0


@pytest.mark.parametrize("name", CASES)
def test_evaluator_and_checker_agree_on_the_total_cost(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance], name: str
) -> None:
    case = CASES[name]
    instance = compile_case(document, compile_document, case)

    evaluation = evaluate(instance, case.plan)

    expected = check_plan(instance, case.plan).total_cost
    assert evaluation.total_cost == pytest.approx(expected, rel=RELATIVE_TOLERANCE)


@pytest.mark.parametrize("name", CASES)
def test_evaluator_and_checker_agree_on_every_violation_amount(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance], name: str
) -> None:
    case = CASES[name]
    instance = compile_case(document, compile_document, case)

    evaluation = evaluate(instance, case.plan)

    expected = check_plan(instance, case.plan).totals()
    assert dict(evaluation.violations) == pytest.approx(
        expected, rel=AGREEMENT_TOLERANCE, abs=AGREEMENT_TOLERANCE
    )


@pytest.mark.parametrize("name", CASES)
def test_evaluator_and_checker_agree_on_feasibility(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance], name: str
) -> None:
    case = CASES[name]
    instance = compile_case(document, compile_document, case)

    evaluation = evaluate(instance, case.plan)

    expected = check_plan(instance, case.plan).feasible
    assert evaluation.feasible is expected

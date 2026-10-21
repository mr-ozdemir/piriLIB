import math
from collections.abc import Callable
from typing import Any

import pytest

from pirilib import Dispatch, Evaluation, Load, Plan, PlanError, check_plan, evaluate
from pirilib.constraints import CATALOG, MaxDispatchesPerDay, MaxRouteKm, constraint
from pirilib.spec import LineHaulInstance

ONE = 6371.0 * math.pi / 180.0
A, B, C = 0, 1, 2
BIG, SMALL = 0, 1

FEASIBLE = [
    Dispatch(day=0, kind="owned", source=0, origin=A, stops=(B,), loads=(Load(0, (900.0,)),)),
    Dispatch(day=0, kind="spot", source=SMALL, origin=A, stops=(C,), loads=(Load(1, (250.0,)),)),
    Dispatch(day=1, kind="spot", source=SMALL, origin=B, stops=(A,), loads=(Load(2, (120.0,)),)),
]


def run(instance: LineHaulInstance, dispatches: list[Dispatch], **options: Any) -> Evaluation:
    """Evaluate, and check that the plain-Python checker agrees."""
    plan = Plan.from_dispatches(dispatches, dimensions=len(instance.dimensions))
    evaluation = evaluate(instance, plan, **options)
    report = check_plan(instance, plan)
    assert report.total_cost == pytest.approx(evaluation.total_cost, rel=1e-12)
    for name, amount in report.totals().items():
        assert evaluation.violations[name] == pytest.approx(amount, rel=1e-9, abs=1e-9), name
    assert report.feasible == (not any(evaluation.violations[name] for name in CATALOG))
    return evaluation


def with_rules(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance], **rules: Any
) -> LineHaulInstance:
    document["rules"] = rules
    return compile_document(document)


def test_feasible_plan(instance: LineHaulInstance) -> None:
    evaluation = run(instance, FEASIBLE)
    assert evaluation.feasible
    assert evaluation.violated() == {}
    assert tuple(evaluation.violations) == tuple(CATALOG)
    assert evaluation.vehicle_type.tolist() == [0, 1, 1]
    assert evaluation.route_km.tolist() == pytest.approx([ONE, 2 * ONE, ONE])
    assert evaluation.direct_km.tolist() == pytest.approx([ONE, 2 * ONE, ONE])
    assert evaluation.load.tolist() == [[900.0], [250.0], [120.0]]
    assert evaluation.capacity.tolist() == [[1000.0], [400.0], [400.0]]
    assert evaluation.fill.tolist() == pytest.approx([0.9, 0.625, 0.3])
    assert evaluation.dispatch_cost.tolist() == pytest.approx(
        [ONE, 150 + 1.5 * 2 * ONE, 150 + 1.5 * ONE]
    )
    assert evaluation.owned_fixed_cost == 200.0
    assert evaluation.total_cost == pytest.approx(500 + 5.5 * ONE)
    assert evaluation.total_km == pytest.approx(4 * ONE)
    assert evaluation.unserved.sum() == 0.0
    assert repr(evaluation).startswith("Evaluation(total_cost=1111.57, dispatches=3, ")


def test_empty_plan_serves_nothing(instance: LineHaulInstance) -> None:
    evaluation = run(instance, [])
    assert evaluation.total_cost == 200.0
    assert evaluation.violated() == {"coverage": 1270.0}
    assert evaluation.unserved.tolist() == [[900.0], [250.0], [120.0]]


def test_non_mandatory_owned_vehicles_pay_per_dispatch(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    document["fleet"]["owned"][0]["mandatory"] = False
    evaluation = run(compile_document(document), FEASIBLE)
    assert evaluation.owned_fixed_cost == 0.0
    assert evaluation.dispatch_cost[0] == pytest.approx(100 + ONE)


def test_routes_must_be_well_formed(instance: LineHaulInstance) -> None:
    dispatches = [
        *FEASIBLE,
        Dispatch(day=1, kind="spot", source=SMALL, origin=A, stops=()),
        Dispatch(day=1, kind="spot", source=SMALL, origin=A, stops=(A, B, B)),
    ]
    violated = run(instance, dispatches).violated()
    assert violated["route"] == 3


def test_loads_must_match_the_route(instance: LineHaulInstance) -> None:
    dispatches = [
        FEASIBLE[0],
        Dispatch(
            day=0, kind="spot", source=SMALL, origin=A, stops=(B,), loads=(Load(1, (250.0,)),)
        ),
        Dispatch(
            day=0, kind="spot", source=SMALL, origin=B, stops=(A,), loads=(Load(2, (120.0,)),)
        ),
    ]
    assert run(instance, dispatches).violated() == {"load_route": 1, "deferral": 1}


def test_bans(instance: LineHaulInstance) -> None:
    dispatches = [
        FEASIBLE[0],
        Dispatch(day=0, kind="spot", source=BIG, origin=A, stops=(C,), loads=(Load(1, (250.0,)),)),
        Dispatch(
            day=1, kind="spot", source=BIG, origin=C, stops=(B, A), loads=(Load(2, (120.0,)),)
        ),
    ]
    violated = run(instance, dispatches).violated()
    assert violated == {"ban": 2, "load_route": 1}


def test_owned_vehicles_keep_to_their_line(instance: LineHaulInstance) -> None:
    dispatches = [
        Dispatch(day=0, kind="owned", source=0, origin=B, stops=(C,)),
        *FEASIBLE[1:],
    ]
    assert run(instance, dispatches).violated() == {"owned_line": 2, "ban": 1, "coverage": 900.0}


def test_availability(instance: LineHaulInstance) -> None:
    extra = [
        Dispatch(day=0, kind="owned", source=0, origin=A, stops=(B,)),
        *(Dispatch(day=1, kind="spot", source=SMALL, origin=A, stops=(B,)) for _ in range(3)),
    ]
    violated = run(instance, [*FEASIBLE, *extra]).violated()
    assert violated == {"owned_availability": 1, "spot_availability": 1}


def test_capacity(instance: LineHaulInstance) -> None:
    dispatches = [
        Dispatch(
            day=0, kind="spot", source=SMALL, origin=A, stops=(B,), loads=(Load(0, (900.0,)),)
        ),
        *FEASIBLE[1:],
    ]
    assert run(instance, dispatches).violated() == {"capacity": 500.0}


def test_min_fill(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    instance = with_rules(document, compile_document, min_fill={"spot": 0.5, "owned": 0.95})
    violated = run(instance, FEASIBLE).violated()
    assert violated == {"min_fill": pytest.approx(130.0)}


def test_stopovers_disabled(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    instance = with_rules(document, compile_document, stopover={"enabled": False})
    dispatches = [
        FEASIBLE[0],
        Dispatch(
            day=0, kind="spot", source=SMALL, origin=A, stops=(B, C), loads=(Load(1, (250.0,)),)
        ),
        FEASIBLE[2],
    ]
    assert run(instance, dispatches).violated() == {"stopover": 1}


def test_owned_groups_without_stopovers(instance: LineHaulInstance) -> None:
    dispatches = [
        Dispatch(day=0, kind="owned", source=0, origin=A, stops=(C, B), loads=(Load(0, (900.0,)),)),
        *FEASIBLE[1:],
    ]
    violated = run(instance, dispatches).violated()
    assert violated["stopover"] == 1
    assert violated["forward_only"] == 1


def test_max_stops_and_stretch(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    instance = with_rules(
        document,
        compile_document,
        stopover={"max_stops": 1, "max_stretch": 1.5, "forward_only": False},
    )
    dispatches = [
        FEASIBLE[0],
        FEASIBLE[1],
        Dispatch(
            day=1, kind="spot", source=SMALL, origin=B, stops=(C, A), loads=(Load(2, (120.0,)),)
        ),
    ]
    violated = run(instance, dispatches).violated()
    assert violated == {"max_stops": 1, "stretch": pytest.approx(1.5 * ONE)}


def test_unsplittable_demand(
    document: dict[str, Any], compile_document: Callable[..., LineHaulInstance]
) -> None:
    instance = with_rules(document, compile_document, split_demand=False)
    dispatches = [
        Dispatch(day=0, kind="owned", source=0, origin=A, stops=(B,), loads=(Load(0, (500.0,)),)),
        Dispatch(day=0, kind="spot", source=BIG, origin=A, stops=(B,), loads=(Load(0, (400.0,)),)),
        *FEASIBLE[1:],
    ]
    assert run(instance, dispatches).violated() == {"split": 1}


def test_coverage_and_overdelivery(instance: LineHaulInstance) -> None:
    dispatches = [
        FEASIBLE[0],
        Dispatch(
            day=0, kind="spot", source=SMALL, origin=A, stops=(C,), loads=(Load(1, (300.0,)),)
        ),
    ]
    assert run(instance, dispatches).violated() == {"coverage": 120.0, "overdelivery": 50.0}


def test_rounding_is_tolerated(instance: LineHaulInstance) -> None:
    thirds = tuple(Load(0, (300.0 + 1e-10,)) for _ in range(3))
    dispatches = [
        Dispatch(day=0, kind="owned", source=0, origin=A, stops=(B,), loads=thirds),
        *FEASIBLE[1:],
    ]
    assert run(instance, dispatches).feasible


def test_constraints_of_your_own(instance: LineHaulInstance) -> None:
    own = [
        MaxRouteKm(1.5 * ONE),
        MaxDispatchesPerDay(1, kind="spot"),
        constraint("no_small", lambda e: float((e.vehicle_type == 1).sum())),
    ]
    evaluation = run(instance, FEASIBLE, constraints=own)
    assert evaluation.violated() == {
        "max_route_km": pytest.approx(0.5 * ONE),
        "no_small": 2.0,
    }
    assert list(evaluation.violations)[-3:] == [
        "max_route_km",
        "max_dispatches_per_day",
        "no_small",
    ]
    assert not evaluation.feasible


def test_constraint_names_must_be_unique(instance: LineHaulInstance) -> None:
    plan = Plan.from_dispatches(FEASIBLE, 1)
    with pytest.raises(ValueError, match="'capacity' is already in use"):
        evaluate(instance, plan, [constraint("capacity", lambda e: 0.0)])


@pytest.mark.parametrize("amount", [-1.0, math.nan, math.inf])
def test_constraint_amounts_must_be_valid(instance: LineHaulInstance, amount: float) -> None:
    plan = Plan.from_dispatches(FEASIBLE, 1)
    with pytest.raises(ValueError, match="expected a number >= 0"):
        evaluate(instance, plan, [constraint("odd", lambda e: amount)])


def test_plans_must_fit_the_instance(instance: LineHaulInstance) -> None:
    plan = Plan.from_dispatches([Dispatch(day=3, kind="spot", source=0, origin=A, stops=(B,))], 1)
    with pytest.raises(PlanError, match=r"day index outside 0..1"):
        evaluate(instance, plan)


def test_results_are_read_only(instance: LineHaulInstance) -> None:
    evaluation = evaluate(instance, Plan.from_dispatches(FEASIBLE, 1))
    with pytest.raises(ValueError, match="read-only"):
        evaluation.route_km[0] = 0.0
    with pytest.raises(TypeError):
        evaluation.violations["capacity"] = 1.0

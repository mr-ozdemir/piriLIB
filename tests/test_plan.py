import numpy as np
import pytest

from pirilib import Dispatch, Load, Plan, PlanError
from pirilib.plan import OWNED, SPOT
from pirilib.spec import LineHaulInstance

DISPATCHES = [
    Dispatch(day=0, kind="owned", source=0, origin=0, stops=(1,), loads=(Load(0, (900.0,)),)),
    Dispatch(day=0, kind="spot", source=1, origin=0, stops=(1, 2), loads=(Load(1, (250.0,)),)),
    Dispatch(day=1, kind="spot", source=0, origin=1, stops=()),
]


def test_arrays_from_dispatches() -> None:
    plan = Plan.from_dispatches(DISPATCHES, dimensions=1)
    assert len(plan) == 3
    assert plan.dimensions == 1
    assert plan.day.tolist() == [0, 0, 1]
    assert plan.kind.tolist() == [OWNED, SPOT, SPOT]
    assert plan.stop_start.tolist() == [0, 1, 3, 3]
    assert plan.stop_center.tolist() == [1, 1, 2]
    assert plan.stops(1).tolist() == [1, 2]
    assert plan.load_dispatch.tolist() == [0, 1]
    assert plan.load_quantity.tolist() == [[900.0], [250.0]]
    assert repr(plan) == "Plan(dispatches=3, stops=3, loads=2, dimensions=1)"


def test_dispatches_round_trip() -> None:
    plan = Plan.from_dispatches(DISPATCHES, dimensions=1)
    assert list(plan.dispatches()) == DISPATCHES


def test_empty_plan() -> None:
    plan = Plan.empty(dimensions=2)
    assert len(plan) == 0
    assert plan.load_quantity.shape == (0, 2)
    assert plan.stop_start.tolist() == [0]


def test_arrays_are_converted_and_read_only() -> None:
    plan = Plan(
        day=[0],
        kind=[1],
        source=[0],
        origin=[0],
        stop_start=[0, 1],
        stop_center=[2],
        load_dispatch=[0],
        load_demand=[1],
        load_quantity=[[250]],
    )
    assert plan.kind.dtype == np.int8
    assert plan.load_quantity.dtype == np.float64
    with pytest.raises(ValueError, match="read-only"):
        plan.day[0] = 1


def test_malformed_arrays_are_reported_together() -> None:
    with pytest.raises(PlanError) as info:
        Plan(
            day=np.array([0, 0]),
            kind=np.array([1, 3]),
            source=np.array([0]),
            origin=np.array([0, 0]),
            stop_start=np.array([0, 2, 1]),
            stop_center=np.array([1]),
            load_dispatch=np.array([0]),
            load_demand=np.array([0, 1]),
            load_quantity=np.array([[-1.0]]),
        )
    assert str(info.value).splitlines() == [
        "malformed plan",
        "- source has 1 entries, day has 2",
        "- stop_start must rise from 0 to the number of stops (1) without decreasing",
        "- load_demand must have one entry per load (1)",
        "- load quantities must be finite and not negative; see loads 0",
        "- kind must be 0 (owned) or 1 (spot); see dispatches 1",
    ]


def test_bad_dispatches_are_reported() -> None:
    dispatches = [
        Dispatch(day=0, kind="rented", source=0, origin=0, stops=(1,)),
        Dispatch(day=0, kind="spot", source=0, origin=0, stops=(1,), loads=(Load(0, (1.0, 2.0)),)),
    ]
    with pytest.raises(PlanError) as info:
        Plan.from_dispatches(dispatches, dimensions=1)
    assert str(info.value).splitlines()[1:] == [
        "- dispatch 0: kind must be 'owned' or 'spot', not 'rented'",
        "- dispatch 1: load of demand 0 has 2 quantities, expected 1",
    ]


def test_indices_are_checked_against_the_instance(instance: LineHaulInstance) -> None:
    dispatches = [
        Dispatch(day=5, kind="owned", source=1, origin=0, stops=(3,)),
        Dispatch(day=0, kind="spot", source=2, origin=-1, stops=(1,), loads=(Load(7, (1.0,)),)),
    ]
    plan = Plan.from_dispatches(dispatches, dimensions=1)
    with pytest.raises(PlanError) as info:
        plan.check(instance)
    assert str(info.value).splitlines() == [
        "plan does not fit the instance",
        "- day index outside 0..1; see dispatches 0",
        "- origin index outside 0..2; see dispatches 1",
        "- stop index outside 0..2; see stops 0",
        "- owned group index outside 0..0; see dispatches 0",
        "- spot offer index outside 0..1; see dispatches 1",
        "- load demand index outside 0..2; see loads 0",
    ]


def test_dimensions_are_checked_against_the_instance(instance: LineHaulInstance) -> None:
    with pytest.raises(PlanError, match="loads have 2 quantity dimensions, the instance has 1"):
        Plan.empty(dimensions=2).check(instance)

"""The plain-Python checker, and its agreement with the numpy evaluator."""

import datetime as dt
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pirilib import Dispatch, Load, Plan, check_plan, evaluate
from pirilib.spec import LineHaulInstance, parse_spec


def test_messages_explain_violations(instance: LineHaulInstance) -> None:
    dispatches = [
        Dispatch(day=0, kind="spot", source=0, origin=0, stops=(2, 1), loads=(Load(1, (1250.0,)),)),
    ]
    report = check_plan(instance, Plan.from_dispatches(dispatches, 1))
    assert not report.feasible
    label = "dispatch 0 (2026-01-05, spot big from A)"
    assert str(report).splitlines() == [
        f"total cost {report.total_cost:.2f}, 6 violation(s)",
        f"- ban: {label}: big is banned at C",
        f"- forward_only: {label}: B is not farther from the origin than C",
        f"- capacity: {label} carries 1250 desi, capacity is 1000",
        "- coverage: demand D000001: 900 desi not carried",
        "- overdelivery: demand D000002: 1000 desi too many",
        "- coverage: demand D000003: 120 desi not carried",
    ]
    assert report.totals()["coverage"] == 1020.0


_CENTERS = [(0.0, 0.0), (0.1, 1.0), (-0.1, 2.0), (0.2, 3.0), (0.0, 4.0)]
_RULES: list[dict[str, Any]] = [
    {},
    {
        "primary_dimension": "kg",
        "min_fill": {"spot": 0.03, "owned": 0.1},
        "stopover": {"max_stretch": 1.2, "max_stops": 2},
        "split_demand": False,
    },
    {"stopover": {"enabled": False, "forward_only": False}},
    {
        "service": {"due_hours": 6, "penalty": 0.5, "max_deferral_days": 1},
        "handling": {"minutes_per_unit": 0.05},
    },
]
_TIMED = [False, False, False, True]


def _document(rules: dict[str, Any], timed: bool = False) -> dict[str, Any]:
    rng = np.random.default_rng(11)
    names = "ABCDE"
    demand = []
    for _ in range(14):
        origin, destination = rng.choice(5, size=2, replace=False)
        demand.append(
            {
                "origin": names[origin],
                "destination": names[destination],
                "date": (dt.date(2026, 3, 2) + dt.timedelta(days=int(rng.integers(3)))).isoformat(),
                "quantity": {
                    "desi": round(float(rng.uniform(50, 380)), 1),
                    "kg": round(float(rng.uniform(30, 280)), 1),
                },
            }
        )
    return {
        "schema": "pirilib/linehaul@0.1",
        "calendar": {"start": "2026-03-02", "end": "2026-03-04"},
        "centers": [
            {"id": name, "lat": lat, "lon": lon, "banned_vehicle_types": ["big"] * (name == "E")}
            for name, (lat, lon) in zip(names, _CENTERS, strict=True)
        ],
        "matrix": {"source": "haversine", "factor": 1.2},
        "vehicle_types": [
            {"name": "big", "capacity": {"desi": 1000.0, "kg": 800.0}},
            {"name": "small", "capacity": {"desi": 400.0, "kg": 300.0}},
        ],
        "fleet": {
            "owned": [
                {
                    "type": "big",
                    "origin": "A",
                    "destination": "D",
                    "count": 2,
                    "stopovers": True,
                    "cost": {"fixed": 900.0, "per_km": 3.0},
                },
                {
                    "type": "small",
                    "origin": "B",
                    "destination": "A",
                    "count": 1,
                    "mandatory": False,
                    "cost": {"fixed": 200.0, "per_km": 1.0},
                },
            ],
            "spot": [
                {"type": "big", "cost": {"fixed": 1500.0, "per_km": 5.0}},
                {"type": "small", "cost": {"fixed": 600.0, "per_km": 2.5}, "available": 4},
            ],
        },
        "demand": demand,
        "rules": rules,
    }


def _timed(document: dict[str, Any]) -> dict[str, Any]:
    """Speeds, hourly costs, ready and due times, handling capacity and vehicle limits."""
    rng = np.random.default_rng(5)
    document["schema"] = "pirilib/linehaul@0.2"
    for vehicle, speed in zip(document["vehicle_types"], (70.0, 80.0), strict=True):
        vehicle["speed_kmh"] = speed
    for entry in [*document["fleet"]["owned"], *document["fleet"]["spot"]]:
        entry["cost"]["per_hour"] = float(rng.uniform(5, 40))
    for row in document["demand"]:
        row["ready"] = str(rng.choice(["09:00", "17:00", "00:00"]))
        if rng.random() < 0.2:
            row["due"] = f"{row['date']}T23:30:00"
    document["centers"][0]["handling_capacity"] = 900.0
    document["centers"][1]["handling_capacity"] = 1500.0
    document["centers"][3]["vehicle_limits"] = {"small": 2, "big": 1}
    return document


_INSTANCES = [
    parse_spec(_timed(_document(rules)) if timed else _document(rules)).compile()
    for rules, timed in zip(_RULES, _TIMED, strict=True)
]


def random_plan(instance: LineHaulInstance, rng: np.random.Generator) -> Plan:
    """A random plan: mostly sensible dispatches, with every kind of mistake now and then."""
    centers, days = len(instance.centers), len(instance.days)
    dispatches = []
    for _ in range(int(rng.integers(0, 10))):
        owned = bool(rng.random() < 0.4)
        source = int(rng.integers(instance.owned_type.size if owned else instance.spot_type.size))
        day = 0 if rng.random() < 0.5 else int(rng.integers(days))
        if owned and rng.random() < 0.8:
            origin = int(instance.owned_origin[source])
        else:
            origin = int(rng.integers(centers))
        stops = [
            int(c)
            for c in rng.choice(
                centers, size=int(rng.integers(0, 4)), replace=bool(rng.random() < 0.2)
            )
        ]
        if owned and stops and rng.random() < 0.7:
            stops[-1] = int(instance.owned_destination[source])
        matching = np.flatnonzero((instance.demand_origin == origin) & (instance.demand_day == day))
        loads = []
        for _ in range(int(rng.integers(0, 4))):
            pool = (
                matching
                if matching.size and rng.random() < 0.8
                else np.arange(len(instance.demand_ids))
            )
            demand = int(rng.choice(pool))
            scale = float(rng.choice([1.0, 0.5, rng.uniform(0.0, 1.5)]))
            loads.append(
                Load(demand, tuple(float(q) * scale for q in instance.demand_quantity[demand]))
            )
        dispatches.append(
            Dispatch(
                day=day,
                kind="owned" if owned else "spot",
                source=source,
                origin=origin,
                stops=tuple(stops),
                loads=tuple(loads),
                start=float(rng.uniform(0, 1500)) if rng.random() < 0.3 else None,
            )
        )
    return Plan.from_dispatches(dispatches, len(instance.dimensions))


def direct_plan(instance: LineHaulInstance, rng: np.random.Generator) -> Plan:
    """Every demand on its own spot vehicles, with quantities a hair off now and then.

    Many of these plans are feasible, so the tolerance boundaries get exercised.
    """
    dispatches = []
    factor = float(rng.choice([1.0, 1.0, 1 + 1e-12, 1 - 1e-12, 1 + 1e-6, 1 - 1e-6]))
    for row in range(len(instance.demand_ids)):
        origin = int(instance.demand_origin[row])
        destination = int(instance.demand_destination[row])
        allowed = [
            offer
            for offer, vehicle in enumerate(instance.spot_type)
            if not instance.banned[origin, vehicle] and not instance.banned[destination, vehicle]
        ]
        offer = allowed[0]
        quantity = instance.demand_quantity[row]
        capacity = instance.capacity[instance.spot_type[offer]]
        parts = (
            1 if not instance.rules.split_demand else max(1, math.ceil(max(quantity / capacity)))
        )
        for _ in range(parts):
            share = tuple(float(q) / parts * factor for q in quantity)
            dispatches.append(
                Dispatch(
                    day=int(instance.demand_day[row]),
                    kind="spot",
                    source=offer,
                    origin=origin,
                    stops=(destination,),
                    loads=(Load(row, share),),
                )
            )
    return Plan.from_dispatches(dispatches, len(instance.dimensions))


@pytest.mark.parametrize("make", [direct_plan, random_plan], ids=["direct", "random"])
@pytest.mark.parametrize("instance_index", range(len(_INSTANCES)))
@given(seed=st.integers(0, 2**32 - 1))
@settings(max_examples=150, deadline=None)
def test_evaluator_agrees_with_checker(
    make: Callable[[LineHaulInstance, np.random.Generator], Plan], instance_index: int, seed: int
) -> None:
    instance = _INSTANCES[instance_index]
    plan = make(instance, np.random.default_rng(seed))
    evaluation = evaluate(instance, plan)
    report = check_plan(instance, plan)
    assert evaluation.total_cost == pytest.approx(report.total_cost, rel=1e-12)
    assert evaluation.violations == pytest.approx(report.totals(), rel=1e-9, abs=1e-6)
    assert evaluation.feasible == report.feasible
    np.testing.assert_array_equal(evaluation.route_km >= evaluation.direct_km * (1 - 1e-12), True)
    assert list(Plan.from_dispatches(plan.dispatches(), plan.dimensions).dispatches()) == list(
        plan.dispatches()
    )

import copy
import datetime as dt
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pirilib.spec import LineHaulInstance, parse_spec

EXAMPLE = Path(__file__).parents[1] / "examples" / "linehaul_mini" / "problem.yaml"

_DOCUMENT: dict[str, Any] = {
    "schema": "pirilib/linehaul@0.1",
    "calendar": {"start": "2026-01-05", "end": "2026-01-06"},
    "centers": [
        {"id": "A", "lat": 0.0, "lon": 0.0},
        {"id": "B", "lat": 0.0, "lon": 1.0},
        {"id": "C", "lat": 0.0, "lon": 2.0, "banned_vehicle_types": ["big"]},
    ],
    "matrix": {"source": "haversine"},
    "vehicle_types": [
        {"name": "big", "capacity": {"desi": 1000.0}},
        {"name": "small", "capacity": {"desi": 400.0}},
    ],
    "fleet": {
        "owned": [
            {
                "type": "big",
                "origin": "A",
                "destination": "B",
                "count": 1,
                "cost": {"fixed": 100.0, "per_km": 1.0},
            }
        ],
        "spot": [
            {"type": "big", "cost": {"fixed": 300.0, "per_km": 2.0}},
            {"type": "small", "cost": {"fixed": 150.0, "per_km": 1.5}, "available": 3},
        ],
    },
    "demand": [
        {"origin": "A", "destination": "B", "date": "2026-01-05", "quantity": {"desi": 900.0}},
        {"origin": "A", "destination": "C", "date": "2026-01-05", "quantity": {"desi": 250.0}},
        {"origin": "B", "destination": "A", "date": "2026-01-06", "quantity": {"desi": 120.0}},
    ],
}


@pytest.fixture
def document() -> dict[str, Any]:
    """A small valid problem document with inline tables; every test gets its own copy."""
    return copy.deepcopy(_DOCUMENT)


@pytest.fixture
def example_path() -> Path:
    """The problem file of the bundled example."""
    return EXAMPLE


@pytest.fixture
def compile_document() -> Callable[[dict[str, Any]], LineHaulInstance]:
    """Validate and compile a problem document."""
    return lambda document: parse_spec(document).compile()


@pytest.fixture
def instance(document: dict[str, Any]) -> LineHaulInstance:
    """The compiled form of :func:`document`."""
    return parse_spec(document).compile()


def random_document(seed: int) -> dict[str, Any]:
    """A small random problem that passes validation, with varied rules."""
    rng = np.random.default_rng(seed)
    names = [f"N{i}" for i in range(int(rng.integers(3, 8)))]
    two = bool(rng.random() < 0.3)
    big = {"desi": 1000.0, **({"kg": 700.0} if two else {})}
    small = {"desi": 400.0, **({"kg": 300.0} if two else {})}
    split = bool(rng.random() < 0.7)
    centers = [
        {
            "id": name,
            "lat": float(rng.uniform(38, 41)),
            "lon": float(rng.uniform(27, 33)),
            "banned_vehicle_types": ["big"] if rng.random() < 0.2 else [],
        }
        for name in names
    ]
    allowed_big = [c["id"] for c in centers if not c["banned_vehicle_types"]]
    owned = []
    for _ in range(int(rng.integers(0, 3))):
        use_big = len(allowed_big) >= 2 and rng.random() < 0.5
        pool = allowed_big if use_big else names
        origin, destination = rng.choice(pool, size=2, replace=False)
        owned.append(
            {
                "type": "big" if use_big else "small",
                "origin": str(origin),
                "destination": str(destination),
                "count": int(rng.integers(1, 3)),
                "mandatory": bool(rng.random() < 0.6),
                "stopovers": bool(rng.random() < 0.6),
                "cost": {
                    "fixed": float(rng.uniform(100, 600)),
                    "per_km": float(rng.uniform(0.5, 2)),
                },
            }
        )
    limited = rng.random() < 0.3
    spot = [
        {"type": "big", "cost": {"fixed": 900.0, "per_km": 3.0}},
        {
            "type": "small",
            "cost": {"fixed": 500.0, "per_km": 2.0},
            **({"available": int(rng.integers(1, 4))} if limited else {}),
        },
    ]
    days = int(rng.integers(1, 4))
    demand = []
    for _ in range(int(rng.integers(3, 20))):
        origin, destination = rng.choice(names, size=2, replace=False)
        top = 2500.0 if split else 400.0
        quantity = {"desi": round(float(rng.uniform(20, top)), 1)}
        if two:
            quantity["kg"] = round(float(rng.uniform(10, top * 0.7)), 1)
        demand.append(
            {
                "origin": str(origin),
                "destination": str(destination),
                "date": (
                    dt.date(2026, 6, 1) + dt.timedelta(days=int(rng.integers(days)))
                ).isoformat(),
                "quantity": quantity,
            }
        )
    stopover: dict[str, Any] = {"enabled": bool(rng.random() < 0.8)}
    if rng.random() < 0.5:
        stopover["max_stretch"] = float(rng.uniform(1.0, 1.6))
    if rng.random() < 0.3:
        stopover["forward_only"] = False
    if rng.random() < 0.3:
        stopover["max_stops"] = int(rng.integers(1, 4))
    return {
        "schema": "pirilib/linehaul@0.1",
        "calendar": {"start": "2026-06-01", "end": f"2026-06-0{days}"},
        "centers": centers,
        "matrix": {"source": "haversine", "factor": 1.2},
        "vehicle_types": [{"name": "big", "capacity": big}, {"name": "small", "capacity": small}],
        "fleet": {"owned": owned, "spot": spot},
        "demand": demand,
        "rules": {
            "min_fill": {"spot": float(rng.choice([0.0, 0.1, 0.3]))},
            "stopover": stopover,
            "split_demand": split,
        },
    }


@pytest.fixture(name="random_document")
def random_document_fixture() -> Callable[[int], dict[str, Any]]:
    """A function from a seed to a small random problem document."""
    return random_document

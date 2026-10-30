import copy
import datetime as dt
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from hypothesis import settings

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
    """A small valid problem document with inline tables; every test gets its own copy.

    Three centers on the equator one degree apart (``A``, ``B``, ``C``); ``C`` bans the
    ``big`` type. ``big`` holds 1000 desi and ``small`` 400. One owned ``big`` vehicle runs
    ``A`` to ``B`` (fixed 100, 1 per km); the spot market offers ``big`` (fixed 300, 2 per km,
    unlimited) and ``small`` (fixed 150, 1.5 per km, 3 available). Demand over the two days:
    ``A`` to ``B`` 900 desi (only ``big`` can carry it), ``A`` to ``C`` 250 desi (``small``
    carries it, ``big`` may not enter ``C``) and ``B`` to ``A`` 120 desi on the second day.
    """
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
    """A small random problem that passes validation, drawn from ``seed`` alone.

    The same seed always gives the same document. The numbers only need to give a varied,
    valid problem of a size the solvers finish quickly; none of them carries a meaning of its
    own beyond the reason given here.

    Centers: 3 to 7, named ``N0``, ``N1``, ... at latitude 38 to 41 and longitude 27 to 33
    (an area the size of a region, so distances are tens to hundreds of kilometres); each
    bans the ``big`` type with probability 0.2. Road distances are 1.2 times the haversine
    distance (a usual road to straight line ratio).

    Vehicle types: ``big`` holds 1000 desi and ``small`` 400. With probability 0.3 the
    capacities have a second dimension ``kg`` (``big`` 700 kg, ``small`` 300 kg).

    Owned offers: 0 to 2. An offer is ``big`` with probability 0.5 when two centers allow
    ``big`` and runs between such centers, otherwise ``small`` between any two centers. Each
    has 1 or 2 vehicles, is mandatory with probability 0.6 and may stop over with
    probability 0.6; it costs 100 to 600 fixed and 0.5 to 2 per km.

    Spot offers: ``big`` (fixed 900, 3 per km) and ``small`` (fixed 500, 2 per km) are always
    offered (at least as dear per km as any owned offer); with probability 0.3 the ``small``
    ones are limited to 1 to 3.

    Demand: 3 to 19 items over 1 to 3 days from 2026-06-01, the calendar ending on the last
    of those days. With probability 0.7 demand may be split and an item holds 20 to 2500
    desi, otherwise 20 to the 400 desi a ``small`` vehicle holds, so every item fits one
    vehicle. With two dimensions an item also weighs 10 kg up to 0.7 times that desi upper
    bound in kg (the ``kg`` to desi ratio of the ``big`` type).

    Rules: ``min_fill`` for spot vehicles is one of 0, 0.1 and 0.3. Stopovers are enabled
    with probability 0.8; with probability 0.5 ``max_stretch`` is set to 1.0 to 1.6, with
    probability 0.3 ``forward_only`` is turned off and with probability 0.3 ``max_stops`` is
    set to 1 to 3.
    """
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


EXAMPLES = 100
"""Examples per Hypothesis test in the ``dev`` and ``ci`` profiles (the Hypothesis default)."""

FUZZ_EXAMPLES = 2000
"""Examples per Hypothesis test in the long ``fuzz`` profile.

The ``fuzz`` profile is the one exception to the rule that the tests are deterministic: it is
an exploration run (the weekly quality workflow) and draws new examples on every run on
purpose, so that it reaches inputs the fixed ``dev`` and ``ci`` runs never see. It keeps no
example database, so a run leaves no state behind, and ``print_blob`` prints the
``@reproduce_failure`` line of a failing example, which replays that failure exactly.
"""

settings.register_profile("dev", deadline=None, derandomize=True, max_examples=EXAMPLES)
settings.register_profile(
    "ci", deadline=None, derandomize=True, max_examples=EXAMPLES, print_blob=True
)
settings.register_profile(
    "fuzz", deadline=None, database=None, max_examples=FUZZ_EXAMPLES, print_blob=True
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


class SuiteLevels:
    """The level and technique markers of every test module.

    A module is ``unit`` unless :attr:`levels` names it: ``integration`` modules work with
    files, optional packages or several components together, ``system`` modules drive the
    command line, the documentation examples or the installed package. A module directly in
    ``tests/fuzz``, ``tests/model`` or ``tests/performance`` also carries the marker named
    after that directory, and every module in ``tests/performance`` is a ``system`` test.

    Attributes:
        levels: Module name (without ``.py``) to level, for the modules that are not ``unit``
        directory_levels: Directory name to the level of every module directly in it
        techniques: Directory names that are technique markers as well
    """

    levels: ClassVar[dict[str, str]] = {
        "test_adapters": "integration",
        "test_bench": "integration",
        "test_exact": "integration",
        "test_forecast": "integration",
        "test_loader": "integration",
        "test_planio": "integration",
        "test_schedule": "integration",
        "test_tuning": "integration",
        "test_waves": "integration",
        "test_cli": "system",
        "test_docs_examples": "system",
        "test_package": "system",
    }
    directory_levels: ClassVar[dict[str, str]] = {"performance": "system"}
    techniques = frozenset({"fuzz", "model", "performance"})

    @classmethod
    def markers(cls, path: Path) -> list[str]:
        """The markers of the module at ``path``: its level, then its technique if it has one."""
        directory = path.parent.name
        level = cls.directory_levels.get(directory, cls.levels.get(path.stem, "unit"))
        return [level, *sorted(cls.techniques & {directory})]


@pytest.fixture
def suite_markers() -> Callable[[Path], list[str]]:
    """A function from a test module's path to the markers the suite gives its tests."""
    return SuiteLevels.markers


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add ``--performance``, which also runs the performance tests."""
    parser.addoption(
        "--performance",
        action="store_true",
        help="also run the performance tests (marked 'performance'), which time the solvers",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark every test with its level and technique; skip performance tests unless asked."""
    skip = pytest.mark.skip(reason="performance test: run with --performance")
    run_performance = config.getoption("--performance")
    for item in items:
        for name in SuiteLevels.markers(item.path):
            item.add_marker(name)
        if "performance" in item.keywords and not run_performance:
            item.add_marker(skip)

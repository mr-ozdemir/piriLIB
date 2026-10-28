"""Tests of the helpers shared through ``tests/conftest.py``.

``random_document`` holds logic, so it has tests of its own: every document it builds for
the seeds the suite uses must pass the library's validation and compile, it must follow
each draw its docstring states, and a seed must fix the whole document. ``SuiteLevels``
maps a module's path to its markers, so its table and its rule that only the module's own
directory counts are tested through the ``suite_markers`` fixture.
"""

from collections.abc import Callable
from operator import itemgetter
from pathlib import Path
from typing import Any

import pytest

from pirilib.spec import LineHaulInstance, parse_spec

type RandomDocument = Callable[[int], dict[str, Any]]
type SuiteMarkers = Callable[[Path], list[str]]

SEEDS_IN_USE = range(60)
"""The seeds other tests pass to ``random_document``; the widest range they use is 0 to 59."""

SEED = 7
"""Any seed; the property holds for every one."""

OTHER_SEED = 8
"""A seed other than :data:`SEED`."""

FEWEST_CENTERS = 3
MOST_CENTERS = 7
FEWEST_DEMANDS = 3
MOST_DEMANDS = 19
FEWEST_OWNED = 0
MOST_OWNED = 2
"""Count bounds stated in the ``random_document`` docstring."""

CENTER_NAMES = ("N0", "N1", "N2", "N3", "N4", "N5", "N6")
"""The center names in order, for the most centers a document has."""

SOUTHMOST, NORTHMOST = 38.0, 41.0
WESTMOST, EASTMOST = 27.0, 33.0
"""The region the centers lie in."""

BANS = {(), ("big",)}
"""A center bans nothing or only the ``big`` type."""

MATRIX = {"source": "haversine", "factor": 1.2}
"""Haversine distances times the road factor 1.2."""

ONE_DIMENSION_TYPES = [
    {"name": "big", "capacity": {"desi": 1000.0}},
    {"name": "small", "capacity": {"desi": 400.0}},
]
TWO_DIMENSION_TYPES = [
    {"name": "big", "capacity": {"desi": 1000.0, "kg": 700.0}},
    {"name": "small", "capacity": {"desi": 400.0, "kg": 300.0}},
]
"""The two vehicle type tables a document can have."""

OWNED_TYPES = {"big", "small"}
OWNED_COUNTS = {1, 2}
CHEAPEST_OWNED_FIXED, DEAREST_OWNED_FIXED = 100.0, 600.0
CHEAPEST_OWNED_PER_KM, DEAREST_OWNED_PER_KM = 0.5, 2.0
"""What an owned offer can be."""

SPOT_TYPES = ["big", "small"]
SPOT_COSTS = [{"fixed": 900.0, "per_km": 3.0}, {"fixed": 500.0, "per_km": 2.0}]
"""The spot offers every document has, in order."""

SPOT_SMALL_AVAILABLE = {None, 1, 2, 3}
"""Spot ``small`` is unlimited (no ``available``) or limited to 1 to 3."""

CALENDARS = [
    {"start": "2026-06-01", "end": "2026-06-01"},
    {"start": "2026-06-01", "end": "2026-06-02"},
    {"start": "2026-06-01", "end": "2026-06-03"},
]
"""A calendar of 1 to 3 days from 2026-06-01."""

CALENDAR_START = "2026-06-01"
"""The first day of every calendar; ISO dates compare in calendar order as strings."""

LIGHTEST_DESI = 20.0
"""The least desi a demand item holds."""

HEAVIEST_DESI = {True: 2500.0, False: 400.0}
"""The most desi an item holds, keyed by ``split_demand``: unsplit items fit a ``small``."""

MIN_FILLS = [{"spot": 0.0}, {"spot": 0.1}, {"spot": 0.3}]
"""The ``min_fill`` rules a document can have."""

OPTIONAL_STOPOVER_RULES = {"max_stretch", "forward_only", "max_stops"}
"""The stopover rules a document may set besides ``enabled``, which it always sets."""

SHORTEST_STRETCH, LONGEST_STRETCH = 1.0, 1.6
"""The range of ``max_stretch``; an unset rule reads as the shortest."""

FORWARD_ONLY = {None, False}
"""``forward_only`` is unset or turned off."""

MAX_STOPS = {None, 1, 2, 3}
"""``max_stops`` is unset or 1 to 3."""


@pytest.mark.parametrize("seed", SEEDS_IN_USE)
def test_random_document_compiles(seed: int, random_document: RandomDocument) -> None:
    document = random_document(seed)

    instance = parse_spec(document).compile()

    assert isinstance(instance, LineHaulInstance)


@pytest.mark.parametrize("seed", SEEDS_IN_USE)
def test_random_document_has_three_to_seven_centers(
    seed: int, random_document: RandomDocument
) -> None:
    document = random_document(seed)

    centers = len(document["centers"])

    assert FEWEST_CENTERS <= centers <= MOST_CENTERS


@pytest.mark.parametrize("seed", SEEDS_IN_USE)
def test_random_document_names_centers_in_order(seed: int, random_document: RandomDocument) -> None:
    document = random_document(seed)

    names = list(map(itemgetter("id"), document["centers"]))

    assert names == list(CENTER_NAMES[: len(names)])


@pytest.mark.parametrize("seed", SEEDS_IN_USE)
def test_random_document_places_centers_in_the_latitude_band(
    seed: int, random_document: RandomDocument
) -> None:
    document = random_document(seed)

    latitudes = list(map(itemgetter("lat"), document["centers"]))

    assert SOUTHMOST <= min(latitudes) <= max(latitudes) <= NORTHMOST


@pytest.mark.parametrize("seed", SEEDS_IN_USE)
def test_random_document_places_centers_in_the_longitude_band(
    seed: int, random_document: RandomDocument
) -> None:
    document = random_document(seed)

    longitudes = list(map(itemgetter("lon"), document["centers"]))

    assert WESTMOST <= min(longitudes) <= max(longitudes) <= EASTMOST


@pytest.mark.parametrize("seed", SEEDS_IN_USE)
def test_random_document_bans_only_the_big_type(seed: int, random_document: RandomDocument) -> None:
    document = random_document(seed)

    bans = set(map(tuple, map(itemgetter("banned_vehicle_types"), document["centers"])))

    assert bans <= BANS


@pytest.mark.parametrize("seed", SEEDS_IN_USE)
def test_random_document_scales_haversine_by_the_road_factor(
    seed: int, random_document: RandomDocument
) -> None:
    document = random_document(seed)

    matrix = document["matrix"]

    assert matrix == MATRIX


@pytest.mark.parametrize("seed", SEEDS_IN_USE)
def test_random_document_has_one_of_the_two_vehicle_type_tables(
    seed: int, random_document: RandomDocument
) -> None:
    document = random_document(seed)

    vehicle_types = document["vehicle_types"]

    assert vehicle_types in (ONE_DIMENSION_TYPES, TWO_DIMENSION_TYPES)


@pytest.mark.parametrize("seed", SEEDS_IN_USE)
def test_random_document_has_zero_to_two_owned_offers(
    seed: int, random_document: RandomDocument
) -> None:
    document = random_document(seed)

    owned = len(document["fleet"]["owned"])

    assert FEWEST_OWNED <= owned <= MOST_OWNED


@pytest.mark.parametrize("seed", SEEDS_IN_USE)
def test_random_document_owned_offers_are_big_or_small(
    seed: int, random_document: RandomDocument
) -> None:
    document = random_document(seed)

    types = set(map(itemgetter("type"), document["fleet"]["owned"]))

    assert types <= OWNED_TYPES


@pytest.mark.parametrize("seed", SEEDS_IN_USE)
def test_random_document_owned_offers_have_one_or_two_vehicles(
    seed: int, random_document: RandomDocument
) -> None:
    document = random_document(seed)

    counts = set(map(itemgetter("count"), document["fleet"]["owned"]))

    assert counts <= OWNED_COUNTS


@pytest.mark.parametrize("seed", SEEDS_IN_USE)
def test_random_document_owned_fixed_costs_lie_in_their_range(
    seed: int, random_document: RandomDocument
) -> None:
    """An empty owned fleet reads as the cheapest cost, which lies in the range."""
    document = random_document(seed)

    costs = list(map(itemgetter("fixed"), map(itemgetter("cost"), document["fleet"]["owned"])))

    assert (
        CHEAPEST_OWNED_FIXED
        <= min(costs, default=CHEAPEST_OWNED_FIXED)
        <= max(costs, default=CHEAPEST_OWNED_FIXED)
        <= DEAREST_OWNED_FIXED
    )


@pytest.mark.parametrize("seed", SEEDS_IN_USE)
def test_random_document_owned_per_km_costs_lie_in_their_range(
    seed: int, random_document: RandomDocument
) -> None:
    """An empty owned fleet reads as the cheapest cost, which lies in the range."""
    document = random_document(seed)

    costs = list(map(itemgetter("per_km"), map(itemgetter("cost"), document["fleet"]["owned"])))

    assert (
        CHEAPEST_OWNED_PER_KM
        <= min(costs, default=CHEAPEST_OWNED_PER_KM)
        <= max(costs, default=CHEAPEST_OWNED_PER_KM)
        <= DEAREST_OWNED_PER_KM
    )


@pytest.mark.parametrize("seed", SEEDS_IN_USE)
def test_random_document_always_offers_spot_big_and_small(
    seed: int, random_document: RandomDocument
) -> None:
    document = random_document(seed)

    types = list(map(itemgetter("type"), document["fleet"]["spot"]))

    assert types == SPOT_TYPES


@pytest.mark.parametrize("seed", SEEDS_IN_USE)
def test_random_document_prices_spot_offers_at_fixed_costs(
    seed: int, random_document: RandomDocument
) -> None:
    document = random_document(seed)

    costs = list(map(itemgetter("cost"), document["fleet"]["spot"]))

    assert costs == SPOT_COSTS


@pytest.mark.parametrize("seed", SEEDS_IN_USE)
def test_random_document_limits_spot_small_to_one_to_three_if_at_all(
    seed: int, random_document: RandomDocument
) -> None:
    document = random_document(seed)

    available = document["fleet"]["spot"][1].get("available")

    assert available in SPOT_SMALL_AVAILABLE


@pytest.mark.parametrize("seed", SEEDS_IN_USE)
def test_random_document_calendar_spans_one_to_three_days(
    seed: int, random_document: RandomDocument
) -> None:
    document = random_document(seed)

    calendar = document["calendar"]

    assert calendar in CALENDARS


@pytest.mark.parametrize("seed", SEEDS_IN_USE)
def test_random_document_has_three_to_nineteen_demands(
    seed: int, random_document: RandomDocument
) -> None:
    document = random_document(seed)

    demands = len(document["demand"])

    assert FEWEST_DEMANDS <= demands <= MOST_DEMANDS


@pytest.mark.parametrize("seed", SEEDS_IN_USE)
def test_random_document_dates_demand_within_the_calendar(
    seed: int, random_document: RandomDocument
) -> None:
    document = random_document(seed)

    dates = list(map(itemgetter("date"), document["demand"]))

    assert CALENDAR_START <= min(dates) <= max(dates) <= document["calendar"]["end"]


@pytest.mark.parametrize("seed", SEEDS_IN_USE)
def test_random_document_measures_demand_in_the_capacity_dimensions(
    seed: int, random_document: RandomDocument
) -> None:
    document = random_document(seed)

    dimensions = set(map(frozenset, map(itemgetter("quantity"), document["demand"])))

    assert dimensions == {frozenset(document["vehicle_types"][0]["capacity"])}


@pytest.mark.parametrize("seed", SEEDS_IN_USE)
def test_random_document_demand_desi_fits_the_split_rule(
    seed: int, random_document: RandomDocument
) -> None:
    document = random_document(seed)

    desi = list(map(itemgetter("desi"), map(itemgetter("quantity"), document["demand"])))

    assert (
        LIGHTEST_DESI <= min(desi) <= max(desi) <= HEAVIEST_DESI[document["rules"]["split_demand"]]
    )


@pytest.mark.parametrize("seed", SEEDS_IN_USE)
def test_random_document_min_fill_is_one_of_three_levels(
    seed: int, random_document: RandomDocument
) -> None:
    document = random_document(seed)

    min_fill = document["rules"]["min_fill"]

    assert min_fill in MIN_FILLS


@pytest.mark.parametrize("seed", SEEDS_IN_USE)
def test_random_document_sets_enabled_and_only_known_stopover_rules(
    seed: int, random_document: RandomDocument
) -> None:
    document = random_document(seed)

    rules = document["rules"]["stopover"].keys()

    assert rules - OPTIONAL_STOPOVER_RULES == {"enabled"}


@pytest.mark.parametrize("seed", SEEDS_IN_USE)
def test_random_document_max_stretch_lies_in_its_range(
    seed: int, random_document: RandomDocument
) -> None:
    document = random_document(seed)

    stretch = document["rules"]["stopover"].get("max_stretch", SHORTEST_STRETCH)

    assert SHORTEST_STRETCH <= stretch <= LONGEST_STRETCH


@pytest.mark.parametrize("seed", SEEDS_IN_USE)
def test_random_document_only_turns_forward_only_off(
    seed: int, random_document: RandomDocument
) -> None:
    document = random_document(seed)

    forward_only = document["rules"]["stopover"].get("forward_only")

    assert forward_only in FORWARD_ONLY


@pytest.mark.parametrize("seed", SEEDS_IN_USE)
def test_random_document_max_stops_is_one_to_three_if_set(
    seed: int, random_document: RandomDocument
) -> None:
    document = random_document(seed)

    max_stops = document["rules"]["stopover"].get("max_stops")

    assert max_stops in MAX_STOPS


def test_random_document_is_the_same_for_the_same_seed(random_document: RandomDocument) -> None:
    first = random_document(SEED)

    second = random_document(SEED)

    assert second == first


def test_random_document_changes_with_the_seed(random_document: RandomDocument) -> None:
    first = random_document(SEED)

    other = random_document(OTHER_SEED)

    assert other != first


@pytest.mark.parametrize(
    ("module", "markers"),
    [
        ("tests/test_aco.py", ["unit"]),
        ("tests/spec/test_models.py", ["unit"]),
        ("tests/spec/test_loader.py", ["integration"]),
        ("tests/test_planio.py", ["integration"]),
        ("tests/test_cli.py", ["system"]),
        ("tests/fuzz/test_fuzz_inputs.py", ["unit", "fuzz"]),
        ("tests/model/test_ledger_model.py", ["unit", "model"]),
        ("tests/performance/test_solver_time.py", ["system", "performance"]),
    ],
)
def test_suite_markers_follow_the_level_table(
    module: str, markers: list[str], suite_markers: SuiteMarkers
) -> None:
    path = Path(module)

    found = suite_markers(path)

    assert found == markers


@pytest.mark.parametrize(
    ("module", "markers"),
    [
        ("model/pirilib/tests/test_aco.py", ["unit"]),
        ("fuzz/pirilib/tests/spec/test_loader.py", ["integration"]),
        ("performance/pirilib/tests/test_cli.py", ["system"]),
        ("performance/pirilib/tests/test_aco.py", ["unit"]),
    ],
)
def test_suite_markers_ignore_directories_above_the_module(
    module: str, markers: list[str], suite_markers: SuiteMarkers
) -> None:
    """A checkout under a directory named like a technique must not mark every test with it.

    Regression test: the markers once came from every part of the module's path, so a
    repository under ``model``, ``fuzz`` or ``performance`` gave all its tests that marker
    (and skipped all of them for ``performance``).
    """
    path = Path(module)

    found = suite_markers(path)

    assert found == markers


def test_collection_marks_this_module_as_unit(request: pytest.FixtureRequest) -> None:
    node = request.node

    marker = node.get_closest_marker("unit")

    assert marker is not None

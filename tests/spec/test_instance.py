"""Tests of compiling a problem document into its read-only array form.

Most tests compile the ``document`` fixture of ``tests/conftest.py``, whose docstring gives
every value it holds. Its names become positions in the order the document lists them:
centers ``A``, ``B``, ``C`` are 0, 1, 2, vehicle types ``big``, ``small`` are 0, 1, and the
calendar days 2026-01-05 and 2026-01-06 are days 0 and 1. The fixture is a 0.1 document, so
it sets none of the time fields; their arrays hold the documented "not given" values.

The last tests compile the bundled example (``examples/linehaul_mini``) through the public
loader, the way a user does; their expected values are read off the example's files.
"""

import dataclasses
import datetime as dt
import math
from operator import attrgetter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pirilib import load_spec
from pirilib.spec import UNLIMITED, LineHaulInstance, parse_spec

EARTH_RADIUS_KM = 6371.0
"""The mean Earth radius the haversine distances use."""

ONE_DEGREE_KM = EARTH_RADIUS_KM * math.pi / 180.0
"""The length of one degree of a great circle, worked out by hand from the radius."""

DEGREES_APART = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
"""Degrees between the fixture's centers, which lie on the equator at longitude 0, 1 and 2."""

HAVERSINE_KM = ONE_DEGREE_KM * DEGREES_APART
"""The great circle distances between the fixture's centers."""

ROAD_FACTOR = 1.3
"""A road to straight line ratio other than the default 1."""

RELATIVE_TOLERANCE = 1e-12
"""Distances agree with the hand worked values up to floating point rounding."""

VALUES = {
    "centers": ("A", "B", "C"),
    "days": (dt.date(2026, 1, 5), dt.date(2026, 1, 6)),
    "dimensions": ("desi",),
    "primary_dimension": 0,
    "vehicle_types": ("big", "small"),
    "demand_ids": ("D000001", "D000002", "D000003"),
    "timed": False,
}
"""The fixture's plain (not array) values: names in document order, generated demand ids,
the only dimension as the primary one, and no time dependent cost or due time."""

ARRAY_VALUES = {
    "capacity": [[1000.0], [400.0]],
    "banned": [[False, False], [False, False], [True, False]],
    "spot_type": [0, 1],
    "spot_fixed": [300.0, 150.0],
    "spot_per_km": [2.0, 1.5],
    "spot_available": [UNLIMITED, 3],
    "spot_per_hour": [0.0, 0.0],
    "owned_type": [0],
    "owned_origin": [0],
    "owned_destination": [1],
    "owned_count": [1],
    "owned_fixed": [100.0],
    "owned_per_km": [1.0],
    "owned_per_hour": [0.0],
    "owned_mandatory": [True],
    "owned_stopovers": [False],
    "demand_origin": [0, 0, 1],
    "demand_destination": [1, 2, 0],
    "demand_day": [0, 0, 1],
    "demand_quantity": [[900.0], [250.0], [120.0]],
    "speed_kmh": [math.nan, math.nan],
    "handling_capacity": [math.inf, math.inf, math.inf],
    "vehicle_limit": [[UNLIMITED, UNLIMITED], [UNLIMITED, UNLIMITED], [UNLIMITED, UNLIMITED]],
    "demand_ready": [0.0, 0.0, 1440.0],
    "demand_due": [math.inf, math.inf, math.inf],
}
"""The fixture's arrays, by position. Only ``C`` bans ``big``; the unlimited spot ``big`` reads
:data:`UNLIMITED`; ``owned_mandatory`` and ``owned_stopovers`` are the schema defaults. The
time fields are not given: no speed (NaN), no per hour cost, no handling or vehicle limit
(inf, :data:`UNLIMITED`), ready at 00:00 of the demand's day (1440 minutes is one day) and no
due time (inf). ``distance_km`` has tests of its own below."""

ARRAY_DTYPES = {
    "distance_km": np.float64,
    "capacity": np.float64,
    "banned": np.bool_,
    "spot_type": np.int64,
    "spot_fixed": np.float64,
    "spot_per_km": np.float64,
    "spot_available": np.int64,
    "owned_type": np.int64,
    "owned_origin": np.int64,
    "owned_destination": np.int64,
    "owned_count": np.int64,
    "owned_fixed": np.float64,
    "owned_per_km": np.float64,
    "owned_mandatory": np.bool_,
    "owned_stopovers": np.bool_,
    "demand_origin": np.int64,
    "demand_destination": np.int64,
    "demand_day": np.int64,
    "demand_quantity": np.float64,
    "speed_kmh": np.float64,
    "spot_per_hour": np.float64,
    "owned_per_hour": np.float64,
    "handling_capacity": np.float64,
    "vehicle_limit": np.int64,
    "demand_ready": np.float64,
    "demand_due": np.float64,
}
"""Every array field of :class:`LineHaulInstance` and the dtype its annotation states."""

OTHER_FIELDS = (
    "centers",
    "days",
    "dimensions",
    "primary_dimension",
    "vehicle_types",
    "demand_ids",
    "rules",
    "timed",
)
"""Every field of :class:`LineHaulInstance` that is not an array."""

EMPTY_DEMAND_SHAPES = {
    "demand_ids": (0,),
    "demand_origin": (0,),
    "demand_destination": (0,),
    "demand_day": (0,),
    "demand_ready": (0,),
    "demand_due": (0,),
    "demand_quantity": (0, 1),
}
"""Shapes of the demand fields without demand; quantities keep their one dimension column."""

EXAMPLE_REPR = (
    "LineHaulInstance(centers=6, days=2, vehicle_types=3, spot_offers=3, owned_groups=1, "
    "demands=18, dimensions=('desi',))"
)
"""The example has 6 centers, a 2 day calendar, 3 vehicle types each offered on the spot
market, 1 owned group and 18 demand rows in one dimension."""

EXAMPLE_BANNED = [
    [False, False, False],
    [False, False, False],
    [False, False, False],
    [False, False, False],
    [False, False, False],
    [True, False, False],
]
"""Bans of the example's centers (İstanbul, Kocaeli, Yalova, Bilecik, Eskişehir, Tekirdağ) on
its vehicle types (``tir``, ``kamyon``, ``kamyonet``): only Tekirdağ bans ``tir``."""

EXAMPLE_TOTAL_DESI = 156_696.0
"""The sum of the ``Desi`` column of ``examples/linehaul_mini/demand.csv``."""


@pytest.fixture
def example_instance(example_path: Path) -> LineHaulInstance:
    """The bundled example, loaded from its files and compiled."""
    return load_spec(example_path).compile()


@pytest.mark.parametrize(("field", "expected"), VALUES.items(), ids=VALUES)
def test_plain_field_holds_the_document_value(
    field: str, expected: object, document: dict[str, Any]
) -> None:
    spec = parse_spec(document)

    instance = spec.compile()

    assert getattr(instance, field) == expected


@pytest.mark.parametrize(("field", "expected"), ARRAY_VALUES.items(), ids=ARRAY_VALUES)
def test_array_field_holds_the_document_value_by_position(
    field: str, expected: list[Any], document: dict[str, Any]
) -> None:
    spec = parse_spec(document)

    instance = spec.compile()

    np.testing.assert_array_equal(getattr(instance, field), expected)


def test_rules_are_the_document_rules(document: dict[str, Any]) -> None:
    spec = parse_spec(document)

    instance = spec.compile()

    assert instance.rules == spec.rules


@pytest.mark.parametrize(("field", "dtype"), ARRAY_DTYPES.items(), ids=ARRAY_DTYPES)
def test_array_field_is_an_array_of_its_dtype(
    field: str, dtype: type[np.generic], document: dict[str, Any]
) -> None:
    spec = parse_spec(document)

    array = getattr(spec.compile(), field)

    assert (type(array), array.dtype) == (np.ndarray, np.dtype(dtype))


@pytest.mark.parametrize("field", ARRAY_DTYPES)
def test_array_field_is_read_only(field: str, document: dict[str, Any]) -> None:
    spec = parse_spec(document)

    array = getattr(spec.compile(), field)

    assert array.flags.writeable is False


@pytest.mark.parametrize("field", OTHER_FIELDS)
def test_other_field_is_not_an_array(field: str, document: dict[str, Any]) -> None:
    spec = parse_spec(document)

    value = getattr(spec.compile(), field)

    assert not isinstance(value, np.ndarray)


def test_array_and_other_fields_are_all_the_fields() -> None:
    fields = set(map(attrgetter("name"), dataclasses.fields(LineHaulInstance)))

    assert fields == set(ARRAY_DTYPES) | set(OTHER_FIELDS)


def test_writing_into_an_array_raises(document: dict[str, Any]) -> None:
    instance = parse_spec(document).compile()

    with pytest.raises(ValueError, match="read-only"):
        instance.distance_km[0, 1] = 0.0


def test_haversine_distances_are_great_circle_kilometres(document: dict[str, Any]) -> None:
    spec = parse_spec(document)

    km = spec.compile().distance_km

    np.testing.assert_allclose(km, HAVERSINE_KM, rtol=RELATIVE_TOLERANCE, atol=0.0)


def test_haversine_distances_are_exactly_symmetric(document: dict[str, Any]) -> None:
    spec = parse_spec(document)

    km = spec.compile().distance_km

    np.testing.assert_array_equal(km, km.T)


def test_distance_factor_scales_every_haversine_distance(document: dict[str, Any]) -> None:
    document["matrix"]["factor"] = ROAD_FACTOR
    spec = parse_spec(document)

    km = spec.compile().distance_km

    np.testing.assert_allclose(km, ROAD_FACTOR * HAVERSINE_KM, rtol=RELATIVE_TOLERANCE, atol=0.0)


def test_explicit_matrix_fills_missing_reverse_arcs_and_applies_the_factor(
    document: dict[str, Any],
) -> None:
    """``B`` to ``A`` is missing and mirrors ``A`` to ``B``; ``C`` to ``A`` is given (25 km), so
    the mirror of ``A`` to ``C`` (20 km) does not replace it; the diagonal is 0; every given
    or mirrored length is doubled by the factor 2."""
    document["matrix"] = {
        "source": "explicit",
        "symmetric": True,
        "factor": 2.0,
        "arcs": [
            {"origin": "A", "destination": "B", "km": 10.0},
            {"origin": "A", "destination": "C", "km": 20.0},
            {"origin": "C", "destination": "A", "km": 25.0},
            {"origin": "B", "destination": "C", "km": 15.0},
        ],
    }
    spec = parse_spec(document)

    km = spec.compile().distance_km

    assert km.tolist() == [[0.0, 20.0, 40.0], [20.0, 0.0, 30.0], [50.0, 30.0, 0.0]]


@pytest.mark.parametrize(("field", "shape"), EMPTY_DEMAND_SHAPES.items(), ids=EMPTY_DEMAND_SHAPES)
def test_empty_demand_gives_empty_demand_fields(
    field: str, shape: tuple[int, ...], document: dict[str, Any]
) -> None:
    document["demand"] = []
    spec = parse_spec(document)

    instance = spec.compile()

    assert np.shape(getattr(instance, field)) == shape


def test_example_repr_counts_its_parts(example_instance: LineHaulInstance) -> None:
    text = repr(example_instance)

    assert text == EXAMPLE_REPR


def test_example_bans_come_from_the_centers_table(example_instance: LineHaulInstance) -> None:
    banned = example_instance.banned

    assert banned.tolist() == EXAMPLE_BANNED


def test_example_demand_quantities_come_from_the_demand_table(
    example_instance: LineHaulInstance,
) -> None:
    total = example_instance.demand_quantity.sum()

    assert total == pytest.approx(EXAMPLE_TOTAL_DESI)

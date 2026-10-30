"""Tests of :mod:`pirilib.geo`, the great-circle distance matrix.

The expected behaviour comes from the docstring of :func:`pirilib.geo.haversine_matrix`
(``lat`` and ``lon`` are 1-D sequences of equal length in decimal degrees, entry ``[i, j]`` is
the great-circle distance in km from point ``i`` to point ``j``), from the mean earth radius of
6371 km and from spherical geometry, which gives the expected distances independently of the
code under test.

Input classes: valid coordinates (no point, one point, several points; coincident points,
points on one meridian, on the equator, across the antimeridian, over a pole, antipodes) and
invalid shapes (unequal lengths, 0-D, 2-D), which raise :class:`ValueError`. Coordinates outside
``[-90, 90]`` x ``[-180, 180]`` and NaN are left out on purpose: the docstring does not define
them and :class:`pirilib.spec.LineHaulSpec` rejects them before they reach this function.
"""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.typing import ArrayLike, NDArray

from pirilib.geo import EARTH_RADIUS_KM, haversine_matrix

Coordinates = tuple[NDArray[np.float64], NDArray[np.float64]]
"""A ``(lat, lon)`` pair of equal-length coordinate arrays in decimal degrees."""

MEAN_EARTH_RADIUS_KM = 6371.0
"""The mean earth radius, written out rather than taken from the code under test."""

DEGREE_KM = MEAN_EARTH_RADIUS_KM * math.pi / 180.0
"""The length of one degree of a great circle on that sphere."""

RELATIVE_TOLERANCE = 1e-12
"""Relative error allowed on a nonzero distance: a few thousand float64 roundings."""

ZERO_KM = 1e-9
"""Absolute error, in km (one micrometre), allowed where the exact distance is 0.

The degree to radian conversion is not exact, so two writings of one place (``lon`` 180 and
-180, or two longitudes at a pole) come out about ``1e-12`` km apart instead of 0.
"""

HALF_CIRCUMFERENCE_SLACK_KM = 1e-9
"""Rounding allowed above half a great circle, the longest possible distance."""

TRIANGLE_SLACK_KM = 1e-3
"""Rounding allowed in the triangle inequality, in km (one metre).

Near antipodes ``arcsin(sqrt(a))`` has an infinite slope, so a rounding of ``a`` by one unit in
the last place moves the distance by about ``R * sqrt(eps)``, roughly ``1e-4`` km.
"""

MIRROR_SLACK_KM = 1e-9
"""Rounding allowed between the distances of a point set and of its mirror image."""

MAX_POINTS = 8
"""Largest number of points drawn by the property tests; the triangle check is cubic in it."""

coordinates = st.integers(1, MAX_POINTS).flatmap(
    lambda size: st.tuples(
        arrays(np.float64, size, elements=st.floats(-90.0, 90.0)),
        arrays(np.float64, size, elements=st.floats(-180.0, 180.0)),
    )
)
"""``(lat, lon)`` arrays of 1 to ``MAX_POINTS`` points anywhere in the valid coordinate range."""


@pytest.mark.parametrize(
    ("lat", "lon", "expected_km"),
    [
        pytest.param([0.0, 0.0], [0.0, 1.0], DEGREE_KM, id="one-degree-of-equator"),
        pytest.param([0.0, 0.0], [0.0, 90.0], 90.0 * DEGREE_KM, id="quarter-of-equator"),
        pytest.param([-10.0, -40.0], [30.0, 30.0], 30.0 * DEGREE_KM, id="southern-meridian"),
        pytest.param([0.0, 90.0], [0.0, 0.0], 90.0 * DEGREE_KM, id="equator-to-north-pole"),
        pytest.param([0.0, 0.0], [179.0, -179.0], 2.0 * DEGREE_KM, id="across-antimeridian"),
        pytest.param([60.0, 60.0], [0.0, 180.0], 60.0 * DEGREE_KM, id="over-north-pole"),
        pytest.param([0.0, 45.0], [0.0, 45.0], 60.0 * DEGREE_KM, id="right-spherical-triangle"),
        pytest.param([0.0, 0.0], [0.0, 180.0], 180.0 * DEGREE_KM, id="antipodes-on-equator"),
        pytest.param([90.0, -90.0], [0.0, 0.0], 180.0 * DEGREE_KM, id="pole-to-pole"),
        pytest.param([45.0, -45.0], [0.0, -180.0], 180.0 * DEGREE_KM, id="antipodes-off-axis"),
    ],
)
def test_distance_between_two_points_is_their_great_circle_distance(
    lat: list[float], lon: list[float], expected_km: float
) -> None:
    """Entry ``[0, 1]`` is the great-circle distance on the mean earth sphere.

    Each expected value is an arc of a great circle in degrees: along the equator or a
    meridian, the difference of the coordinates; over the pole, ``(90 - 60) * 2``; from
    ``(0, 0)`` to ``(45, 45)``, the right spherical triangle ``cos c = cos 45 * cos 45 = 1/2``
    gives 60. For the off-axis antipodes the haversine term rounds to just above 1, and the
    distance must still be half a great circle, not NaN.
    """
    km = haversine_matrix(lat, lon)

    assert km[0, 1] == pytest.approx(expected_km, rel=RELATIVE_TOLERANCE)


@pytest.mark.parametrize(
    ("lat", "lon"),
    [
        pytest.param([0.0, 0.0], [180.0, -180.0], id="antimeridian-written-twice"),
        pytest.param([90.0, 90.0], [0.0, 120.0], id="north-pole-at-two-longitudes"),
        pytest.param([-90.0, -90.0], [-180.0, 45.0], id="south-pole-at-two-longitudes"),
    ],
)
def test_one_place_written_two_ways_is_zero_km_away(lat: list[float], lon: list[float]) -> None:
    """Two coordinate pairs that name the same place on the sphere are 0 km apart."""
    km = haversine_matrix(lat, lon)

    assert km[0, 1] == pytest.approx(0.0, abs=ZERO_KM)


def test_matrix_entry_i_j_is_the_distance_from_point_i_to_point_j() -> None:
    """Every entry of a three-point matrix, with three different distances, is in its place.

    The points are on the equator at longitudes 0, 1 and 180, so the pairs are 1, 180 and
    179 degrees apart.
    """
    lat = [0.0, 0.0, 0.0]
    lon = [0.0, 1.0, 180.0]
    expected = DEGREE_KM * np.array(
        [
            [0.0, 1.0, 180.0],
            [1.0, 0.0, 179.0],
            [180.0, 179.0, 0.0],
        ]
    )

    km = haversine_matrix(lat, lon)

    np.testing.assert_allclose(km, expected, rtol=RELATIVE_TOLERANCE, atol=0.0)


def test_radius_argument_sets_the_sphere() -> None:
    """On the unit sphere a quarter of the equator is ``pi / 2`` long."""
    lat = [0.0, 0.0]
    lon = [0.0, 90.0]

    km = haversine_matrix(lat, lon, radius_km=1.0)

    assert km[0, 1] == pytest.approx(math.pi / 2.0, rel=RELATIVE_TOLERANCE)


def test_earth_radius_is_the_mean_radius() -> None:
    """The exported default radius is the mean earth radius, 6371 km."""
    radius = EARTH_RADIUS_KM

    assert radius == MEAN_EARTH_RADIUS_KM


def test_no_points_give_an_empty_square_matrix() -> None:
    """Empty sequences are a valid input with no pair, so the result has shape ``(0, 0)``."""
    km = haversine_matrix([], [])

    assert km.shape == (0, 0)


def test_one_point_gives_a_single_zero() -> None:
    """A single point has one pair, itself, at distance 0."""
    km = haversine_matrix([41.0], [29.0])

    np.testing.assert_array_equal(km, np.array([[0.0]]))


def test_coincident_points_are_exactly_zero_km_apart() -> None:
    """Two points with the same coordinates are 0 km apart off the diagonal too, not nearly 0."""
    km = haversine_matrix([41.0, 41.0], [29.0, 29.0])

    np.testing.assert_array_equal(km, np.zeros((2, 2)))


def test_integer_degrees_give_a_float64_matrix() -> None:
    """Integer coordinates are accepted and the result is a float64 matrix."""
    km = haversine_matrix([0, 0], [0, 1])

    assert km.dtype == np.float64


def test_integer_degrees_mean_the_same_degrees_as_floats() -> None:
    """One integer degree along the equator is one degree of a great circle."""
    km = haversine_matrix([0, 0], [0, 1])

    assert km[0, 1] == pytest.approx(DEGREE_KM, rel=RELATIVE_TOLERANCE)


@pytest.mark.parametrize(
    ("lat", "lon"),
    [
        pytest.param([0.0, 1.0], [0.0], id="fewer-longitudes"),
        pytest.param([0.0], [0.0, 1.0], id="fewer-latitudes"),
        pytest.param(0.0, 0.0, id="scalars"),
        pytest.param([[0.0, 1.0]], [[0.0, 1.0]], id="two-dimensional"),
    ],
)
def test_rejects_coordinates_that_are_not_two_equal_1d_sequences(
    lat: ArrayLike, lon: ArrayLike
) -> None:
    """``lat`` and ``lon`` of unequal length or not 1-D raise a ValueError that says so."""
    with pytest.raises(ValueError, match="must be 1-D sequences of equal length"):
        haversine_matrix(lat, lon)


@given(coordinates)
def test_matrix_has_one_row_and_one_column_per_point(points: Coordinates) -> None:
    """The result is square, with one row and one column per point."""
    lat, lon = points

    km = haversine_matrix(lat, lon)

    assert km.shape == (lat.size, lat.size)


@given(coordinates)
def test_distances_are_not_negative(points: Coordinates) -> None:
    """No distance is below 0."""
    lat, lon = points

    km = haversine_matrix(lat, lon)

    assert km.min() >= 0.0


@given(coordinates)
def test_distances_are_at_most_half_a_great_circle(points: Coordinates) -> None:
    """No two points are further apart than antipodes, ``pi`` times the earth radius."""
    lat, lon = points

    km = haversine_matrix(lat, lon)

    assert km.max() <= MEAN_EARTH_RADIUS_KM * math.pi + HALF_CIRCUMFERENCE_SLACK_KM


@given(coordinates)
def test_every_point_is_exactly_zero_km_from_itself(points: Coordinates) -> None:
    """The diagonal is exactly 0 for any points."""
    lat, lon = points

    km = haversine_matrix(lat, lon)

    np.testing.assert_array_equal(km.diagonal(), np.zeros(lat.size))


@given(coordinates)
def test_distance_from_i_to_j_equals_distance_from_j_to_i(points: Coordinates) -> None:
    """The matrix is exactly symmetric."""
    lat, lon = points

    km = haversine_matrix(lat, lon)

    np.testing.assert_array_equal(km, km.T)


@given(coordinates)
def test_detour_through_a_third_point_is_never_shorter(points: Coordinates) -> None:
    """Triangle inequality: ``km[i, j] <= km[i, k] + km[k, j]`` for every ``i``, ``j``, ``k``.

    ``excess[i, k, j]`` is ``km[i, j] - (km[i, k] + km[k, j])``; its largest value is the worst
    violation.
    """
    lat, lon = points

    km = haversine_matrix(lat, lon)

    excess = km[:, None, :] - (km[:, :, None] + km[None, :, :])
    assert excess.max() <= TRIANGLE_SLACK_KM


@given(coordinates)
def test_mirroring_longitudes_keeps_the_distances(points: Coordinates) -> None:
    """Reflecting every point across the prime meridian is an isometry of the sphere."""
    lat, lon = points
    original = haversine_matrix(lat, lon)

    mirrored = haversine_matrix(lat, -lon)

    np.testing.assert_allclose(mirrored, original, rtol=0.0, atol=MIRROR_SLACK_KM)


@given(coordinates)
def test_mirroring_latitudes_keeps_the_distances(points: Coordinates) -> None:
    """Reflecting every point across the equator is an isometry of the sphere."""
    lat, lon = points
    original = haversine_matrix(lat, lon)

    mirrored = haversine_matrix(-lat, lon)

    np.testing.assert_allclose(mirrored, original, rtol=0.0, atol=MIRROR_SLACK_KM)

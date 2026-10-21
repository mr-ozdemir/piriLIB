import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pirilib.geo import EARTH_RADIUS_KM, haversine_matrix

points = st.lists(
    st.tuples(st.floats(-90.0, 90.0), st.floats(-180.0, 180.0)), min_size=1, max_size=8
)


def test_known_distances() -> None:
    km = haversine_matrix([0.0, 0.0, 0.0, 90.0], [0.0, 1.0, 180.0, 0.0])
    assert km[0, 1] == pytest.approx(EARTH_RADIUS_KM * math.pi / 180.0, rel=1e-12)
    assert km[0, 2] == pytest.approx(EARTH_RADIUS_KM * math.pi, rel=1e-12)
    assert km[0, 3] == pytest.approx(EARTH_RADIUS_KM * math.pi / 2.0, rel=1e-12)


def test_rejects_mismatched_input() -> None:
    with pytest.raises(ValueError, match="equal length"):
        haversine_matrix([0.0, 1.0], [0.0])


@given(points)
def test_is_a_metric(coordinates: list[tuple[float, float]]) -> None:
    lat, lon = zip(*coordinates, strict=True)
    km = haversine_matrix(lat, lon)
    assert km.shape == (len(coordinates), len(coordinates))
    assert np.all(km >= 0.0)
    assert np.all(km <= EARTH_RADIUS_KM * math.pi + 1e-9)
    assert np.all(km.diagonal() == 0.0)
    assert np.array_equal(km, km.T)
    assert np.all(km[:, None, :] <= km[:, :, None] + km[None, :, :] + 1e-3)

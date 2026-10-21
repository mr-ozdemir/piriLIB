"""Great-circle distances between coordinates."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

EARTH_RADIUS_KM = 6371.0


def haversine_matrix(
    lat: ArrayLike, lon: ArrayLike, radius_km: float = EARTH_RADIUS_KM
) -> NDArray[np.float64]:
    """Return the great-circle distance in km between every pair of points.

    ``lat`` and ``lon`` are 1-D sequences in decimal degrees. Entry ``[i, j]`` of
    the result is the distance from point ``i`` to point ``j``.
    """
    phi = np.radians(np.asarray(lat, dtype=np.float64))
    lam = np.radians(np.asarray(lon, dtype=np.float64))
    if phi.ndim != 1 or phi.shape != lam.shape:
        raise ValueError("lat and lon must be 1-D sequences of equal length")
    half_dphi = (phi[:, None] - phi[None, :]) / 2.0
    half_dlam = (lam[:, None] - lam[None, :]) / 2.0
    a = np.sin(half_dphi) ** 2 + np.outer(np.cos(phi), np.cos(phi)) * np.sin(half_dlam) ** 2
    distance = np.asarray(2.0 * radius_km * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))), np.float64)
    np.fill_diagonal(distance, 0.0)
    return distance

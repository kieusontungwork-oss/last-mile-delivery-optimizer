"""Geospatial utility functions."""

import numpy as np


EARTH_RADIUS_KM = 6371.0


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute haversine distance in km between two points."""
    return float(haversine_vectorized(
        np.array([lat1]), np.array([lon1]),
        np.array([lat2]), np.array([lon2]),
    )[0])


def haversine_vectorized(
    lats1: np.ndarray, lons1: np.ndarray,
    lats2: np.ndarray, lons2: np.ndarray,
) -> np.ndarray:
    """Vectorized haversine distance in km."""
    lat1_r = np.radians(lats1)
    lat2_r = np.radians(lats2)
    dlat = np.radians(lats2 - lats1)
    dlon = np.radians(lons2 - lons1)

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute initial bearing in degrees from point 1 to point 2."""
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlon_r = np.radians(lon2 - lon1)

    x = np.sin(dlon_r) * np.cos(lat2_r)
    y = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon_r)
    return float(np.degrees(np.arctan2(x, y)) % 360)


def manhattan_ratio(road_distance: float, haversine_distance: float) -> float:
    """Compute ratio of road distance to haversine distance.

    Returns 1.0 if haversine_distance is near zero to avoid division by zero.
    """
    if haversine_distance < 1e-6:
        return 1.0
    return road_distance / haversine_distance

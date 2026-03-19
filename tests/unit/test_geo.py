"""Tests for src/utils/geo.py."""

import numpy as np
import pytest

from src.utils.geo import bearing, haversine, haversine_vectorized, manhattan_ratio


class TestHaversine:
    def test_known_distance_nyc_to_la(self):
        """NYC (40.7128, -74.0060) to LA (34.0522, -118.2437) ~3940 km."""
        d = haversine(40.7128, -74.0060, 34.0522, -118.2437)
        assert 3900 < d < 4000

    def test_same_point_is_zero(self):
        d = haversine(40.7128, -74.0060, 40.7128, -74.0060)
        assert d == pytest.approx(0, abs=1e-6)

    def test_short_manhattan_distance(self):
        """Times Square to Grand Central ~1 km."""
        d = haversine(40.7580, -73.9855, 40.7527, -73.9772)
        assert 0.5 < d < 2.0

    def test_vectorized_matches_scalar(self):
        lats1 = np.array([40.7128, 40.7580])
        lons1 = np.array([-74.0060, -73.9855])
        lats2 = np.array([34.0522, 40.7527])
        lons2 = np.array([-118.2437, -73.9772])

        d_vec = haversine_vectorized(lats1, lons1, lats2, lons2)
        d1 = haversine(40.7128, -74.0060, 34.0522, -118.2437)
        d2 = haversine(40.7580, -73.9855, 40.7527, -73.9772)

        assert d_vec[0] == pytest.approx(d1, rel=1e-6)
        assert d_vec[1] == pytest.approx(d2, rel=1e-6)


class TestBearing:
    def test_north(self):
        """Point due north should have bearing ~0."""
        b = bearing(40.0, -74.0, 41.0, -74.0)
        assert b == pytest.approx(0, abs=1)

    def test_east(self):
        """Point due east should have bearing ~90."""
        b = bearing(40.0, -74.0, 40.0, -73.0)
        assert 89 < b < 91

    def test_south(self):
        b = bearing(41.0, -74.0, 40.0, -74.0)
        assert 179 < b < 181


class TestManhattanRatio:
    def test_normal_ratio(self):
        assert manhattan_ratio(5.0, 3.0) == pytest.approx(5.0 / 3.0)

    def test_zero_haversine(self):
        assert manhattan_ratio(0.0, 0.0) == 1.0

    def test_near_zero_haversine(self):
        assert manhattan_ratio(1.0, 1e-9) == 1.0

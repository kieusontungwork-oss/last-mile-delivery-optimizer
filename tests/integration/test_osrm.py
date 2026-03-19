"""Integration tests requiring a running OSRM server."""

import numpy as np
import pytest

from src.optimization.osrm_client import OSRMClient

pytestmark = pytest.mark.integration


@pytest.fixture
def osrm():
    client = OSRMClient(base_url="http://localhost:5000")
    if not client.health_check():
        pytest.skip("OSRM server not available at localhost:5000")
    yield client
    client.close()


class TestOSRMClient:
    def test_health_check(self, osrm):
        assert osrm.health_check() is True

    def test_duration_matrix_shape(self, osrm):
        locations = [
            (40.7580, -73.9855),
            (40.7614, -73.9776),
            (40.7527, -73.9772),
        ]
        matrix = osrm.get_duration_matrix(locations)
        assert matrix.shape == (3, 3)

    def test_duration_matrix_diagonal_zero(self, osrm):
        locations = [
            (40.7580, -73.9855),
            (40.7614, -73.9776),
        ]
        matrix = osrm.get_duration_matrix(locations)
        assert matrix[0, 0] == pytest.approx(0, abs=1)
        assert matrix[1, 1] == pytest.approx(0, abs=1)

    def test_duration_matrix_positive_off_diagonal(self, osrm):
        locations = [
            (40.7580, -73.9855),
            (40.7128, -74.0060),
        ]
        matrix = osrm.get_duration_matrix(locations)
        assert matrix[0, 1] > 0
        assert matrix[1, 0] > 0

    def test_route_returns_geometry(self, osrm):
        route = osrm.get_route(
            origin=(40.7580, -73.9855),
            destination=(40.7128, -74.0060),
        )
        assert route["duration"] > 0
        assert route["distance"] > 0
        assert len(route["geometry"]) > 2

    def test_distance_and_duration_matrices(self, osrm):
        locations = [
            (40.7580, -73.9855),
            (40.7614, -73.9776),
        ]
        dur, dist = osrm.get_duration_and_distance_matrices(locations)
        assert dur.shape == (2, 2)
        assert dist.shape == (2, 2)
        assert dist[0, 1] > 0

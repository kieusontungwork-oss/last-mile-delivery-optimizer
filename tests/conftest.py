"""Shared pytest fixtures."""

import numpy as np
import pytest

from src.optimization.vrp_solver import Stop


@pytest.fixture
def sample_stops() -> list[Stop]:
    """5 Manhattan delivery stops."""
    return [
        Stop(id="S1", lat=40.7580, lng=-73.9855, demand=5),
        Stop(id="S2", lat=40.7614, lng=-73.9776, demand=3),
        Stop(id="S3", lat=40.7527, lng=-73.9772, demand=8),
        Stop(id="S4", lat=40.7489, lng=-73.9680, demand=4),
        Stop(id="S5", lat=40.7411, lng=-74.0018, demand=6),
    ]


@pytest.fixture
def sample_cost_matrix() -> np.ndarray:
    """6x6 cost matrix (depot + 5 stops), scaled by 100."""
    return np.array([
        [0, 300, 400, 350, 500, 600],
        [300, 0, 200, 250, 400, 500],
        [400, 200, 0, 150, 350, 450],
        [350, 250, 150, 0, 300, 400],
        [500, 400, 350, 300, 0, 350],
        [600, 500, 450, 400, 350, 0],
    ], dtype=int)

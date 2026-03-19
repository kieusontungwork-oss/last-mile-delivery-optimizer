"""Tests for API Pydantic schemas."""

import pytest
from pydantic import ValidationError

from api.schemas.optimization import (
    DeliveryStop,
    Location,
    OptimizeConfig,
    OptimizeRequest,
)


class TestLocation:
    def test_valid_location(self):
        loc = Location(lat=40.7484, lng=-73.9857, name="Depot")
        assert loc.lat == 40.7484

    def test_lat_out_of_range(self):
        with pytest.raises(ValidationError):
            Location(lat=91.0, lng=-73.9857)

    def test_lng_out_of_range(self):
        with pytest.raises(ValidationError):
            Location(lat=40.0, lng=-181.0)


class TestDeliveryStop:
    def test_default_values(self):
        stop = DeliveryStop(id="S1", lat=40.758, lng=-73.985)
        assert stop.demand == 1
        assert stop.service_time_minutes == 5
        assert stop.time_window is None


class TestOptimizeConfig:
    def test_defaults(self):
        config = OptimizeConfig()
        assert config.use_ml is True
        assert config.max_solve_time_seconds == 30
        assert config.solver.value == "pyvrp"


class TestOptimizeRequest:
    def test_valid_request(self):
        req = OptimizeRequest(
            depot=Location(lat=40.748, lng=-73.985),
            stops=[DeliveryStop(id="S1", lat=40.758, lng=-73.985)],
            vehicles=[{"id": "V1", "capacity": 100}],
        )
        assert len(req.stops) == 1

    def test_empty_stops_rejected(self):
        with pytest.raises(ValidationError):
            OptimizeRequest(
                depot=Location(lat=40.748, lng=-73.985),
                stops=[],
                vehicles=[{"id": "V1", "capacity": 100}],
            )

"""Integration tests for the FastAPI application."""

import time

import pytest
from fastapi.testclient import TestClient

from api.main import app

pytestmark = pytest.mark.integration

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "osrm_connected" in data
        assert "model_loaded" in data


class TestOptimizationEndpoints:
    def test_submit_returns_202(self):
        payload = {
            "depot": {"lat": 40.7484, "lng": -73.9857},
            "stops": [
                {"id": "S1", "lat": 40.758, "lng": -73.985, "demand": 5},
                {"id": "S2", "lat": 40.761, "lng": -73.977, "demand": 3},
            ],
            "vehicles": [{"id": "V1", "capacity": 100}],
            "config": {"use_ml": False, "max_solve_time_seconds": 5},
        }
        resp = client.post("/api/optimize", json=payload)
        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert data["status"] in ("pending", "running")

    def test_poll_nonexistent_job(self):
        resp = client.get("/api/optimize/nonexistent-id")
        assert resp.status_code == 404


class TestPredictionEndpoints:
    def test_predict_eta(self):
        payload = {
            "origin_lat": 40.758,
            "origin_lng": -73.985,
            "destination_lat": 40.712,
            "destination_lng": -74.006,
        }
        resp = client.post("/api/predict-eta", json=payload)
        # May return 503 if OSRM is not available
        assert resp.status_code in (200, 503)

    def test_cost_matrix(self):
        payload = {
            "locations": [
                {"lat": 40.758, "lng": -73.985},
                {"lat": 40.712, "lng": -74.006},
            ],
            "use_ml": False,
        }
        resp = client.post("/api/cost-matrix", json=payload)
        assert resp.status_code in (200, 503)

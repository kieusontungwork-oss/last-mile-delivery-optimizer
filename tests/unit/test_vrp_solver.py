"""Tests for src/optimization/vrp_solver.py."""

import numpy as np
import pytest

from src.optimization.vrp_solver import ORToolsSolver, PyVRPSolver, Stop


class TestPyVRPSolver:
    def test_single_stop(self):
        """Single stop should return one route with that stop."""
        stops = [Stop(id="A", lat=0, lng=0, demand=5)]
        matrix = np.array([[0, 100], [100, 0]], dtype=int)

        solver = PyVRPSolver()
        sol = solver.solve(
            cost_matrix=matrix, stops=stops,
            vehicle_capacity=10, num_vehicles=1, max_runtime=5,
        )
        assert sol.num_vehicles == 1
        assert sol.routes[0].stop_ids == ["A"]

    def test_all_stops_visited(self, sample_stops, sample_cost_matrix):
        solver = PyVRPSolver()
        sol = solver.solve(
            cost_matrix=sample_cost_matrix, stops=sample_stops,
            vehicle_capacity=30, num_vehicles=3, max_runtime=10,
        )
        visited = set()
        for route in sol.routes:
            visited.update(route.stop_ids)
        assert visited == {s.id for s in sample_stops}

    def test_no_duplicate_visits(self, sample_stops, sample_cost_matrix):
        solver = PyVRPSolver()
        sol = solver.solve(
            cost_matrix=sample_cost_matrix, stops=sample_stops,
            vehicle_capacity=30, num_vehicles=3, max_runtime=10,
        )
        all_ids = []
        for route in sol.routes:
            all_ids.extend(route.stop_ids)
        assert len(all_ids) == len(set(all_ids))

    def test_capacity_respected(self, sample_stops, sample_cost_matrix):
        capacity = 15
        solver = PyVRPSolver()
        sol = solver.solve(
            cost_matrix=sample_cost_matrix, stops=sample_stops,
            vehicle_capacity=capacity, num_vehicles=5, max_runtime=10,
        )
        stop_map = {s.id: s for s in sample_stops}
        for route in sol.routes:
            total_demand = sum(stop_map[sid].demand for sid in route.stop_ids)
            assert total_demand <= capacity, (
                f"Route {route.vehicle_id} has demand {total_demand} > capacity {capacity}"
            )

    def test_solution_feasible(self, sample_stops, sample_cost_matrix):
        solver = PyVRPSolver()
        sol = solver.solve(
            cost_matrix=sample_cost_matrix, stops=sample_stops,
            vehicle_capacity=50, num_vehicles=3, max_runtime=10,
        )
        assert sol.is_feasible


class TestORToolsSolver:
    def test_all_stops_visited(self, sample_stops, sample_cost_matrix):
        solver = ORToolsSolver()
        sol = solver.solve(
            cost_matrix=sample_cost_matrix, stops=sample_stops,
            vehicle_capacity=30, num_vehicles=3, max_runtime=10,
        )
        visited = set()
        for route in sol.routes:
            visited.update(route.stop_ids)
        assert visited == {s.id for s in sample_stops}

    def test_capacity_respected(self, sample_stops, sample_cost_matrix):
        capacity = 15
        solver = ORToolsSolver()
        sol = solver.solve(
            cost_matrix=sample_cost_matrix, stops=sample_stops,
            vehicle_capacity=capacity, num_vehicles=5, max_runtime=10,
        )
        stop_map = {s.id: s for s in sample_stops}
        for route in sol.routes:
            total_demand = sum(stop_map[sid].demand for sid in route.stop_ids)
            assert total_demand <= capacity

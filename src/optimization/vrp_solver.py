"""VRP solver wrappers for PyVRP and OR-Tools."""

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from pyvrp import Model
from pyvrp.stop import MaxRuntime

logger = logging.getLogger(__name__)


@dataclass
class Stop:
    """Delivery stop definition."""
    id: str
    lat: float
    lng: float
    demand: int = 1
    tw_early: int = 0  # seconds from start of planning horizon
    tw_late: int = 86400  # 24 hours (no constraint by default)
    service_time: int = 300  # 5 minutes default


@dataclass
class Route:
    """Single vehicle route result."""
    vehicle_id: str
    stop_ids: list[str] = field(default_factory=list)
    distance: float = 0.0
    duration: float = 0.0
    arrival_times: dict[str, float] = field(default_factory=dict)


@dataclass
class VRPSolution:
    """Complete VRP solution."""
    routes: list[Route] = field(default_factory=list)
    total_distance: float = 0.0
    total_duration: float = 0.0
    num_vehicles: int = 0
    solve_time: float = 0.0
    is_feasible: bool = False


class PyVRPSolver:
    """VRP solver using PyVRP (Hybrid Genetic Search)."""

    def solve(
        self,
        cost_matrix: np.ndarray,
        stops: list[Stop],
        depot_coords: tuple[float, float] = (0.0, 0.0),
        vehicle_capacity: int = 100,
        num_vehicles: int = 10,
        max_runtime: int = 30,
        scaling_factor: int = 100,
    ) -> VRPSolution:
        """Solve CVRPTW using PyVRP.

        Args:
            cost_matrix: NxN integer cost matrix (depot is index 0, stops are 1..N).
                Already scaled to integers.
            stops: List of delivery stops.
            depot_coords: (lat, lng) of the depot (used for coordinates, not routing).
            vehicle_capacity: Maximum capacity per vehicle.
            num_vehicles: Maximum number of vehicles available.
            max_runtime: Maximum solve time in seconds.
            scaling_factor: Factor used to scale costs (for converting back).

        Returns:
            VRPSolution with routes and metrics.
        """
        n_stops = len(stops)
        n_locations = n_stops + 1  # depot + stops
        assert cost_matrix.shape == (n_locations, n_locations), (
            f"Cost matrix shape {cost_matrix.shape} doesn't match "
            f"{n_locations} locations (1 depot + {n_stops} stops)"
        )

        start_time = time.time()

        # Build PyVRP model
        m = Model()

        # Add depot (index 0)
        depot = m.add_depot(x=0, y=0)

        # Add vehicle types
        m.add_vehicle_type(
            num_available=num_vehicles,
            capacity=[vehicle_capacity],
            start_depot=depot,
            end_depot=depot,
        )

        # Add clients (indices 1..n)
        clients = []
        for stop in stops:
            client = m.add_client(
                x=0,  # Coordinates are not used for routing (we use the cost matrix)
                y=0,
                delivery=stop.demand,
                tw_early=stop.tw_early,
                tw_late=stop.tw_late,
                service_duration=stop.service_time,
            )
            clients.append(client)

        # Add edges from cost matrix
        all_locations = [depot] + clients
        for i, frm in enumerate(all_locations):
            for j, to in enumerate(all_locations):
                if i == j:
                    continue
                m.add_edge(
                    frm,
                    to,
                    distance=int(cost_matrix[i][j]),
                    duration=int(cost_matrix[i][j]),
                )

        # Solve
        result = m.solve(stop=MaxRuntime(max_runtime), display=False)
        solve_time = time.time() - start_time

        # Extract solution
        solution = VRPSolution(solve_time=solve_time)

        if result.best is None:
            logger.warning("PyVRP found no feasible solution")
            return solution

        solution.is_feasible = result.is_feasible()
        total_dist = 0.0
        total_dur = 0.0

        for route_idx, route in enumerate(result.best.routes()):
            visit_indices = route.visits()
            # visit_indices are location indices: clients are 1..n
            # Map to stop list: stops[idx - 1]
            stop_ids = [stops[idx - 1].id for idx in visit_indices]

            route_dist = float(route.distance()) / scaling_factor
            route_dur = float(route.duration()) / scaling_factor

            total_dist += route_dist
            total_dur += route_dur

            r = Route(
                vehicle_id=f"vehicle_{route_idx + 1}",
                stop_ids=stop_ids,
                distance=route_dist,
                duration=route_dur,
            )
            solution.routes.append(r)

        solution.total_distance = total_dist
        solution.total_duration = total_dur
        solution.num_vehicles = len(solution.routes)

        logger.info(
            "PyVRP solved: %d routes, total_duration=%.1fs, feasible=%s, solve_time=%.1fs",
            solution.num_vehicles, solution.total_duration,
            solution.is_feasible, solution.solve_time,
        )
        return solution


class ORToolsSolver:
    """VRP solver using Google OR-Tools."""

    def solve(
        self,
        cost_matrix: np.ndarray,
        stops: list[Stop],
        depot_coords: tuple[float, float] = (0.0, 0.0),
        vehicle_capacity: int = 100,
        num_vehicles: int = 10,
        max_runtime: int = 30,
        scaling_factor: int = 100,
    ) -> VRPSolution:
        """Solve CVRP using OR-Tools.

        Same interface as PyVRPSolver.solve().
        """
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2

        n_locations = len(stops) + 1
        assert cost_matrix.shape == (n_locations, n_locations)

        start_time = time.time()

        # Create routing index manager (depot is node 0)
        manager = pywrapcp.RoutingIndexManager(n_locations, num_vehicles, 0)
        routing = pywrapcp.RoutingModel(manager)

        # Transit callback
        def transit_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(cost_matrix[from_node][to_node])

        transit_callback_index = routing.RegisterTransitCallback(transit_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Capacity constraint
        def demand_callback(from_index):
            node = manager.IndexToNode(from_index)
            if node == 0:
                return 0
            return stops[node - 1].demand

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            [vehicle_capacity] * num_vehicles,
            True,  # start cumul to zero
            "Capacity",
        )

        # Search parameters
        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_params.time_limit.seconds = max_runtime

        # Solve
        assignment = routing.SolveWithParameters(search_params)
        solve_time = time.time() - start_time

        solution = VRPSolution(solve_time=solve_time)

        if assignment is None:
            logger.warning("OR-Tools found no feasible solution")
            return solution

        solution.is_feasible = True
        total_dist = 0.0
        total_dur = 0.0

        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            stop_ids = []
            route_dist = 0.0

            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                if node != 0:
                    stop_ids.append(stops[node - 1].id)
                next_index = assignment.Value(routing.NextVar(index))
                route_dist += routing.GetArcCostForVehicle(index, next_index, vehicle_id)
                index = next_index

            if not stop_ids:
                continue

            route_dur = route_dist / scaling_factor
            route_dist_scaled = route_dist / scaling_factor
            total_dist += route_dist_scaled
            total_dur += route_dur

            r = Route(
                vehicle_id=f"vehicle_{vehicle_id + 1}",
                stop_ids=stop_ids,
                distance=route_dist_scaled,
                duration=route_dur,
            )
            solution.routes.append(r)

        solution.total_distance = total_dist
        solution.total_duration = total_dur
        solution.num_vehicles = len(solution.routes)

        logger.info(
            "OR-Tools solved: %d routes, total_duration=%.1fs, solve_time=%.1fs",
            solution.num_vehicles, solution.total_duration, solution.solve_time,
        )
        return solution

"""Test VRP solvers on Solomon benchmarks and a small NYC problem."""

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.optimization.vrp_solver import ORToolsSolver, PyVRPSolver, Stop

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_solomon_c101():
    """Solve Solomon C101 and compare with best-known solution."""
    import vrplib

    solomon_dir = PROJECT_ROOT / "data" / "raw" / "solomon"
    instance_path = solomon_dir / "C101.txt"
    solution_path = solomon_dir / "C101.sol"

    if not instance_path.exists():
        logger.warning("Solomon C101 not found. Downloading...")
        solomon_dir.mkdir(parents=True, exist_ok=True)
        vrplib.download_instance("C101", str(instance_path))
        vrplib.download_solution("C101", str(solution_path))

    instance = vrplib.read_instance(str(instance_path), instance_format="solomon")
    n = instance["dimension"]
    coords = instance["node_coord"]
    demands = instance["demand"]
    time_windows = instance["time_window"]
    service_times = instance["service_time"]
    capacity = instance["capacity"]

    # Build Euclidean distance matrix
    dist_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            # PyVRP uses truncated integer distances for Solomon
            dist_matrix[i][j] = int(np.sqrt(dx * dx + dy * dy) * 10)

    # Create stops (exclude depot at index 0)
    stops = []
    for i in range(1, n):
        stops.append(Stop(
            id=f"C{i}",
            lat=float(coords[i][0]),
            lng=float(coords[i][1]),
            demand=int(demands[i]),
            tw_early=int(time_windows[i][0]) * 10,
            tw_late=int(time_windows[i][1]) * 10,
            service_time=int(service_times[i]) * 10,
        ))

    scaling_factor = 10

    # Solve with PyVRP
    logger.info("Solving Solomon C101 with PyVRP (%d stops)...", len(stops))
    pyvrp_solver = PyVRPSolver()
    pyvrp_sol = pyvrp_solver.solve(
        cost_matrix=dist_matrix,
        stops=stops,
        vehicle_capacity=capacity,
        num_vehicles=25,
        max_runtime=30,
        scaling_factor=scaling_factor,
    )

    logger.info(
        "PyVRP: %d routes, total_cost=%.1f, feasible=%s, time=%.1fs",
        pyvrp_sol.num_vehicles, pyvrp_sol.total_duration,
        pyvrp_sol.is_feasible, pyvrp_sol.solve_time,
    )

    # Solve with OR-Tools
    logger.info("Solving Solomon C101 with OR-Tools...")
    ortools_solver = ORToolsSolver()
    ortools_sol = ortools_solver.solve(
        cost_matrix=dist_matrix,
        stops=stops,
        vehicle_capacity=capacity,
        num_vehicles=25,
        max_runtime=30,
        scaling_factor=scaling_factor,
    )

    logger.info(
        "OR-Tools: %d routes, total_cost=%.1f, feasible=%s, time=%.1fs",
        ortools_sol.num_vehicles, ortools_sol.total_duration,
        ortools_sol.is_feasible, ortools_sol.solve_time,
    )

    # Verify all stops visited
    pyvrp_visited = set()
    for route in pyvrp_sol.routes:
        pyvrp_visited.update(route.stop_ids)
    assert len(pyvrp_visited) == len(stops), (
        f"PyVRP visited {len(pyvrp_visited)}/{len(stops)} stops"
    )
    logger.info("All %d stops visited by PyVRP.", len(stops))


def test_small_problem():
    """Test with a small 5-stop problem using a synthetic cost matrix."""
    logger.info("Testing small 5-stop problem...")

    stops = [
        Stop(id="A", lat=40.758, lng=-73.985, demand=10),
        Stop(id="B", lat=40.761, lng=-73.977, demand=15),
        Stop(id="C", lat=40.753, lng=-73.977, demand=20),
        Stop(id="D", lat=40.749, lng=-73.968, demand=10),
        Stop(id="E", lat=40.741, lng=-74.001, demand=5),
    ]

    # Synthetic cost matrix (6x6: depot + 5 stops), already scaled by 100
    cost_matrix = np.array([
        [0, 300, 400, 350, 500, 600],
        [300, 0, 200, 250, 400, 500],
        [400, 200, 0, 150, 350, 450],
        [350, 250, 150, 0, 300, 400],
        [500, 400, 350, 300, 0, 350],
        [600, 500, 450, 400, 350, 0],
    ], dtype=int)

    for solver_name, solver_cls in [("PyVRP", PyVRPSolver), ("OR-Tools", ORToolsSolver)]:
        solver = solver_cls()
        sol = solver.solve(
            cost_matrix=cost_matrix,
            stops=stops,
            vehicle_capacity=30,
            num_vehicles=3,
            max_runtime=10,
            scaling_factor=100,
        )

        visited = set()
        for route in sol.routes:
            visited.update(route.stop_ids)

        logger.info(
            "%s: %d routes, total_duration=%.1f, all_visited=%s",
            solver_name, sol.num_vehicles, sol.total_duration,
            visited == {s.id for s in stops},
        )

        for route in sol.routes:
            logger.info("  %s: %s (dur=%.1f)", route.vehicle_id, route.stop_ids, route.duration)


if __name__ == "__main__":
    test_small_problem()
    print()
    test_solomon_c101()

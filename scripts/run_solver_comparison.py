"""Compare PyVRP vs OR-Tools on the same problem instances."""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.optimization.vrp_solver import ORToolsSolver, PyVRPSolver, Stop

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_solomon_instance(name: str) -> dict:
    """Load a Solomon VRPTW instance."""
    import vrplib

    solomon_dir = PROJECT_ROOT / "data" / "raw" / "solomon"
    instance_path = solomon_dir / f"{name}.txt"

    if not instance_path.exists():
        solomon_dir.mkdir(parents=True, exist_ok=True)
        vrplib.download_instance(name, str(instance_path))

    return vrplib.read_instance(str(instance_path), instance_format="solomon")


def build_solomon_problem(instance: dict) -> tuple[np.ndarray, list[Stop], int]:
    """Convert Solomon instance to our solver format."""
    n = instance["dimension"]
    coords = instance["node_coord"]
    demands = instance["demand"]
    time_windows = instance["time_window"]
    service_times = instance["service_time"]
    capacity = instance["capacity"]

    # Euclidean distance matrix (scaled by 10)
    dist_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dist_matrix[i][j] = int(np.sqrt(dx * dx + dy * dy) * 10)

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

    return dist_matrix, stops, capacity


def main():
    instances = ["C101", "C201", "R101", "R201", "RC101", "RC201"]
    results = []

    for name in instances:
        logger.info("=== Instance: %s ===", name)
        try:
            instance = load_solomon_instance(name)
        except Exception as e:
            logger.warning("Could not load %s: %s", name, e)
            continue

        dist_matrix, stops, capacity = build_solomon_problem(instance)
        scaling_factor = 10

        for solver_name, solver_cls in [("PyVRP", PyVRPSolver), ("OR-Tools", ORToolsSolver)]:
            solver = solver_cls()
            sol = solver.solve(
                cost_matrix=dist_matrix,
                stops=stops,
                vehicle_capacity=capacity,
                num_vehicles=25,
                max_runtime=30,
                scaling_factor=scaling_factor,
            )

            visited = set()
            for route in sol.routes:
                visited.update(route.stop_ids)

            results.append({
                "instance": name,
                "solver": solver_name,
                "total_duration": sol.total_duration,
                "num_vehicles": sol.num_vehicles,
                "solve_time": sol.solve_time,
                "is_feasible": sol.is_feasible,
                "all_visited": len(visited) == len(stops),
            })

            logger.info(
                "  %s: duration=%.1f, vehicles=%d, time=%.1fs, feasible=%s",
                solver_name, sol.total_duration, sol.num_vehicles,
                sol.solve_time, sol.is_feasible,
            )

    # Summary table
    df = pd.DataFrame(results)
    output_path = PROJECT_ROOT / "data" / "processed" / "solver_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info("\n%s", df.to_string(index=False))
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()

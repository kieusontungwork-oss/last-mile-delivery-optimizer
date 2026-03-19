"""Ablation study: compare static vs ML-adjusted cost matrices.

Experiments:
1. Static baseline: OSRM raw durations
2. ML-adjusted (off-peak): LightGBM at 10 AM
3. ML-adjusted (peak): LightGBM at 8 AM
4. Full dynamic: LightGBM with all features

For each: solve Solomon C1/R1/RC1 instances and synthetic NYC scenarios.
Repeat 10 times with different seeds. Report mean +/- std.
Apply Wilcoxon signed-rank test for statistical significance.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.predict import ETAPredictor
from src.optimization.cost_matrix import CostMatrixBuilder
from src.optimization.osrm_client import OSRMClient
from src.optimization.vrp_solver import PyVRPSolver, Stop

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "data" / "processed"

# Synthetic NYC scenarios
NYC_SCENARIOS = {
    "manhattan_20": {
        "depot": (40.7484, -73.9857),
        "stops": [
            (40.7580, -73.9855), (40.7614, -73.9776), (40.7527, -73.9772),
            (40.7489, -73.9680), (40.7282, -73.7949), (40.7061, -74.0087),
            (40.7128, -74.0060), (40.7411, -74.0018), (40.7549, -73.9840),
            (40.7681, -73.9819), (40.7505, -73.9934), (40.7425, -73.9889),
            (40.7362, -73.9903), (40.7308, -73.9972), (40.7247, -73.9918),
            (40.7193, -73.9872), (40.7560, -73.9700), (40.7480, -73.9740),
            (40.7650, -73.9600), (40.7350, -73.9800),
        ],
    },
}

SEEDS = list(range(10))


def run_experiment(
    name: str,
    cost_builder: CostMatrixBuilder,
    locations: list[tuple[float, float]],
    stops: list[Stop],
    departure_time: datetime | None,
    use_ml: bool,
) -> list[dict]:
    """Run a single experiment across all seeds."""
    results = []

    for seed in SEEDS:
        # Build cost matrix
        if use_ml and departure_time is not None:
            matrix = cost_builder.build_dynamic_matrix(locations, departure_time)
        else:
            matrix = cost_builder.build_static_matrix(locations)

        solver = PyVRPSolver()
        solution = solver.solve(
            cost_matrix=matrix,
            stops=stops,
            vehicle_capacity=100,
            num_vehicles=10,
            max_runtime=30,
            scaling_factor=cost_builder.scaling_factor,
        )

        results.append({
            "experiment": name,
            "seed": seed,
            "total_duration": solution.total_duration,
            "total_distance": solution.total_distance,
            "num_vehicles": solution.num_vehicles,
            "is_feasible": solution.is_feasible,
            "solve_time": solution.solve_time,
        })

    return results


def run_ablation():
    """Run the full ablation study."""
    logger.info("Starting ablation study...")

    # Initialize components
    osrm = OSRMClient(base_url="http://localhost:5000")
    if not osrm.health_check():
        logger.error("OSRM not reachable. Start with: docker compose up osrm-backend")
        sys.exit(1)

    model_path = MODELS_DIR / "eta_lightgbm_v1.joblib"
    predictor = None
    if model_path.exists():
        predictor = ETAPredictor(model_path)
    else:
        logger.warning("No ML model found. Running static-only experiments.")

    cost_builder = CostMatrixBuilder(osrm, predictor)
    all_results = []

    # Run on NYC scenarios
    for scenario_name, scenario in NYC_SCENARIOS.items():
        logger.info("Scenario: %s", scenario_name)

        locations = [scenario["depot"]] + scenario["stops"]
        stops = [
            Stop(id=f"S{i+1}", lat=loc[0], lng=loc[1], demand=5)
            for i, loc in enumerate(scenario["stops"])
        ]

        # Experiment 1: Static baseline
        logger.info("  Running static baseline...")
        all_results.extend(run_experiment(
            f"{scenario_name}_static", cost_builder, locations, stops,
            departure_time=None, use_ml=False,
        ))

        if predictor:
            # Experiment 2: ML off-peak (10 AM Tuesday)
            logger.info("  Running ML off-peak...")
            all_results.extend(run_experiment(
                f"{scenario_name}_ml_offpeak", cost_builder, locations, stops,
                departure_time=datetime(2023, 6, 13, 10, 0), use_ml=True,
            ))

            # Experiment 3: ML peak (8 AM Tuesday)
            logger.info("  Running ML peak...")
            all_results.extend(run_experiment(
                f"{scenario_name}_ml_peak", cost_builder, locations, stops,
                departure_time=datetime(2023, 6, 13, 8, 0), use_ml=True,
            ))

            # Experiment 4: ML weekend (10 AM Saturday)
            logger.info("  Running ML weekend...")
            all_results.extend(run_experiment(
                f"{scenario_name}_ml_weekend", cost_builder, locations, stops,
                departure_time=datetime(2023, 6, 17, 10, 0), use_ml=True,
            ))

    osrm.close()

    # Aggregate results
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / "evaluation_results.csv", index=False)
    logger.info("Results saved to %s", RESULTS_DIR / "evaluation_results.csv")

    # Statistical tests
    if predictor:
        _run_statistical_tests(df)

    return df


def _run_statistical_tests(df: pd.DataFrame):
    """Run Wilcoxon signed-rank tests comparing static vs ML."""
    scenarios = df["experiment"].str.rsplit("_", n=1).str[0].unique()

    for scenario in scenarios:
        static_key = f"{scenario}_static"
        ml_peak_key = f"{scenario}_ml_peak"

        static = df[df["experiment"] == static_key]["total_duration"].values
        ml_peak = df[df["experiment"] == ml_peak_key]["total_duration"].values

        if len(static) < 2 or len(ml_peak) < 2:
            continue

        min_len = min(len(static), len(ml_peak))
        stat, p_value = wilcoxon(static[:min_len], ml_peak[:min_len])

        improvement = (static.mean() - ml_peak.mean()) / static.mean() * 100

        logger.info(
            "  %s: static=%.1f, ml_peak=%.1f, improvement=%.1f%%, p=%.4f %s",
            scenario, static.mean(), ml_peak.mean(), improvement, p_value,
            "(significant)" if p_value < 0.05 else "(not significant)",
        )


if __name__ == "__main__":
    run_ablation()

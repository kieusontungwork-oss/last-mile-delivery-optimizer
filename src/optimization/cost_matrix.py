"""Dynamic and static cost matrix construction."""

import logging
from datetime import datetime

import numpy as np

from src.features.engineering import FeatureEngineer
from src.models.predict import ETAPredictor
from src.optimization.osrm_client import OSRMClient
from src.utils.geo import haversine_vectorized

logger = logging.getLogger(__name__)


class CostMatrixBuilder:
    """Builds VRP cost matrices using OSRM base times and optional ML adjustment."""

    def __init__(
        self,
        osrm_client: OSRMClient,
        eta_predictor: ETAPredictor | None = None,
        scaling_factor: int = 100,
    ):
        self.osrm = osrm_client
        self.predictor = eta_predictor
        self.scaling_factor = scaling_factor
        self._feature_engineer = FeatureEngineer()

    def build_static_matrix(
        self, locations: list[tuple[float, float]]
    ) -> np.ndarray:
        """Build cost matrix using raw OSRM travel times (baseline).

        Args:
            locations: List of (lat, lng) tuples. First element is depot.

        Returns:
            NxN integer matrix (seconds × scaling_factor).
        """
        duration_matrix = self.osrm.get_duration_matrix(locations)
        return (duration_matrix * self.scaling_factor).astype(int)

    def build_dynamic_matrix(
        self,
        locations: list[tuple[float, float]],
        departure_time: datetime,
    ) -> np.ndarray:
        """Build ML-adjusted cost matrix.

        Args:
            locations: List of (lat, lng) tuples. First element is depot.
            departure_time: When the routes will start.

        Returns:
            NxN integer matrix (ML-predicted seconds × scaling_factor).
        """
        if self.predictor is None:
            logger.warning("No ML model loaded, falling back to static matrix")
            return self.build_static_matrix(locations)

        n = len(locations)

        # Step 1: Get OSRM base durations and distances
        duration_matrix, distance_matrix = self.osrm.get_duration_and_distance_matrices(locations)

        # Step 2: Build feature vectors for all (i,j) pairs (excluding diagonal)
        lats = np.array([loc[0] for loc in locations])
        lngs = np.array([loc[1] for loc in locations])

        # Create all pairs (i, j) where i != j
        rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        mask = rows != cols
        src_idx = rows[mask]
        dst_idx = cols[mask]

        osrm_times = duration_matrix[src_idx, dst_idx]
        osrm_dists = distance_matrix[src_idx, dst_idx]
        origin_lats = lats[src_idx]
        origin_lngs = lngs[src_idx]
        dest_lats = lats[dst_idx]
        dest_lngs = lngs[dst_idx]

        # Step 3: Build features
        hour = departure_time.hour + departure_time.minute / 60.0
        dow = departure_time.weekday()
        month = departure_time.month

        features = self._feature_engineer.transform_for_prediction(
            osrm_base_time=osrm_times,
            osrm_base_distance=osrm_dists,
            origin_lats=origin_lats,
            origin_lngs=origin_lngs,
            dest_lats=dest_lats,
            dest_lngs=dest_lngs,
            departure_hour=hour,
            departure_dow=dow,
            departure_month=month,
        )

        # Step 4: ML prediction
        predictions = self.predictor.predict(features)

        # Step 5: Build the NxN matrix
        ml_matrix = np.zeros((n, n), dtype=float)
        ml_matrix[src_idx, dst_idx] = predictions

        # Ensure non-negative and scale to integers
        ml_matrix = np.maximum(ml_matrix, 0)
        return (ml_matrix * self.scaling_factor).astype(int)

    def build_both(
        self,
        locations: list[tuple[float, float]],
        departure_time: datetime,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build both static and dynamic matrices for comparison.

        Returns:
            Tuple of (static_matrix, dynamic_matrix), both integer-scaled.
        """
        static = self.build_static_matrix(locations)
        dynamic = self.build_dynamic_matrix(locations, departure_time)
        return static, dynamic

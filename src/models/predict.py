"""ETA prediction / inference module."""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.engineering import get_feature_names

logger = logging.getLogger(__name__)


class ETAPredictor:
    """Loads a trained model and provides ETA predictions."""

    def __init__(self, model_path: str | Path, metadata_path: str | Path | None = None):
        """Load model and validate feature alignment.

        Args:
            model_path: Path to the joblib model file.
            metadata_path: Path to the metadata JSON. If None, inferred from model_path.
        """
        model_path = Path(model_path)
        if metadata_path is None:
            metadata_path = model_path.with_name(model_path.stem + "_metadata.json")
        else:
            metadata_path = Path(metadata_path)

        # Load model
        self.model = joblib.load(model_path)
        logger.info("Loaded model from %s", model_path)

        # Load and validate metadata
        self.metadata = {}
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)

            # Validate feature alignment
            expected = get_feature_names()
            stored = self.metadata.get("feature_names", [])
            if stored and stored != expected:
                raise ValueError(
                    f"Feature mismatch! Model was trained with {stored}, "
                    f"but current feature list is {expected}"
                )

        self._feature_names = get_feature_names()

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict travel times for a batch of feature vectors.

        Args:
            features: DataFrame with columns matching FEATURE_NAMES.

        Returns:
            Array of predicted durations in seconds.
        """
        X = features[self._feature_names]
        predictions = self.model.predict(X)
        # Clamp to reasonable range (0 to 2 hours)
        return np.clip(predictions, 0, 7200)

    def predict_single(
        self,
        osrm_base_time: float,
        osrm_base_distance: float,
        origin_lat: float,
        origin_lng: float,
        dest_lat: float,
        dest_lng: float,
        departure_hour: float,
        departure_dow: int,
        departure_month: int,
    ) -> float:
        """Predict travel time for a single origin-destination pair.

        Returns:
            Predicted duration in seconds.
        """
        from src.features.engineering import FeatureEngineer

        fe = FeatureEngineer()
        features = fe.transform_for_prediction(
            osrm_base_time=np.array([osrm_base_time]),
            osrm_base_distance=np.array([osrm_base_distance]),
            origin_lats=np.array([origin_lat]),
            origin_lngs=np.array([origin_lng]),
            dest_lats=np.array([dest_lat]),
            dest_lngs=np.array([dest_lng]),
            departure_hour=departure_hour,
            departure_dow=departure_dow,
            departure_month=departure_month,
        )
        return float(self.predict(features)[0])

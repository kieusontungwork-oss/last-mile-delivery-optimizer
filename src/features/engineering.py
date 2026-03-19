"""Feature engineering pipeline for ETA prediction."""

import numpy as np
import pandas as pd

from src.utils.geo import haversine_vectorized


FEATURE_NAMES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month",
    "is_weekend",
    "is_rush_hour",
    "pickup_zone",
    "dropoff_zone",
    "haversine_distance_km",
    "bearing",
    "osrm_base_time_seconds",
    "osrm_base_distance_m",
    "avg_speed_mps",
]

CATEGORICAL_FEATURES = ["pickup_zone", "dropoff_zone"]


def get_feature_names() -> list[str]:
    """Return ordered list of feature names used by the model."""
    return list(FEATURE_NAMES)


def get_categorical_features() -> list[str]:
    """Return list of categorical feature names."""
    return list(CATEGORICAL_FEATURES)


class FeatureEngineer:
    """Transforms raw trip data into ML features."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature transformations.

        Expects columns: pickup_datetime, pickup_zone, dropoff_zone,
        pickup_lat, pickup_lng, dropoff_lat, dropoff_lng,
        osrm_base_time_seconds, osrm_base_distance_m.

        Returns DataFrame with exactly the columns in FEATURE_NAMES.
        """
        features = pd.DataFrame(index=df.index)

        # Temporal features (cyclical encoding)
        dt = pd.to_datetime(df["pickup_datetime"])
        hour = dt.dt.hour + dt.dt.minute / 60.0
        features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        features["hour_cos"] = np.cos(2 * np.pi * hour / 24)

        dow = dt.dt.dayofweek  # Monday=0, Sunday=6
        features["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        features["dow_cos"] = np.cos(2 * np.pi * dow / 7)

        features["month"] = dt.dt.month

        features["is_weekend"] = (dow >= 5).astype(int)

        hour_int = dt.dt.hour
        features["is_rush_hour"] = (
            ((hour_int >= 7) & (hour_int < 9)) | ((hour_int >= 16) & (hour_int < 19))
        ).astype(int)

        # Zone IDs (categorical, must be int-typed for LightGBM)
        features["pickup_zone"] = df["pickup_zone"].astype(int)
        features["dropoff_zone"] = df["dropoff_zone"].astype(int)

        # Spatial features
        features["haversine_distance_km"] = haversine_vectorized(
            df["pickup_lat"].values,
            df["pickup_lng"].values,
            df["dropoff_lat"].values,
            df["dropoff_lng"].values,
        )

        features["bearing"] = _vectorized_bearing(
            df["pickup_lat"].values,
            df["pickup_lng"].values,
            df["dropoff_lat"].values,
            df["dropoff_lng"].values,
        )

        # OSRM-based features
        features["osrm_base_time_seconds"] = df["osrm_base_time_seconds"]
        features["osrm_base_distance_m"] = df["osrm_base_distance_m"]

        # Average speed (handle zero division)
        osrm_time = df["osrm_base_time_seconds"].values
        osrm_dist = df["osrm_base_distance_m"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            avg_speed = np.where(osrm_time > 0, osrm_dist / osrm_time, 0.0)
        features["avg_speed_mps"] = avg_speed

        # Ensure column order matches FEATURE_NAMES
        return features[FEATURE_NAMES]

    def transform_for_prediction(
        self,
        osrm_base_time: np.ndarray,
        osrm_base_distance: np.ndarray,
        origin_lats: np.ndarray,
        origin_lngs: np.ndarray,
        dest_lats: np.ndarray,
        dest_lngs: np.ndarray,
        departure_hour: float,
        departure_dow: int,
        departure_month: int,
        pickup_zone: int = 0,
        dropoff_zone: int = 0,
    ) -> pd.DataFrame:
        """Build feature DataFrame for cost matrix prediction (batch of pairs).

        Args:
            osrm_base_time: Array of OSRM base travel times (seconds).
            osrm_base_distance: Array of OSRM base distances (meters).
            origin_lats, origin_lngs: Origin coordinates.
            dest_lats, dest_lngs: Destination coordinates.
            departure_hour: Hour of day (float, e.g. 8.5 for 8:30 AM).
            departure_dow: Day of week (0=Monday, 6=Sunday).
            departure_month: Month (1-12).
            pickup_zone, dropoff_zone: Zone IDs (default 0 for unknown).

        Returns:
            DataFrame with FEATURE_NAMES columns, one row per pair.
        """
        n = len(osrm_base_time)

        features = pd.DataFrame({
            "hour_sin": np.full(n, np.sin(2 * np.pi * departure_hour / 24)),
            "hour_cos": np.full(n, np.cos(2 * np.pi * departure_hour / 24)),
            "dow_sin": np.full(n, np.sin(2 * np.pi * departure_dow / 7)),
            "dow_cos": np.full(n, np.cos(2 * np.pi * departure_dow / 7)),
            "month": np.full(n, departure_month, dtype=int),
            "is_weekend": np.full(n, int(departure_dow >= 5)),
            "is_rush_hour": np.full(n, int(
                (7 <= departure_hour < 9) or (16 <= departure_hour < 19)
            )),
            "pickup_zone": np.full(n, pickup_zone, dtype=int),
            "dropoff_zone": np.full(n, dropoff_zone, dtype=int),
            "haversine_distance_km": haversine_vectorized(
                origin_lats, origin_lngs, dest_lats, dest_lngs,
            ),
            "bearing": _vectorized_bearing(
                origin_lats, origin_lngs, dest_lats, dest_lngs,
            ),
            "osrm_base_time_seconds": osrm_base_time,
            "osrm_base_distance_m": osrm_base_distance,
        })

        with np.errstate(divide="ignore", invalid="ignore"):
            features["avg_speed_mps"] = np.where(
                osrm_base_time > 0, osrm_base_distance / osrm_base_time, 0.0
            )

        return features[FEATURE_NAMES]


def _vectorized_bearing(
    lats1: np.ndarray, lons1: np.ndarray,
    lats2: np.ndarray, lons2: np.ndarray,
) -> np.ndarray:
    """Compute initial bearing in degrees (vectorized)."""
    lat1_r = np.radians(lats1)
    lat2_r = np.radians(lats2)
    dlon_r = np.radians(lons2 - lons1)

    x = np.sin(dlon_r) * np.cos(lat2_r)
    y = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon_r)
    return np.degrees(np.arctan2(x, y)) % 360

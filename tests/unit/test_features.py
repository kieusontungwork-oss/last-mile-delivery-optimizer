"""Tests for src/features/engineering.py."""

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    FEATURE_NAMES,
    FeatureEngineer,
    get_categorical_features,
    get_feature_names,
)


class TestFeatureNames:
    def test_feature_count(self):
        assert len(get_feature_names()) == 14

    def test_categorical_in_features(self):
        for cat in get_categorical_features():
            assert cat in get_feature_names()

    def test_feature_names_immutable(self):
        """Modifying returned list shouldn't affect the original."""
        names = get_feature_names()
        names.append("extra")
        assert len(get_feature_names()) == 14


class TestFeatureEngineer:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "pickup_datetime": pd.to_datetime([
                "2023-01-10 08:30:00",
                "2023-06-17 14:00:00",
                "2023-10-03 00:00:00",
            ]),
            "pickup_zone": [100, 200, 50],
            "dropoff_zone": [150, 100, 200],
            "pickup_lat": [40.758, 40.712, 40.748],
            "pickup_lng": [-73.985, -74.006, -73.968],
            "dropoff_lat": [40.753, 40.741, 40.761],
            "dropoff_lng": [-73.977, -74.001, -73.977],
            "osrm_base_time_seconds": [300.0, 600.0, 450.0],
            "osrm_base_distance_m": [1500.0, 3000.0, 2200.0],
        })

    def test_transform_produces_all_features(self, sample_df):
        fe = FeatureEngineer()
        result = fe.transform(sample_df)
        assert list(result.columns) == FEATURE_NAMES
        assert len(result) == len(sample_df)

    def test_no_nan_values(self, sample_df):
        fe = FeatureEngineer()
        result = fe.transform(sample_df)
        assert not result.isna().any().any()

    def test_cyclical_encoding_range(self, sample_df):
        fe = FeatureEngineer()
        result = fe.transform(sample_df)
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
            assert result[col].between(-1, 1).all()

    def test_rush_hour_detection(self, sample_df):
        fe = FeatureEngineer()
        result = fe.transform(sample_df)
        # Row 0: 8:30 AM -> rush hour
        assert result.iloc[0]["is_rush_hour"] == 1
        # Row 1: 2:00 PM -> not rush hour
        assert result.iloc[1]["is_rush_hour"] == 0
        # Row 2: midnight -> not rush hour
        assert result.iloc[2]["is_rush_hour"] == 0

    def test_weekend_detection(self, sample_df):
        fe = FeatureEngineer()
        result = fe.transform(sample_df)
        # Row 0: 2023-01-10 is Tuesday -> not weekend
        assert result.iloc[0]["is_weekend"] == 0
        # Row 1: 2023-06-17 is Saturday -> weekend
        assert result.iloc[1]["is_weekend"] == 1

    def test_zones_are_int(self, sample_df):
        fe = FeatureEngineer()
        result = fe.transform(sample_df)
        assert result["pickup_zone"].dtype in [np.int32, np.int64]
        assert result["dropoff_zone"].dtype in [np.int32, np.int64]


class TestTransformForPrediction:
    def test_batch_prediction_features(self):
        fe = FeatureEngineer()
        n = 5
        result = fe.transform_for_prediction(
            osrm_base_time=np.array([300.0] * n),
            osrm_base_distance=np.array([1500.0] * n),
            origin_lats=np.array([40.758] * n),
            origin_lngs=np.array([-73.985] * n),
            dest_lats=np.array([40.753] * n),
            dest_lngs=np.array([-73.977] * n),
            departure_hour=8.5,
            departure_dow=1,
            departure_month=6,
        )
        assert list(result.columns) == FEATURE_NAMES
        assert len(result) == n
        assert not result.isna().any().any()

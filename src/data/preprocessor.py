"""Data preprocessing and splitting for NYC TLC dataset."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class NycTlcPreprocessor:
    """Preprocessor for NYC TLC Yellow Taxi trip data."""

    # Valid trip duration range (seconds)
    MIN_DURATION = 60
    MAX_DURATION = 7200

    # Valid trip distance range (miles)
    MIN_DISTANCE = 0.1
    MAX_DISTANCE = 100.0

    def filter_trips(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out invalid trips."""
        n_before = len(df)

        mask = (
            df["trip_duration_seconds"].between(self.MIN_DURATION, self.MAX_DURATION)
            & df["trip_distance"].between(self.MIN_DISTANCE, self.MAX_DISTANCE)
            & df["pickup_zone"].notna()
            & df["dropoff_zone"].notna()
            & (df["pickup_zone"] > 0)
            & (df["dropoff_zone"] > 0)
            # Exclude zone 264+ (unknown zones)
            & (df["pickup_zone"] <= 263)
            & (df["dropoff_zone"] <= 263)
        )

        df = df.loc[mask].copy()
        logger.info("Filtered %d -> %d trips (removed %d)", n_before, len(df), n_before - len(df))
        return df

    def add_zone_centroids(
        self, df: pd.DataFrame, zones_gdf: "gpd.GeoDataFrame"
    ) -> pd.DataFrame:
        """Merge zone centroid coordinates into trip data."""
        zone_coords = zones_gdf[["LocationID", "centroid_lat", "centroid_lng"]].copy()

        # Merge pickup zone
        df = df.merge(
            zone_coords.rename(columns={
                "LocationID": "pickup_zone",
                "centroid_lat": "pickup_lat",
                "centroid_lng": "pickup_lng",
            }),
            on="pickup_zone",
            how="left",
        )

        # Merge dropoff zone
        df = df.merge(
            zone_coords.rename(columns={
                "LocationID": "dropoff_zone",
                "centroid_lat": "dropoff_lat",
                "centroid_lng": "dropoff_lng",
            }),
            on="dropoff_zone",
            how="left",
        )

        # Drop rows where centroids couldn't be found
        n_before = len(df)
        df = df.dropna(subset=["pickup_lat", "dropoff_lat"])
        if len(df) < n_before:
            logger.warning("Dropped %d rows with missing centroids", n_before - len(df))

        return df

    def temporal_split(
        self,
        df: pd.DataFrame,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data chronologically (no shuffle to avoid temporal leakage)."""
        assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

        df = df.sort_values("pickup_datetime").reset_index(drop=True)
        n = len(df)

        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))

        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]

        logger.info(
            "Split: train=%d, val=%d, test=%d",
            len(train), len(val), len(test),
        )
        return train, val, test

    def save_splits(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        output_dir: str | Path,
    ) -> None:
        """Save train/val/test splits as Parquet files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, df in [("train", train), ("val", val), ("test", test)]:
            path = output_dir / f"{name}.parquet"
            df.to_parquet(path, index=False)
            logger.info("Saved %s (%d rows) to %s", name, len(df), path)

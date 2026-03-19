"""Data processing pipeline: load, preprocess, add OSRM times, engineer features, save."""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_nyc_taxi_zones, load_nyc_tlc
from src.data.preprocessor import NycTlcPreprocessor
from src.features.engineering import FeatureEngineer
from src.optimization.osrm_client import OSRMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def build_zone_osrm_matrix(
    zones_gdf: "gpd.GeoDataFrame",
    osrm: OSRMClient,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute zone-to-zone OSRM durations and distances.

    Returns:
        Tuple of (duration_matrix, distance_matrix) indexed by LocationID.
        Index 0 corresponds to LocationID 1, etc.
    """
    # Get zone centroids ordered by LocationID
    zone_info = (
        zones_gdf[["LocationID", "centroid_lat", "centroid_lng"]]
        .sort_values("LocationID")
        .reset_index(drop=True)
    )

    locations = list(zip(zone_info["centroid_lat"], zone_info["centroid_lng"]))
    logger.info("Computing OSRM matrix for %d zones...", len(locations))

    duration_matrix, distance_matrix = osrm.get_duration_and_distance_matrices(locations)
    logger.info("Zone OSRM matrix computed: shape=%s", duration_matrix.shape)

    return duration_matrix, distance_matrix, zone_info["LocationID"].values


def add_osrm_features(
    df: pd.DataFrame,
    duration_matrix: np.ndarray,
    distance_matrix: np.ndarray,
    zone_ids: np.ndarray,
) -> pd.DataFrame:
    """Add OSRM base time and distance using precomputed zone-to-zone matrix."""
    # Create zone_id -> matrix_index mapping
    zone_to_idx = {int(zid): i for i, zid in enumerate(zone_ids)}

    pu_idx = df["pickup_zone"].astype(int).map(zone_to_idx)
    do_idx = df["dropoff_zone"].astype(int).map(zone_to_idx)

    # Drop rows where zone mapping failed
    valid = pu_idx.notna() & do_idx.notna()
    df = df.loc[valid].copy()
    pu_idx = pu_idx.loc[valid].astype(int).values
    do_idx = do_idx.loc[valid].astype(int).values

    df["osrm_base_time_seconds"] = duration_matrix[pu_idx, do_idx]
    df["osrm_base_distance_m"] = distance_matrix[pu_idx, do_idx]

    # Filter out unreachable pairs
    unreachable = df["osrm_base_time_seconds"] >= 99000
    if unreachable.any():
        logger.info("Removing %d trips with unreachable OSRM pairs", unreachable.sum())
        df = df.loc[~unreachable]

    return df


def main():
    # Step 1: Load raw data
    tlc_files = sorted(RAW_DIR.glob("nyc_tlc/yellow_tripdata_*.parquet"))
    if not tlc_files:
        logger.error("No NYC TLC files found in %s", RAW_DIR / "nyc_tlc")
        logger.info("Run scripts/download_data.sh first.")
        sys.exit(1)

    logger.info("Found %d TLC files: %s", len(tlc_files), [f.name for f in tlc_files])
    df = load_nyc_tlc(tlc_files)

    # Step 2: Load taxi zones
    zones_dir = RAW_DIR / "nyc_tlc" / "taxi_zones"
    shapefile = zones_dir / "taxi_zones.shp"
    if not shapefile.exists():
        logger.error("Taxi zone shapefile not found at %s", shapefile)
        sys.exit(1)

    zones_gdf = load_nyc_taxi_zones(shapefile)

    # Step 3: Preprocess
    preprocessor = NycTlcPreprocessor()
    df = preprocessor.filter_trips(df)
    df = preprocessor.add_zone_centroids(df, zones_gdf)

    # Step 4: Compute zone-to-zone OSRM matrix
    osrm = OSRMClient(base_url="http://localhost:5000")
    if not osrm.health_check():
        logger.error("OSRM server not reachable at http://localhost:5000")
        logger.info("Start OSRM with: docker compose up osrm-backend")
        sys.exit(1)

    dur_matrix, dist_matrix, zone_ids = build_zone_osrm_matrix(zones_gdf, osrm)
    osrm.close()

    # Step 5: Add OSRM features to each trip
    df = add_osrm_features(df, dur_matrix, dist_matrix, zone_ids)
    logger.info("After OSRM features: %d trips", len(df))

    # Step 6: Engineer features
    fe = FeatureEngineer()
    features_df = fe.transform(df)

    # Combine features with target and metadata
    output_df = features_df.copy()
    output_df["trip_duration_seconds"] = df["trip_duration_seconds"].values
    output_df["pickup_datetime"] = df["pickup_datetime"].values

    # Step 7: Temporal split
    train, val, test = preprocessor.temporal_split(output_df)

    # Step 8: Save
    preprocessor.save_splits(train, val, test, PROCESSED_DIR)

    logger.info("Processing complete!")
    logger.info("  Train: %d rows", len(train))
    logger.info("  Val:   %d rows", len(val))
    logger.info("  Test:  %d rows", len(test))
    logger.info("  Features: %s", list(features_df.columns))


if __name__ == "__main__":
    main()

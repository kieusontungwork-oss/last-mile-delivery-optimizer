"""Data loading utilities for NYC TLC and Amazon Last Mile datasets."""

import json
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)

# NYC TLC columns we need
TLC_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_distance",
]


def load_nyc_tlc(paths: list[str | Path]) -> pd.DataFrame:
    """Load NYC TLC Yellow Taxi Parquet files.

    Args:
        paths: List of paths to Parquet files.

    Returns:
        DataFrame with pickup/dropoff datetime, zone IDs, trip distance, and duration.
    """
    frames = []
    for p in paths:
        logger.info("Loading %s", p)
        df = pd.read_parquet(p, columns=TLC_COLUMNS)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    # Compute trip duration in seconds
    df["trip_duration_seconds"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds()

    # Rename for consistency
    df = df.rename(columns={
        "tpep_pickup_datetime": "pickup_datetime",
        "tpep_dropoff_datetime": "dropoff_datetime",
        "PULocationID": "pickup_zone",
        "DOLocationID": "dropoff_zone",
    })

    logger.info("Loaded %d trips from %d files", len(df), len(paths))
    return df


def load_nyc_taxi_zones(shapefile_path: str | Path) -> gpd.GeoDataFrame:
    """Load NYC taxi zone shapefile and compute zone centroids.

    Returns:
        GeoDataFrame with columns: LocationID, zone, borough, geometry, centroid_lat, centroid_lng.
    """
    gdf = gpd.read_file(shapefile_path)
    # Reproject to WGS84 if needed
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    centroids = gdf.geometry.centroid
    gdf["centroid_lat"] = centroids.y
    gdf["centroid_lng"] = centroids.x

    logger.info("Loaded %d taxi zones", len(gdf))
    return gdf


def load_amazon_routes(data_dir: str | Path) -> pd.DataFrame:
    """Load Amazon Last Mile routing data and extract stop-to-stop travel times.

    Args:
        data_dir: Path to the Amazon Last Mile dataset root.

    Returns:
        DataFrame with columns for origin/destination coordinates, travel time, and features.
    """
    data_dir = Path(data_dir)
    route_dir = data_dir / "model_apply_inputs" / "new_route_data"
    travel_time_dir = data_dir / "model_apply_inputs" / "new_travel_times"

    if not route_dir.exists():
        # Try alternative directory structures
        route_dir = data_dir / "model_build_inputs" / "route_data"
        travel_time_dir = data_dir / "model_build_inputs" / "travel_times"

    if not route_dir.exists():
        raise FileNotFoundError(
            f"Could not find route data in {data_dir}. "
            "Expected model_apply_inputs/new_route_data or model_build_inputs/route_data"
        )

    records = []
    route_files = sorted(route_dir.glob("*.json"))
    logger.info("Found %d route files", len(route_files))

    for route_file in route_files:
        route_id = route_file.stem

        with open(route_file) as f:
            route_data = json.load(f)

        # Load travel times for this route
        tt_file = travel_time_dir / f"{route_id}.json"
        if not tt_file.exists():
            continue

        with open(tt_file) as f:
            travel_times = json.load(f)

        # Extract stops with coordinates
        stops = {}
        for stop_id, stop_info in route_data.get("stops", {}).items():
            stops[stop_id] = {
                "lat": stop_info.get("lat", 0),
                "lng": stop_info.get("lng", 0),
                "zone_id": stop_info.get("zone_id", ""),
            }

        # Extract pairwise travel times
        for origin_id, destinations in travel_times.items():
            if origin_id not in stops:
                continue
            for dest_id, tt in destinations.items():
                if dest_id not in stops or tt is None:
                    continue
                records.append({
                    "route_id": route_id,
                    "origin_lat": stops[origin_id]["lat"],
                    "origin_lng": stops[origin_id]["lng"],
                    "dest_lat": stops[dest_id]["lat"],
                    "dest_lng": stops[dest_id]["lng"],
                    "origin_zone": stops[origin_id]["zone_id"],
                    "dest_zone": stops[dest_id]["zone_id"],
                    "travel_time_seconds": float(tt),
                })

    df = pd.DataFrame(records)
    logger.info("Extracted %d stop-to-stop travel time records", len(df))
    return df

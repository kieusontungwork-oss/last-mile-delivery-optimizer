"""OSRM REST API client for routing and travel time matrices."""

import logging

import httpx
import numpy as np

logger = logging.getLogger(__name__)

# Large fallback for unreachable pairs
UNREACHABLE_DURATION = 99999.0
UNREACHABLE_DISTANCE = 99999.0


class OSRMClient:
    """Client for the OSRM routing engine REST API."""

    def __init__(self, base_url: str = "http://localhost:5000", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def health_check(self) -> bool:
        """Check if OSRM server is reachable."""
        try:
            resp = self._client.get(f"{self.base_url}/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def get_duration_matrix(
        self, locations: list[tuple[float, float]]
    ) -> np.ndarray:
        """Compute NxN travel time matrix using OSRM Table API.

        Args:
            locations: List of (lat, lng) tuples.

        Returns:
            NxN numpy array of travel times in seconds.
            Unreachable pairs get UNREACHABLE_DURATION.
        """
        return self._get_table(locations, annotations="duration", fallback=UNREACHABLE_DURATION)

    def get_distance_matrix(
        self, locations: list[tuple[float, float]]
    ) -> np.ndarray:
        """Compute NxN distance matrix using OSRM Table API.

        Args:
            locations: List of (lat, lng) tuples.

        Returns:
            NxN numpy array of distances in meters.
        """
        return self._get_table(locations, annotations="distance", fallback=UNREACHABLE_DISTANCE)

    def get_duration_and_distance_matrices(
        self, locations: list[tuple[float, float]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get both duration and distance matrices in one API call.

        Returns:
            Tuple of (duration_matrix, distance_matrix).
        """
        # OSRM supports annotations=duration,distance
        coords = ";".join(f"{lng},{lat}" for lat, lng in locations)
        url = f"{self.base_url}/table/v1/driving/{coords}"
        resp = self._client.get(url, params={"annotations": "duration,distance"})
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != "Ok":
            raise RuntimeError(f"OSRM Table API error: {data.get('code')}: {data.get('message')}")

        durations = self._parse_matrix(data["durations"], UNREACHABLE_DURATION)
        distances = self._parse_matrix(data["distances"], UNREACHABLE_DISTANCE)
        return durations, distances

    def get_route(
        self, origin: tuple[float, float], destination: tuple[float, float]
    ) -> dict:
        """Get route between two points.

        Args:
            origin: (lat, lng) tuple.
            destination: (lat, lng) tuple.

        Returns:
            Dict with keys: duration (seconds), distance (meters), geometry (list of [lat, lng]).
        """
        coords = f"{origin[1]},{origin[0]};{destination[1]},{destination[0]}"
        url = f"{self.base_url}/route/v1/driving/{coords}"
        resp = self._client.get(url, params={
            "overview": "full",
            "geometries": "geojson",
        })
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != "Ok" or not data.get("routes"):
            return {"duration": 0, "distance": 0, "geometry": []}

        route = data["routes"][0]
        # Convert GeoJSON [lng, lat] to [lat, lng]
        coords_geojson = route["geometry"]["coordinates"]
        geometry = [[c[1], c[0]] for c in coords_geojson]

        return {
            "duration": route["duration"],
            "distance": route["distance"],
            "geometry": geometry,
        }

    def get_route_geometries(
        self, waypoint_sequences: list[list[tuple[float, float]]]
    ) -> list[list[list[float]]]:
        """Get route geometries for multiple sequences of waypoints.

        Args:
            waypoint_sequences: List of routes, each a list of (lat, lng) waypoints.

        Returns:
            List of geometries, each a list of [lat, lng] coordinate pairs.
        """
        geometries = []
        for waypoints in waypoint_sequences:
            if len(waypoints) < 2:
                geometries.append([])
                continue

            # Build multi-waypoint route
            coords = ";".join(f"{lng},{lat}" for lat, lng in waypoints)
            url = f"{self.base_url}/route/v1/driving/{coords}"
            resp = self._client.get(url, params={
                "overview": "full",
                "geometries": "geojson",
            })
            resp.raise_for_status()
            data = resp.json()

            if data.get("code") != "Ok" or not data.get("routes"):
                geometries.append([])
                continue

            coords_geojson = data["routes"][0]["geometry"]["coordinates"]
            geometry = [[c[1], c[0]] for c in coords_geojson]
            geometries.append(geometry)

        return geometries

    def _get_table(
        self,
        locations: list[tuple[float, float]],
        annotations: str,
        fallback: float,
    ) -> np.ndarray:
        """Internal: call OSRM Table API."""
        # CRITICAL: OSRM uses lng,lat order (not lat,lng)
        coords = ";".join(f"{lng},{lat}" for lat, lng in locations)
        url = f"{self.base_url}/table/v1/driving/{coords}"
        resp = self._client.get(url, params={"annotations": annotations})
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != "Ok":
            raise RuntimeError(f"OSRM Table API error: {data.get('code')}: {data.get('message')}")

        key = annotations.split(",")[0] + "s"  # "duration" -> "durations"
        return self._parse_matrix(data[key], fallback)

    @staticmethod
    def _parse_matrix(matrix: list[list[float | None]], fallback: float) -> np.ndarray:
        """Replace null values with fallback."""
        arr = np.array(matrix, dtype=float)
        arr = np.where(np.isnan(arr), fallback, arr)
        # Also replace None-converted zeros on diagonal... actually diagonal should be 0
        return arr

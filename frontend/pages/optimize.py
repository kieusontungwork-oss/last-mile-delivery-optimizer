"""Optimize page: submit delivery stops, get optimized routes."""

import os
import time

import requests
import streamlit as st
from streamlit_folium import st_folium

from frontend.components.map_display import create_empty_map, create_route_map

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

# Preset scenarios
PRESETS = {
    "Manhattan 10 stops": {
        "depot": {"lat": 40.7484, "lng": -73.9857, "name": "Midtown Depot"},
        "stops": [
            {"id": "S1", "lat": 40.7580, "lng": -73.9855, "demand": 5},
            {"id": "S2", "lat": 40.7614, "lng": -73.9776, "demand": 3},
            {"id": "S3", "lat": 40.7527, "lng": -73.9772, "demand": 8},
            {"id": "S4", "lat": 40.7489, "lng": -73.9680, "demand": 4},
            {"id": "S5", "lat": 40.7282, "lng": -73.7949, "demand": 6},
            {"id": "S6", "lat": 40.7061, "lng": -74.0087, "demand": 2},
            {"id": "S7", "lat": 40.7128, "lng": -74.0060, "demand": 7},
            {"id": "S8", "lat": 40.7411, "lng": -74.0018, "demand": 3},
            {"id": "S9", "lat": 40.7549, "lng": -73.9840, "demand": 5},
            {"id": "S10", "lat": 40.7681, "lng": -73.9819, "demand": 4},
        ],
    },
    "Brooklyn 20 stops": {
        "depot": {"lat": 40.6892, "lng": -73.9857, "name": "Brooklyn Depot"},
        "stops": [
            {"id": f"B{i+1}", "lat": 40.67 + i * 0.005, "lng": -73.97 + (i % 5) * 0.005, "demand": (i % 5) + 1}
            for i in range(20)
        ],
    },
}


def render():
    st.header("Route Optimization")

    # Initialize session state for results persistence
    if "opt_result" not in st.session_state:
        st.session_state.opt_result = None
    if "opt_depot" not in st.session_state:
        st.session_state.opt_depot = None

    # Sidebar config
    with st.sidebar:
        st.subheader("Configuration")
        preset = st.selectbox("Preset scenario", ["Custom"] + list(PRESETS.keys()))
        num_vehicles = st.number_input("Vehicles", min_value=1, max_value=20, value=3)
        capacity = st.number_input("Vehicle capacity", min_value=1, max_value=1000, value=50)
        max_solve_time = st.slider("Max solve time (s)", 1, 120, 30)
        use_ml = st.checkbox("Use ML-adjusted costs", value=True)
        solver = st.selectbox("Solver", ["pyvrp", "ortools"])

    # Input section
    if preset != "Custom" and preset in PRESETS:
        scenario = PRESETS[preset]
        depot = scenario["depot"]
        stops = scenario["stops"]
        st.success(f"Loaded preset: {preset} ({len(stops)} stops)")
    else:
        st.subheader("Enter stops (CSV: id,lat,lng,demand)")
        csv_input = st.text_area(
            "Stops CSV",
            value="S1,40.7580,-73.9855,5\nS2,40.7614,-73.9776,3\nS3,40.7527,-73.9772,8",
            height=150,
        )
        depot_lat = st.number_input("Depot lat", value=40.7484, format="%.4f")
        depot_lng = st.number_input("Depot lng", value=-73.9857, format="%.4f")
        depot = {"lat": depot_lat, "lng": depot_lng, "name": "Depot"}
        stops = _parse_csv(csv_input)

    # Submit
    if st.button("Optimize Routes", type="primary"):
        if not stops:
            st.error("No stops provided.")
            return

        payload = {
            "depot": depot,
            "stops": stops,
            "vehicles": [{"id": f"V{i+1}", "capacity": capacity} for i in range(num_vehicles)],
            "config": {
                "use_ml": use_ml,
                "max_solve_time_seconds": max_solve_time,
                "solver": solver,
            },
        }

        with st.spinner("Optimizing routes..."):
            try:
                # Submit job
                resp = requests.post(f"{API_BASE}/api/optimize", json=payload, timeout=5)
                resp.raise_for_status()
                job_id = resp.json()["job_id"]

                # Poll for result
                result = _poll_job(job_id, timeout=max_solve_time + 10)

                if result["status"] == "completed" and result.get("result"):
                    st.session_state.opt_result = result["result"]
                    st.session_state.opt_depot = depot
                elif result["status"] == "failed":
                    st.error(f"Optimization failed: {result.get('error', 'Unknown error')}")
                    st.session_state.opt_result = None
                else:
                    st.warning(f"Job status: {result['status']}")
                    st.session_state.opt_result = None

            except requests.ConnectionError:
                st.error(
                    f"Cannot connect to API at {API_BASE}. "
                    "Make sure the API server is running."
                )
            except Exception as e:
                st.error(f"Error: {e}")

    # Display results or empty map
    if st.session_state.opt_result and st.session_state.opt_depot:
        _display_results(st.session_state.opt_depot, st.session_state.opt_result)
    else:
        m = create_empty_map()
        st_folium(m, width=800, height=500)


def _parse_csv(csv_text: str) -> list[dict]:
    """Parse CSV text into stop dicts."""
    stops = []
    for line in csv_text.strip().split("\n"):
        parts = line.strip().split(",")
        if len(parts) >= 3:
            stops.append({
                "id": parts[0].strip(),
                "lat": float(parts[1].strip()),
                "lng": float(parts[2].strip()),
                "demand": int(parts[3].strip()) if len(parts) > 3 else 1,
            })
    return stops


def _poll_job(job_id: str, timeout: int = 60) -> dict:
    """Poll for job completion."""
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.get(f"{API_BASE}/api/optimize/{job_id}", timeout=5)
        data = resp.json()
        if data["status"] in ("completed", "failed"):
            return data
        time.sleep(1)
    return {"status": "timeout"}


def _display_results(depot: dict, result: dict):
    """Display optimization results with map and metrics."""
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Vehicles Used", result["num_vehicles_used"])
    col2.metric("Total Distance", f"{result['total_distance_km']:.1f} km")
    col3.metric("Total Time", f"{result['total_time_minutes']:.0f} min")
    col4.metric("Solve Time", f"{result['solve_time_seconds']:.1f}s")

    # Map
    routes_for_map = []
    for r in result["routes"]:
        routes_for_map.append({
            "vehicle_id": r["vehicle_id"],
            "stops": [{"id": s["id"], "lat": s["lat"], "lng": s["lng"]} for s in r["stops"]],
            "geometry": r.get("geometry", []),
            "total_distance_km": r["total_distance_km"],
            "total_time_minutes": r["total_time_minutes"],
        })

    m = create_route_map(depot, routes_for_map)
    st_folium(m, width=800, height=500)

    # Route details table
    st.subheader("Route Details")
    for r in result["routes"]:
        with st.expander(f"{r['vehicle_id']} - {len(r['stops'])} stops"):
            stop_ids = [s["id"] for s in r["stops"]]
            st.write(f"**Sequence**: {' -> '.join(stop_ids)}")
            st.write(f"**Distance**: {r['total_distance_km']:.1f} km")
            st.write(f"**Time**: {r['total_time_minutes']:.0f} min")

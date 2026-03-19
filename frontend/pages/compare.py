"""Compare page: side-by-side static vs ML-adjusted routing."""

import os
import time

import requests
import streamlit as st
from streamlit_folium import st_folium

from frontend.components.map_display import create_route_map

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")


def render():
    st.header("Static vs ML-Adjusted Comparison")

    with st.sidebar:
        st.subheader("Comparison Config")
        num_vehicles = st.number_input("Vehicles", min_value=1, max_value=20, value=3, key="cmp_v")
        capacity = st.number_input("Capacity", min_value=1, max_value=1000, value=50, key="cmp_c")
        max_solve_time = st.slider("Max solve time (s)", 1, 120, 30, key="cmp_t")
        solver = st.selectbox("Solver", ["pyvrp", "ortools"], key="cmp_s")

    # Use Manhattan preset
    depot = {"lat": 40.7484, "lng": -73.9857, "name": "Midtown Depot"}
    stops = [
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
    ]

    if st.button("Run Comparison", type="primary"):
        vehicles = [{"id": f"V{i+1}", "capacity": capacity} for i in range(num_vehicles)]

        with st.spinner("Running static optimization..."):
            static_result = _run_optimization(
                depot, stops, vehicles, max_solve_time, solver, use_ml=False
            )

        with st.spinner("Running ML-adjusted optimization..."):
            ml_result = _run_optimization(
                depot, stops, vehicles, max_solve_time, solver, use_ml=True
            )

        if static_result and ml_result:
            _display_comparison(depot, static_result, ml_result)
        else:
            st.error("One or both optimizations failed. Make sure the API is running.")


def _run_optimization(
    depot: dict, stops: list, vehicles: list,
    max_solve_time: int, solver: str, use_ml: bool,
) -> dict | None:
    """Submit optimization and wait for result."""
    payload = {
        "depot": depot,
        "stops": stops,
        "vehicles": vehicles,
        "config": {
            "use_ml": use_ml,
            "max_solve_time_seconds": max_solve_time,
            "solver": solver,
        },
    }

    try:
        resp = requests.post(f"{API_BASE}/api/optimize", json=payload, timeout=5)
        resp.raise_for_status()
        job_id = resp.json()["job_id"]

        start = time.time()
        while time.time() - start < max_solve_time + 10:
            resp = requests.get(f"{API_BASE}/api/optimize/{job_id}", timeout=5)
            data = resp.json()
            if data["status"] == "completed":
                return data["result"]
            if data["status"] == "failed":
                return None
            time.sleep(1)
    except Exception:
        return None
    return None


def _display_comparison(depot: dict, static: dict, ml: dict):
    """Display side-by-side comparison."""
    # Metrics comparison
    st.subheader("Metrics Comparison")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Metric**")
        st.write("Total Distance")
        st.write("Total Time")
        st.write("Vehicles Used")
        st.write("Solve Time")
    with col2:
        st.write("**Static**")
        st.write(f"{static['total_distance_km']:.1f} km")
        st.write(f"{static['total_time_minutes']:.0f} min")
        st.write(str(static["num_vehicles_used"]))
        st.write(f"{static['solve_time_seconds']:.1f}s")
    with col3:
        st.write("**ML-Adjusted**")
        st.write(f"{ml['total_distance_km']:.1f} km")
        st.write(f"{ml['total_time_minutes']:.0f} min")
        st.write(str(ml["num_vehicles_used"]))
        st.write(f"{ml['solve_time_seconds']:.1f}s")

    # Improvement
    if static["total_time_minutes"] > 0:
        time_imp = (
            (static["total_time_minutes"] - ml["total_time_minutes"])
            / static["total_time_minutes"] * 100
        )
        st.metric("Time Improvement", f"{time_imp:.1f}%")

    # Side-by-side maps
    st.subheader("Route Maps")
    left, right = st.columns(2)

    with left:
        st.write("**Static Routes**")
        static_routes = _format_routes(static)
        m1 = create_route_map(depot, static_routes)
        st_folium(m1, width=500, height=400, key="static_map")

    with right:
        st.write("**ML-Adjusted Routes**")
        ml_routes = _format_routes(ml)
        m2 = create_route_map(depot, ml_routes)
        st_folium(m2, width=500, height=400, key="ml_map")


def _format_routes(result: dict) -> list[dict]:
    """Convert API result routes into format expected by create_route_map."""
    return [
        {
            "vehicle_id": r["vehicle_id"],
            "stops": [{"id": s["id"], "lat": s["lat"], "lng": s["lng"]} for s in r["stops"]],
            "geometry": r.get("geometry", []),
            "total_distance_km": r["total_distance_km"],
            "total_time_minutes": r["total_time_minutes"],
        }
        for r in result["routes"]
    ]

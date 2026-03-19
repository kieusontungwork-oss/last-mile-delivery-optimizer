"""Folium map rendering for route visualization."""

import folium

VEHICLE_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]


def create_route_map(
    depot: dict,
    routes: list[dict],
    center: list[float] | None = None,
    zoom_start: int = 13,
) -> folium.Map:
    """Create a Folium map with depot, stops, and route polylines.

    Args:
        depot: Dict with "lat" and "lng" keys.
        routes: List of route dicts, each with "vehicle_id", "stops", "geometry",
                "total_distance_km", "total_time_minutes".
        center: Optional [lat, lng] for map center.
        zoom_start: Initial zoom level.

    Returns:
        Folium Map object.
    """
    if center is None:
        center = [depot["lat"], depot["lng"]]

    m = folium.Map(location=center, zoom_start=zoom_start, tiles="CartoDB positron")

    # Depot marker
    folium.Marker(
        location=[depot["lat"], depot["lng"]],
        popup="<b>Depot</b>",
        icon=folium.Icon(color="green", icon="home", prefix="fa"),
    ).add_to(m)

    # Routes
    for idx, route in enumerate(routes):
        color = VEHICLE_COLORS[idx % len(VEHICLE_COLORS)]

        # Stop markers
        for i, stop in enumerate(route.get("stops", [])):
            folium.CircleMarker(
                location=[stop["lat"], stop["lng"]],
                radius=7,
                color=color,
                fill=True,
                fill_opacity=0.8,
                popup=(
                    f"<b>Stop {stop['id']}</b><br>"
                    f"Order: {i + 1}<br>"
                    f"ETA: {stop.get('arrival_time', 'N/A')}"
                ),
                tooltip=f"{stop['id']} (#{i + 1})",
            ).add_to(m)

        # Route polyline
        geometry = route.get("geometry", [])
        if geometry:
            folium.PolyLine(
                geometry,
                weight=4,
                color=color,
                opacity=0.8,
                popup=(
                    f"<b>{route['vehicle_id']}</b><br>"
                    f"Distance: {route.get('total_distance_km', 0):.1f} km<br>"
                    f"Time: {route.get('total_time_minutes', 0):.0f} min<br>"
                    f"Stops: {len(route.get('stops', []))}"
                ),
            ).add_to(m)

    # Fit bounds to all points
    all_points = [[depot["lat"], depot["lng"]]]
    for route in routes:
        for stop in route.get("stops", []):
            all_points.append([stop["lat"], stop["lng"]])
    if len(all_points) > 1:
        m.fit_bounds(all_points)

    return m


def create_empty_map(
    center: list[float] | None = None,
    zoom_start: int = 12,
) -> folium.Map:
    """Create an empty map centered on Manhattan."""
    if center is None:
        center = [40.7580, -73.9855]
    return folium.Map(location=center, zoom_start=zoom_start, tiles="CartoDB positron")

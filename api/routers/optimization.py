"""Optimization endpoints: submit and poll VRP solutions."""

import asyncio
import logging
import uuid

from fastapi import APIRouter, HTTPException

from api.main import app_state
from api.schemas.optimization import (
    OptimizeRequest,
    OptimizeResponse,
    OptimizeResult,
    RouteResult,
    StopResult,
)
from src.optimization import get_solver
from src.optimization.vrp_solver import Stop

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/optimize", status_code=202)
async def create_optimization(request: OptimizeRequest) -> OptimizeResponse:
    """Submit a route optimization job. Returns immediately with a job ID."""
    job_id = str(uuid.uuid4())
    response = OptimizeResponse(job_id=job_id, status="pending")
    app_state.jobs[job_id] = response

    loop = asyncio.get_event_loop()
    loop.run_in_executor(app_state.executor, _run_optimization, job_id, request)

    return response


@router.get("/optimize/{job_id}")
async def get_optimization(job_id: str) -> OptimizeResponse:
    """Poll for optimization result."""
    if job_id not in app_state.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return app_state.jobs[job_id]


def _run_optimization(job_id: str, request: OptimizeRequest) -> None:
    """Execute optimization in a background thread."""
    try:
        app_state.jobs[job_id].status = "running"

        # Build locations list: depot first, then stops
        locations = [(request.depot.lat, request.depot.lng)]
        stops = []
        stop_map = {}  # stop_id -> DeliveryStop

        for s in request.stops:
            locations.append((s.lat, s.lng))
            stop_map[s.id] = s
            stops.append(Stop(
                id=s.id,
                lat=s.lat,
                lng=s.lng,
                demand=s.demand,
                service_time=s.service_time_minutes * 60,
            ))

        # Build cost matrix
        config = request.config
        if config.use_ml and app_state.cost_builder:
            cost_matrix = app_state.cost_builder.build_dynamic_matrix(
                locations, config.departure_time
            )
        elif app_state.cost_builder:
            cost_matrix = app_state.cost_builder.build_static_matrix(locations)
        else:
            raise RuntimeError("Cost matrix builder not initialized")

        # Solve VRP
        vehicle_capacity = request.vehicles[0].capacity if request.vehicles else 100
        num_vehicles = len(request.vehicles)

        solver = get_solver(config.solver.value)
        solution = solver.solve(
            cost_matrix=cost_matrix,
            stops=stops,
            depot_coords=(request.depot.lat, request.depot.lng),
            vehicle_capacity=vehicle_capacity,
            num_vehicles=num_vehicles,
            max_runtime=config.max_solve_time_seconds,
            scaling_factor=app_state.cost_builder.scaling_factor,
        )

        # Fetch route geometries from OSRM
        route_results = []
        for route in solution.routes:
            # Build waypoint sequence: depot -> stops -> depot
            waypoints = [(request.depot.lat, request.depot.lng)]
            stop_results = []

            for sid in route.stop_ids:
                ds = stop_map[sid]
                waypoints.append((ds.lat, ds.lng))
                stop_results.append(StopResult(id=sid, lat=ds.lat, lng=ds.lng))

            waypoints.append((request.depot.lat, request.depot.lng))

            # Get road-following geometry
            geometry = []
            if app_state.osrm_client:
                geoms = app_state.osrm_client.get_route_geometries([waypoints])
                if geoms and geoms[0]:
                    geometry = geoms[0]

            route_results.append(RouteResult(
                vehicle_id=route.vehicle_id,
                stops=stop_results,
                total_distance_km=route.distance / 1000 if route.distance > 100 else route.distance,
                total_time_minutes=route.duration / 60,
                geometry=geometry,
            ))

        result = OptimizeResult(
            routes=route_results,
            total_distance_km=sum(r.total_distance_km for r in route_results),
            total_time_minutes=sum(r.total_time_minutes for r in route_results),
            num_vehicles_used=solution.num_vehicles,
            solve_time_seconds=solution.solve_time,
            ml_enabled=config.use_ml,
            solver_used=config.solver.value,
        )

        app_state.jobs[job_id].status = "completed"
        app_state.jobs[job_id].result = result

    except Exception as e:
        logger.exception("Optimization job %s failed", job_id)
        app_state.jobs[job_id].status = "failed"
        app_state.jobs[job_id].error = str(e)

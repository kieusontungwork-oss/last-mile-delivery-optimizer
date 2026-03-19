"""ETA prediction and cost matrix endpoints."""

import logging

from fastapi import APIRouter, HTTPException

from api.main import app_state
from api.schemas.prediction import (
    CostMatrixRequest,
    CostMatrixResponse,
    ETAPredictionRequest,
    ETAPredictionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/predict-eta")
async def predict_eta(request: ETAPredictionRequest) -> ETAPredictionResponse:
    """Predict travel time between two points."""
    if not app_state.osrm_client:
        raise HTTPException(status_code=503, detail="OSRM client not available")

    # Get OSRM base time
    route = app_state.osrm_client.get_route(
        origin=(request.origin_lat, request.origin_lng),
        destination=(request.destination_lat, request.destination_lng),
    )
    osrm_base = route["duration"]

    # ML prediction
    if app_state.predictor:
        predicted = app_state.predictor.predict_single(
            osrm_base_time=osrm_base,
            osrm_base_distance=route["distance"],
            origin_lat=request.origin_lat,
            origin_lng=request.origin_lng,
            dest_lat=request.destination_lat,
            dest_lng=request.destination_lng,
            departure_hour=request.departure_time.hour + request.departure_time.minute / 60,
            departure_dow=request.departure_time.weekday(),
            departure_month=request.departure_time.month,
        )
        adjustment = predicted / osrm_base if osrm_base > 0 else 1.0
    else:
        predicted = osrm_base
        adjustment = 1.0

    return ETAPredictionResponse(
        predicted_seconds=predicted,
        predicted_minutes=predicted / 60,
        osrm_base_seconds=osrm_base,
        ml_adjustment_factor=adjustment,
    )


@router.post("/cost-matrix")
async def build_cost_matrix(request: CostMatrixRequest) -> CostMatrixResponse:
    """Build NxN cost matrix for given locations."""
    if not app_state.cost_builder:
        raise HTTPException(status_code=503, detail="Cost matrix builder not available")

    locations = [(loc["lat"], loc["lng"]) for loc in request.locations]

    if request.use_ml:
        matrix = app_state.cost_builder.build_dynamic_matrix(
            locations, request.departure_time
        )
    else:
        matrix = app_state.cost_builder.build_static_matrix(locations)

    # Convert back from integer-scaled to seconds
    sf = app_state.cost_builder.scaling_factor
    duration_matrix = (matrix / sf).tolist()

    return CostMatrixResponse(
        duration_matrix=duration_matrix,
        matrix_size=len(locations),
        ml_enabled=request.use_ml,
    )

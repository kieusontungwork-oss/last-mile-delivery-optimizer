"""Pydantic schemas for prediction endpoints."""

from datetime import datetime

from pydantic import BaseModel, Field


class ETAPredictionRequest(BaseModel):
    origin_lat: float = Field(..., ge=-90, le=90)
    origin_lng: float = Field(..., ge=-180, le=180)
    destination_lat: float = Field(..., ge=-90, le=90)
    destination_lng: float = Field(..., ge=-180, le=180)
    departure_time: datetime = Field(default_factory=datetime.now)


class ETAPredictionResponse(BaseModel):
    predicted_seconds: float
    predicted_minutes: float
    osrm_base_seconds: float
    ml_adjustment_factor: float


class CostMatrixRequest(BaseModel):
    locations: list[dict] = Field(..., min_length=2, max_length=500)
    departure_time: datetime = Field(default_factory=datetime.now)
    use_ml: bool = True


class CostMatrixResponse(BaseModel):
    duration_matrix: list[list[float]]
    distance_matrix: list[list[float]] | None = None
    matrix_size: int
    ml_enabled: bool

"""Pydantic schemas for the optimization API."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Location(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)
    name: str | None = None


class TimeWindow(BaseModel):
    earliest: str  # "HH:MM" format
    latest: str


class DeliveryStop(BaseModel):
    id: str
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)
    demand: int = Field(default=1, ge=0)
    time_window: TimeWindow | None = None
    service_time_minutes: int = Field(default=5, ge=0)


class Vehicle(BaseModel):
    id: str
    capacity: int = Field(default=100, gt=0)


class SolverType(str, Enum):
    PYVRP = "pyvrp"
    ORTOOLS = "ortools"


class OptimizeConfig(BaseModel):
    use_ml: bool = True
    departure_time: datetime = Field(default_factory=datetime.now)
    max_solve_time_seconds: int = Field(default=30, ge=1, le=300)
    solver: SolverType = SolverType.PYVRP


class OptimizeRequest(BaseModel):
    depot: Location
    stops: list[DeliveryStop] = Field(..., min_length=1, max_length=500)
    vehicles: list[Vehicle] = Field(..., min_length=1)
    config: OptimizeConfig = Field(default_factory=OptimizeConfig)


class StopResult(BaseModel):
    id: str
    lat: float
    lng: float
    arrival_time: str | None = None


class RouteResult(BaseModel):
    vehicle_id: str
    stops: list[StopResult]
    total_distance_km: float
    total_time_minutes: float
    geometry: list[list[float]]  # [[lat, lng], ...]


class OptimizeResult(BaseModel):
    routes: list[RouteResult]
    total_distance_km: float
    total_time_minutes: float
    num_vehicles_used: int
    solve_time_seconds: float
    ml_enabled: bool
    solver_used: str


class OptimizeResponse(BaseModel):
    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    result: OptimizeResult | None = None
    error: str | None = None

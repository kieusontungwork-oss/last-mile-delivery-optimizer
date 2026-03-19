"""Health check endpoint."""

from fastapi import APIRouter

from api.main import app_state

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    """Check system health: OSRM connectivity, model status, active jobs."""
    osrm_ok = False
    if app_state.osrm_client:
        osrm_ok = app_state.osrm_client.health_check()

    model_loaded = app_state.predictor is not None
    active_jobs = len([j for j in app_state.jobs.values() if j.status == "running"])

    return {
        "status": "healthy" if (osrm_ok and model_loaded) else "degraded",
        "osrm_connected": osrm_ok,
        "model_loaded": model_loaded,
        "active_jobs": active_jobs,
    }

"""FastAPI application for the Last-Mile Delivery Optimizer."""

import logging
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.models.predict import ETAPredictor
from src.optimization.cost_matrix import CostMatrixBuilder
from src.optimization.osrm_client import OSRMClient
from src.utils.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AppState:
    """Shared application state initialized at startup."""

    predictor: ETAPredictor | None = None
    osrm_client: OSRMClient | None = None
    cost_builder: CostMatrixBuilder | None = None
    executor: ThreadPoolExecutor | None = None
    jobs: dict = {}  # job_id -> OptimizeResponse


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML model and initialize services at startup."""
    settings = get_settings()

    # Initialize OSRM client
    app_state.osrm_client = OSRMClient(
        base_url=settings.osrm.base_url,
        timeout=settings.osrm.timeout,
    )

    # Load ML model (if available)
    model_path = Path(settings.model.model_path)
    if model_path.exists():
        try:
            app_state.predictor = ETAPredictor(
                model_path=settings.model.model_path,
                metadata_path=settings.model.metadata_path,
            )
            logger.info("ML model loaded successfully")
        except Exception as e:
            logger.warning("Failed to load ML model: %s. Running without ML.", e)
    else:
        logger.info("No ML model found at %s. Running without ML.", model_path)

    # Build cost matrix builder
    app_state.cost_builder = CostMatrixBuilder(
        osrm_client=app_state.osrm_client,
        eta_predictor=app_state.predictor,
        scaling_factor=settings.model.cost_scaling_factor,
    )

    # Thread pool for VRP solving
    app_state.executor = ThreadPoolExecutor(max_workers=4)

    osrm_ok = app_state.osrm_client.health_check()
    logger.info("Startup complete. OSRM=%s, ML=%s", osrm_ok, app_state.predictor is not None)

    yield

    # Cleanup
    if app_state.osrm_client:
        app_state.osrm_client.close()
    if app_state.executor:
        app_state.executor.shutdown(wait=False)


app = FastAPI(
    title="Last-Mile Delivery Optimizer",
    description="ML-driven dynamic routing for last-mile delivery optimization",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
from api.routers import health, optimization, prediction  # noqa: E402

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(optimization.router, prefix="/api", tags=["optimization"])
app.include_router(prediction.router, prefix="/api", tags=["prediction"])

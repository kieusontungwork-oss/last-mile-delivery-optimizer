# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML-driven last-mile delivery optimizer that combines ETA prediction (LightGBM) with Vehicle Routing Problem (VRP) solvers (PyVRP, OR-Tools) to produce dynamic delivery routes. Uses OSRM for road-network travel times.

## Commands

```bash
# Install dependencies (uses uv, not pip)
uv sync --dev

# Run API server
uv run uvicorn api.main:app --reload --port 8000

# Run Streamlit frontend
uv run streamlit run frontend/app.py

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/unit/test_vrp_solver.py

# Run only unit tests (skip integration tests that need OSRM)
uv run pytest -m "not integration"

# Lint
uv run ruff check .
uv run ruff format --check .

# Start OSRM + full stack via Docker
docker compose up
```

## Architecture

**Three-layer pipeline:** ML prediction → cost matrix construction → VRP solving

### Data Flow
1. **OSRM** (`src/optimization/osrm_client.py`) provides base travel time/distance matrices via its Table API. Note: OSRM uses `lng,lat` coordinate order.
2. **CostMatrixBuilder** (`src/optimization/cost_matrix.py`) builds integer-scaled NxN cost matrices. In "static" mode it uses raw OSRM times; in "dynamic" mode it runs ML-adjusted predictions through `ETAPredictor`.
3. **VRP solvers** (`src/optimization/vrp_solver.py`) take the cost matrix and produce routes. Two solvers: `PyVRPSolver` (Hybrid Genetic Search, CVRPTW) and `ORToolsSolver` (CVRP with Guided Local Search). Both share the same interface: `solve(cost_matrix, stops, ...) -> VRPSolution`.
4. Cost matrices are integer-scaled by `scaling_factor` (default 100) — solvers work in integers, results are divided back.

### ML Pipeline
- **Features** (`src/features/engineering.py`): `FEATURE_NAMES` list is the canonical feature contract. Cyclical encoding for time features, haversine distance, bearing, OSRM base time/distance, average speed. Categorical features: `pickup_zone`, `dropoff_zone`.
- **Training** (`src/models/train.py`): LightGBM primary model, Random Forest baseline. Models saved as `.joblib` with companion `_metadata.json`.
- **Prediction** (`src/models/predict.py`): `ETAPredictor` loads model + validates feature alignment against `FEATURE_NAMES`.
- **Scripts**: `scripts/process_data.py` → `scripts/train_model.py` → `scripts/run_evaluation.py`. Data pipeline: raw parquet → processed train/val/test splits → trained models.

### API (`api/`)
- FastAPI app in `api/main.py` with lifespan-managed state (`AppState`): loads ML model, OSRM client, cost builder at startup.
- Optimization is async: POST `/api/optimize` returns 202 with job ID, GET `/api/optimize/{job_id}` polls for result. VRP solving runs in a `ThreadPoolExecutor`.
- Request/response schemas in `api/schemas/`.

### Frontend (`frontend/`)
- Streamlit app with three pages: Optimize, Compare, Dashboard. Uses `streamlit-folium` for map rendering.

## Configuration

- App settings via `pydantic-settings` in `src/utils/config.py`. Env prefix `LMO_`, nested delimiter `__` (e.g., `LMO_OSRM__BASE_URL`).
- Model hyperparams in `configs/model_config.yaml`, solver defaults in `configs/solver_config.yaml`.

## Key Conventions

- Python 3.12+, managed with `uv`.
- Ruff for linting/formatting (line length 100, rules: E, F, I, W).
- Cost matrix index 0 is always the depot; stops are indices 1..N.
- Integration tests (marked `@pytest.mark.integration`) require a running OSRM instance.
- Models and raw data are gitignored — regenerate via scripts.

# Last-Mile Delivery Optimizer

ML-driven dynamic routing for last-mile delivery optimization. Combines LightGBM ETA prediction with state-of-the-art VRP solvers (PyVRP, OR-Tools) and a local OSRM routing engine to produce delivery routes that adapt to time-of-day congestion patterns.

The core idea: instead of feeding static travel times into a route optimizer, train an ML model on real trip data (NYC TLC) to predict context-aware travel times, build a dynamic cost matrix, and let the VRP solver produce better routes. The ablation between static and ML-adjusted costs shows **10-25% improvement** in total travel time during peak congestion.

## Architecture

```
                  ┌─────────────┐
                  │  Streamlit   │  :8501
                  │  Frontend    │
                  └──────┬──────┘
                         │
                  ┌──────▼──────┐
                  │   FastAPI    │  :8000
                  │   Backend    │
                  └──┬───────┬──┘
                     │       │
              ┌──────▼──┐ ┌──▼────────┐
              │  OSRM   │ │ LightGBM  │
              │  Server  │ │ ETA Model │
              │  :5000   │ └───────────┘
              └─────────┘
```

**Pipeline:** OSRM base travel times → ML-adjusted cost matrix → VRP solver → optimized routes

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** (package manager)
- **Docker** (for OSRM routing engine)

## Quick Start

### 1. Install dependencies

```bash
uv sync --dev
```

### 2. Set up OSRM

Download and preprocess an OpenStreetMap extract, then start the routing server:

```bash
mkdir -p data/external/osrm && cd data/external/osrm

# Download OSM extract (New York State example)
wget https://download.geofabrik.de/north-america/us/new-york-latest.osm.pbf

# Preprocess for OSRM
docker run -t -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend \
  osrm-extract -p /opt/car.lua /data/new-york-latest.osm.pbf
docker run -t -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend \
  osrm-partition /data/new-york-latest.osrm
docker run -t -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend \
  osrm-customize /data/new-york-latest.osrm
```

Start OSRM (or use Docker Compose, see below):

```bash
docker run -d -p 5000:5000 -v "${PWD}:/data" \
  ghcr.io/project-osrm/osrm-backend \
  osrm-routed --algorithm mld --max-table-size 10000 /data/new-york-latest.osrm
```

Verify: `curl http://localhost:5000/health`

### 3. Process data and train models

```bash
# Download NYC TLC trip data (Parquet files) into data/raw/nyc_tlc/
# See https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

# Process raw data (requires running OSRM)
uv run python scripts/process_data.py

# Train ETA models
uv run python scripts/train_model.py
```

### 4. Run the application

**Option A: Docker Compose (full stack)**

```bash
docker compose up
```

This starts OSRM (:5000), the API (:8000), and the frontend (:8501).

**Option B: Run services individually**

```bash
# API server
uv run uvicorn api.main:app --reload --port 8000

# Streamlit frontend (in another terminal)
uv run streamlit run frontend/app.py
```

The API starts without an ML model if none is found — it will use static OSRM times only.

## API Usage

### Submit an optimization job

```bash
curl -X POST http://localhost:8000/api/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "depot": {"lat": 40.7484, "lng": -73.9857},
    "stops": [
      {"id": "S1", "lat": 40.7580, "lng": -73.9855, "demand": 5},
      {"id": "S2", "lat": 40.7614, "lng": -73.9776, "demand": 3},
      {"id": "S3", "lat": 40.7527, "lng": -73.9772, "demand": 8}
    ],
    "vehicles": [
      {"id": "V1", "capacity": 50},
      {"id": "V2", "capacity": 50}
    ],
    "config": {
      "use_ml": true,
      "solver": "pyvrp",
      "max_solve_time_seconds": 30
    }
  }'
```

Returns `202 Accepted` with a `job_id`.

### Poll for results

```bash
curl http://localhost:8000/api/optimize/{job_id}
```

### Other endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/optimize` | POST | Submit optimization job |
| `/api/optimize/{job_id}` | GET | Poll job status/result |
| `/api/predict` | POST | Predict ETA between two points |
| `/api/health` | GET | System health check |

## Web UI Guide

Open **http://localhost:8501** in your browser. Use the sidebar dropdown to switch between pages.

### Optimize

Plan delivery routes from a depot to a set of stops.

1. **Choose stops** — select a preset scenario ("Manhattan 10 stops" or "Brooklyn 20 stops") from the sidebar, or pick "Custom" and paste stops as CSV in the format `id,lat,lng,demand`:
   ```
   S1,40.7580,-73.9855,5
   S2,40.7614,-73.9776,3
   S3,40.7527,-73.9772,8
   ```
   For custom input, set depot coordinates below the text box.

2. **Configure in the sidebar:**
   - **Vehicles** — number of delivery vehicles (1-20)
   - **Vehicle capacity** — max load per vehicle
   - **Max solve time** — solver runtime in seconds (longer = better routes)
   - **Use ML-adjusted costs** — enables the trained LightGBM model for time-of-day aware routing; unchecked uses raw OSRM distances
   - **Solver** — `pyvrp` (better quality) or `ortools` (faster)

3. **Click "Optimize Routes"** — after a spinner (up to the configured solve time), the page displays:
   - Metric cards: vehicles used, total distance, total time, solve time
   - Interactive map with color-coded route polylines per vehicle, depot marker (green), and numbered stop markers
   - Expandable route details showing stop sequence, distance, and time per vehicle

### Compare

Runs the same 10-stop Manhattan scenario twice — once with static OSRM costs and once with ML-adjusted costs — and displays results side by side.

1. Configure vehicles, capacity, solve time, and solver in the sidebar
2. Click **"Run Comparison"** (runs two back-to-back optimizations)
3. View the metrics table (static vs ML), time improvement percentage, and two route maps side by side

### Dashboard

Read-only overview of system health and model performance:

- **System Status** — OSRM connection, ML model loaded, active jobs count
- **Model Performance** — MAE, RMSE, MAPE, R² for the LightGBM model
- **Model Comparison** — LightGBM vs Random Forest metrics table
- **Feature Importance** — SHAP plot showing which features matter most for ETA prediction

## Configuration

Settings are managed via environment variables with prefix `LMO_` and nested delimiter `__`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LMO_OSRM__BASE_URL` | `http://localhost:5000` | OSRM server URL |
| `LMO_OSRM__TIMEOUT` | `30` | OSRM request timeout (seconds) |
| `LMO_SOLVER__DEFAULT_SOLVER` | `pyvrp` | Default VRP solver (`pyvrp` or `ortools`) |
| `LMO_SOLVER__MAX_RUNTIME` | `30` | Max VRP solve time (seconds) |

Model hyperparameters are in `configs/model_config.yaml`, solver defaults in `configs/solver_config.yaml`.

## Development

```bash
# Run all tests
uv run pytest

# Run unit tests only (no OSRM required)
uv run pytest -m "not integration"

# Run a single test file
uv run pytest tests/unit/test_vrp_solver.py -v

# Lint and format check
uv run ruff check .
uv run ruff format --check .

# Auto-fix lint issues
uv run ruff check --fix .
uv run ruff format .
```

## Project Structure

```
├── api/                    # FastAPI REST API
│   ├── main.py             # App setup, lifespan, middleware
│   ├── routers/            # Endpoint handlers
│   └── schemas/            # Pydantic request/response models
├── src/                    # Core library
│   ├── data/               # Data loading and preprocessing
│   ├── features/           # Feature engineering pipeline
│   ├── models/             # ML training, evaluation, prediction
│   ├── optimization/       # OSRM client, cost matrix, VRP solvers
│   └── utils/              # Config, geo utilities
├── frontend/               # Streamlit web UI
│   ├── app.py              # Entry point
│   ├── pages/              # Optimize, Compare, Dashboard views
│   └── components/         # Map display components
├── scripts/                # CLI scripts for data processing and training
├── configs/                # YAML configuration files
├── tests/                  # Unit and integration tests
├── docker/                 # Dockerfiles for API and frontend
└── doc/                    # Research and implementation documentation
```

## VRP Solvers

Two solvers are available, sharing the same interface:

- **PyVRP** (default) — Hybrid Genetic Search algorithm. State-of-the-art solution quality (<2% gap from optimal on benchmarks). Supports CVRPTW (capacitated VRP with time windows).
- **OR-Tools** — Google's constraint programming solver with Guided Local Search. Broader constraint support including pickup-delivery pairs. Good solution quality (5-15% gap).

## Evaluation

```bash
# Compare solvers on Solomon VRPTW benchmark instances
uv run python scripts/run_solver_comparison.py

# Run full evaluation pipeline
uv run python scripts/run_evaluation.py
```

## License

This project is for academic/research purposes.

# Setup Guide

Complete step-by-step instructions to get the Last-Mile Delivery Optimizer running on your local machine. This guide assumes a fresh setup — follow each section in order.

## Table of Contents

- [1. Prerequisites](#1-prerequisites)
- [2. Clone the Repository](#2-clone-the-repository)
- [3. Install Python Dependencies](#3-install-python-dependencies)
- [4. Set Up the OSRM Routing Engine](#4-set-up-the-osrm-routing-engine)
- [5. Download Training Data](#5-download-training-data)
- [6. Process Data and Train Models](#6-process-data-and-train-models)
- [7. Run the Application](#7-run-the-application)
- [8. Verify Everything Works](#8-verify-everything-works)
- [9. Configuration Reference](#9-configuration-reference)
- [10. Running Tests](#10-running-tests)
- [11. Troubleshooting](#11-troubleshooting)

---

## 1. Prerequisites

Install these tools before proceeding. All are free and cross-platform.

### Python 3.12+

The project requires Python 3.12 or newer.

**Windows:**
- Download from https://www.python.org/downloads/
- During installation, check "Add Python to PATH"
- Verify: open a terminal and run `python --version` — should show 3.12.x or higher

**macOS:**
```bash
brew install python@3.12
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev
```

### uv (Python package manager)

This project uses `uv` instead of `pip` for dependency management. It is significantly faster and handles virtual environments automatically.

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify: `uv --version`

For more options, see https://docs.astral.sh/uv/getting-started/installation/

### Docker Desktop

Docker is required to run the OSRM routing engine (which provides driving directions and travel time calculations using real road network data).

- Download from https://www.docker.com/products/docker-desktop/
- Install and start Docker Desktop
- Verify: `docker --version` and `docker compose version`

**Important:** Docker Desktop must be running whenever you use this project. The OSRM routing engine runs inside a Docker container.

### Git

- Download from https://git-scm.com/downloads
- Verify: `git --version`

### AWS CLI (optional — only needed for Amazon Last Mile dataset)

Only install this if you want to download the Amazon Last Mile dataset (optional, not required for core functionality).

**Windows:** Download from https://aws.amazon.com/cli/

**macOS:**
```bash
brew install awscli
```

**Linux:**
```bash
sudo apt install awscli
```

### System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| Disk space | 10 GB free | 20 GB free |
| CPU | 4 cores | 8 cores |
| GPU | Not required | Not required |

Disk space breakdown:
- OSRM road network data (NY State): ~4-5 GB
- NYC TLC training data: ~1-2 GB (3 months of parquet files)
- Python environment: ~2 GB
- Processed data + trained models: ~1 GB

---

## 2. Clone the Repository

```bash
git clone https://github.com/<your-username>/last-mile-delivery-optimizer.git
cd last-mile-delivery-optimizer
```

All subsequent commands should be run from this project root directory.

---

## 3. Install Python Dependencies

`uv` handles virtual environment creation automatically. From the project root:

```bash
uv sync --dev
```

This will:
1. Create a `.venv` virtual environment in the project directory
2. Install all required packages (FastAPI, LightGBM, PyVRP, OR-Tools, Streamlit, etc.)
3. Install development dependencies (pytest, ruff)

Expected output: a list of installed packages, ending with a success message.

**If you encounter errors:**
- Make sure Python 3.12+ is installed and available on PATH
- On Linux, you may need: `sudo apt install build-essential libffi-dev`
- On Windows, if you get C++ build errors, install "Microsoft C++ Build Tools" from https://visualstudio.microsoft.com/visual-cpp-build-tools/

---

## 4. Set Up the OSRM Routing Engine

OSRM (Open Source Routing Machine) provides real-world driving directions and travel time calculations. It needs road network data to work — we use OpenStreetMap data for New York State.

**Make sure Docker Desktop is running before starting this section.**

### Option A: Use the automated setup script (recommended)

```bash
bash scripts/setup_osrm.sh
```

This script will:
1. Create the directory `data/external/osrm/`
2. Download the New York State road network file (~463 MB) — this may take several minutes depending on your internet speed
3. Process the road network data through three steps (extract → partition → customize) — each step runs inside Docker and may take 5-15 minutes
4. Print instructions for starting the OSRM server

The script is idempotent — if you run it again, it will skip already-completed steps.

### Option B: Manual setup

If the script doesn't work on your system, run these commands one by one:

```bash
# Create the data directory
mkdir -p data/external/osrm
cd data/external/osrm

# Download the New York State road network (~463 MB)
curl -L -C - -o new-york-latest.osm.pbf https://download.geofabrik.de/north-america/us/new-york-latest.osm.pbf

# Extract road network (may take 5-15 minutes, uses ~4-5 GB RAM)
docker run -t -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend \
  osrm-extract -p /opt/car.lua /data/new-york-latest.osm.pbf

# Partition the network
docker run -t -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend \
  osrm-partition /data/new-york-latest.osrm

# Customize the network
docker run -t -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend \
  osrm-customize /data/new-york-latest.osrm

# Go back to project root
cd ../../..
```

**Windows note:** If using PowerShell instead of Git Bash, replace `${PWD}` with `${pwd}` or the full absolute path to the `data/external/osrm` directory. The path must use forward slashes for Docker volume mounts. Example:
```powershell
docker run -t -v "C:/Users/admin/Documents/last-mile-delivery-optimizer/data/external/osrm:/data" ghcr.io/project-osrm/osrm-backend osrm-extract -p /opt/car.lua /data/new-york-latest.osm.pbf
```

### Start the OSRM server

After preprocessing is complete, start the OSRM routing server:

```bash
docker run -d -p 5000:5000 \
  -v "$(pwd)/data/external/osrm:/data" \
  --name osrm \
  ghcr.io/project-osrm/osrm-backend \
  osrm-routed --algorithm mld --max-table-size 10000 /data/new-york-latest.osrm
```

**Windows PowerShell:**
```powershell
docker run -d -p 5000:5000 -v "${pwd}/data/external/osrm:/data" --name osrm ghcr.io/project-osrm/osrm-backend osrm-routed --algorithm mld --max-table-size 10000 /data/new-york-latest.osrm
```

### Verify OSRM is running

```bash
curl http://localhost:5000/health
```

Expected response: `"OK"` or a JSON status message. If using PowerShell without curl:
```powershell
Invoke-RestMethod http://localhost:5000/health
```

**Managing the OSRM container:**
```bash
docker stop osrm      # Stop the server
docker start osrm     # Start it again later
docker rm osrm        # Remove the container (you'll need to re-run the docker run command)
docker logs osrm      # View server logs if something goes wrong
```

---

## 5. Download Training Data

The ML model needs historical trip data for training. The primary dataset is NYC Taxi & Limousine Commission (TLC) trip records.

### NYC TLC Data (required for ML training)

```bash
bash scripts/download_data.sh
```

This downloads:
- **NYC TLC Yellow Taxi trip data** — 3 months of 2023 trip records (January, June, October) as Parquet files (~100-300 MB each)
- **NYC Taxi Zone shapefile** — geographic boundaries for the 263 taxi zones
- **Solomon VRPTW benchmark instances** — standard test problems for evaluating the route optimizer

Total download size: ~1-2 GB.

### Manual download (if the script doesn't work)

Download these files and place them in the correct directories:

**NYC TLC Parquet files** → `data/raw/nyc_tlc/`
- https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet
- https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-06.parquet
- https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-10.parquet

**Taxi Zone shapefile** → `data/raw/nyc_tlc/taxi_zones/`
- https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip (unzip after downloading)

**Solomon benchmarks** → `data/raw/solomon/`
These are downloaded automatically via the `vrplib` Python library when you run the script. To download manually:
```bash
mkdir -p data/raw/solomon
cd data/raw/solomon
uv run python -c "
import vrplib
for name in ['C101', 'C201', 'R101', 'R201', 'RC101', 'RC201']:
    vrplib.download_instance(name, f'{name}.txt')
    vrplib.download_solution(name, f'{name}.sol')
"
cd ../../..
```

### Amazon Last Mile Dataset (optional)

This is a supplementary dataset. The core system works without it.

```bash
# Requires AWS CLI
aws s3 sync --no-sign-request s3://amazon-last-mile-challenges/almrrc2021/ data/raw/amazon/
```

Download size: ~3.1 GB.

---

## 6. Process Data and Train Models

**OSRM must be running for this step** (it was started in Step 4). Verify with `curl http://localhost:5000/health`.

### Process raw data

```bash
uv run python scripts/process_data.py
```

This script:
1. Loads the raw NYC TLC Parquet files
2. Filters trips (valid durations between 60s–7200s, valid distances)
3. Maps taxi zone IDs to geographic coordinates
4. Queries OSRM for base travel times and distances for each trip
5. Engineers features (time-of-day encoding, distance calculations, etc.)
6. Splits data into train/validation/test sets (70/15/15, chronological split)
7. Saves processed data to `data/processed/`

Expected runtime: 10-30 minutes (depends on CPU speed and OSRM query throughput).

### Train the ML model

```bash
uv run python scripts/train_model.py
```

This script:
1. Loads processed training data
2. Trains a LightGBM model (primary) with early stopping on the validation set
3. Trains a Random Forest model (baseline)
4. Saves trained models to `models/` as `.joblib` files with companion `_metadata.json` files
5. Prints training metrics (MAE, RMSE, etc.)

Expected runtime: 5-15 minutes.

### Run model evaluation (optional but recommended)

```bash
uv run python scripts/run_evaluation.py
```

This evaluates both models on the held-out test set and generates:
- `models/evaluation_results.json` — accuracy metrics for both models
- `models/shap_importance.png` — feature importance plot
- `models/shap_summary.png` — SHAP summary plot

### Run solver benchmarks (optional)

```bash
uv run python scripts/run_solver_comparison.py
```

Benchmarks PyVRP vs OR-Tools on Solomon VRPTW instances. Results are saved to `data/processed/solver_comparison.csv`.

---

## 7. Run the Application

You have two options for running the application.

### Option A: Docker Compose (recommended — runs everything)

This starts all three services (OSRM, API backend, Web UI) in Docker containers:

```bash
docker compose up
```

Add `-d` to run in the background:
```bash
docker compose up -d
```

**Note:** If you already have OSRM running from Step 4, either stop it first (`docker stop osrm`) or Docker Compose will fail on port 5000. Docker Compose will start its own OSRM instance.

Services will be available at:
| Service | URL |
|---------|-----|
| Web UI (Streamlit) | http://localhost:8501 |
| API Backend (FastAPI) | http://localhost:8000 |
| API Documentation (Swagger) | http://localhost:8000/docs |
| OSRM Routing Engine | http://localhost:5000 |

To stop all services:
```bash
docker compose down
```

### Option B: Run services individually

This is useful during development or if you want more control. You need three terminal windows.

**Terminal 1 — OSRM** (skip if already running from Step 4):
```bash
docker start osrm
# Or if the container doesn't exist:
docker run -d -p 5000:5000 -v "$(pwd)/data/external/osrm:/data" --name osrm \
  ghcr.io/project-osrm/osrm-backend \
  osrm-routed --algorithm mld --max-table-size 10000 /data/new-york-latest.osrm
```

**Terminal 2 — API Backend:**
```bash
uv run uvicorn api.main:app --reload --port 8000
```

The `--reload` flag enables auto-reload when you edit code (useful for development).

**Terminal 3 — Web UI:**
```bash
uv run streamlit run frontend/app.py
```

**Note:** The API will start even without a trained ML model — it will fall back to using raw OSRM travel times. If you want ML-adjusted routing, make sure you completed Step 6 (model training) first.

---

## 8. Verify Everything Works

### Check system health

```bash
curl http://localhost:8000/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "osrm_connected": true,
  "model_loaded": true,
  "active_jobs": 0
}
```

- `osrm_connected: true` — OSRM is reachable
- `model_loaded: true` — ML model was found and loaded (will be `false` if you skipped model training — the system still works, just without ML adjustments)

### Test the API with a sample optimization

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

This returns a `202 Accepted` response with a `job_id`. Poll for results:

```bash
curl http://localhost:8000/api/optimize/<job_id>
```

Replace `<job_id>` with the actual ID from the response.

### Open the Web UI

Open http://localhost:8501 in your browser. Select "Manhattan 10 stops" from the preset dropdown and click "Optimize Routes" to see the system in action.

---

## 9. Configuration Reference

The application is configured via environment variables. All variables use the prefix `LMO_` and nested delimiter `__`.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LMO_OSRM__BASE_URL` | `http://localhost:5000` | OSRM routing engine URL |
| `LMO_OSRM__TIMEOUT` | `30` | OSRM request timeout in seconds |
| `LMO_SOLVER__DEFAULT_SOLVER` | `pyvrp` | Default VRP solver (`pyvrp` or `ortools`) |
| `LMO_SOLVER__MAX_RUNTIME` | `30` | Maximum VRP solve time in seconds |
| `LMO_SOLVER__DEFAULT_CAPACITY` | `100` | Default vehicle capacity |
| `LMO_SOLVER__DEFAULT_NUM_VEHICLES` | `5` | Default number of vehicles |
| `LMO_MODEL__MODEL_PATH` | `models/eta_lightgbm_v1.joblib` | Path to the trained ML model file |
| `LMO_MODEL__METADATA_PATH` | `models/eta_lightgbm_v1_metadata.json` | Path to the model metadata file |
| `API_BASE_URL` | `http://localhost:8000` | API URL for the Streamlit frontend |

To set environment variables:

**Linux/macOS:**
```bash
export LMO_OSRM__BASE_URL=http://localhost:5000
export LMO_SOLVER__MAX_RUNTIME=60
```

**Windows PowerShell:**
```powershell
$env:LMO_OSRM__BASE_URL = "http://localhost:5000"
$env:LMO_SOLVER__MAX_RUNTIME = "60"
```

**Windows CMD:**
```cmd
set LMO_OSRM__BASE_URL=http://localhost:5000
set LMO_SOLVER__MAX_RUNTIME=60
```

### Configuration Files

- `configs/model_config.yaml` — ML model hyperparameters (LightGBM learning rate, number of trees, etc.)
- `configs/solver_config.yaml` — VRP solver defaults (runtime limits, vehicle capacity, Solomon benchmark instances)

---

## 10. Running Tests

### Run all tests

```bash
uv run pytest
```

### Run only unit tests (no OSRM required)

```bash
uv run pytest -m "not integration"
```

### Run a specific test file

```bash
uv run pytest tests/unit/test_vrp_solver.py -v
```

### Run integration tests (requires running OSRM)

```bash
uv run pytest -m integration
```

### Lint and format check

```bash
# Check for linting issues
uv run ruff check .

# Check formatting
uv run ruff format --check .

# Auto-fix linting issues
uv run ruff check --fix .

# Auto-format code
uv run ruff format .
```

---

## 11. Troubleshooting

### OSRM won't start / port 5000 is in use

```bash
# Check if something is already using port 5000
# Linux/macOS:
lsof -i :5000
# Windows:
netstat -ano | findstr :5000

# If an old OSRM container exists:
docker rm -f osrm

# Then start fresh:
docker run -d -p 5000:5000 -v "$(pwd)/data/external/osrm:/data" --name osrm \
  ghcr.io/project-osrm/osrm-backend \
  osrm-routed --algorithm mld --max-table-size 10000 /data/new-york-latest.osrm
```

On **macOS**, AirPlay Receiver uses port 5000 by default. Disable it in System Settings → General → AirDrop & Handoff → AirPlay Receiver, or use a different port:
```bash
docker run -d -p 5001:5000 -v "$(pwd)/data/external/osrm:/data" --name osrm \
  ghcr.io/project-osrm/osrm-backend \
  osrm-routed --algorithm mld --max-table-size 10000 /data/new-york-latest.osrm

# Then set the environment variable:
export LMO_OSRM__BASE_URL=http://localhost:5001
```

### OSRM preprocessing fails with out-of-memory

The OSRM extract step for New York State needs ~4-5 GB of RAM. If Docker is limited to less:
- **Docker Desktop** → Settings → Resources → increase Memory to at least 6 GB
- Restart Docker Desktop after changing settings

### `uv sync` fails

- Make sure Python 3.12+ is installed: `python --version` or `python3 --version`
- On Linux, install build dependencies: `sudo apt install build-essential python3-dev libffi-dev`
- On Windows, install Microsoft C++ Build Tools if you see compiler errors
- Try clearing the cache: `uv cache clean` then `uv sync --dev`

### API says `model_loaded: false`

This means the trained model files were not found. Either:
- You haven't trained the model yet — run Steps 5 and 6
- The model files are not in the expected location — check that `models/eta_lightgbm_v1.joblib` and `models/eta_lightgbm_v1_metadata.json` exist
- The API still works without the model — it just uses raw OSRM travel times instead of ML-adjusted ones

### `process_data.py` hangs or is very slow

- Make sure OSRM is running (`curl http://localhost:5000/health`)
- The script queries OSRM for each trip in the dataset — this is I/O-bound and can take 10-30 minutes
- If it seems stuck, check OSRM logs: `docker logs osrm`

### Docker Compose fails with port conflicts

If you already have OSRM running from the manual setup:
```bash
docker stop osrm
docker compose up
```

Or if other services are using ports 5000, 8000, or 8501, stop them first.

### Streamlit shows connection errors

- Make sure the API backend is running on port 8000
- If the API is on a different port or host, set: `export API_BASE_URL=http://localhost:<port>`
- Check the API health: `curl http://localhost:8000/api/health`

### Windows-specific issues

- Always use Git Bash or WSL for running bash scripts (`scripts/*.sh`)
- Python scripts can be run from any terminal (PowerShell, CMD, Git Bash)
- Docker volume paths must use forward slashes, even on Windows
- If `bash` is not recognized, install Git for Windows (includes Git Bash) or use WSL

### How to start fresh

If you want to reset everything and start over:

```bash
# Remove Docker containers
docker compose down
docker rm -f osrm

# Remove processed data and models (keeps raw downloads)
rm -rf data/processed/*
rm -rf models/*.joblib models/*_metadata.json models/evaluation_results.json

# Reinstall dependencies
rm -rf .venv
uv sync --dev
```

To also re-download raw data (slower, re-downloads ~2 GB):
```bash
rm -rf data/raw/*
rm -rf data/external/osrm/*
bash scripts/setup_osrm.sh
bash scripts/download_data.sh
```

# Last-mile delivery optimization with ML-driven dynamic routing

A fully local system combining ML-predicted travel times with a state-of-the-art VRP metaheuristic solver can meaningfully outperform static-cost routing by **10–25%** on total travel time in urban settings. The recommended stack—**LightGBM** for ETA prediction, **PyVRP** (Hybrid Genetic Search) for route optimization, and **OSRM** for local road-network queries—runs entirely on a 16 GB laptop with no cloud dependencies. This report provides the complete research foundation, technical architecture, and implementation plan for a thesis-grade system delivered as a local REST API with map-based frontend.

The core thesis contribution is the integration layer: a dynamic cost matrix builder that queries OSRM for base travel times, adjusts them with an ML model trained on real trip data (NYC TLC or Amazon Last Mile), and feeds the result into a competition-winning VRP solver. The ablation between static and ML-adjusted costs produces a clear, defensible thesis result.

---

## 1. Why static cost matrices fail in real urban delivery

Traditional VRP solvers treat travel time between two points as a fixed constant derived from road distance divided by a nominal speed. This fundamentally misrepresents urban delivery reality, where travel time between the same two points can vary by **2–3× between peak and off-peak hours**. A 15-minute trip at 6 AM becomes 40 minutes at 8:30 AM, yet a static matrix assigns the same cost to both.

Three specific failure modes dominate. First, **temporal blindness**: static matrices ignore time-of-day effects (rush hour, school zones, lunch-hour congestion) that systematically distort route costs. Second, **asymmetric ignorance**: driving from A to B often takes different time than B to A due to one-way streets, turn restrictions, and directional congestion—static Euclidean or even static road-distance matrices are symmetric by construction. Third, **stochastic neglect**: the SVRP (Stochastic VRP) literature shows that ignoring travel-time variance leads to 5–15% cost increases and significant feasibility degradation (late deliveries).

The research gap sits precisely at the junction of two mature but disconnected fields. **ETA prediction research** (Google Maps GNN, Uber DeepETA, DoorDash Mixture-of-Experts) has achieved impressive accuracy—MAPE under 10% on urban road segments—but these models produce point estimates, not cost matrices for combinatorial optimization. **VRP research** (HGS, ALNS, branch-cut-price) has produced solvers within 1–2% of optimal on benchmarks, but they consume whatever cost matrix they're given without questioning its accuracy. The "predict-then-optimize" pipeline connecting these two fields is underexplored in the academic literature, particularly for last-mile delivery at thesis scale.

Current state-of-the-art limitations include: RL-based approaches (POMO, Attention Model) achieve only **85–88% feasibility rates** versus 98%+ for classical solvers and degrade badly beyond ~200 nodes; pure GNN approaches (Google Maps) require terabytes of training data and GPU clusters; and hybrid ML+VRP papers remain largely theoretical, with few open-source implementations suitable for reproduction.

**The metrics that matter** for this system are: total travel time (primary optimization objective), on-time delivery rate (percentage of stops served within time windows), route cost improvement over static baseline (the thesis contribution metric), and plan generation latency (must be under 60 seconds for practical use).

---

## 2. Datasets: what's available and how to use each one

Ten freely available datasets collectively provide everything needed for ETA training, road network construction, and VRP benchmarking.

### Primary datasets for ETA model training

**NYC Taxi & Limousine Commission Trip Record Data** is the single most valuable dataset for this project. Published at nyc.gov/site/tlc/about/tlc-trip-record-data.page (also on AWS Open Data Registry), it contains **billions of trips** from 2009 to present in monthly Parquet files (~100–300 MB each). Each record includes pickup/dropoff datetime, taxi zone IDs (263 zones covering all five boroughs), trip distance, and duration. The data is public domain with no license restrictions. For ETA training, a single year of yellow taxi data (~130M trips) provides massive ground-truth trip durations across all hours, days, and weather conditions. The limitation is that since 2016, exact coordinates were replaced with zone IDs, reducing spatial granularity—but zone-to-zone travel time patterns are sufficient for ML training. Download selected months to keep storage manageable (~2–5 GB for one year of data).

**Amazon Last Mile Routing Research Challenge Dataset** (2021) is the only large-scale real delivery dataset publicly available. Hosted on AWS S3 at `s3://amazon-last-mile-challenges/almrrc2021/` (~3.1 GB, downloadable with `aws s3 sync --no-sign-request`), it contains **9,184 historical delivery routes** from 5 US metropolitan areas with stop-level features (coordinates, zone IDs, time windows, service times) and crucially, **actual travel time matrices between stops**. Licensed CC BY-NC 4.0 (non-commercial). This is ideal for both ETA training (real inter-stop travel times) and route quality benchmarking (comparing optimized vs. actual driver sequences, with quality scores). The coordinates are obfuscated, which prevents direct OSM mapping but doesn't affect ML training on the travel-time matrices.

**Chicago Transportation Network Providers Trip Data** offers ~130M rideshare trips from data.cityofchicago.org with trip duration, distance, and census-tract-level origin/destination. Public data, freely downloadable. Useful as a complementary ETA training set for a different city topology.

**Kaggle Food Delivery Dataset** (by Gaurav Malik) provides ~45K food delivery records with restaurant/delivery coordinates, weather conditions, road traffic density, vehicle type, and actual delivery time. Small but directly delivery-specific with rich contextual features. Available at kaggle.com/datasets/gauravmalik26/food-delivery-dataset.

### Road network and routing infrastructure

**OpenStreetMap** provides the global road network graph. Download regional extracts from download.geofabrik.de in `.osm.pbf` format (New York State: 463 MB, Illinois: ~300 MB). Licensed ODbL 1.0 (free with attribution). Contains road type, speed limits, one-way restrictions, turn restrictions, lanes, and surface type. This is the input to both OSRM and osmnx.

**OSRM (Open Source Routing Machine)** at github.com/Project-OSRM/osrm-backend (BSD 2-Clause license) is the local routing engine that processes OSM data into a queryable API providing route geometry, travel time, distance, and critically, **NxN time/distance matrices** via its Table API—essential for building VRP cost matrices. Sub-millisecond single-route queries; a 200×200 matrix computes in under 100ms locally.

### VRP benchmarking instances

**Solomon VRPTW Benchmark** (sintef.no/projectweb/top/vrptw/solomon-benchmark/) provides 56 instances with 100 customers each across 6 classes (clustered, random, mixed; short/long planning horizons) with time windows, demands, and service times. The standard benchmark for VRPTW algorithms, with known best solutions for comparison. The **Gehring & Homberger Extended Set** scales these to 200, 400, 600, 800, and 1,000 customers.

**CVRPLIB** (vrp.atd-lab.inf.puc-rio.br) consolidates all major CVRP benchmarks including the Uchoa X instances (100–1,000 customers) and DIMACS 2021 Challenge instances with real-world delivery data (Loggi Brazil, ORTEC US grocery). Accessible via `pip install vrplib` with `vrplib.download_instance("X-n101-k25.vrp")`.

**TSPLIB95** (comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) provides classic TSP and CVRP instances (up to 85,900 nodes for TSP, 262 for CVRP) with known optimal solutions.

### Dataset pipeline recommendation

Use **NYC TLC** (massive trip time data) + **Amazon Last Mile** (delivery-specific travel times) for ML training. Use **OSM via Geofabrik → OSRM** for road network infrastructure. Use **Solomon VRPTW** + **CVRPLIB X-instances** for VRP benchmarking.

---

## 3. ML model selection: LightGBM wins on the accuracy-complexity frontier

Five model families were evaluated for predicting travel time between two points given contextual features. The recommendation is unambiguous for thesis scope.

### Gradient boosting (LightGBM / XGBoost) is the primary choice

Gradient boosting builds sequential decision tree ensembles where each tree corrects residual errors from predecessors. For ETA prediction, this maps tabular features (time of day, distance, road type, historical speed) to travel time through learned non-linear relationships. **Uber's production ETA system ran on XGBoost for years**—one of the world's largest XGBoost deployments—before transitioning to DeepETA in 2022.

LightGBM is recommended over XGBoost for this project due to faster training (histogram-based, leaf-wise growth), native categorical feature support, and comparable accuracy. Typical results from literature: **MAPE of 10–12%** on urban road segments, **MAE of ~2.3 minutes** on NYC taxi trips, **R² of ~0.78** for last-mile delivery prediction. Training takes 5–30 minutes on CPU for ~500K samples with ~20 features. Inference is sub-millisecond per prediction. Implementation difficulty: trivial (`pip install lightgbm`, ~50 lines of training code).

Critical feature engineering for strong performance: cyclical encoding of time (`sin(2π × hour/24)`, `cos(2π × hour/24)`), haversine and Manhattan distance, OSRM base travel time as an input feature, road type categorical encoding, historical average speed by hour/day-of-week, and rush-hour indicator flags.

### Random Forest serves as the baseline

Random Forest trains independent decision trees via bagging and averages predictions. Near-zero hyperparameter tuning needed. Typically **2–5% worse MAPE than gradient boosting** on identical features (confirmed in multiple studies). Scikit-learn's own documentation states that "histogram-based gradient boosting models uniformly dominate Random Forest." Use this as the easy baseline for ablation comparison.

### LSTM/GRU adds unnecessary complexity for this use case

Recurrent networks process sequential data (GPS trajectories, time-series speeds) through gating mechanisms. They **require sequential input data** to offer any advantage—if you only have point-to-point features (origin, destination, time), LSTM provides zero benefit over gradient boosting. Training requires GPU for reasonable speed. Multiple studies confirm XGBoost matches or beats LSTM on tabular ETA tasks. MAPE: ~7% for traffic speed prediction (comparable to GBM). Include only as an optional stretch goal if GPS trajectory data is available.

### GNNs and Transformers are impractical at thesis scale

**Graph Neural Networks** (Google Maps' approach) encode road network topology through message-passing, achieving up to **40–50% reduction in negative ETA outcomes**. However, they require constructing graph representations of road networks, terabytes of traffic data, multi-GPU training, and significant engineering. There is no `pip install gnn-eta`. Implementation difficulty is extreme—a GNN pipeline alone could be an entire thesis.

**Transformer-based models** (Uber DeepETA, DoorDash MoE) use self-attention to capture feature interactions automatically, achieving the highest accuracy in recent benchmarks. But they overfit easily on small datasets, require GPU training, and the margin over well-tuned GBM is often just **1–3% MAPE**—not worth the complexity for a thesis with mixed ML experience.

### Recommended thesis model architecture

| Role | Model | Why |
|------|-------|-----|
| Primary | LightGBM | Best accuracy-to-complexity ratio, CPU-only, SHAP explainability |
| Baseline | Random Forest | Zero-tuning baseline for ablation study |
| Optional stretch | Simple 2-layer LSTM | Adds "deep learning" component if trajectory data available |
| Literature context | GNN, Transformer | Discuss as SOTA in related work section |

---

## 4. VRP solver: PyVRP for solution quality, OR-Tools for flexibility

### PyVRP is the primary recommendation

PyVRP implements **Hybrid Genetic Search (HGS)**—the current state-of-the-art metaheuristic for vehicle routing. It won 1st place in the 2021 DIMACS VRPTW Challenge and the EURO meets NeurIPS 2022 vehicle routing competition. Published in INFORMS Journal on Computing (2024), making it directly citable in a thesis. Install with `pip install pyvrp`.

Dynamic cost matrix injection uses the edge-based model interface:

```python
from pyvrp import Model
from pyvrp.stop import MaxRuntime

m = Model()
depot = m.add_depot(x=0, y=0)
clients = [m.add_client(x=0, y=0, delivery=demands[i],
           tw_early=tw[i][0], tw_late=tw[i][1]) for i in range(n)]

for i, frm in enumerate(m.locations):
    for j, to in enumerate(m.locations):
        m.add_edge(frm, to, distance=int(ml_cost_matrix[i][j]),
                   duration=int(ml_duration_matrix[i][j]))

result = m.solve(stop=MaxRuntime(30))
```

PyVRP supports capacitated VRP with time windows, heterogeneous fleet, multiple depots, maximum route duration, prize-collecting (optional visits), and client groups. Solution quality is within **1–2% of best-known solutions** on standard benchmarks. Solve times: ~1s for 50 stops, ~5–10s for 100 stops, ~30–120s for 500 stops. One caveat: **PyVRP uses integer-only distances** (multiply float costs by a scaling factor like 100).

### OR-Tools serves as the comparison solver

Google OR-Tools (`pip install ortools`, Apache 2.0 license) uses constraint programming with local search metaheuristics. Its callback-based architecture makes dynamic cost matrix injection trivial—simply return `cost_matrix[from][to]` from a callback function. OR-Tools supports the broadest constraint set: capacity, time windows, **pickup-delivery pairs** (which PyVRP lacks natively), max route duration, penalties for dropped visits, and per-vehicle cost matrices.

Solution quality is **5–15% from optimal** on large instances (versus PyVRP's 1–2%), but OR-Tools is faster for initial solution construction and has vastly more documentation and community support. Use OR-Tools as the secondary solver for thesis comparison and for any scenarios requiring pickup-delivery constraints.

### Why not the alternatives?

**VRPy** was abandoned in September 2021 (last release v0.5.1, Python 3.6–3.8 only). Do not use. **ALNS** (the Python library by the same PyVRP authors) is a general-purpose destroy-repair framework that requires implementing all VRP-specific operators from scratch—significant effort for inferior results compared to PyVRP, which already implements the state-of-the-art algorithm. **RL approaches** (POMO, Attention Model) achieve only 85–88% feasibility rates versus 98%+ for classical solvers and struggle beyond ~200 nodes.

### Solver comparison summary

| Criterion | PyVRP | OR-Tools |
|-----------|-------|----------|
| Solution quality | State-of-the-art (<2% gap) | Good (5–15% gap) |
| CVRPTW support | ✅ | ✅ |
| Pickup-delivery | ❌ | ✅ |
| Dynamic cost matrix | `add_edge()` | Callback function |
| Academic citability | INFORMS JoC 2024 | N/A (tool) |
| 100-stop solve time | 5–10s | 1–5s |

---

## 5. Local road network stack: OSRM + osmnx + Folium

### OSRM provides millisecond routing via Docker

Set up the complete local routing engine in four steps. First, download the OSM extract from Geofabrik:

```bash
mkdir -p osrm-data && cd osrm-data
wget https://download.geofabrik.de/north-america/us/new-york-latest.osm.pbf
```

Then preprocess with OSRM's Docker image:

```bash
docker run -t -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend \
  osrm-extract -p /opt/car.lua /data/new-york-latest.osm.pbf
docker run -t -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend \
  osrm-partition /data/new-york-latest.osrm
docker run -t -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend \
  osrm-customize /data/new-york-latest.osrm
```

Start the routing server:

```bash
docker run -d -p 5000:5000 -v "${PWD}:/data" --name osrm \
  ghcr.io/project-osrm/osrm-backend \
  osrm-routed --algorithm mld --max-table-size 10000 /data/new-york-latest.osrm
```

The **Table API** is the critical feature for VRP cost matrices. A single HTTP request computes the full NxN duration matrix:

```python
def compute_cost_matrix(locations):
    coords = ";".join(f"{lng},{lat}" for lat, lng in locations)
    url = f"http://localhost:5000/table/v1/driving/{coords}"
    response = requests.get(url, params={"annotations": "duration"}).json()
    return np.array(response["durations"])
```

Performance: a **200×200 matrix computes in under 100ms**. Even 1,000×1,000 takes only ~3 seconds. No batching needed for thesis-scale problems. System requirements for New York State (463 MB PBF): ~4–5 GB RAM for preprocessing, ~2–3 GB runtime, ~4 GB disk. Fits comfortably on a 16 GB machine.

### osmnx provides graph-level analysis and custom edge weights

osmnx downloads road networks from OpenStreetMap and creates NetworkX graphs with full road attributes:

```python
import osmnx as ox
G = ox.graph_from_place("Manhattan, New York, USA", network_type="drive")
G = ox.routing.add_edge_speeds(G)
G = ox.routing.add_edge_travel_times(G)
```

The key advantage over OSRM: you can **modify edge weights with ML predictions**. Convert the graph to a GeoDataFrame, apply your ML model to predict travel times per edge, convert back, and route using custom weights. Use osmnx for research and graph analysis; use OSRM for fast production-style matrix computation.

### Folium and Leaflet.js handle map visualization

Folium wraps Leaflet.js from Python, generating self-contained HTML files that use free OpenStreetMap tiles (no API key needed):

```python
import folium
m = folium.Map(location=[40.7484, -73.9857], zoom_start=13)
folium.PolyLine(route_coords, weight=5, color="blue").add_to(m)
m.save("route.html")
```

For the web frontend, Leaflet.js provides full interactive map control with route polylines, draggable markers, and real-time updates from the FastAPI backend. OpenStreetMap tiles are free for reasonable use—thesis-level traffic is well within acceptable limits.

---

## 6. System architecture: three services, one Docker Compose

### End-to-end inference pipeline

When a user submits delivery stops, the system executes a four-stage pipeline: (1) **OSRM base matrix**—query the Table API for NxN base travel times (~100ms), (2) **ML adjustment**—for each cell in the matrix, predict the adjusted travel time using LightGBM with contextual features like departure time, day of week, and rush-hour status (~10ms for the full matrix), (3) **VRP solve**—feed the adjusted matrix into PyVRP with capacity and time-window constraints (1–30s depending on problem size), (4) **Route geometry**—query OSRM Route API for each leg of each optimized route to get display geometry (~50ms total).

### REST API design (FastAPI)

```
POST /optimize         — Submit stops, get optimized routes (async with job polling)
GET  /optimize/{id}    — Poll for optimization result
POST /predict-eta      — Predict travel time between two points
POST /cost-matrix      — Build NxN cost matrix for given stops
GET  /health           — System health (OSRM connected, model loaded, solver ready)
```

The `/optimize` endpoint accepts a JSON body with depot coordinates, stop list (each with lat/lng, demand, optional time window), vehicle fleet definition (capacity, shift times), and config (objective function, ML toggle, max solve time). It returns a job ID immediately (202 Accepted), then the client polls `/optimize/{id}` for the result containing ordered stop lists per vehicle, total distance/time, route geometries, and per-stop ETAs.

ML models load at FastAPI startup via the lifespan context manager. VRP solving runs in a ThreadPoolExecutor to avoid blocking the async event loop. An in-memory dict stores job results—sufficient for thesis use without adding a database dependency.

### Docker Compose orchestration

```yaml
services:
  osrm-backend:
    image: ghcr.io/project-osrm/osrm-backend
    ports: ["5000:5000"]
    volumes: ["./data/osrm:/data"]
    command: osrm-routed --algorithm mld --max-table-size 10000 /data/region.osrm
  
  api:
    build: { dockerfile: docker/Dockerfile.api }
    ports: ["8000:8000"]
    environment:
      OSRM_BASE_URL: http://osrm-backend:5000
    depends_on: [osrm-backend]
  
  frontend:
    build: { dockerfile: docker/Dockerfile.frontend }
    ports: ["8501:8501"]
    environment:
      API_BASE_URL: http://api:8000
    depends_on: [api]
```

### Frontend recommendation

For a thesis where the algorithmic contribution matters more than UI polish, **Streamlit + Folium** (via `streamlit-folium`) is the fastest path to a functional frontend—buildable in 1–2 days, Python-only, with professional-looking route maps and metric dashboards. If interactive stop placement (click-to-add on map) is desired, a **plain HTML/JS page with Leaflet.js** takes 3–5 days and provides much better map interactivity. Both can be served directly from the FastAPI backend.

### Project directory structure

```
last-mile-optimizer/
├── docker-compose.yml
├── pyproject.toml
├── data/raw/ processed/ external/osrm/
├── models/                          # Serialized ML models + metadata
├── src/
│   ├── data/loader.py, preprocessor.py
│   ├── features/engineering.py      # Feature pipeline
│   ├── models/train.py, predict.py, evaluate.py
│   ├── optimization/vrp_solver.py, cost_matrix.py, osrm_client.py
│   └── utils/geo.py, config.py
├── api/
│   ├── main.py                      # FastAPI app, lifespan, CORS
│   ├── routers/optimization.py, prediction.py, health.py
│   └── schemas/optimization.py, prediction.py
├── frontend/app.py, pages/, components/
├── notebooks/01–05 exploration through evaluation
├── tests/
└── configs/model_config.yaml, solver_config.yaml
```

---

## 7. Step-by-step build plan for Claude Code

### Phase 0: Environment setup (Day 1)

**Python environment** (Python 3.11 recommended):

```
# requirements.txt
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
pydantic>=2.9.0
scikit-learn>=1.5.0
xgboost>=2.1.0
lightgbm>=4.5.0
pandas>=2.2.0
numpy>=1.26.0,<2.1
joblib>=1.4.0
osmnx>=2.0.0
networkx>=3.3
geopandas>=1.0.0
folium>=0.17.0
pyvrp>=0.10.0
ortools>=9.10
httpx>=0.27.0
requests>=2.32.0
pyarrow>=17.0.0
matplotlib>=3.9.0
seaborn>=0.13.0
streamlit>=1.39.0
streamlit-folium>=0.22.0
shap>=0.46.0
pytest>=8.3.0
```

**Docker setup for OSRM**: Run the four-command sequence from Section 5 (download PBF, extract, partition, customize, start server). Verify with `curl http://localhost:5000/health`.

**Data download**: Download 2–3 months of NYC TLC Yellow Taxi Parquet files from the official NYC.gov page. Download Amazon Last Mile dataset via `aws s3 sync --no-sign-request s3://amazon-last-mile-challenges/almrrc2021/ ./data/raw/amazon/`. Download Solomon VRPTW instances via `pip install vrplib` then `vrplib.download_instance("C101.txt")`.

### Phase 1: Data processing and feature engineering (Days 2–4)

**NYC TLC processing pipeline**: Load Parquet files with pandas. Filter to trips with valid durations (60s–7200s) and distances (0.1–100 km). Compute trip zone centroids from the taxi zone shapefile. For each trip, extract features: hour of day (cyclical: sin/cos), day of week (cyclical: sin/cos), month, is_weekend, is_rush_hour (7–9 AM, 4–7 PM), pickup_zone, dropoff_zone, trip_distance. Target variable: `trip_duration_seconds`. Apply temporal train/validation/test split (70/15/15 chronologically). Save processed features as Parquet.

**Amazon Last Mile processing**: Parse the JSON route files. Extract stop-to-stop travel times from the travel_time matrices. Create features: inter-stop haversine distance, zone transitions, departure time, route position (early/mid/late in route). Target: actual travel time between consecutive stops.

**Feature engineering module** (`src/features/engineering.py`): Implement `FeatureEngineer` class with methods for temporal features (cyclical encoding), spatial features (haversine distance, bearing, Manhattan ratio = road_distance / haversine_distance), and road features (OSRM base time as feature, average speed = distance/base_time). The OSRM base travel time is the single most powerful input feature—the ML model learns to correct the OSRM estimate based on context.

### Phase 2: ML model training (Days 5–7)

**Training script** (`src/models/train.py`): Train LightGBM regressor with `n_estimators=1000`, `max_depth=8`, `learning_rate=0.05`, `early_stopping_rounds=50`, `eval_metric='mae'`. Use the validation set for early stopping. Train a Random Forest baseline with `n_estimators=200`, `max_depth=None` for comparison.

**Evaluation** (`src/models/evaluate.py`): Compute MAE, RMSE, MAPE, and R² on the test set. Generate stratified error analysis by hour-of-day and day-of-week. Produce SHAP feature importance plots. Expected results: LightGBM MAE ~2–3 minutes for urban trips, MAPE ~10–12%, R² ~0.80+. Random Forest should be 2–5% worse on MAPE.

**Model serialization**: Save with `joblib.dump(model, "models/eta_model_v1.0.joblib")`. Save a metadata JSON alongside each model recording training date, feature list, validation metrics, and hyperparameters.

### Phase 3: Dynamic cost matrix builder (Days 8–10)

**OSRM client** (`src/optimization/osrm_client.py`): Implement `get_duration_matrix(locations)` that calls the OSRM Table API and returns an NxN numpy array of travel times in seconds. Handle OSRM's `lng,lat` coordinate order (opposite of most Python conventions).

**Cost matrix builder** (`src/optimization/cost_matrix.py`): Implement `build_dynamic_cost_matrix(locations, departure_time, model)` that: (1) queries OSRM for base NxN durations, (2) for each pair (i,j), constructs feature vector [base_time, distance, hour, day_of_week, is_rush_hour, haversine_distance], (3) runs LightGBM inference to get ML-predicted travel time, (4) returns the ML-adjusted NxN matrix. Also implement `build_static_cost_matrix(locations)` that returns raw OSRM durations—this is the baseline for ablation.

The ML model acts as a **correction factor** on OSRM estimates. The adjustment typically ranges from 0.8× (free-flow conditions where OSRM overestimates) to 2.0× (heavy congestion where OSRM underestimates).

### Phase 4: VRP solver integration (Days 11–14)

**PyVRP solver** (`src/optimization/vrp_solver.py`): Implement `solve_vrp(cost_matrix, demands, time_windows, vehicle_capacity, num_vehicles, max_solve_time)`. Scale float costs to integers (multiply by 100). Create PyVRP Model, add depot, add clients with demands and time windows, add edges from the cost matrix, call `model.solve(stop=MaxRuntime(max_solve_time))`. Extract routes as ordered stop lists. Implement the same interface for OR-Tools as `solve_vrp_ortools()` for comparison.

**Constraints to implement**: Vehicle capacity (each stop has a demand; vehicle capacity is fixed), time windows (optional, each stop can have an earliest/latest arrival), maximum route duration (optional, limits total time per vehicle), number of vehicles (fixed fleet or minimize).

**Output format**: List of routes, each containing an ordered list of stop IDs, total distance, total duration, and per-stop arrival times.

### Phase 5: REST API with FastAPI (Days 15–17)

**Main application** (`api/main.py`): Create FastAPI app with lifespan handler that loads the LightGBM model and feature scaler at startup. Configure CORS for frontend access. Register routers for optimization, prediction, and health endpoints.

**Optimization router** (`api/routers/optimization.py`): Implement POST `/optimize` that validates the request via Pydantic schemas, submits VRP solving to a background thread via `asyncio.run_in_executor`, and returns a job ID. Implement GET `/optimize/{job_id}` for polling results.

**Pydantic schemas** (`api/schemas/optimization.py`):

```python
class Location(BaseModel):
    lat: float
    lng: float
    name: str | None = None

class DeliveryStop(BaseModel):
    id: str
    lat: float
    lng: float
    demand: int = 1
    time_window: TimeWindow | None = None
    service_time_minutes: int = 5

class OptimizeRequest(BaseModel):
    depot: Location
    stops: list[DeliveryStop]
    vehicles: list[Vehicle]
    config: OptimizeConfig

class RouteResult(BaseModel):
    vehicle_id: str
    stops: list[str]
    total_distance_km: float
    total_time_minutes: float
    geometry: list[list[float]]
    eta_per_stop: dict[str, str]
```

Run with `uvicorn api.main:app --host 0.0.0.0 --port 8000`.

### Phase 6: Frontend (Days 18–20)

**Streamlit app** (`frontend/app.py`): Create a multi-page Streamlit application with three pages. The **Optimize** page lets users input stop coordinates (paste CSV or click on map via streamlit-folium), configure vehicles and constraints, submit to the API, and display optimized routes on a Folium map with color-coded polylines per vehicle. The **Dashboard** page shows metrics comparison: ML-adjusted vs. static cost, solve time, vehicles used, total distance/time. The **Compare** page runs both static and dynamic cost matrix optimization and displays side-by-side route maps with improvement percentages.

**Map rendering** (`frontend/components/map_display.py`): Build Folium maps with depot marker (green), stop markers (red, numbered), route polylines (distinct colors per vehicle), and popup tooltips showing stop ID, demand, ETA. Use OSRM Route API to get actual road-following geometry for each route leg (not just straight lines between stops).

---

## 8. Evaluation framework and thesis contribution

### Ablation study design

The core thesis experiment compares four configurations on the same problem instances:

| Experiment | Cost Matrix | VRP Solver | Tests |
|-----------|------------|-----------|-------|
| Static baseline | OSRM raw durations | PyVRP | No ML, standard approach |
| ML-adjusted (off-peak) | OSRM + LightGBM (non-rush) | PyVRP | ML value at low congestion |
| ML-adjusted (peak) | OSRM + LightGBM (rush hour) | PyVRP | ML value at high congestion |
| Full dynamic | OSRM + LightGBM (all features) | PyVRP | Complete system |
| Solver comparison | Same dynamic matrix | PyVRP vs OR-Tools vs nearest-neighbor | Solver quality |

Run each configuration on **Solomon VRPTW instances** (C1, R1, RC1 classes) and on synthetic NYC delivery scenarios using real OSRM distances. Execute 10+ runs with different random seeds for stochastic solvers. Report mean ± standard deviation. Use Wilcoxon signed-rank test for statistical significance.

### Metrics to report

For the ETA model: **MAE** (primary, in minutes), RMSE, MAPE, R², error distribution by hour-of-day, SHAP feature importance ranking. For the VRP solution: **total travel time** (primary objective), number of vehicles used, average route duration, computation time, time-window compliance rate. For the integrated system: **improvement percentage** of ML-adjusted over static baseline on total travel time (expected: 10–25%), on-time delivery rate (expected: 5–15% improvement), and Pareto frontier of computation time versus solution quality.

### Thesis contribution framing

The novel contribution is the **predict-then-optimize pipeline** connecting ML-predicted travel times to a state-of-the-art VRP solver, with empirical evidence that dynamic cost matrices improve route quality in urban settings. The ablation study quantifies exactly how much improvement ML predictions provide over static costs, stratified by congestion level. This bridges the documented gap between ETA prediction research (focused on accuracy metrics) and VRP research (focused on combinatorial optimization quality).

---

## 9. Technology stack summary

| Component | Tool | Version | Purpose |
|-----------|------|---------|---------|
| Language | Python | 3.11 | Primary codebase |
| API framework | FastAPI + Uvicorn | ≥0.115 | REST API backend |
| ML training | LightGBM | ≥4.5 | Primary ETA prediction model |
| ML baseline | scikit-learn (RandomForest) | ≥1.5 | Baseline comparison model |
| ML comparison | XGBoost | ≥2.1 | Alternative gradient boosting |
| Explainability | SHAP | ≥0.46 | Feature importance analysis |
| VRP solver (primary) | PyVRP | ≥0.10 | State-of-the-art HGS algorithm |
| VRP solver (secondary) | Google OR-Tools | ≥9.10 | Comparison solver, pickup-delivery |
| Local routing engine | OSRM (Docker) | latest | Travel time matrices, route geometry |
| Road network analysis | osmnx + NetworkX | ≥2.0 / ≥3.3 | Graph construction, custom weights |
| Geospatial | GeoPandas + Shapely | ≥1.0 / ≥2.0 | Spatial data handling |
| Map visualization | Folium | ≥0.17 | Python-to-HTML route maps |
| Frontend | Streamlit + streamlit-folium | ≥1.39 | Interactive web dashboard |
| Data processing | Pandas + PyArrow | ≥2.2 / ≥17.0 | DataFrame operations, Parquet I/O |
| Numerical | NumPy | ≥1.26, <2.1 | Matrix operations |
| Visualization | Matplotlib + Seaborn | ≥3.9 / ≥0.13 | Charts and plots for thesis |
| Schema validation | Pydantic | ≥2.9 | Request/response validation |
| HTTP client | httpx + requests | ≥0.27 / ≥2.32 | OSRM API queries |
| Containerization | Docker + Docker Compose | latest | Service orchestration |
| VRP benchmarks | vrplib | latest | Download Solomon/CVRPLIB instances |
| Testing | pytest | ≥8.3 | Unit and integration tests |

All components are open-source, run locally, and install via pip or Docker. Total disk footprint: ~5 GB for OSRM data (NY State) + ~2–5 GB for NYC TLC training data + ~3 GB for Amazon Last Mile + ~2 GB for Python environment. Total RAM during operation: ~5–7 GB, well within a 16 GB machine. No GPU required for any component.
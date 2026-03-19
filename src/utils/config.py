"""Application configuration via Pydantic Settings."""

from pathlib import Path
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class OSRMSettings(BaseSettings):
    base_url: str = "http://localhost:5000"
    timeout: int = 30


class ModelSettings(BaseSettings):
    model_path: str = str(PROJECT_ROOT / "models" / "eta_lightgbm_v1.joblib")
    metadata_path: str = str(PROJECT_ROOT / "models" / "eta_lightgbm_v1_metadata.json")
    cost_scaling_factor: int = 100


class SolverSettings(BaseSettings):
    default_solver: str = "pyvrp"
    max_runtime: int = 30
    default_capacity: int = 100
    default_num_vehicles: int = 5


class Settings(BaseSettings):
    osrm: OSRMSettings = Field(default_factory=OSRMSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    solver: SolverSettings = Field(default_factory=SolverSettings)

    data_raw_dir: str = str(PROJECT_ROOT / "data" / "raw")
    data_processed_dir: str = str(PROJECT_ROOT / "data" / "processed")

    model_config = {"env_prefix": "LMO_", "env_nested_delimiter": "__"}


@lru_cache
def get_settings() -> Settings:
    return Settings()

"""Model training for ETA prediction."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.features.engineering import get_categorical_features, get_feature_names

logger = logging.getLogger(__name__)


def train_lightgbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str = "trip_duration_seconds",
    params: dict | None = None,
) -> lgb.LGBMRegressor:
    """Train a LightGBM regressor for ETA prediction.

    Args:
        train_df: Training data with features and target.
        val_df: Validation data for early stopping.
        target_col: Name of the target column.
        params: Optional hyperparameter overrides.

    Returns:
        Trained LGBMRegressor.
    """
    feature_names = get_feature_names()
    cat_features = get_categorical_features()

    default_params = {
        "n_estimators": 1000,
        "max_depth": 8,
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
        "random_state": 42,
    }
    if params:
        default_params.update(params)

    model = lgb.LGBMRegressor(**default_params)

    X_train = train_df[feature_names]
    y_train = train_df[target_col]
    X_val = val_df[feature_names]
    y_val = val_df[target_col]

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ],
        categorical_feature=cat_features,
    )

    logger.info("LightGBM trained. Best iteration: %d", model.best_iteration_)
    return model


def train_random_forest(
    train_df: pd.DataFrame,
    target_col: str = "trip_duration_seconds",
    params: dict | None = None,
) -> RandomForestRegressor:
    """Train a Random Forest baseline.

    Args:
        train_df: Training data with features and target.
        target_col: Name of the target column.
        params: Optional hyperparameter overrides.

    Returns:
        Trained RandomForestRegressor.
    """
    feature_names = get_feature_names()

    default_params = {
        "n_estimators": 200,
        "max_depth": None,
        "n_jobs": -1,
        "random_state": 42,
    }
    if params:
        default_params.update(params)

    model = RandomForestRegressor(**default_params)

    X_train = train_df[feature_names]
    y_train = train_df[target_col]

    logger.info("Training Random Forest with %d samples...", len(X_train))
    model.fit(X_train, y_train)
    logger.info("Random Forest trained. n_estimators=%d", model.n_estimators)
    return model


def save_model(
    model: lgb.LGBMRegressor | RandomForestRegressor,
    model_path: str | Path,
    val_metrics: dict | None = None,
    hyperparams: dict | None = None,
) -> None:
    """Save model and metadata.

    Args:
        model: Trained model.
        model_path: Path for the joblib file. Metadata saved alongside as _metadata.json.
        val_metrics: Validation metrics dict.
        hyperparams: Hyperparameters used for training.
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(model, model_path)
    logger.info("Model saved to %s", model_path)

    # Save metadata
    metadata = {
        "model_type": type(model).__name__,
        "feature_names": get_feature_names(),
        "categorical_features": get_categorical_features(),
        "training_date": datetime.now(timezone.utc).isoformat(),
        "val_metrics": val_metrics or {},
        "hyperparams": hyperparams or {},
    }

    metadata_path = model_path.with_name(model_path.stem + "_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved to %s", metadata_path)

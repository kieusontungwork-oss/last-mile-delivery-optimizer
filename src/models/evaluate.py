"""Model evaluation utilities."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features.engineering import get_feature_names

logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    test_df: pd.DataFrame,
    target_col: str = "trip_duration_seconds",
) -> dict:
    """Compute evaluation metrics on test data.

    Returns:
        Dict with MAE, RMSE, MAPE, R2 (all in seconds for time-based metrics).
    """
    feature_names = get_feature_names()
    X_test = test_df[feature_names]
    y_test = test_df[target_col].values

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # MAPE (exclude near-zero actuals to avoid division issues)
    mask = y_test > 60  # at least 1 minute
    mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100

    metrics = {
        "mae_seconds": float(mae),
        "mae_minutes": float(mae / 60),
        "rmse_seconds": float(rmse),
        "rmse_minutes": float(rmse / 60),
        "mape_percent": float(mape),
        "r2": float(r2),
        "n_samples": int(len(y_test)),
    }

    logger.info(
        "Evaluation: MAE=%.1fs (%.1f min), RMSE=%.1fs, MAPE=%.1f%%, R2=%.3f",
        mae, mae / 60, rmse, mape, r2,
    )
    return metrics


def stratified_evaluation(
    model,
    test_df: pd.DataFrame,
    target_col: str = "trip_duration_seconds",
) -> pd.DataFrame:
    """Compute metrics stratified by hour-of-day and day-of-week.

    Returns:
        DataFrame with columns: group_type, group_value, mae_seconds, mape_percent, n_samples.
    """
    feature_names = get_feature_names()
    X_test = test_df[feature_names]
    y_test = test_df[target_col].values
    y_pred = model.predict(X_test)

    dt = pd.to_datetime(test_df["pickup_datetime"])

    results = []

    # By hour
    hours = dt.dt.hour
    for h in sorted(hours.unique()):
        mask = hours == h
        if mask.sum() < 10:
            continue
        yt, yp = y_test[mask], y_pred[mask]
        mae = mean_absolute_error(yt, yp)
        valid = yt > 60
        mape = np.mean(np.abs((yt[valid] - yp[valid]) / yt[valid])) * 100 if valid.any() else 0
        results.append({
            "group_type": "hour",
            "group_value": int(h),
            "mae_seconds": float(mae),
            "mape_percent": float(mape),
            "n_samples": int(mask.sum()),
        })

    # By day of week
    dows = dt.dt.dayofweek
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for d in range(7):
        mask = dows == d
        if mask.sum() < 10:
            continue
        yt, yp = y_test[mask], y_pred[mask]
        mae = mean_absolute_error(yt, yp)
        valid = yt > 60
        mape = np.mean(np.abs((yt[valid] - yp[valid]) / yt[valid])) * 100 if valid.any() else 0
        results.append({
            "group_type": "day_of_week",
            "group_value": dow_names[d],
            "mae_seconds": float(mae),
            "mape_percent": float(mape),
            "n_samples": int(mask.sum()),
        })

    return pd.DataFrame(results)


def generate_shap_analysis(
    model,
    test_df: pd.DataFrame,
    output_dir: str | Path,
    max_samples: int = 5000,
) -> None:
    """Generate SHAP feature importance plots.

    Args:
        model: Trained model (LightGBM or RF).
        test_df: Test data with feature columns.
        output_dir: Directory to save plots.
        max_samples: Max samples for SHAP computation (for speed).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = get_feature_names()
    X = test_df[feature_names]

    if len(X) > max_samples:
        X = X.sample(n=max_samples, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False, max_display=14)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Bar plot (mean absolute SHAP)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=14)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("SHAP plots saved to %s", output_dir)


def compare_models(
    models: dict,
    test_df: pd.DataFrame,
    target_col: str = "trip_duration_seconds",
) -> pd.DataFrame:
    """Compare multiple models side by side.

    Args:
        models: Dict of {model_name: model_object}.
        test_df: Test data.

    Returns:
        DataFrame with one row per model and metric columns.
    """
    results = []
    for name, model in models.items():
        metrics = evaluate_model(model, test_df, target_col)
        metrics["model"] = name
        results.append(metrics)
    return pd.DataFrame(results).set_index("model")

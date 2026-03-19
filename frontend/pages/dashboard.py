"""Dashboard page: model metrics and system status."""

import json
import os
from pathlib import Path

import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def render():
    st.header("Dashboard")

    # System status
    st.subheader("System Status")
    try:
        resp = requests.get(f"{API_BASE}/api/health", timeout=3)
        health = resp.json()
        col1, col2, col3 = st.columns(3)
        col1.metric("OSRM", "Connected" if health["osrm_connected"] else "Disconnected")
        col2.metric("ML Model", "Loaded" if health["model_loaded"] else "Not loaded")
        col3.metric("Active Jobs", health["active_jobs"])
    except requests.ConnectionError:
        st.warning(f"Cannot connect to API at {API_BASE}")

    # Model metrics
    st.subheader("Model Performance")
    eval_path = PROJECT_ROOT / "models" / "evaluation_results.json"
    if eval_path.exists():
        with open(eval_path) as f:
            eval_data = json.load(f)

        if "lightgbm" in eval_data:
            lgb_metrics = eval_data["lightgbm"]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAE", f"{lgb_metrics.get('mae_minutes', 0):.1f} min")
            col2.metric("RMSE", f"{lgb_metrics.get('rmse_minutes', 0):.1f} min")
            col3.metric("MAPE", f"{lgb_metrics.get('mape_percent', 0):.1f}%")
            col4.metric("R2", f"{lgb_metrics.get('r2', 0):.3f}")

        # Show comparison if available
        if "random_forest" in eval_data:
            st.write("**Model Comparison**")
            rf = eval_data["random_forest"]
            lgb = eval_data.get("lightgbm", {})
            st.table({
                "Metric": ["MAE (min)", "RMSE (min)", "MAPE (%)", "R2"],
                "LightGBM": [
                    f"{lgb.get('mae_minutes', 0):.2f}",
                    f"{lgb.get('rmse_minutes', 0):.2f}",
                    f"{lgb.get('mape_percent', 0):.2f}",
                    f"{lgb.get('r2', 0):.4f}",
                ],
                "Random Forest": [
                    f"{rf.get('mae_minutes', 0):.2f}",
                    f"{rf.get('rmse_minutes', 0):.2f}",
                    f"{rf.get('mape_percent', 0):.2f}",
                    f"{rf.get('r2', 0):.4f}",
                ],
            })
    else:
        st.info("No evaluation results found. Train a model first.")

    # SHAP plots
    st.subheader("Feature Importance")
    shap_path = PROJECT_ROOT / "models" / "shap_importance.png"
    if shap_path.exists():
        st.image(str(shap_path), caption="SHAP Feature Importance")
    else:
        st.info("No SHAP analysis available. Run model evaluation first.")

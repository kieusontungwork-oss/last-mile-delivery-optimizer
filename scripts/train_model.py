"""Train ETA prediction models and save results."""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.engineering import get_feature_names
from src.models.evaluate import compare_models, evaluate_model, generate_shap_analysis
from src.models.train import save_model, train_lightgbm, train_random_forest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def main():
    # Load processed data
    train_path = PROCESSED_DIR / "train.parquet"
    val_path = PROCESSED_DIR / "val.parquet"
    test_path = PROCESSED_DIR / "test.parquet"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            logger.error("Data file not found: %s", p)
            logger.info("Run scripts/process_data.py first.")
            sys.exit(1)

    logger.info("Loading processed data...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    logger.info("Train: %d, Val: %d, Test: %d", len(train_df), len(val_df), len(test_df))

    # Train LightGBM
    logger.info("Training LightGBM...")
    lgb_model = train_lightgbm(train_df, val_df)
    lgb_metrics = evaluate_model(lgb_model, test_df)

    save_model(
        lgb_model,
        MODELS_DIR / "eta_lightgbm_v1.joblib",
        val_metrics=lgb_metrics,
        hyperparams=lgb_model.get_params(),
    )

    # Train Random Forest
    logger.info("Training Random Forest...")
    rf_model = train_random_forest(train_df)
    rf_metrics = evaluate_model(rf_model, test_df)

    save_model(
        rf_model,
        MODELS_DIR / "eta_rf_baseline_v1.joblib",
        val_metrics=rf_metrics,
    )

    # Compare models
    logger.info("Comparing models...")
    comparison = compare_models(
        {"lightgbm": lgb_model, "random_forest": rf_model},
        test_df,
    )
    logger.info("\n%s", comparison.to_string())

    # Save evaluation results
    eval_results = {
        "lightgbm": lgb_metrics,
        "random_forest": rf_metrics,
    }
    eval_path = MODELS_DIR / "evaluation_results.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    logger.info("Evaluation results saved to %s", eval_path)

    # SHAP analysis
    logger.info("Generating SHAP analysis...")
    generate_shap_analysis(lgb_model, test_df, MODELS_DIR)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()

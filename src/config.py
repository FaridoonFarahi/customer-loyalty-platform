"""Paths and fixed settings for the redemption-model pipeline.

The modeling settings repeat the choices made in notebooks 03 to 06 and saved in
``models/modeling_config.json``. ``tests/test_config.py`` checks that the two stay in step.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Where the raw dunnhumby CSV files live.
RAW_DIR = Path(os.environ.get("LOYALTY_RAW_DIR", PROJECT_ROOT / "data" / "raw"))

# Where the pipeline writes models and reports. Tests point this at a temporary folder.
OUTPUT_ROOT = Path(os.environ.get("LOYALTY_OUTPUT_ROOT", PROJECT_ROOT))
MODELS_DIR = OUTPUT_ROOT / "models"
REPORTS_DIR = OUTPUT_ROOT / "reports" / "pipeline"
MONITORING_DIR = OUTPUT_ROOT / "reports" / "monitoring"

MODEL_PATH = MODELS_DIR / "redemption_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "redemption_model_metadata.json"

# MLflow: a local SQLite tracking store with its artifacts beside it.
MLFLOW_DIR = Path(os.environ.get("LOYALTY_MLFLOW_DIR", PROJECT_ROOT / "mlflow_tracking"))
MLFLOW_EXPERIMENT = "customer_loyalty_redemption"
REGISTERED_MODEL_NAME = "customer_loyalty_redemption"
CHAMPION_ALIAS = "champion"

# ---------------------------------------------------------------- modeling settings
TARGET = "redeemed"
HISTORY_WINDOW_DAYS = 90

# Campaigns ending after this day are left out: their outcome window is not fully observed.
OUTCOME_CUTOFF_DAY = 704

# Split by time. Campaigns that cross a boundary are excluded.
VALIDATION_START_DAY = 463
TEST_START_DAY = 587

NUMERIC_FEATURES = [
    "baskets_90d",
    "retailer_sales_90d",
    "recency_days_90d",
    "has_transactions_90d",
    "avg_retailer_sales_per_basket_90d",
    "active_days_90d",
]
CATEGORICAL_FEATURES = ["DESCRIPTION"]  # campaign type: TypeA, TypeB, TypeC
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
CAMPAIGN_TYPES = ["TypeA", "TypeB", "TypeC"]

# Chosen on validation campaigns in notebooks 04 and 05 (see docs/model_selection.md).
SELECTED_MODEL = "logistic_with_campaign_type"

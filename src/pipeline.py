"""Run the redemption-model pipeline from raw files to a registered model.

    python -m src.pipeline            # build features, train, validate, log to MLflow, save
    python -m src.pipeline --no-mlflow

Steps: raw CSV files -> modeling table -> split by time -> fit the candidates -> validation
metrics -> MLflow runs and Model Registry -> saved model and metadata.

The pipeline stops at validation. The held-out test campaigns were scored once, in
notebooks/06_final_evaluation.ipynb, and are not scored again here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import UTC, datetime

import joblib
import pandas as pd

from src import config, features, train


def table_fingerprint(table: pd.DataFrame) -> str:
    """A short hash of the modeling table, so a later run can tell if the data changed."""
    hashes = pd.util.hash_pandas_object(
        table[["household_key", "CAMPAIGN", config.TARGET, *config.NUMERIC_FEATURES]].round(6),
        index=False,
    )
    return hashlib.sha256(hashes.to_numpy().tobytes()).hexdigest()[:16]


def package_versions() -> dict[str, str]:
    import duckdb
    import sklearn
    import xgboost

    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "duckdb": duckdb.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    }


def run(log_to_mlflow: bool = True) -> dict:
    table = features.build_training_table()
    splits = train.split_by_time(table)
    results = train.train_candidates(splits)

    tracking_info = {"run_ids": {}, "registered_version": None}
    if log_to_mlflow:
        from src import tracking

        tracking_info = tracking.log_candidates(results, splits)

    selected = results[config.SELECTED_MODEL]
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected["pipeline"], config.MODEL_PATH)

    validation_table = pd.DataFrame([
        {"model": name, "selected": name == config.SELECTED_MODEL, **result["validation_metrics"]}
        for name, result in results.items()
    ])
    validation_table.to_csv(config.REPORTS_DIR / "validation_metrics.csv", index=False)

    metadata = {
        "trained_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "selected_model": config.SELECTED_MODEL,
        "model_features": config.MODEL_FEATURES,
        "target": config.TARGET,
        "history_window_days": config.HISTORY_WINDOW_DAYS,
        "validation_start_day": config.VALIDATION_START_DAY,
        "test_start_day": config.TEST_START_DAY,
        "modeling_rows": len(table),
        "data_fingerprint": table_fingerprint(table),
        "split_examples": {name: len(frame) for name, frame in splits.items()},
        "split_campaigns": {name: int(frame["CAMPAIGN"].nunique()) for name, frame in splits.items()},
        "training_redemption_rate": selected["training_redemption_rate"],
        "validation_metrics": {name: result["validation_metrics"] for name, result in results.items()},
        "validation_constant_baseline_brier": selected["constant_baseline_brier"],
        "mlflow_run_ids": tracking_info["run_ids"],
        "mlflow_registered_model": config.REGISTERED_MODEL_NAME if tracking_info["registered_version"] else None,
        "mlflow_registered_version": tracking_info["registered_version"],
        "package_versions": package_versions(),
    }
    config.MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the redemption-model pipeline.")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging and the registry.")
    args = parser.parse_args()

    metadata = run(log_to_mlflow=not args.no_mlflow)

    print(f"Modeling rows: {metadata['modeling_rows']:,}")
    print("Examples per split:", metadata["split_examples"])
    print("\nValidation results (four later campaigns):")
    for name, metrics in metadata["validation_metrics"].items():
        marker = "  <- selected" if name == metadata["selected_model"] else ""
        print(
            f"  {name:<30} ROC-AUC {metrics['roc_auc']:.4f}  "
            f"AP {metrics['average_precision']:.4f}  Brier {metrics['brier_score']:.4f}{marker}"
        )
    print(f"  constant-rate baseline          Brier {metadata['validation_constant_baseline_brier']:.4f}")
    if metadata["mlflow_registered_version"]:
        print(
            f"\nMLflow: registered '{metadata['mlflow_registered_model']}' "
            f"version {metadata['mlflow_registered_version']} as '{config.CHAMPION_ALIAS}'."
        )
    print(f"Saved model: {config.MODEL_PATH.relative_to(config.OUTPUT_ROOT)}")
    print(f"Saved metadata: {config.MODEL_METADATA_PATH.relative_to(config.OUTPUT_ROOT)}")


if __name__ == "__main__":
    main()

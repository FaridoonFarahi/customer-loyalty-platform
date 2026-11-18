"""MLflow logging: one run per candidate, and the selected model in the Model Registry."""
from __future__ import annotations

import warnings

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature

from src import config


def setup_mlflow() -> str:
    """Point MLflow at the local store and return the experiment id."""
    config.MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    artifact_dir = config.MLFLOW_DIR / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(config.MLFLOW_DIR / 'mlflow.db').as_posix()}")
    experiment = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            config.MLFLOW_EXPERIMENT, artifact_location=artifact_dir.as_uri()
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def log_candidates(results: dict[str, dict], splits: dict[str, pd.DataFrame]) -> dict:
    """Log every candidate as it is trained. Register the selected one and mark it champion."""
    setup_mlflow()
    run_ids = {}
    registered_version = None
    split_sizes = {f"{name}_examples": len(frame) for name, frame in splits.items()}
    split_campaigns = {f"{name}_campaigns": frame["CAMPAIGN"].nunique() for name, frame in splits.items()}

    for name, result in results.items():
        selected = name == config.SELECTED_MODEL
        with mlflow.start_run(run_name=f"pipeline_{name}") as run:
            mlflow.set_tags({
                "logging_mode": "pipeline",
                "selected_for_final_evaluation": str(selected).lower(),
                "selection_basis": "validation_metrics (docs/model_selection.md)",
            })
            mlflow.log_params({
                "history_window_days": config.HISTORY_WINDOW_DAYS,
                "outcome_cutoff_day": config.OUTCOME_CUTOFF_DAY,
                "validation_start_day": config.VALIDATION_START_DAY,
                "test_start_day": config.TEST_START_DAY,
                **split_sizes,
                **split_campaigns,
            })
            mlflow.log_params({
                f"model__{key}": value
                for key, value in result["pipeline"].named_steps["model"].get_params().items()
                if value is not None
            })
            mlflow.log_metrics({f"validation_{key}": value for key, value in result["validation_metrics"].items()})
            mlflow.log_metric("validation_constant_baseline_brier", result["constant_baseline_brier"])
            mlflow.log_dict({"input_features": result["columns"]}, "input_features.json")

            if selected:
                example = splits["train"][result["columns"]].head(5)
                signature = infer_signature(example, result["pipeline"].predict_proba(example)[:, 1])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mlflow.sklearn.log_model(
                        sk_model=result["pipeline"],
                        name="model",
                        signature=signature,
                        input_example=example,
                        registered_model_name=config.REGISTERED_MODEL_NAME,
                        # MLflow saves scikit-learn models in the skops format, which lists
                        # every type it will load. numpy.dtype is plain array metadata.
                        skops_trusted_types=["numpy.dtype"],
                    )
                client = MlflowClient()
                versions = client.search_model_versions(f"name='{config.REGISTERED_MODEL_NAME}'")
                registered_version = max(int(version.version) for version in versions)
                client.set_registered_model_alias(
                    config.REGISTERED_MODEL_NAME, config.CHAMPION_ALIAS, str(registered_version)
                )
            run_ids[name] = run.info.run_id

    return {"run_ids": run_ids, "registered_version": registered_version}


def log_final_metrics(run_id: str, metrics: dict[str, float]) -> None:
    """Attach held-out results to the selected model's run."""
    setup_mlflow()
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
        mlflow.set_tag("final_test_evaluated", "true")

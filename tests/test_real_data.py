"""Checks against the real dunnhumby files. Skipped when data/raw is not present (for example in CI)."""
import numpy as np
import pandas as pd
import pytest

from src import config, features, train

RAW_PRESENT = (config.RAW_DIR / "transaction_data.csv").is_file()
pytestmark = pytest.mark.skipif(not RAW_PRESENT, reason="data/raw is not present")


@pytest.fixture(scope="module")
def real_table() -> pd.DataFrame:
    return features.build_training_table()


def test_pipeline_table_equals_the_notebook_table(real_table):
    notebook_path = config.PROJECT_ROOT / "data" / "processed" / "household_campaign_modeling.parquet"
    if not notebook_path.is_file():
        pytest.skip("notebook modeling table not present")
    keys = ["household_key", "CAMPAIGN"]
    notebook = pd.read_parquet(notebook_path).sort_values(keys).reset_index(drop=True)
    pipeline = real_table.sort_values(keys).reset_index(drop=True)
    assert len(pipeline) == len(notebook) == 7091
    assert (pipeline[keys].to_numpy() == notebook[keys].to_numpy()).all()
    assert (pipeline[config.TARGET].to_numpy() == notebook[config.TARGET].to_numpy()).all()
    for column in config.NUMERIC_FEATURES:
        np.testing.assert_allclose(
            pipeline[column].to_numpy(), notebook[column].astype(float).to_numpy(),
            rtol=0, atol=1e-8, equal_nan=True,
        )


def test_real_split_sizes(real_table):
    splits = train.split_by_time(real_table)
    assert {name: len(frame) for name, frame in splits.items()} == {
        "train": 2499, "validation": 1584, "test": 2218,
    }
    assert {name: frame["CAMPAIGN"].nunique() for name, frame in splits.items()} == {
        "train": 13, "validation": 4, "test": 7,
    }


def test_logistic_validation_metrics_match_the_notebooks(real_table):
    reported_path = config.PROJECT_ROOT / "reports" / "model_evaluation" / "model_comparison_validation.csv"
    if not reported_path.is_file():
        pytest.skip("notebook validation report not present")
    reported = pd.read_csv(reported_path).set_index("model")
    results = train.train_candidates(train.split_by_time(real_table))
    names = {
        "logistic_numeric_baseline": "Numerical baseline",
        "logistic_with_campaign_type": "With campaign type",
    }
    for pipeline_name, notebook_name in names.items():
        for metric in ("roc_auc", "average_precision", "brier_score"):
            assert results[pipeline_name]["validation_metrics"][metric] == pytest.approx(
                reported.loc[notebook_name, metric], abs=1e-6
            )

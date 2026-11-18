"""Split the modeling table by time, fit the candidates, and score them on validation campaigns.

The candidates and their settings are the ones from notebooks 04 and 05. This module does not
choose a model: the choice was made on validation results and is fixed in ``config.SELECTED_MODEL``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src import config


def split_by_time(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Earlier campaigns train the model; later campaigns check it.

    A campaign belongs to a split only if it lies fully inside that period. Campaigns that
    cross a boundary are left out, so no outcome window overlaps the next period.
    """
    train = table.loc[table["END_DAY"] < config.VALIDATION_START_DAY]
    validation = table.loc[
        (table["START_DAY"] >= config.VALIDATION_START_DAY)
        & (table["END_DAY"] < config.TEST_START_DAY)
    ]
    holdout = table.loc[table["START_DAY"] >= config.TEST_START_DAY]
    splits = {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": holdout.reset_index(drop=True),
    }
    check_split(splits)
    return splits


def check_split(splits: dict[str, pd.DataFrame]) -> None:
    """Fail loudly if campaigns are shared between splits or the periods overlap."""
    campaigns = {name: set(frame["CAMPAIGN"]) for name, frame in splits.items()}
    if campaigns["train"] & campaigns["validation"] or campaigns["train"] & campaigns["test"] \
            or campaigns["validation"] & campaigns["test"]:
        raise ValueError("A campaign appears in more than one split.")
    if any(frame.empty for frame in splits.values()):
        raise ValueError("A split is empty.")
    if splits["train"]["END_DAY"].max() >= splits["validation"]["START_DAY"].min():
        raise ValueError("Training campaigns overlap the validation period.")
    if splits["validation"]["END_DAY"].max() >= splits["test"]["START_DAY"].min():
        raise ValueError("Validation campaigns overlap the test period.")


def build_candidates() -> dict[str, tuple[Pipeline, list[str]]]:
    """Each candidate with the columns it uses. Settings are fixed, as in the notebooks."""
    numeric = config.NUMERIC_FEATURES
    scaled_numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return {
        "logistic_numeric_baseline": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000)),
            ]),
            numeric,
        ),
        "logistic_with_campaign_type": (
            Pipeline([
                ("preprocessor", ColumnTransformer([
                    ("numeric", scaled_numeric, numeric),
                    ("campaign_type", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
                ])),
                ("model", LogisticRegression(max_iter=1000)),
            ]),
            config.MODEL_FEATURES,
        ),
        "xgboost_with_campaign_type": (
            Pipeline([
                ("preprocessor", ColumnTransformer([
                    ("numeric", SimpleImputer(strategy="median"), numeric),
                    ("campaign_type", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                     config.CATEGORICAL_FEATURES),
                ])),
                ("model", XGBClassifier(
                    objective="binary:logistic",
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    tree_method="hist",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=4,
                )),
            ]),
            config.MODEL_FEATURES,
        ),
    }


def probability_metrics(y_true, probabilities) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }


def train_candidates(splits: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Fit every candidate on training campaigns and score it on validation campaigns."""
    train, validation = splits["train"], splits["validation"]
    y_train, y_validation = train[config.TARGET], validation[config.TARGET]

    constant_rate = float(y_train.mean())
    constant_brier = float(brier_score_loss(y_validation, np.full(len(y_validation), constant_rate)))

    results = {}
    for name, (pipeline, columns) in build_candidates().items():
        pipeline.fit(train[columns], y_train)
        validation_probabilities = pipeline.predict_proba(validation[columns])[:, 1]
        results[name] = {
            "pipeline": pipeline,
            "columns": columns,
            "validation_metrics": probability_metrics(y_validation, validation_probabilities),
            "validation_probabilities": validation_probabilities,
            "constant_baseline_brier": constant_brier,
            "training_redemption_rate": constant_rate,
        }
    return results

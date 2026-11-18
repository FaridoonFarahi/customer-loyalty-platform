"""Scoring behaviour of the saved model. Skipped until the pipeline has produced a model."""
import numpy as np
import pandas as pd
import pytest

from src import config, predict

pytestmark = pytest.mark.skipif(not config.MODEL_PATH.is_file(), reason="run `python -m src.pipeline` first")


def example_rows() -> pd.DataFrame:
    return pd.DataFrame([
        {"baskets_90d": 30, "retailer_sales_90d": 900.0, "recency_days_90d": 2,
         "has_transactions_90d": 1, "avg_retailer_sales_per_basket_90d": 30.0,
         "active_days_90d": 25, "DESCRIPTION": "TypeA"},
        {"baskets_90d": 0, "retailer_sales_90d": 0.0, "recency_days_90d": np.nan,
         "has_transactions_90d": 0, "avg_retailer_sales_per_basket_90d": np.nan,
         "active_days_90d": 0, "DESCRIPTION": "TypeB"},
    ])


def test_probabilities_are_valid_and_missing_history_is_handled():
    probabilities = predict.score(predict.load_model(), example_rows())
    assert probabilities.shape == (2,)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert not np.isnan(probabilities).any()
    assert probabilities[0] > probabilities[1]  # an active TypeA household outranks an inactive TypeB one


def test_missing_input_column_is_rejected():
    with pytest.raises(KeyError):
        predict.score(predict.load_model(), example_rows().drop(columns=["recency_days_90d"]))


def test_ranking_flags_the_requested_share():
    scored = pd.DataFrame({"household_key": range(10), "redemption_probability": np.linspace(0.9, 0.0, 10)})
    ranked = predict.rank_households(scored, top_share=0.3)
    assert ranked["contact"].sum() == 3
    assert list(ranked.loc[ranked["contact"], "household_key"]) == [0, 1, 2]

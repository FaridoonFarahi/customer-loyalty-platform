"""Feature and label rules, checked on hand-made data with known answers."""
import numpy as np
import pandas as pd

from src import config, features
from tests.conftest import build_table


def row(table: pd.DataFrame, household: int, campaign: int) -> pd.Series:
    match = table[(table["household_key"] == household) & (table["CAMPAIGN"] == campaign)]
    assert len(match) == 1
    return match.iloc[0]


def test_one_row_per_targeted_pair_inside_cutoff(tiny_table):
    # Campaign 12 ends on day 710, after the day-704 cutoff, so its pair is left out.
    assert len(tiny_table) == 5
    assert 12 not in set(tiny_table["CAMPAIGN"])
    assert not tiny_table.duplicated(["household_key", "CAMPAIGN"]).any()


def test_features_use_only_the_90_days_before_the_start(tiny_table):
    household_1 = row(tiny_table, 1, 10)
    assert household_1["baskets_90d"] == 2            # baskets 901 and 902
    assert household_1["retailer_sales_90d"] == 22.0  # 5 + 7 + 10; not the 50, 99, or 40
    assert household_1["active_days_90d"] == 2        # days 150 and 199
    assert household_1["recency_days_90d"] == 1       # start day 200 minus last purchase day 199
    assert household_1["avg_retailer_sales_per_basket_90d"] == 11.0
    assert household_1["has_transactions_90d"] == 1
    assert household_1["history_start_day"] == 110
    assert household_1["history_end_day"] == 199


def test_household_with_no_history_is_kept_and_flagged(tiny_table):
    household_2 = row(tiny_table, 2, 10)
    assert household_2["baskets_90d"] == 0
    assert household_2["retailer_sales_90d"] == 0.0
    assert household_2["has_transactions_90d"] == 0
    assert np.isnan(household_2["recency_days_90d"])
    assert np.isnan(household_2["avg_retailer_sales_per_basket_90d"])


def test_label_counts_only_redemptions_during_the_campaign(tiny_table):
    assert row(tiny_table, 1, 10)["redeemed"] == 1  # day 215, inside days 200 to 230
    assert row(tiny_table, 2, 10)["redeemed"] == 0  # day 240, after the campaign ended
    assert row(tiny_table, 3, 10)["redeemed"] == 0
    assert row(tiny_table, 2, 11)["redeemed"] == 1  # the first campaign day counts
    assert row(tiny_table, 1, 11)["redeemed"] == 0


def test_no_leakage_from_the_campaign_period(tiny_sources, tiny_table):
    """Add heavy shopping on and after every start day. No feature may move."""
    leaky = dict(tiny_sources)
    future_rows = []
    for campaign in tiny_sources["campaigns"].itertuples():
        for household in (1, 2, 3):
            for offset in (0, 1, 10):
                future_rows.append(
                    (household, 5000 + len(future_rows), campaign.START_DAY + offset, 1000.0)
                )
    leaky["transactions"] = pd.concat([
        tiny_sources["transactions"],
        pd.DataFrame(future_rows, columns=["household_key", "BASKET_ID", "DAY", "SALES_VALUE"]),
    ])
    leaky_table = build_table(leaky)

    # Purchases after one campaign starts are real history for a LATER campaign, so compare
    # each campaign only against purchases added on or after its own start day.
    first_campaign = tiny_table[tiny_table["CAMPAIGN"] == 10].reset_index(drop=True)
    first_campaign_leaky = leaky_table[leaky_table["CAMPAIGN"] == 10].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        first_campaign[config.NUMERIC_FEATURES], first_campaign_leaky[config.NUMERIC_FEATURES]
    )


def test_truncating_data_at_the_start_day_changes_nothing(tiny_sources, tiny_table):
    """Features for a campaign must be the same if all later data is deleted."""
    for campaign in tiny_sources["campaigns"].itertuples():
        if campaign.CAMPAIGN not in set(tiny_table["CAMPAIGN"]):
            continue
        truncated = dict(tiny_sources)
        truncated["transactions"] = tiny_sources["transactions"].query("DAY < @campaign.START_DAY")
        truncated_table = build_table(truncated)
        full = tiny_table[tiny_table["CAMPAIGN"] == campaign.CAMPAIGN].reset_index(drop=True)
        cut = truncated_table[truncated_table["CAMPAIGN"] == campaign.CAMPAIGN].reset_index(drop=True)
        pd.testing.assert_frame_equal(full[config.NUMERIC_FEATURES], cut[config.NUMERIC_FEATURES])


def test_scoring_features_match_training_features(tiny_sources, tiny_table):
    """Scoring a planned campaign on day 200 must give household 1 the training features."""
    con = features.connect()
    try:
        features.register_source_tables(con, **tiny_sources)
        features.create_scoring_pairs(con, start_day=200, campaign_type="TypeA")
        scoring = features.add_history_features(con)
    finally:
        con.close()
    assert set(scoring["household_key"]) == {1, 3}  # household 2 had not shopped before day 200
    trained = row(tiny_table, 1, 10)
    scored = scoring[scoring["household_key"] == 1].iloc[0]
    for column in config.NUMERIC_FEATURES:
        assert scored[column] == trained[column]


def test_model_inputs_have_fixed_types(tiny_table):
    for column in config.NUMERIC_FEATURES:
        assert tiny_table[column].dtype == "float64"

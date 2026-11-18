"""Small hand-made source tables, so feature tests run without the real data."""
import pandas as pd
import pytest

from src import features


@pytest.fixture
def tiny_sources() -> dict[str, pd.DataFrame]:
    campaigns = pd.DataFrame({
        "CAMPAIGN": [10, 11, 12],
        "DESCRIPTION": ["TypeA", "TypeB", "TypeC"],
        "START_DAY": [200, 300, 650],
        "END_DAY": [230, 330, 710],  # campaign 12 ends after the day-704 cutoff
    })
    campaign_households = pd.DataFrame({
        "household_key": [1, 2, 3, 1, 2, 1],
        "CAMPAIGN": [10, 10, 10, 11, 11, 12],
    })
    transactions = pd.DataFrame(
        [
            # household 1, seen from campaign 10 (history window is days 110 to 199)
            (1, 900, 100, 50.0),   # older than 90 days: outside the window
            (1, 901, 150, 5.0),    # basket 901 has two lines
            (1, 901, 150, 7.0),
            (1, 902, 199, 10.0),   # last day inside the window
            (1, 903, 200, 99.0),   # campaign start day: must NOT count
            (1, 904, 215, 40.0),   # during the campaign: must NOT count
            # household 2 first shops after campaign 10 starts
            (2, 910, 250, 20.0),
            (2, 911, 299, 30.0),
            # household 3 never shops inside any window
            (3, 920, 50, 15.0),
        ],
        columns=["household_key", "BASKET_ID", "DAY", "SALES_VALUE"],
    )
    redemptions = pd.DataFrame({
        "household_key": [1, 2, 2],
        "CAMPAIGN": [10, 10, 11],
        "DAY": [215, 240, 300],  # inside window; after campaign 10 ended; on campaign 11's first day
    })
    return {
        "transactions": transactions,
        "campaigns": campaigns,
        "campaign_households": campaign_households,
        "redemptions": redemptions,
    }


def build_table(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    con = features.connect()
    try:
        features.register_source_tables(con, **sources)
        features.create_training_pairs(con)
        return features.add_history_features(con)
    finally:
        con.close()


@pytest.fixture
def tiny_table(tiny_sources) -> pd.DataFrame:
    return build_table(tiny_sources)

"""The time-based split: no shared campaigns, no overlap, boundary campaigns excluded."""
import pandas as pd
import pytest

from src import config, train


def campaign_rows(campaign: int, start: int, end: int, n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({
        "household_key": range(n), "CAMPAIGN": campaign, "START_DAY": start, "END_DAY": end,
        config.TARGET: [1] + [0] * (n - 1),
    })


def test_split_follows_time_and_drops_boundary_campaigns():
    table = pd.concat([
        campaign_rows(1, 300, 340),                                   # train
        campaign_rows(2, 440, config.VALIDATION_START_DAY + 5),       # crosses into validation: dropped
        campaign_rows(3, config.VALIDATION_START_DAY, 500),           # validation
        campaign_rows(4, 560, config.TEST_START_DAY + 10),            # crosses into test: dropped
        campaign_rows(5, config.TEST_START_DAY, 640),                 # test
    ])
    splits = train.split_by_time(table)
    assert set(splits["train"]["CAMPAIGN"]) == {1}
    assert set(splits["validation"]["CAMPAIGN"]) == {3}
    assert set(splits["test"]["CAMPAIGN"]) == {5}
    assert splits["train"]["END_DAY"].max() < splits["validation"]["START_DAY"].min()
    assert splits["validation"]["END_DAY"].max() < splits["test"]["START_DAY"].min()


def test_split_fails_when_a_period_has_no_campaigns():
    only_early = campaign_rows(1, 300, 340)
    with pytest.raises(ValueError):
        train.split_by_time(only_early)

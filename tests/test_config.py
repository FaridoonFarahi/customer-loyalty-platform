"""config.py must agree with the settings saved by the notebooks."""
import json

import pytest

from src import config


def test_config_matches_saved_modeling_config():
    saved_path = config.PROJECT_ROOT / "models" / "modeling_config.json"
    if not saved_path.is_file():
        pytest.skip("models/modeling_config.json not present")
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    assert saved["target"] == config.TARGET
    assert saved["numeric_features"] == config.NUMERIC_FEATURES
    assert saved["campaign_type_features"] == config.MODEL_FEATURES
    assert saved["history_window_days"] == config.HISTORY_WINDOW_DAYS
    assert saved["eligible_campaign_end_day_cutoff"] == config.OUTCOME_CUTOFF_DAY
    assert saved["validation_start_day"] == config.VALIDATION_START_DAY
    assert saved["test_start_day"] == config.TEST_START_DAY


def test_selected_model_is_a_known_candidate():
    from src import train

    assert config.SELECTED_MODEL in train.build_candidates()

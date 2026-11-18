"""Drift measure and the retraining decision."""
import numpy as np
import pandas as pd

from src import monitor


def test_psi_is_near_zero_for_the_same_distribution():
    rng = np.random.default_rng(0)
    reference, current = rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)
    assert monitor.psi(reference, current) < 0.02


def test_psi_is_large_for_a_shifted_distribution():
    rng = np.random.default_rng(0)
    reference, current = rng.normal(0, 1, 5000), rng.normal(1.5, 1, 5000)
    assert monitor.psi(reference, current) > monitor.PSI_ACT


def test_psi_counts_a_change_in_missing_values():
    reference = pd.Series(np.arange(1000, dtype=float))
    current = reference.copy()
    current.iloc[:300] = np.nan
    assert monitor.psi(reference, current) > monitor.PSI_WATCH


def test_psi_handles_a_constant_reference():
    assert monitor.psi(np.ones(100), np.ones(100)) == 0.0


def drift_row(period, psi_value):
    return {"period": period, "feature": "retailer_sales_90d", "psi": psi_value, "status": monitor.status(psi_value)}


def accuracy_row(campaign, observed, predicted, flag):
    return {"CAMPAIGN": campaign, "observed_rate": observed, "mean_predicted": predicted,
            "calibration_flag": flag, "auc_flag": False}


def test_stable_inputs_and_accurate_scores_keep_the_model():
    decision = monitor.decide(
        pd.DataFrame([drift_row("later campaigns", 0.03)]), pd.DataFrame(),
        pd.DataFrame([accuracy_row(1, 0.12, 0.11, False)]),
    )
    assert decision["recommendation"] == "keep the current model"
    assert decision["reasons"] == []


def test_input_drift_triggers_retraining():
    decision = monitor.decide(pd.DataFrame([drift_row("later campaigns", 0.40)]), pd.DataFrame(), None)
    assert decision["recommendation"] == "retrain"


def test_miscalibration_triggers_retraining():
    decision = monitor.decide(
        pd.DataFrame([drift_row("later campaigns", 0.03)]), pd.DataFrame(),
        pd.DataFrame([accuracy_row(18, 0.19, 0.13, True)]),
    )
    assert decision["recommendation"] == "retrain"
    assert "18" in decision["reasons"][0]


def test_planned_population_shift_is_a_warning_not_a_retraining_trigger():
    decision = monitor.decide(
        pd.DataFrame([drift_row("later campaigns", 0.03), drift_row("planned TypeA campaign", 0.60)]),
        pd.DataFrame(), None,
    )
    assert decision["recommendation"] == "keep the current model"
    assert len(decision["warnings"]) == 1

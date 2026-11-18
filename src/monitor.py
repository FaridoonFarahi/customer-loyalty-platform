"""Monitoring for the redemption model: has the data moved, and is the model still accurate?

    python -m src.monitor

Three checks, each with a written threshold (see docs/monitoring_and_retraining.md):

1. Input drift. Compare each later period's features with the training campaigns, using the
   population stability index (PSI). This needs no outcomes, so it can run before a campaign ends.
2. Score drift. The same comparison on the model's predicted probabilities.
3. Accuracy by campaign. Once outcomes are known, compare predicted and observed redemption.
   This reads the predictions already saved by notebooks/06_final_evaluation.ipynb. It does not
   score those campaigns again.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config, features, predict, train

PSI_WATCH = 0.10
PSI_ACT = 0.25
CALIBRATION_GAP_ACT = 0.05     # |observed rate - mean predicted probability|
AUC_DROP_ACT = 0.05            # below the within-campaign ROC-AUC seen on validation campaigns
MIN_REDEEMERS_FOR_AUC = 10        # fewer than this and the ROC-AUC is not reported
MIN_REDEEMERS_FOR_AUC_FLAG = 50   # fewer than this and the ROC-AUC is too noisy to act on
SAVED_PREDICTIONS = config.PROJECT_ROOT / "reports" / "model_evaluation" / "final_test_predictions.csv"


def psi(reference, current, bins: int = 10) -> float:
    """Population stability index between two samples of one numeric variable.

    Bin edges come from the reference sample's quantiles. Missing values get their own bin.
    0 means identical distributions; above 0.25 is usually treated as a real shift.
    """
    reference = pd.Series(reference, dtype="float64")
    current = pd.Series(current, dtype="float64")
    edges = np.unique(np.quantile(reference.dropna(), np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
    edges[0], edges[-1] = -np.inf, np.inf

    def shares(values: pd.Series) -> np.ndarray:
        counts = np.histogram(values.dropna(), bins=edges)[0].astype(float)
        counts = np.append(counts, values.isna().sum())
        return counts / max(len(values), 1)

    reference_share, current_share = shares(reference), shares(current)
    keep = (reference_share > 0) | (current_share > 0)
    reference_share = np.clip(reference_share[keep], 1e-4, None)
    current_share = np.clip(current_share[keep], 1e-4, None)
    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def category_psi(reference: pd.Series, current: pd.Series) -> float:
    levels = sorted(set(reference.unique()) | set(current.unique()))

    def shares(values: pd.Series) -> np.ndarray:
        return np.clip(values.value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy(), 1e-4, None)

    reference_share, current_share = shares(reference), shares(current)
    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def status(value: float) -> str:
    if value >= PSI_ACT:
        return "act"
    return "watch" if value >= PSI_WATCH else "stable"


def drift_tables(model, reference: pd.DataFrame, periods: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_scores = predict.score(model, reference)
    feature_rows, score_rows = [], []
    for period, frame in periods.items():
        for column in config.NUMERIC_FEATURES:
            value = psi(reference[column], frame[column])
            feature_rows.append({
                "period": period, "feature": column, "psi": value, "status": status(value),
                "reference_median": float(reference[column].median()), "period_median": float(frame[column].median()),
            })
        score_value = psi(reference_scores, predict.score(model, frame))
        score_rows.append({
            "period": period, "households_or_pairs": len(frame),
            "score_psi": score_value, "status": status(score_value),
            "campaign_type_psi": category_psi(reference["DESCRIPTION"], frame["DESCRIPTION"]),
            "mean_predicted_probability": float(predict.score(model, frame).mean()),
        })
    return pd.DataFrame(feature_rows), pd.DataFrame(score_rows)


def within_campaign_auc(frame: pd.DataFrame, probabilities: np.ndarray) -> float:
    """Example-weighted mean of the ROC-AUC inside each campaign that has enough redeemers.

    A pooled ROC-AUC mixes two things: ranking households inside a campaign, and telling
    high-redemption campaigns from low ones. Monitoring one campaign at a time needs the first.
    """
    scored = frame[["CAMPAIGN", config.TARGET]].assign(probability=probabilities)
    total, weight = 0.0, 0
    for _, group in scored.groupby("CAMPAIGN"):
        redeemers = int(group[config.TARGET].sum())
        if MIN_REDEEMERS_FOR_AUC <= redeemers < len(group):
            total += roc_auc_score(group[config.TARGET], group["probability"]) * len(group)
            weight += len(group)
    return total / weight if weight else float("nan")


def accuracy_by_campaign(saved: pd.DataFrame, reference_auc: float) -> pd.DataFrame:
    rows = []
    for campaign, group in saved.groupby("CAMPAIGN"):
        redeemers = int(group["redeemed"].sum())
        observed = float(group["redeemed"].mean())
        predicted = float(group["predicted_redemption_probability"].mean())
        auc = (
            float(roc_auc_score(group["redeemed"], group["predicted_redemption_probability"]))
            if redeemers >= MIN_REDEEMERS_FOR_AUC and redeemers < len(group) else np.nan
        )
        rows.append({
            "CAMPAIGN": int(campaign), "start_day": int(group["START_DAY"].iloc[0]), "examples": len(group),
            "redeemers": redeemers, "observed_rate": observed, "mean_predicted": predicted,
            "calibration_gap": observed - predicted, "roc_auc": auc,
            # Act on a gap only when it is both large and more than two standard errors from zero.
            "calibration_flag": bool(
                abs(observed - predicted) > CALIBRATION_GAP_ACT
                and abs(observed - predicted) > 2 * np.sqrt(observed * (1 - observed) / len(group))
            ),
            "auc_flag": bool(
                redeemers >= MIN_REDEEMERS_FOR_AUC_FLAG and not np.isnan(auc)
                and auc < reference_auc - AUC_DROP_ACT
            ),
        })
    return pd.DataFrame(rows).sort_values("start_day").reset_index(drop=True)


def decide(feature_drift: pd.DataFrame, score_drift: pd.DataFrame, accuracy: pd.DataFrame | None) -> dict:
    """Turn the three checks into one recommendation."""
    campaign_drift = feature_drift[~feature_drift["period"].str.startswith("planned")]
    reasons = []
    acted = campaign_drift.loc[campaign_drift["status"] == "act"]
    if len(acted):
        reasons.append(f"input drift above {PSI_ACT} for: " + ", ".join(sorted(set(acted["feature"]))))
    if accuracy is not None:
        flagged = accuracy.loc[accuracy["calibration_flag"]]
        if len(flagged):
            reasons.append(
                f"predicted and observed redemption differ by more than {CALIBRATION_GAP_ACT:.0%} in campaigns "
                + ", ".join(str(c) for c in flagged["CAMPAIGN"])
            )
        weak = accuracy.loc[accuracy["auc_flag"]]
        if len(weak):
            reasons.append("ranking accuracy fell in campaigns " + ", ".join(str(c) for c in weak["CAMPAIGN"]))
    watch = sorted(set(campaign_drift.loc[campaign_drift["status"] == "watch", "feature"]))
    planned = feature_drift[feature_drift["period"].str.startswith("planned") & (feature_drift["status"] == "act")]
    warnings = []
    if len(planned):
        warnings.append(
            "the planned scoring population differs strongly from the training population on "
            + ", ".join(sorted(set(planned["feature"])))
            + ". The model was trained only on households the retailer chose to target, so scores "
            "for all households are an extrapolation."
        )
    return {
        "recommendation": "retrain" if reasons else "keep the current model",
        "reasons": reasons,
        "features_to_watch": watch,
        "warnings": warnings,
    }


def run() -> dict:
    model = predict.load_model("file")
    splits = train.split_by_time(features.build_training_table())
    reference = splits["train"]
    reference_auc = within_campaign_auc(splits["validation"], predict.score(model, splits["validation"]))
    periods = {
        "validation campaigns (days 463-586)": splits["validation"],
        "later campaigns (day 587 on)": splits["test"],
        "planned TypeA campaign, all households, day 712": features.build_scoring_table(712, "TypeA"),
    }
    feature_drift, score_drift = drift_tables(model, reference, periods)

    accuracy = None
    if SAVED_PREDICTIONS.is_file():
        accuracy = accuracy_by_campaign(pd.read_csv(SAVED_PREDICTIONS), reference_auc)

    decision = decide(feature_drift, score_drift, accuracy)
    decision["reference_within_campaign_auc"] = reference_auc
    decision["thresholds"] = {
        "psi_watch": PSI_WATCH, "psi_act": PSI_ACT,
        "calibration_gap_act": CALIBRATION_GAP_ACT, "auc_drop_act": AUC_DROP_ACT,
    }

    config.MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    feature_drift.to_csv(config.MONITORING_DIR / "feature_drift.csv", index=False)
    score_drift.to_csv(config.MONITORING_DIR / "score_drift.csv", index=False)
    if accuracy is not None:
        accuracy.to_csv(config.MONITORING_DIR / "accuracy_by_campaign.csv", index=False)
    (config.MONITORING_DIR / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return {"feature_drift": feature_drift, "score_drift": score_drift, "accuracy": accuracy, "decision": decision}


def main() -> None:
    result = run()
    pd.set_option("display.width", 200)
    print("Input drift (PSI against the training campaigns):")
    print(result["feature_drift"].pivot(index="feature", columns="period", values="psi").round(3).to_string())
    print("\nScore drift:")
    print(result["score_drift"].round(3).to_string(index=False))
    if result["accuracy"] is not None:
        print("\nAccuracy by campaign (from the saved held-out predictions):")
        print(result["accuracy"].round(3).to_string(index=False))
    decision = result["decision"]
    print(f"\nRecommendation: {decision['recommendation']}")
    for reason in decision["reasons"]:
        print(f"  - {reason}")
    if decision["features_to_watch"]:
        print("  Watch:", ", ".join(decision["features_to_watch"]))
    for warning in decision["warnings"]:
        print(f"  Warning: {warning}")
    reference_auc = decision["reference_within_campaign_auc"]
    print(f"  Reference within-campaign ROC-AUC on validation campaigns: {reference_auc:.3f}")
    print(f"Saved: {config.MONITORING_DIR.relative_to(config.OUTPUT_ROOT)}")


if __name__ == "__main__":
    main()

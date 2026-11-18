"""Customer loyalty platform: a small app on top of the saved results and the redemption model.

    streamlit run streamlit_app.py

The app trains nothing. It reads the tables in reports/ and the model saved by `python -m src.pipeline`.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.stats.power import TTestIndPower

from src import config, predict

ROOT = config.PROJECT_ROOT
REPORTS = ROOT / "reports"

st.set_page_config(page_title="Customer loyalty platform", layout="wide")


@st.cache_data
def read_csv(relative_path: str) -> pd.DataFrame | None:
    path = REPORTS / relative_path
    return pd.read_csv(path) if path.is_file() else None


@st.cache_resource
def load_model():
    return predict.load_model("file") if config.MODEL_PATH.is_file() else None


@st.cache_data
def load_metadata() -> dict:
    path = config.MODEL_METADATA_PATH
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def missing(what: str) -> None:
    st.info(f"{what} is not available yet. Run the notebooks or `python -m src.pipeline` to create it.")


def gains(predictions: pd.DataFrame, share: float) -> dict:
    ranked = predictions.sort_values("predicted_redemption_probability", ascending=False)
    top = ranked.head(int(round(len(ranked) * share)))
    overall = predictions["redeemed"].mean()
    return {
        "contacted": len(top),
        "rate": top["redeemed"].mean(),
        "overall": overall,
        "lift": top["redeemed"].mean() / overall,
        "reached": top["redeemed"].sum() / predictions["redeemed"].sum(),
        "redeemers": int(top["redeemed"].sum()),
    }


# ---------------------------------------------------------------------------------- pages
def page_overview() -> None:
    st.title("Customer loyalty and marketing optimization")
    st.write(
        "Built on dunnhumby's public dataset *The Complete Journey*: 2,500 households, about 2.6 million "
        "transaction rows, and 30 coupon campaigns. This app shows the results and lets you try the model."
    )
    predictions = read_csv("model_evaluation/final_test_predictions.csv")
    test_metrics = read_csv("model_evaluation/final_test_metrics.csv")
    sample_size = read_csv("ab_test_design/baseline_adjustment_sample_size.csv")
    uplift_test = read_csv("uplift/final_test_effect_rmse.csv")

    one, two, three, four = st.columns(4)
    if test_metrics is not None:
        one.metric("Held-out ROC-AUC", f"{test_metrics['roc_auc'].iloc[0]:.3f}")
    if predictions is not None:
        top = gains(predictions, 0.10)
        two.metric("Top 10% redemption rate", f"{top['rate']:.1%}", f"{top['lift']:.2f}x the average")
    if sample_size is not None:
        row = sample_size.iloc[0]
        three.metric(
            "Households needed per test group",
            f"{int(row['required_per_group_adjusted']):,}",
            f"down from {int(row['required_per_group_unadjusted']):,}",
            delta_color="off",
        )
    if uplift_test is not None:
        errors = uplift_test.set_index("model")["effect_rmse"]
        four.metric(
            "Uplift error (simulated)",
            f"{errors['Ridge-plus-trees T-learner']:.2f}",
            f"baseline {errors['Constant training-group difference']:.2f}",
            delta_color="off",
        )

    st.subheader("Five questions")
    st.table(pd.DataFrame([
        ("Who redeems a coupon?", "Logistic regression, trained on earlier campaigns and tested on later ones", "Who to target"),
        ("What would a household score?", "The saved model, called live", "Score a household"),
        ("Did campaign 18 change spending?", "Difference in changes with overlap weighting. An association, not a cause.", "Campaign effect and uplift"),
        ("How should the next campaign be tested?", "Power analysis with baseline adjustment", "Plan an experiment"),
        ("Who should get the campaign?", "Uplift T-learner on simulated data with a known effect", "Campaign effect and uplift"),
    ], columns=["Question", "Method", "Page"]))
    st.caption(
        "Limits: the redemption model predicts coupon use, not extra sales. The campaign analysis is observational. "
        "The experiment and uplift results are simulations and are labeled as such."
    )


def page_targeting() -> None:
    st.title("Who to target")
    predictions = read_csv("model_evaluation/final_test_predictions.csv")
    if predictions is None:
        return missing("The held-out test predictions file")
    st.write(
        "These are the model's scores on the seven held-out test campaigns, with what really happened. "
        "Choose how many households you can afford to contact. The model picks the highest scores first."
    )
    share = st.slider("Share of households to contact", 5, 100, 20, step=5, format="%d%%") / 100
    result = gains(predictions, share)

    one, two, three, four = st.columns(4)
    one.metric("Contacted", f"{result['contacted']:,}")
    two.metric("Redemption rate", f"{result['rate']:.1%}", f"average is {result['overall']:.1%}", delta_color="off")
    three.metric("Lift over average", f"{result['lift']:.2f}x")
    four.metric("Redeemers reached", f"{result['reached']:.1%}", f"{result['redeemers']:,} redeemers", delta_color="off")

    ranked = predictions.sort_values("predicted_redemption_probability", ascending=False).reset_index(drop=True)
    curve = pd.DataFrame({
        "Share contacted": (np.arange(len(ranked)) + 1) / len(ranked),
        "Model ranking": ranked["redeemed"].cumsum() / ranked["redeemed"].sum(),
    })
    curve["No ranking ability"] = curve["Share contacted"]
    st.subheader("Share of redeemers reached")
    st.line_chart(curve.iloc[:: max(1, len(curve) // 200)].set_index("Share contacted"))

    st.subheader("Households the model would contact first")
    st.dataframe(
        ranked.head(result["contacted"])[["household_key", "CAMPAIGN", "predicted_redemption_probability", "redeemed"]]
        .rename(columns={"predicted_redemption_probability": "predicted probability"}),
        hide_index=True, width="stretch",
    )
    st.caption("This shows who redeems a coupon. It does not show who buys more because of the campaign.")


def page_score() -> None:
    st.title("Score a household")
    model = load_model()
    if model is None:
        return missing("The saved model")
    st.write("Describe a household's last 90 days before a campaign starts. The model returns its redemption probability.")

    left, right = st.columns(2)
    campaign_type = left.selectbox("Campaign type", config.CAMPAIGN_TYPES)
    baskets = left.number_input("Shopping baskets", min_value=0, max_value=500, value=25)
    sales = left.number_input("Total sales", min_value=0.0, max_value=20000.0, value=800.0, step=50.0)
    active_days = right.number_input("Days with a purchase", min_value=0, max_value=90, value=20)
    recency = right.number_input("Days since the last purchase", min_value=1, max_value=90, value=3)

    if baskets == 0:
        sales, active_days, recency_value = 0.0, 0, np.nan
        st.caption("With no baskets, sales, active days, and recency are treated as empty.")
    else:
        recency_value = float(recency)
        active_days = max(1, min(active_days, baskets))

    row = pd.DataFrame([{
        "baskets_90d": baskets, "retailer_sales_90d": sales, "recency_days_90d": recency_value,
        "has_transactions_90d": int(baskets > 0),
        "avg_retailer_sales_per_basket_90d": sales / baskets if baskets > 0 else np.nan,
        "active_days_90d": active_days, "DESCRIPTION": campaign_type,
    }])
    probability = float(predict.score(model, row)[0])
    average = load_metadata().get("training_redemption_rate")

    one, two = st.columns(2)
    one.metric("Redemption probability", f"{probability:.1%}")
    if average:
        two.metric("Compared with the training average", f"{probability / average:.2f}x", f"average {average:.1%}", delta_color="off")

    scores = read_csv("pipeline/scores_day712_TypeA.csv")
    if scores is not None:
        st.subheader("Ranked list for a planned campaign")
        st.write("Every household scored for a TypeA campaign starting on day 712, from `python -m src.predict`.")
        st.dataframe(scores.head(200), hide_index=True, width="stretch")
        st.download_button("Download the full list (CSV)", scores.to_csv(index=False), "scores_day712_TypeA.csv", "text/csv")


def page_experiment() -> None:
    st.title("Plan an experiment")
    planning = read_csv("ab_test_design/power_analysis.csv")
    adjustment = read_csv("ab_test_design/baseline_adjustment_sample_size.csv")
    mean_default = float(planning["baseline_mean"].iloc[0]) if planning is not None else 274.30
    sd_default = float(planning["assumed_standard_deviation"].iloc[0]) if planning is not None else 328.46
    correlation_default = float(adjustment["baseline_outcome_correlation"].iloc[0]) if adjustment is not None else 0.85

    st.write(
        "A randomized test compares a campaign group with a holdout group on sales per household over 56 days. "
        "This page works out how many households the test needs. The defaults come from the historical data."
    )
    left, right = st.columns(2)
    lift = left.slider("Smallest sales increase worth detecting", 2, 30, 10, format="%d%%") / 100
    per_group = left.number_input("Households available per group", min_value=50, max_value=100000, value=1249, step=50)
    power_target = left.slider("Power wanted", 50, 95, 80, step=5, format="%d%%") / 100
    mean = right.number_input("Mean sales per household", min_value=1.0, value=round(mean_default, 2))
    sd = right.number_input("Standard deviation of sales", min_value=1.0, value=round(sd_default, 2))
    correlation = right.number_input(
        "Correlation between baseline and outcome sales",
        min_value=0.0, max_value=0.95, value=round(correlation_default, 4), step=0.01, format="%.3f",
    )

    effect = mean * lift
    adjusted_sd = sd * np.sqrt(1 - correlation ** 2)
    solver = TTestIndPower()
    rows = []
    for label, spread in [("Plain comparison", sd), ("With baseline adjustment (CUPED)", adjusted_sd)]:
        rows.append({
            "Analysis": label,
            "Households needed per group": int(np.ceil(solver.solve_power(effect_size=effect / spread, alpha=0.05, power=power_target))),
            "Power with the households available": solver.power(effect_size=effect / spread, nobs1=per_group, alpha=0.05),
        })
    summary = pd.DataFrame(rows)

    one, two = st.columns(2)
    one.metric("Needed per group, plain comparison", f"{summary.loc[0, 'Households needed per group']:,}",
               f"power now {summary.loc[0, 'Power with the households available']:.0%}", delta_color="off")
    two.metric("Needed per group, with baseline adjustment", f"{summary.loc[1, 'Households needed per group']:,}",
               f"power now {summary.loc[1, 'Power with the households available']:.0%}", delta_color="off")

    sizes = np.unique(np.linspace(100, max(3000, int(per_group * 1.5)), 60).astype(int))
    st.subheader("Power by households per group")
    st.line_chart(pd.DataFrame({
        "Households per group": sizes,
        "Plain comparison": [solver.power(effect_size=effect / sd, nobs1=n, alpha=0.05) for n in sizes],
        "With baseline adjustment": [solver.power(effect_size=effect / adjusted_sd, nobs1=n, alpha=0.05) for n in sizes],
    }).set_index("Households per group"))
    st.caption(
        f"Detecting a {lift:.0%} increase means a difference of {effect:.2f} per household. Two-sided test at the 5% level. "
        "No real randomized campaign was run; the simulation study that checks these figures is in docs/ab_test_design.md."
    )


def page_effects() -> None:
    st.title("Campaign effect and uplift")
    st.subheader("Campaign 18: observational comparison")
    results = read_csv("campaign_effectiveness/campaign_18_observational_results.csv")
    exposure = read_csv("campaign_effectiveness/campaign_18_other_campaign_exposure.csv")
    if results is None:
        missing("The campaign 18 results file")
    else:
        st.dataframe(
            results[["method", "sales_change_contrast", "ci_lower", "ci_upper"]].rename(columns={
                "method": "Comparison", "sales_change_contrast": "Difference in sales change",
                "ci_lower": "95% interval, low", "ci_upper": "95% interval, high",
            }).round(2), hide_index=True, width="stretch",
        )
        if exposure is not None:
            weighted = exposure.set_index("targeted")["weighted_other_campaign_pct"]
            st.warning(
                f"These are associations, not the effect of campaign 18. After weighting, {weighted[1]:.1f}% of targeted "
                f"households were also in other overlapping campaigns, against {weighted[0]:.1f}% of the comparison group."
            )

    st.subheader("Uplift model on simulated data")
    targeting = read_csv("uplift/final_test_targeting_value.csv")
    bands = read_csv("uplift/final_test_effect_by_activity_band.csv")
    if targeting is None:
        return missing("The uplift results")
    st.write(
        "The simulation gives each household a known campaign effect. The table shows the average known effect "
        "among the households each rule would contact, on 500 held-out households."
    )
    share = st.select_slider("Share contacted", options=sorted(targeting["share_contacted"]), value=0.4, format_func=lambda v: f"{v:.0%}")
    chosen = targeting[targeting["share_contacted"] == share].drop(columns="share_contacted").T
    chosen.columns = ["Average known effect per contacted household"]
    st.bar_chart(chosen)
    if bands is not None:
        st.dataframe(bands.rename(columns={
            "activity_band": "Baseline activity", "households": "Households",
            "known_effect": "Known effect", "boosted_mean_estimate": "Model estimate",
        }).round(2), hide_index=True, width="stretch")
    st.caption("The effect pattern was written by hand. This shows the method works on a pattern of this kind, not how real households respond.")


def page_model() -> None:
    st.title("Model and pipeline")
    metadata = load_metadata()
    if not metadata:
        return missing("The model metadata")
    one, two, three = st.columns(3)
    one.metric("Selected model", "Logistic regression")
    two.metric("Trained (UTC)", metadata["trained_at_utc"].replace("T", " ").replace("Z", ""))
    three.metric("Registry version", metadata.get("mlflow_registered_version") or "not registered")

    st.subheader("Validation results")
    validation = pd.DataFrame(metadata["validation_metrics"]).T.rename_axis("model").reset_index()
    validation["selected"] = validation["model"] == metadata["selected_model"]
    st.dataframe(validation.round(4), hide_index=True, width="stretch")

    st.subheader("How the model is built")
    st.code(
        "python -m src.pipeline      # raw files -> features -> split -> train -> validate -> MLflow -> saved model\n"
        "python -m pytest            # feature, leakage, split, scoring, and service tests\n"
        "python -m src.predict --start-day 712 --campaign-type TypeA --top-share 0.2\n"
        "uvicorn src.api:app         # scoring service, docs at /docs",
        language="bash",
    )
    st.json({key: metadata[key] for key in (
        "model_features", "history_window_days", "split_examples", "split_campaigns", "data_fingerprint", "package_versions"
    )})


def page_monitoring() -> None:
    st.title("Monitoring")
    decision_path = REPORTS / "monitoring" / "decision.json"
    feature_drift = read_csv("monitoring/feature_drift.csv")
    accuracy = read_csv("monitoring/accuracy_by_campaign.csv")
    if not decision_path.is_file() or feature_drift is None:
        return missing("The monitoring report (`python -m src.monitor`)")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))

    st.metric("Recommendation", decision["recommendation"].capitalize())
    for reason in decision["reasons"]:
        st.error(reason.capitalize() + ".")
    for warning in decision["warnings"]:
        st.warning(warning[0].upper() + warning[1:])

    st.subheader("Input drift against the training campaigns")
    st.write("Population stability index (PSI). Under 0.10 is stable, 0.10 to 0.25 is worth watching, 0.25 or more calls for action.")
    st.dataframe(
        feature_drift.pivot(index="feature", columns="period", values="psi").round(3),
        width="stretch",
    )
    if accuracy is not None:
        st.subheader("Predicted and observed redemption by campaign")
        chart = accuracy.assign(campaign=accuracy["CAMPAIGN"].astype(str)).set_index("campaign")
        st.bar_chart(chart[["observed_rate", "mean_predicted"]], stack=False)
        st.dataframe(
            accuracy[["CAMPAIGN", "examples", "redeemers", "observed_rate", "mean_predicted", "calibration_gap", "roc_auc", "calibration_flag"]].round(3),
            hide_index=True, width="stretch",
        )
    st.caption("The rules and the retraining steps are in docs/monitoring_and_retraining.md.")


PAGES = {
    "Overview": page_overview,
    "Who to target": page_targeting,
    "Score a household": page_score,
    "Plan an experiment": page_experiment,
    "Campaign effect and uplift": page_effects,
    "Model and pipeline": page_model,
    "Monitoring": page_monitoring,
}

choice = st.sidebar.radio("Page", list(PAGES))
st.sidebar.caption("Data: dunnhumby, The Complete Journey. A personal portfolio project.")
PAGES[choice]()

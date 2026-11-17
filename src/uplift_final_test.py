"""Final test for the simulated uplift study. Run ONCE, from notebooks/11_uplift_modeling.ipynb:

    %run -i ../src/uplift_final_test.py

It uses the variables already in the notebook (the fitted Ridge-plus-trees T-learner, the three
household splits, and the validation tables). It scores the frozen test households once through
``guard.final_test``, saves the results, and writes docs/uplift_modeling.md from the real numbers.
"""
import joblib
import numpy as np
import pandas as pd
from IPython.display import display

_required = [
    "guard", "project_root", "uplift_state_dir", "uplift_simulation",
    "uplift_train", "uplift_validation", "uplift_test",
    "uplift_features", "uplift_outcome", "constant_uplift",
    "boosted_holdout_model", "boosted_campaign_model",
    "uplift_validation_results", "validation_band_review", "validation_targeting",
    "validation_observed_uplift", "validation_error_review", "mean_effect_of_selected",
]
_missing = [name for name in _required if name not in globals()]
if _missing:
    raise RuntimeError(
        "Run every cell of 11_uplift_modeling.ipynb first, and start this script with "
        f"`%run -i`. Missing from the notebook: {_missing}"
    )

_validation_chart = uplift_state_dir / "figures" / "12_boosted_validation_effects.png"
if not _validation_chart.is_file():
    raise RuntimeError("Run the validation review cells before the final test.")

uplift_report_folder = project_root / "reports" / "uplift"
uplift_report_folder.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ the single test evaluation
test_known_effect = uplift_test["true_simulated_effect"].to_numpy()
test_scores = {}


def boosted_effect_estimate(features):
    test_scores["boosted"] = (
        boosted_campaign_model.predict(features[uplift_features])
        - boosted_holdout_model.predict(features[uplift_features])
    )
    return test_scores["boosted"]


def effect_rmse(_, estimated_effect):
    return float(np.sqrt(np.mean((estimated_effect - test_known_effect) ** 2)))


final_test_effect_rmse = guard.final_test(
    boosted_effect_estimate,
    uplift_test,
    target=uplift_outcome,
    metric_fn=effect_rmse,
    state_dir=uplift_state_dir,
    pipeline_md=uplift_state_dir / "PIPELINE.md",
)

constant_test_rmse = float(np.sqrt(np.mean((constant_uplift - test_known_effect) ** 2)))

final_test_summary = pd.DataFrame([
    {"model": "Constant training-group difference", "effect_rmse": constant_test_rmse},
    {"model": "Ridge-plus-trees T-learner", "effect_rmse": final_test_effect_rmse},
])

# ------------------------------------------------------------------ views of those same estimates
test_band_review = (
    uplift_test
    .assign(
        activity_band=pd.cut(
            uplift_test["active_days_before"],
            bins=[-1, 4, 14, 56],
            labels=["0-4 active days", "5-14 active days", "15+ active days"],
        ),
        boosted_estimate=test_scores["boosted"],
    )
    .groupby("activity_band", observed=True)
    .agg(
        households=("household_key", "size"),
        known_effect=("true_simulated_effect", "mean"),
        boosted_mean_estimate=("boosted_estimate", "mean"),
    )
)

_test_rules = {
    "Ridge-plus-trees estimate": test_scores["boosted"],
    "Highest baseline sales": uplift_test["sales_before"].to_numpy(),
    "Best possible (known effect)": test_known_effect,
}
_rows = []
for _share in [0.20, 0.40, 0.60]:
    _row = {"share_contacted": _share}
    for _rule, _score in _test_rules.items():
        _row[_rule] = mean_effect_of_selected(_score, test_known_effect, _share)
    _row["No targeting"] = float(test_known_effect.mean())
    _rows.append(_row)
test_targeting = pd.DataFrame(_rows).set_index("share_contacted")

print("Final test: effect RMSE on 500 test households (lower is better)")
display(final_test_summary.round(2))
print("Mean estimated effect by baseline activity, test households")
display(test_band_review.round(2))
print("Average known effect among contacted households, test households")
display(test_targeting.round(2))

# ------------------------------------------------------------------ save results, model, data
final_test_summary.to_csv(uplift_report_folder / "final_test_effect_rmse.csv", index=False)
test_band_review.to_csv(uplift_report_folder / "final_test_effect_by_activity_band.csv")
test_targeting.to_csv(uplift_report_folder / "final_test_targeting_value.csv")

joblib.dump(
    {
        "holdout_model": boosted_holdout_model,
        "campaign_model": boosted_campaign_model,
        "features": uplift_features,
        "outcome": uplift_outcome,
        "note": "Simulated data. Effect estimate = campaign_model - holdout_model.",
    },
    project_root / "models" / "uplift_ridge_plus_trees_t_learner.joblib",
)

_split_labels = pd.concat([
    uplift_train[["household_key"]].assign(split="train"),
    uplift_validation[["household_key"]].assign(split="validation"),
    uplift_test[["household_key"]].assign(split="test"),
])
_simulated_folder = project_root / "data" / "simulated"
_simulated_folder.mkdir(parents=True, exist_ok=True)
uplift_simulation.merge(
    _split_labels, on="household_key", how="left", validate="one_to_one"
).to_parquet(_simulated_folder / "simulated_uplift_households.parquet", index=False)


# ------------------------------------------------------------------ write the document
def _markdown_table(frame, digits=2):
    if not isinstance(frame.index, pd.RangeIndex):
        frame = frame.reset_index()
    lines = [
        "| " + " | ".join(str(column) for column in frame.columns) + " |",
        "|" + "|".join("---" for _ in frame.columns) + "|",
    ]
    for values in frame.itertuples(index=False):
        cells = [
            f"{value:,.{digits}f}" if isinstance(value, (float, np.floating)) else str(value)
            for value in values
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


_document = f"""# Uplift modeling on simulated campaign data

## Purpose

A redemption model predicts who uses a coupon. An uplift model estimates **how much more a household buys because it received the campaign**, which is what decides who should receive it.

The historical data has no randomized campaign, so the true effect of a campaign on any household is unknown. This step therefore uses a simulation with a known effect to test whether the method can recover it. **Nothing here measures a real campaign.**

## Simulation

- **Households:** {len(uplift_simulation):,}, with real baseline sales and active days from days 531 to 586.
- **Assignment:** random 1:1 to campaign or holdout (seed 42).
- **Sales without a campaign:** 0.8 x baseline sales + 5 x baseline active days + random noise.
- **Known campaign effect:** 10 for households with 0 to 4 baseline active days, 40 for 5 to 14 days, 20 for 15 or more days.

## Method

A T-learner fits one sales model on holdout households and one on campaign households. The estimated effect for a household is the difference between the two predictions.

Households were split once into {len(uplift_train):,} training, {len(uplift_validation):,} validation, and {len(uplift_test):,} test households. Candidates were compared on validation households. The test households were scored once, after the candidate was chosen.

## Validation results

Effect RMSE is the typical gap between the estimated and the known effect. Lower is better.

{_markdown_table(uplift_validation_results)}

Mean estimated effect by baseline activity:

{_markdown_table(validation_band_review)}

Ridge is a straight-line model, so it cannot follow an effect that rises and then falls. Adding shallow trees on top of Ridge lets the model follow the steps.

Average known effect among the households each rule would contact:

{_markdown_table(validation_targeting)}

Check that does not use the known effect (campaign minus holdout, inside each third of estimated effect):

{_markdown_table(validation_observed_uplift)}

Errors next to the two jumps in the simulated effect:

{_markdown_table(validation_error_review)}

## Final test

The Ridge-plus-trees T-learner was scored once on the {len(uplift_test):,} test households.

{_markdown_table(final_test_summary)}

Mean estimated effect by baseline activity, test households:

{_markdown_table(test_band_review)}

Average known effect among the households each rule would contact, test households:

{_markdown_table(test_targeting)}

## Limits

- The effect pattern was written by hand. The result shows the method can recover a pattern of this kind. It says nothing about how real households respond.
- The simulated noise is small compared with real sales. Real campaign data would need far more households to estimate effects this precisely.
- The effect depends on one feature. Real effects would depend on many, and some would not be recorded.
- Only one tree-based candidate was tried, with fixed settings. No tuning was done.
- A real uplift model needs a randomized campaign with a holdout group. `docs/ab_test_design.md` describes that experiment.

## Saved evidence

| File | Contents |
|---|---|
| `notebooks/11_uplift_modeling.ipynb` | Simulation, models, and validation review |
| `src/uplift_final_test.py` | The single final test, run from the notebook |
| `reports/uplift/` | Validation and final-test tables |
| `ml_pipeline/uplift/figures/` | Charts, with their explanations in `ml_pipeline/uplift/PIPELINE.md` |
| `models/uplift_ridge_plus_trees_t_learner.joblib` | Fitted holdout and campaign models |
| `data/simulated/simulated_uplift_households.parquet` | Simulated households with their split |
"""

(project_root / "docs" / "uplift_modeling.md").write_text(_document, encoding="utf-8")

print("Saved: reports/uplift (three final-test tables)")
print("Saved: models/uplift_ridge_plus_trees_t_learner.joblib")
print("Saved: data/simulated/simulated_uplift_households.parquet")
print("Saved: docs/uplift_modeling.md")

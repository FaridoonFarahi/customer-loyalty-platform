# Uplift modeling on simulated campaign data

## Purpose

A redemption model predicts who uses a coupon. An uplift model estimates **how much more a household buys because it received the campaign**, which is what decides who should receive it.

The historical data has no randomized campaign, so the true effect of a campaign on any household is unknown. This step therefore uses a simulation with a known effect to test whether the method can recover it. **Nothing here measures a real campaign.**

## Simulation

- **Households:** 2,498, with real baseline sales and active days from days 531 to 586.
- **Assignment:** random 1:1 to campaign or holdout (seed 42).
- **Sales without a campaign:** 0.8 x baseline sales + 5 x baseline active days + random noise.
- **Known campaign effect:** 10 for households with 0 to 4 baseline active days, 40 for 5 to 14 days, 20 for 15 or more days.

## Method

A T-learner fits one sales model on holdout households and one on campaign households. The estimated effect for a household is the difference between the two predictions.

Households were split once into 1,498 training, 500 validation, and 500 test households. Candidates were compared on validation households. The test households were scored once, after the candidate was chosen.

## Validation results

Effect RMSE is the typical gap between the estimated and the known effect. Lower is better.

| model | effect_rmse |
|---|---|
| Constant training-group difference | 14.22 |
| Ridge T-learner | 13.24 |
| Ridge-plus-trees T-learner | 5.71 |

Mean estimated effect by baseline activity:

| activity_band | households | known_effect | ridge_mean_estimate | boosted_mean_estimate | boosted_estimate_sd |
|---|---|---|---|---|---|
| 0-4 active days | 220 | 10.00 | 23.31 | 11.89 | 4.83 |
| 5-14 active days | 200 | 40.00 | 25.65 | 41.06 | 5.82 |
| 15+ active days | 80 | 20.00 | 28.11 | 18.50 | 6.43 |

Ridge is a straight-line model, so it cannot follow an effect that rises and then falls. Adding shallow trees on top of Ridge lets the model follow the steps.

Average known effect among the households each rule would contact:

| share_contacted | Ridge-plus-trees estimate | Ridge estimate | Highest baseline sales | Best possible (known effect) | No targeting |
|---|---|---|---|---|---|
| 0.20 | 40.00 | 30.50 | 29.50 | 40.00 | 23.60 |
| 0.40 | 39.60 | 30.85 | 30.75 | 40.00 | 23.60 |
| 0.60 | 32.10 | 29.70 | 30.03 | 32.67 | 23.60 |

Check that does not use the known effect (campaign minus holdout, inside each third of estimated effect):

| estimated_effect_group | campaign_households | holdout_households | observed_difference | standard_error | known_mean_effect |
|---|---|---|---|---|---|
| Lowest third | 77 | 90 | 9.86 | 2.81 | 10.96 |
| Middle third | 80 | 86 | 14.91 | 3.85 | 20.06 |
| Highest third | 76 | 91 | 38.13 | 3.11 | 39.76 |

Errors next to the two jumps in the simulated effect:

| position | households | mean_absolute_error | rmse |
|---|---|---|---|
| Away from both jumps | 341 | 4.34 | 5.59 |
| Near the jump at 15 days (13-16) | 40 | 6.73 | 7.89 |
| Near the jump at 5 days (3-6) | 119 | 4.31 | 5.16 |

## Final test

The Ridge-plus-trees T-learner was scored once on the 500 test households.

| model | effect_rmse |
|---|---|
| Constant training-group difference | 13.93 |
| Ridge-plus-trees T-learner | 6.36 |

Mean estimated effect by baseline activity, test households:

| activity_band | households | known_effect | boosted_mean_estimate |
|---|---|---|---|
| 0-4 active days | 226 | 10.00 | 12.86 |
| 5-14 active days | 188 | 40.00 | 39.95 |
| 15+ active days | 86 | 20.00 | 17.37 |

Average known effect among the households each rule would contact, test households:

| share_contacted | Ridge-plus-trees estimate | Highest baseline sales | Best possible (known effect) | No targeting |
|---|---|---|---|---|
| 0.20 | 39.60 | 28.00 | 40.00 | 23.00 |
| 0.40 | 38.60 | 29.80 | 38.80 | 23.00 |
| 0.60 | 30.67 | 29.53 | 31.67 | 23.00 |

## Limits

- The effect pattern was written by hand. The result shows the method can recover a pattern of this kind. It says nothing about how real households respond.
- The simulated noise is small compared with real sales. Real campaign data would need far more households to estimate effects this precisely.
- The effect depends on one feature. Real effects would depend on many, and some would not be recorded.
- Only one tree-based candidate was tried, with fixed settings. No tuning was done.
- A real uplift model needs a randomized campaign with a holdout group. `docs/ab_test_design.md` describes that experiment.

## If this were used on a real campaign

Nothing here is deployed. A real version would need the following.

- **A randomized campaign with a holdout group** as the training data, so the effect is measured and not assumed.
- **A check that needs no known effect.** Sort households by estimated effect and compare campaign and holdout households inside each group, as in the validation check above. Repeat it on every new campaign.
- **Watch the inputs.** If baseline sales or active days shift away from the range the model was fitted on, its estimates are less reliable. In this study the estimates already run low for the few households with 30 or more active days.
- **Refit on a schedule.** Refit after each new randomized campaign, using the same split-then-test-once steps.

## Saved evidence

| File | Contents |
|---|---|
| `notebooks/11_uplift_modeling.ipynb` | Simulation, models, and validation review |
| `src/uplift_final_test.py` | The single final test, run from the notebook |
| `reports/uplift/` | Validation and final-test tables |
| `ml_pipeline/uplift/figures/` | Charts, with their explanations in `ml_pipeline/uplift/PIPELINE.md` |
| `models/uplift_ridge_plus_trees_t_learner.joblib` | Fitted holdout and campaign models |
| `data/simulated/simulated_uplift_households.parquet` | Simulated households with their split |

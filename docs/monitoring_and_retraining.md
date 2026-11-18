# Monitoring and retraining

A model that was accurate when it was built can go stale. Customers change, campaigns change, and the data feeding the model changes. This document says what is checked, what counts as a problem, and what to do about it.

Run the checks with:

```bash
python -m src.monitor
```

## What is checked

| Check | Question | Needs outcomes? | Rule for action |
|---|---|---|---|
| Input drift | Do the features in a later period look like the features the model was trained on? | No | Population stability index (PSI) of 0.25 or more on any feature. From 0.10 to 0.25 is "watch". |
| Score drift | Has the spread of predicted probabilities moved? | No | Reported with the same PSI bands |
| Accuracy by campaign | Does predicted redemption match what happened? | Yes | Observed and predicted redemption differ by more than 5 percentage points, and by more than two standard errors |
| Ranking by campaign | Does the model still rank households well inside a campaign? | Yes | ROC-AUC more than 0.05 below the within-campaign level on validation campaigns (0.702), for campaigns with at least 50 redeemers |

PSI compares two distributions bin by bin. Zero means identical. The bins come from the training data, and missing values get their own bin.

The accuracy check reads the predictions already saved by `notebooks/06_final_evaluation.ipynb`. It does not score those campaigns again.

## What the checks found

### Inputs are stable across campaigns

Against the 13 training campaigns, the largest feature PSI is 0.024 for the validation campaigns and 0.040 for the later campaigns. The score PSI is 0.015 and 0.059. All are well under the 0.10 "watch" line. Targeted households in later campaigns look like targeted households in earlier ones.

### The model under-predicts redemption in later campaigns

| Campaign | Examples | Redeemers | Observed rate | Mean predicted | Gap | ROC-AUC | Flagged |
|---|---:|---:|---:|---:|---:|---:|---|
| 18 | 1,133 | 214 | 18.9% | 12.7% | +6.2 | 0.681 | Yes |
| 19 | 130 | 15 | 11.5% | 5.5% | +6.0 | 0.620 | Yes |
| 20 | 244 | 20 | 8.2% | 4.4% | +3.8 | 0.663 | No |
| 21 | 65 | 4 | 6.2% | 6.9% | -0.8 | not reported | No |
| 22 | 276 | 17 | 6.2% | 5.9% | +0.3 | 0.619 | No |
| 23 | 183 | 23 | 12.6% | 7.0% | +5.6 | 0.624 | Yes |
| 25 | 187 | 24 | 12.8% | 7.1% | +5.7 | 0.532 | Yes |

Four of the seven later campaigns are flagged, all in the same direction: households redeemed more often than the model predicted. Redemption rose over time (10.6% in the training campaigns, 14.3% in the later ones), and a model fitted on the earlier period carries the earlier rate.

Ranking held up better than the probabilities. Campaign 18, the only later campaign with enough redeemers to judge, has a ROC-AUC of 0.681 against the 0.702 reference, which is inside the 0.05 allowance. The small campaigns have 15 to 24 redeemers each, so their ROC-AUC figures are too noisy to act on.

**Recommendation from the rule: retrain.** The model still ranks households usefully, but its probabilities are too low for recent campaigns.

### A warning about who gets scored

When every household is scored for a planned campaign (`python -m src.predict`), the inputs look very different from the training data: PSI of 0.41 to 0.63 on baskets, sales, active days, and recency.

The cause is selection. The model was trained only on households the retailer chose to target, and those were heavier shoppers. Scores for all 2,500 households are therefore an extrapolation for the lighter shoppers. This does not trigger retraining, because retraining on the same kind of data would not fix it. The fix is data from a campaign sent to a random sample of households, which is what `docs/ab_test_design.md` proposes.

## Retraining procedure

1. Wait until the newest campaigns have ended and their redemptions are recorded.
2. Move the split forward in `src/config.py`, so training includes the recent campaigns and the newest ones are held out.
3. Run `python -m src.pipeline`. This logs new MLflow runs and registers a new model version.
4. Compare the new version (challenger) with the current `champion` on the newly held-out campaigns.
5. If the challenger is better, move the `champion` alias to it in the MLflow Model Registry. The service and the batch scorer load whatever `champion` points to after the next `python -m src.pipeline`.
6. Run `python -m src.monitor` again.

## What has not been done

The retraining itself has not been run. The later campaigns are this project's only held-out test set, and they have been used once. Training on them now would leave no unseen data to check the new model against. In a live system new campaigns keep arriving, so step 2 always has fresh data to hold out.

## Outputs

| File | Contents |
|---|---|
| `reports/monitoring/feature_drift.csv` | PSI for each feature and period |
| `reports/monitoring/score_drift.csv` | PSI of predicted probabilities and campaign-type mix |
| `reports/monitoring/accuracy_by_campaign.csv` | Observed and predicted redemption and ROC-AUC by campaign |
| `reports/monitoring/decision.json` | The recommendation, its reasons, warnings, and thresholds |

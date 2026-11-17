# Final model evaluation

## Model and task
Logistic regression predicts whether a targeted household will have at least one recorded coupon redemption during a campaign.

Inputs are six purchase-history features calculated before campaign start, plus campaign type.

## Evaluation design
- Training: 2,499 examples across 13 campaigns.
- Validation: 1,584 examples across 4 campaigns.
- Test: 2,218 examples across 7 later campaigns.
- Campaigns crossing split boundaries were excluded.
- Earlier groups’ outcome windows ended before the next evaluation period began.
- The saved model was evaluated without refitting.

Households may appear across groups. This evaluates future campaign response among existing households, not performance on entirely unseen households.

## Final test results

| Measure | Result |
|---|---:|
| Observed redemption rate | 14.29% |
| ROC-AUC | 0.6804 |
| Average precision | 0.2579 |
| Brier score | 0.1191 |
| Constant training-rate baseline Brier score | 0.1239 |

The model improves on the reference baselines, but test performance is lower than validation performance. Results support modest predictive value.

## Limitations
This is one chronological test period containing seven campaigns. Results do not establish statistical significance, profitability, or causal campaign effects.

The day-704 campaign eligibility cutoff was a conservative modeling choice, not a verified redemption collection boundary.

## Saved evidence
Final metrics and individual predictions are stored in `reports/model_evaluation`. The selected pipeline is `models/logistic_with_campaign_type.joblib`.
## Targeting view of the test results

`notebooks/06b_test_lift_table.ipynb` ranks the saved test predictions from highest to lowest probability. Nothing was refitted.

| Share contacted, highest scores first | Redemption rate | Lift vs overall | Share of redeemers reached |
|---|---:|---:|---:|
| Top 10% | 33.78% | 2.36 | 23.66% |
| Top 20% | 27.48% | 1.92 | 38.49% |
| Top 30% | 24.51% | 1.72 | 51.42% |
| Top 50% | 20.20% | 1.41 | 70.66% |
| All | 14.29% | 1.00 | 100.00% |

Within single campaigns, the top 20% of scores had a higher redemption rate than the campaign average in six of the seven test campaigns (lift 1.25 to 2.07). Campaign 25 was below average (lift 0.63). Campaign 18 supplies about half of the test examples; the other campaigns have 4 to 24 redeemers each, so their figures are noisy.

This view describes who redeems a coupon. It does not show who buys more because of a campaign.

Files: `reports/model_evaluation/test_lift_table.csv`, `test_lift_by_campaign.csv`, and `test_cumulative_gains.png`.

# Model selection decision

## Selected candidate
Logistic regression with six numerical purchase-history features and campaign type.

Saved pipeline: `models/logistic_with_campaign_type.joblib`

## Validation evidence

| Model | ROC-AUC ↑ | Average precision ↑ | Brier score ↓ |
|---|---:|---:|---:|
| Numerical logistic baseline | 0.6662 | 0.2145 | 0.1186 |
| Logistic regression with campaign type | 0.7409 | 0.2832 | 0.1138 |
| XGBoost with campaign type | 0.7059 | 0.2601 | 0.1149 |

The selected candidate performed best on all three validation metrics. These results cover four validation campaigns and do not establish statistical superiority or guarantee future performance.

## Final evaluation plan
Evaluate the saved logistic regression pipeline on the held-out test campaigns without refitting or changing its features, preprocessing, or coefficients.

Report ROC-AUC, average precision, and Brier score. Compare probability accuracy against a constant prediction using the training redemption rate.

Do not use test results to choose between these candidates. Any subsequent model changes require a new evaluation plan.

## Scope
The model predicts recorded coupon redemption among targeted households. It does not estimate incremental purchases caused by a campaign.
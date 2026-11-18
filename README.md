# Customer Loyalty and Marketing Optimization

![tests](https://github.com/FaridoonFarahi/customer-loyalty-platform/actions/workflows/ci.yml/badge.svg)

A portfolio project on retail marketing data. It asks five questions a marketing team would ask:

1. Which households are likely to redeem a campaign coupon?
2. What kinds of customers do we have?
3. Did a campaign change what households spent?
4. How should the next campaign be tested so the answer can be trusted?
5. Who should receive a campaign, if the goal is extra sales and not coupon use?

Data: dunnhumby's public dataset **The Complete Journey** (2,500 households, 2,595,732 transaction rows, 30 marketing campaigns, days 1 to 711).

## Results at a glance

| Question | Method | Result |
|---|---|---|
| Who redeems? | Logistic regression vs XGBoost, trained on earlier campaigns and tested on later ones | Test ROC-AUC 0.680. The top 10% of scores redeem at 33.8% against a 14.3% average (lift 2.36). |
| What kinds of customers? | K-means on recency, active days, and sales | 1,430 higher-activity, 896 lower-activity, and 174 inactive households |
| Did campaign 18 change spending? | Difference in changes, with overlap weighting | Positive association (+14.92 unweighted, +47.46 weighted per household), **not** a causal estimate |
| How to test the next campaign? | Power analysis and 2,000 simulated trials | A plain comparison needs 2,252 households per group. Adjusting for baseline sales cuts that to 609. |
| Who should get the campaign? | Uplift T-learner on simulated data with a known effect | Ridge plus shallow trees cut the effect error from 13.93 to 6.36 on held-out test households. |

## 1. Coupon redemption model

**Question.** For a household targeted by a campaign, what is the probability of at least one recorded coupon redemption during that campaign?

**Data.** 7,091 household-campaign pairs across 28 campaigns; 880 (12.4%) redeemed.

**Leakage control.** Every feature uses transactions from the 90 days before the campaign starts. The split is by time: 13 earlier campaigns for training, 4 for validation, 7 later campaigns for the final test. Campaigns that cross a split boundary are excluded.

**Validation results (used to choose the model).**

| Model | ROC-AUC | Average precision | Brier score |
|---|---:|---:|---:|
| Logistic regression, numeric features | 0.666 | 0.215 | 0.1186 |
| Logistic regression with campaign type | 0.741 | 0.283 | 0.1138 |
| XGBoost with campaign type | 0.706 | 0.260 | 0.1149 |

Logistic regression was selected. With six features and about 2,500 training rows, the simpler model generalized better than boosted trees.

**Final test (run once, on the saved model).** ROC-AUC 0.680, average precision 0.258, Brier score 0.1191 against 0.1239 for a constant prediction.

| Share contacted, highest scores first | Redemption rate | Share of redeemers reached |
|---|---:|---:|
| Top 10% | 33.8% | 23.7% |
| Top 20% | 27.5% | 38.5% |
| Top 50% | 20.2% | 70.7% |
| All | 14.3% | 100% |

The model predicts who redeems. It does not show who buys more because of a campaign.

Details: `docs/prediction_target.md`, `docs/model_selection.md`, `docs/model_evaluation.md`.

## 2. Customer segmentation

K-means on three 90-day features (days since last purchase, active days, total sales), log-transformed and standardized. Two clusters had the highest silhouette score (0.450). Households with no activity in the window are kept as a separate group.

| Segment | Households | Median recency | Median active days | Median sales |
|---|---:|---:|---:|---:|
| Higher recent activity | 1,430 | 3 days | 16 | 585.08 |
| Lower recent activity | 896 | 20.5 days | 4 | 102.78 |
| No recorded activity in last 90 days | 174 | n/a | 0 | 0 |

Details: `docs/customer_segmentation.md`.

## 3. Campaign 18: observational analysis

Campaign 18 reached 1,133 households. The comparison group is 1,365 households it did not target. The outcome is the change in sales between the 56 days before and the 56 days during the campaign.

| Comparison | Difference in mean changes | 95% bootstrap interval |
|---|---:|---:|
| Unweighted | 14.92 | [0.74, 29.25] |
| Overlap-weighted | 47.46 | [30.95, 64.43] |

Targeted households were much heavier shoppers before the campaign. Overlap weighting balances pre-campaign sales and active days, but 45.0% of the weighted targeted group was also in other overlapping campaigns, against 16.2% of the comparison group. For that reason these figures are reported as associations, not as the effect of campaign 18.

Details: `docs/campaign_18_observational_analysis.md`.

## 4. A/B test design

Because the historical data cannot separate the campaign from who was chosen for it, the project designs the randomized test that would.

- **Design.** Randomize households 1:1, measure sales per household over 56 days, analyze by assigned group (intention to treat).
- **Power.** Detecting a 10% increase (27.43 per household) at 80% power needs 2,252 households per group. The cohort has 1,249 per group, which gives 55% power.
- **Check by simulation.** Across 2,000 simulated trials the effect was detected in 53.9% of runs, in line with the calculation.
- **Baseline adjustment.** Sales in the previous 56 days correlate 0.855 with sales in the next 56. Adjusting for baseline (CUPED) raised simulated power to 97.8% and cut the required sample to 609 per group. False-positive rates stayed near 5%.

No real randomized campaign was run. All experiment results are simulations and are labeled as such.

Details: `docs/ab_test_design.md`.

## 5. Uplift modeling (simulated)

A redemption model predicts who uses a coupon. An uplift model estimates how much more a household buys **because** it received the campaign. The historical data has no randomized campaign, so this step uses a simulation with a known effect to test whether the method can recover it.

- **Simulation.** 2,498 households with real baseline sales, random 1:1 assignment, and a known effect that depends on baseline activity: 10, 40, or 20 per household.
- **Method.** A T-learner: one sales model for holdout households, one for campaign households, and the estimated effect is the difference.

Validation results (effect RMSE is the typical gap between the estimated and the known effect):

| Model | Effect RMSE |
|---|---:|
| Constant difference | 14.22 |
| Ridge T-learner | 13.24 |
| Ridge-plus-trees T-learner | 5.71 |

Ridge is a straight-line model and missed a pattern that rises and then falls. Adding shallow trees recovered it: mean estimates of 11.9, 41.1, and 18.5 against known effects of 10, 40, and 20.

If the campaign went to the 40% of validation households with the highest estimated effect, the average known effect among them would be 39.6. Contacting the biggest spenders gives 30.8, and no targeting gives 23.6.

**Final test (run once, on 500 held-out households).** Effect RMSE 6.36 against 13.93 for the constant baseline. Mean estimates by activity band were 12.9, 40.0, and 17.4 against known effects of 10, 40, and 20. Contacting the top 40% by estimated effect gives an average known effect of 38.6, against 29.8 for the biggest spenders, 23.0 with no targeting, and 38.8 for the best possible choice.

The effect pattern was written by hand, so this shows the method works on a pattern of this kind. It says nothing about how real households respond. Details: `docs/uplift_modeling.md`.

## 6. Runnable pipeline, tests, and model registry

The redemption model also exists as code that runs from the raw files with one command. It uses the same features, split, and settings as the notebooks, and it does not choose a model.

```bash
python -m src.pipeline      # raw CSV files -> features -> split -> train -> validate -> MLflow -> saved model
python -m pytest            # 43 tests
python -m src.predict --start-day 712 --campaign-type TypeA --top-share 0.2
```

| Step | What it does | Code |
|---|---|---|
| Features | One query builds every feature from the 90 days before each campaign. Training and scoring use the same query. | `src/features.py` |
| Split and train | Split by time, fit the three candidates, score them on validation campaigns | `src/train.py` |
| Tracking and registry | One MLflow run per candidate. The selected model is registered and marked `champion`. | `src/tracking.py` |
| Batch scoring | Scores every household for a planned campaign and writes a ranked contact list | `src/predict.py` |
| Orchestration | Runs the steps in order and saves the model with its metadata | `src/pipeline.py` |

**The pipeline reproduces the notebooks.** Its modeling table equals the notebook table row for row (7,091 rows), and the logistic regression validation metrics match to six decimal places. Both checks are tests.

**Tests.** Hand-made data with known answers checks the feature window, the label window, and the split. Two leakage tests add or delete purchases on and after a campaign's start day and require that no feature changes. Tests that need the real data are skipped when `data/raw/` is absent.

**The pipeline stops at validation.** The held-out test campaigns were scored once, in notebook 06, and the pipeline does not score them again.

## 7. Deployment: scoring service, container, and app

| Form | Command | What it is |
|---|---|---|
| Batch | `python -m src.predict --start-day 712 --campaign-type TypeA --top-share 0.2` | Scores all 2,500 households for a planned campaign and writes a ranked contact list |
| Online | `uvicorn src.api:app` | A FastAPI service. `POST /predict` returns redemption probabilities, `GET /health` reports the loaded model. Interactive docs at `/docs`. |
| Container | `docker build -t loyalty-scoring .` then `docker run -p 8000:8000 loyalty-scoring` | The same service with only the eight packages it needs (`requirements-serve.txt`) |
| App | `streamlit run streamlit_app.py` | Seven pages: results, a targeting slider, live scoring of a household, an experiment planner, campaign effect and uplift, model details, and monitoring |

The service checks its input. It refuses unknown campaign types, negative counts, more active days than the 90-day window holds, and histories that contradict themselves. One test confirms that the service and the batch scorer return the same probability for the same household.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"households": [{"household_key": 1, "campaign_type": "TypeA", "baskets_90d": 30, "retailer_sales_90d": 900, "active_days_90d": 25, "recency_days_90d": 2}]}'
```

The app trains nothing. It reads the saved tables in `reports/` and the saved model.

## 8. Monitoring, retraining, and continuous integration

```bash
python -m src.monitor
```

| Check | Rule for action | Result |
|---|---|---|
| Input drift against the training campaigns (PSI) | 0.25 or more on any feature | Stable. Largest value 0.040. |
| Predicted against observed redemption, by campaign | A gap over 5 points and over two standard errors | Flagged in 4 of 7 later campaigns, always under-predicting (campaign 18: 18.9% observed, 12.7% predicted) |
| Ranking inside a campaign (ROC-AUC) | More than 0.05 below the validation level of 0.702 | Campaign 18 holds at 0.681 |

**The rule says retrain.** The model still ranks households usefully, but redemption rose over time and its probabilities are now too low. The retraining steps are written down in `docs/monitoring_and_retraining.md`. They have not been run, because the later campaigns are the only held-out data and training on them would leave nothing to check a new model against.

**A selection warning.** The model was trained only on households the retailer chose to target, who were heavier shoppers. Scoring all 2,500 households is an extrapolation for lighter shoppers (PSI of 0.41 to 0.63 on four features). A campaign sent to a random sample would fix this, which is what the A/B test design proposes.

**Continuous integration.** GitHub Actions runs the style check and the tests on every push (`.github/workflows/ci.yml`). The raw data is not in the repository, so the three tests that need it are skipped there and the other 40 run.

## Limits

- The redemption model uses six purchase-history features. It has no prior-redemption or product-level features yet.
- The campaign analysis is observational. Targeting was not random, and campaigns overlap.
- Demographics exist for 801 of 2,500 households and are not used in the models.
- Sales are the amount received by the retailer, not profit. Campaign costs and margins are not in the data.

## Project layout

| Path | Contents |
|---|---|
| `notebooks/` | Numbered notebooks, in the order they were run |
| `docs/` | Written decisions and results for each step |
| `reports/` | Saved tables and charts |
| `models/` | Fitted pipelines and the modeling configuration |
| `ml_pipeline/` | Test-set guard, plus a ledger and figures for the uplift and redemption pipelines |
| `src/` | The redemption-model pipeline (features, training, MLflow, scoring, service, monitoring) and the uplift final-test script |
| `tests/` | Feature, leakage, split, scoring, service, app, and monitoring tests |
| `streamlit_app.py` | The app |
| `Dockerfile`, `requirements-serve.txt` | Container for the scoring service |
| `.github/workflows/ci.yml`, `requirements-ci.txt` | Tests on every push |
| `data/` | Not in the repository (see below) |

| Notebook | Step |
|---|---|
| `01_data_understanding` | Profile and clean the eight source files |
| `02_sql_warehouse` | DuckDB warehouse and campaign analysis tables |
| `03_campaign_modeling_table` | One row per household-campaign with pre-campaign features |
| `04_baseline_model` | Time-based split and logistic regression |
| `05_xgboost_model` | XGBoost candidate |
| `06_final_evaluation`, `06b_test_lift_table` | Single test evaluation and the targeting view |
| `07_experiment_tracking` | MLflow runs |
| `08_customer_segmentation` | K-means segments |
| `09_campaign_effectiveness` | Campaign 18 observational analysis |
| `10_ab_test_design`, `10b_ab_test_simulation_study` | Power analysis and simulated trials |
| `11_uplift_modeling` | Uplift T-learner on simulated campaign data |

## How to run

1. Download **The Complete Journey** from [dunnhumby Source Files](https://www.dunnhumby.com/source-files/) and place the CSV files in `data/raw/`.
2. Create an environment and install the packages:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Run the notebooks in number order, or run the redemption model alone with `python -m src.pipeline`.

## Tools

Python, DuckDB (SQL), pandas, scikit-learn, XGBoost, MLflow (tracking and Model Registry), FastAPI, Docker, Streamlit, GitHub Actions, pytest, ruff, SciPy, statsmodels, matplotlib.

## Data source

dunnhumby, *The Complete Journey*. The data and its user guide belong to dunnhumby and are not redistributed here.

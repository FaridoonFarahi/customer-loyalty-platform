# Pipeline ledger: coupon redemption model

The redemption model was built in notebooks 01 to 07 before the pipeline guard was added to this project, so no approvals were recorded at the time. This ledger records what was done and the owner's decision to turn that work into scripts.

## Status

- Step 1 data inspection: done in `notebooks/01_data_understanding.ipynb`. Eight source files profiled; row counts match the published figures.
- Step 2 EDA: done in notebook 01. Figures 01 and 02 below profile the modeling table.
- Step 3 prediction problem: done. `docs/prediction_target.md`. Probability that a targeted household redeems at least one coupon during the campaign.
- Step 4 cleaning: done. `docs/data_quality_notes.md`. Figure 04 below.
- Step 5 data engineering: done in notebooks 02 and 03. One row per household-campaign pair, 7,091 rows.
- Step 6 split: done in notebook 04. By time: 13 campaigns train, 4 validation, 7 test; campaigns that cross a boundary are excluded.
- Step 7 feature engineering: done in notebook 03. Six purchase-history features from the 90 days before campaign start.
- Step 8 preprocessing: done in notebook 04. Median imputation, scaling, and one-hot campaign type inside one scikit-learn pipeline.
- Step 9 baseline: done. Constant training-rate prediction and a numeric-only logistic regression.
- Step 10 model training: done in notebooks 04 and 05. Logistic regression with campaign type; XGBoost with campaign type.
- Step 11 hyperparameter tuning: not done. Fixed settings.
- Step 12 model evaluation: done on validation campaigns. `docs/model_selection.md`. Logistic regression with campaign type selected. Figure 12 below.
- Step 13 error analysis: by-campaign review in notebook 04 and `notebooks/06b_test_lift_table.ipynb`.
- Step 14 (final test) was run once in `notebooks/06_final_evaluation.ipynb`: ROC-AUC 0.6804, average precision 0.2579, Brier 0.1191.
- Step 15 deployment: in progress from 2026-11-17 (`src/` pipeline, scoring service, app).
- Step 16 monitoring and retraining plan: in progress from 2026-11-17.

## Figures

- Figure 01_missingness.png: 2 of 11 columns have missing values; worst is recency_days_90d at 0.4%.
- Figure 02_target_balance.png: redeemed=0 is 87.6% of 7,091 rows, so a majority-class dummy scores 87.6%; the minority share is 12.4%.
- Figure 02_distributions.png: Histograms of 9 numeric features; 6 are strongly skewed (baskets_90d, retailer_sales_90d, recency_days_90d, has_transactions_90d, avg_retailer_sales_per_basket_90d), which matters for scaling and outliers.
- Figure 02_correlations.png: Strongest correlation with redeemed: retailer_sales_90d (0.13); anything above 0.95 is a leakage suspect.
- Figure 02_temporal_coverage.png: Examples per campaign by start day, from day 224 to day 659, with the split boundaries marked. Campaigns differ a lot in size, and later campaigns must not inform earlier ones, so the split is by time.
- Figure 04_rows_kept_and_dropped.png: Counts of household-campaign pairs before and after the preparation rules: 7,208 targeted pairs, 7,091 kept because their campaign ends by day 704, 117 dropped because the outcome window is not fully observed, and 27 kept rows with no purchases in the 90-day window, which get a flag and median imputation later.
- Figure 12_validation_curves.png: ROC and precision-recall curves for the three candidates on the four validation campaigns, with the no-skill lines dashed. The selected logistic regression with campaign type has ROC-AUC 0.741 and average precision 0.283, against a 14.1% redemption rate; all candidates beat the no-skill lines.

Model rationale:
- traits: binary, 7,091 rows, minority 12.4%, temporal (campaign start day), repeated households
- baseline: constant training-rate prediction + numeric-only logistic regression
- candidates: logistic regression with campaign type; XGBoost with shallow trees
- ruled out: neural nets (too little data); random split (campaigns are ordered in time); accuracy as the metric (12.4% positives)
- metric: ROC-AUC and average precision for ranking; Brier score for probability accuracy

## Owner decisions

On 2026-11-17 the owner approved turning the training steps of notebooks 03 to 06 into scripts with the same features, the same settings, and no new model selection.

- Override: Gate A - user approved fitting the existing preprocessing in scripts 2026-11-17 - reason: steps 1 to 8 were completed in notebooks 01 to 04 before the guard was adopted
- Override: Gate B - user approved training the existing models in scripts 2026-11-17 - reason: same features and fixed settings as notebooks 04 and 05; no new candidates and no new selection

The pipeline in `src/` stops at validation. The held-out test campaigns were scored once in notebook 06, and the pipeline does not score them again.

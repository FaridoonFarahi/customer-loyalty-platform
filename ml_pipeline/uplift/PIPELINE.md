- Figure 01_missingness.png: No column has missing values.
- Figure 02_target_balance.png: simulated_sales_56d ranges 1.04 to 3.54e+03 with median 210; a mean-predictor baseline is the number to beat.
- Figure 02_distributions.png: Histograms of 4 numeric features; 2 are strongly skewed (sales_before, active_days_before), which matters for scaling and outliers.
- Figure 02_correlations.png: Strongest correlation with simulated_sales_56d: sales_before (0.99); anything above 0.95 is a leakage suspect.
- Figure 03_ridge_validation_effects.png: Compares Ridge effect estimates with known simulated effects across baseline activity levels, using validation households.

## Pipeline status (uplift on simulated data)

Steps 1 to 11 were run by the project owner in `notebooks/11_uplift_modeling.ipynb`. Gates A and B were not recorded at the time.

- Step 1 data inspection: done. 2,498 households, no missing values, no duplicates.
- Step 2 EDA: done. Figures 01 and 02 above.
- Step 3 prediction problem: done. Estimate the simulated campaign effect on 56-day sales per household; metric is RMSE against the known simulated effect.
- Step 4 cleaning: done. No rules needed.
- Step 5 data engineering: done. One row per household with baseline sales, baseline active days, simulated assignment, and simulated sales.
- Step 6 split: done. Group split by household, 1,498 train / 500 validation / 500 test, seed 2027. Test set frozen.
- Step 7 feature engineering: done. Two features: baseline sales and baseline active days.
- Step 8 preprocessing: done. Standard scaling inside each model pipeline, fitted on training rows.
- Step 9 baseline: done. Constant training-group difference, validation effect RMSE 14.22.
- Step 10 model training: done. Ridge T-learner 13.24; Ridge-plus-trees T-learner 5.71 (validation effect RMSE).
- Step 11 hyperparameter tuning: not done. Fixed settings; one tree-based candidate.
- Step 12 model evaluation: done. Validation effect by band 11.89 / 41.06 / 18.50 against known 10 / 40 / 20; targeting the top 40% gives 39.6 against 23.6 with no targeting.
- Step 13 error analysis: done. Validation RMSE 7.89 near the jump at 15 days against 5.59 away from both jumps; estimates run low for the few households with 30 or more active days.
- Step 14 (final test): done once on 2026-11-17 with the Ridge-plus-trees T-learner. Effect RMSE 6.36 against 13.93 for the constant baseline. The logged line is under Gate approvals.
- Step 15 deployment: not requested.
- Step 16 monitoring and retraining plan: written in docs/uplift_modeling.md (what a real deployment would need). Nothing is deployed.
- Figure 12_boosted_validation_effects.png: Compares Ridge-plus-trees effect estimates with known simulated effects across baseline activity levels, using validation households. It shows whether the model follows the step pattern that Ridge missed.
- Figure 13_boosted_validation_errors.png: Shows Ridge-plus-trees effect errors by baseline active days on validation households, with dashed lines where the simulated effect jumps. It shows whether errors concentrate next to the jumps.

## Gate approvals

Gates A and B were not recorded when steps 1 to 11 were run.

- Gate C: approved 2026-11-17
- Step 14 final test: effect_rmse=6.3579 (touch 1, 2026-11-17)

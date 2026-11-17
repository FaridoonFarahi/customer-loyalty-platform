# Proposed randomized campaign experiment

## Objective

Estimate how assigning a marketing campaign changes recorded retailer sales per household.

This document describes a proposed experiment and a simulated demonstration. **No real randomized campaign experiment was conducted.** The historical data does not establish randomized assignment to campaign 18.

## Experiment design

| Component | Definition |
|---|---|
| Randomization unit | Household |
| Eligibility | Defined and fixed before randomization |
| Allocation | Randomly assign eligible households 1:1 |
| Treatment group | Assigned to receive the proposed campaign |
| Comparison group | Not assigned to receive that campaign; otherwise follows the same business-as-usual marketing rules |
| Measurement period | 56 days from campaign start |
| Primary outcome | Total recorded retailer sales per household during those 56 days |
| Baseline period | The preceding 56 days, measured before assignment |

The primary outcome measures **retailer sales, not profit**. Campaign costs and margins would be needed for a profitability assessment.

## Analysis principle

Compare households according to their assigned group, regardless of whether they redeem a coupon. This is an **intention-to-treat analysis**.

Estimate the difference in mean retailer sales between the campaign and holdout groups. Report a two-sided Welch t-test and a 95% confidence interval after the measurement period is complete.

Include households with verified zero purchases. Do not automatically interpret missing outcome measurement as zero.

Baseline sales may support a separately prespecified precision adjustment. The calculations and simulation reported below do not assume a benefit from such adjustment.

## Power-analysis assumptions

Historical planning data came from 2,498 households with recorded transactions before day 531. Their sales during days 531–586 provided the planning mean and standard deviation.

| Assumption | Value |
|---|---:|
| Outcome window | 56 days |
| Historical mean retailer sales | 274.30 |
| Assumed common standard deviation | 328.46 |
| Illustrative detectable increase | 10% |
| Corresponding absolute difference | 27.43 |
| Two-sided significance level | 5% |
| Desired power | 80% |
| Allocation ratio | 1:1 |

The 10% target is a **tutorial planning assumption**, not a business-approved profitability threshold or a forecast of campaign impact.

Power was calculated using an independent two-sample t-test model, assuming independent households, a common outcome variance, and complete outcome measurement. Historical variability is a planning approximation.

## Power-analysis findings

| Measure | Result |
|---|---:|
| Required households per group | 2,252 |
| Required households in total | 4,504 |
| Households available in the planning cohort | 2,498 |
| Available households per group | 1,249 |
| Power to detect the planned 10% difference with the available sample | 55.02% |
| Detectable absolute difference at 80% power with the available sample | 36.84 |
| Corresponding detectable relative difference | 13.43% |

The available cohort **does not meet the original 80% power target for detecting a 10% difference**.

The 13.43% result describes the available design’s sensitivity; it does not replace the original planning target. A prospective experiment would need to resolve this sample-size shortfall before claiming to meet that target.

Reported values are rounded for readability. Calculations used unrounded values.

## Simulated demonstration

We created a reproducible random assignment of 1,249 households to each group using seed **42**.

Using outcome seed **2026**, we generated synthetic sales by resampling historical baseline sales independently for each household. We then added the planned absolute difference—approximately **27.43**—to campaign-assigned households.

In this simplified simulation:

- The imposed effect is identical for every campaign-assigned household.
- Synthetic outcomes are independent of each household’s own baseline sales.
- Assignment and outcome records are explicitly labeled as simulated.
- Neither assignments nor outcomes represent a real randomized campaign.

### Simulation results

| Measure | Result |
|---|---:|
| Imposed effect | 27.43 |
| Estimated difference in mean sales | 6.88 |
| 95% Welch confidence interval | [−19.03, 32.80] |
| Two-sided p-value | 0.6025 |

The result is **inconclusive at the 5% significance level**. It does not establish zero effect: the interval includes both zero and the known imposed effect.

The difference between the imposed and estimated effects illustrates sampling variability. We retained the original assignment and outcome seeds rather than searching for a statistically significant result.

## Saved evidence

All paths below are relative to the project folder.

| File | Contents |
|---|---|
| `reports/ab_test_design/power_analysis.csv` | Planning assumptions and sample-size calculations |
| `reports/ab_test_design/simulated_assignment_balance.csv` | Baseline summaries by simulated group |
| `reports/ab_test_design/simulated_trial_results.csv` | Simulated effect estimate, confidence interval, and p-value |
| `data/simulated/simulated_campaign_assignment.parquet` | Simulated household assignments |
| `data/simulated/simulated_campaign_trial.parquet` | Simulated assignments and outcomes |
| `notebooks/10_ab_test_design.ipynb` | Reproducible planning and simulation code |
## Simulation study: repeated trials and baseline adjustment

The single simulated trial above can land anywhere, so it cannot show whether the power calculation is right. `notebooks/10b_ab_test_simulation_study.ipynb` repeats the trial 2,000 times. All results in this section are simulated.

### Does the power calculation hold?

Using the same recipe as the single trial (outcomes drawn at random from historical sales, planned effect of 27.43 added to the campaign group, 1,249 households per group):

| Measure | Calculated | Simulated (2,000 trials) |
|---|---:|---:|
| Share of trials that detect the planned effect (power) | 55.02% | 53.90% |
| Share of trials that wrongly detect an effect when none exists | 5.00% | 6.15% |
| Share of 95% intervals that contain the true effect | 95.00% | 95.20% |
| Mean estimated effect | 27.43 | 26.91 |

Simulated power is close to the calculated value. The single trial reported above (estimate 6.88, p = 0.60) is one of the roughly 46% of runs that miss an effect of this size. The false-positive rate with no effect came out slightly above 5% in this recipe; with 2,000 trials the sampling error on that figure is about half a percentage point.

### How much does baseline adjustment help?

Each household's recorded sales in days 475–530 predict its sales in days 531–586 with a correlation of 0.855. In this part each household keeps its own real baseline and outcome sales, and only the group assignment is random. The baseline-adjusted analysis (CUPED) removes the part of each outcome that the baseline already predicts, then applies the same Welch test.

| Analysis | Power (planned effect) | False positives (no effect) | Interval coverage | Mean interval half-width |
|---|---:|---:|---:|---:|
| Welch t-test | 54.60% | 4.40% | 95.00% | 25.77 |
| Baseline-adjusted (CUPED) | 97.80% | 5.05% | 95.10% | 13.38 |

### Sample size with baseline adjustment

| Measure | Unadjusted | Baseline-adjusted |
|---|---:|---:|
| Outcome standard deviation used | 328.46 | 170.58 |
| Households required per group for 80% power | 2,252 | 609 |
| Calculated power with 1,249 per group | 55.02% | 98.02% |

With baseline adjustment, the available cohort of 1,249 households per group would meet the 80% power target for a 10% difference.

### Limits of this study

- The imposed effect is the same for every campaign household.
- The 0.855 correlation comes from one pair of historical windows under business-as-usual marketing. A less predictable period would shrink the gain.
- In a real experiment the baseline adjustment must be written into the analysis plan before outcomes are seen.

### Added evidence

| File | Contents |
|---|---|
| `reports/ab_test_design/simulation_study.csv` | Power, false-positive rate, and interval coverage across 2,000 simulated trials |
| `reports/ab_test_design/baseline_adjustment_sample_size.csv` | Sample-size and power figures with and without baseline adjustment |
| `reports/ab_test_design/simulation_study_estimates.png` | Distribution of estimated effects for both analyses |
| `notebooks/10b_ab_test_simulation_study.ipynb` | Reproducible simulation code |

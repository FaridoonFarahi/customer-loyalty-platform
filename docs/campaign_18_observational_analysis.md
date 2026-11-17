# Campaign 18: observational purchasing analysis

## Design
Campaign 18 was TypeA and ran during days 587–642.

We compared two equal 56-day periods:
- Before: days 531–586.
- During: days 587–642.

The study included households with a recorded transaction before day 531: 1,133 targeted households and 1,365 households not targeted by campaign 18.

The outcome was the change in recorded retailer sales per household. These were observational groups, not randomized assignments.

## Results

| Comparison | Difference in mean changes | 95% household-bootstrap interval |
|---|---:|---:|
| Unweighted | 14.92 | [0.74, 29.25] |
| Overlap-weighted | 47.46 | [30.95, 64.43] |

The weighted result emphasizes a different population: households with greater overlap in measured baseline characteristics. It is not an estimate for all targeted households.

## Adjustment
We estimated targeting propensity using log-transformed pre-campaign sales and active days. Targeted households received weight `1 − propensity`; comparison households received weight `propensity`.

Standardized mean differences improved:
- Pre-campaign sales: 0.987 → 0.089.
- Pre-campaign active days: 1.118 → 0.017.

Effective sample sizes were approximately 854 targeted and 833 comparison households.

## Remaining limitations
Other overlapping campaigns reached 44.97% of the weighted targeted group versus 16.23% of the weighted comparison group.

Weighting does not resolve this difference or unmeasured targeting factors. Eight weekly pre-campaign averages do not establish parallel trends.

Both intervals used 2,000 household bootstrap samples. The weighted bootstrap refitted the targeting model in every sample. These intervals assume independent households and do not account for shared campaign shocks or remove confounding.

## Conclusion
Both comparisons show positive observational associations. Neither establishes incremental sales caused specifically by campaign 18 or campaign profitability.

A randomized holdout experiment would provide a stronger basis for evaluating incremental marketing impact.
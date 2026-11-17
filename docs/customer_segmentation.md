# Customer segmentation

## Snapshot
Snapshot day: 712  
Purchase-history window: days 622–711, inclusive.

All 2,500 households are retained. Of these, 2,326 had recorded transactions during the window; 174 did not.

## Method
K-means uses three features for households with recent activity:
- Days since the latest recorded transaction.
- Number of distinct days with recorded transactions.
- Total recorded retailer sales.

Features were transformed with `log1p` and standardized. We compared 2–6 clusters using silhouette scores, inertia, and cluster sizes.

The two-cluster solution had the highest silhouette score: 0.4504. We selected it after inspecting its behavioral profiles.

## Segments

| Segment | Households | Median recency | Median active days | Median retailer sales |
|---|---:|---:|---:|---:|
| Higher recent activity | 1,430 | 3 days | 16 | 585.08 |
| Lower recent activity | 896 | 20.5 days | 4 | 102.78 |
| No recorded activity in last 90 days | 174 | Not available within window | 0 | 0 |

The no-activity group is assigned separately, not fitted by K-means.

## Interpretation
These are descriptive groups for one historical snapshot. They do not establish loyalty, churn, or responsiveness to marketing.

Cluster numbers are arbitrary; the fitted model’s name mapping is saved. These end-of-data segments must not be used as inputs for predictions made before this snapshot.

## Outputs
Assignments: `data/processed/household_segments.parquet`  
Model and settings: `models/customer_segmentation.joblib`  
Chart and supporting tables: `reports/segmentation/`
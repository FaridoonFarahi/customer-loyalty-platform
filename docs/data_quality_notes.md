# Data quality notes

## Source preservation
The eight original CSV files remain unchanged in `data/raw`. Derived datasets are saved as Parquet files in `data/processed`. Inspection and preparation code is recorded in `notebooks/01_data_understanding.ipynb`.

## Cleaning decisions
- **Coupons:** Removed 5,164 exact repeated coupon–product–campaign combinations, leaving 119,384 eligibility records.
- **Products:** Converted blank text fields and the inspected “no description” placeholders to missing values. Retained all 92,353 products.
- **Transactions:** Retained all 2,595,732 rows, including zero-quantity and zero-sales records. Avoid per-unit calculations where quantity is zero.
- **Promotions:** Combined records into one row per product–store–week while preserving distinct display and mailer codes. Flagged 15,208 combinations containing both “not on display” and a nonzero display code. This aggregation is our analysis rule; the guide does not explain these conflicting records.

## Coverage and interpretation
- Transactions span days 1–711. Campaign 24 ends on day 719, so its final eight days fall outside transaction coverage.
- Promotion records span weeks 9–101; transactions span weeks 1–102. Missing promotion matches must not automatically mean “no promotion.”
- Demographics cover 801 households. Of 1,584 targeted households, 824 lack demographic records; retain these households in the analysis.
- Preserve coded demographic labels and the combined `None/Unknown` category.
- `SALES_VALUE` represents the amount received by the retailer, not necessarily the amount paid by the customer, according to page 3 of the user guide.
- TypeA households received 16 personalized coupons from a larger pool; their exact assigned coupons are unavailable. TypeB and TypeC households received all campaign coupons, according to page 5.
- Household redemption percentages describe observed participation. They do not establish causal campaign effects or profitability.

## Saved outputs
Nine Parquet files were saved. Their metadata row counts match the corresponding notebook results.
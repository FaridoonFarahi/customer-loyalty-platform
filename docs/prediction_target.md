# Campaign redemption prediction

## Prediction question
For a household targeted by a campaign, what is the probability of at least one recorded coupon redemption during that campaign?

## Unit of analysis
One row per household–campaign pair. A household may appear in multiple campaigns.

## Target
- **1:** At least one matching redemption is recorded between the campaign’s start and end days, inclusive.
- **0:** No matching redemption is recorded during that window.

A zero means no recorded coupon redemption. It does not mean the household made no purchases.

## Prediction time
Immediately before the campaign starts.

Purchase-history features must use transactions where `DAY < START_DAY`. Purchases and redemptions during the campaign cannot be used as input features.

## Evaluation requirements
Check observation-window coverage before selecting eligible campaigns. Evaluate on later campaigns rather than randomly mixing past and future records.

## Interpretation
This model estimates redemption propensity among targeted households. It does not estimate whether sending a campaign causes additional purchases.

## Models
Compare a logistic regression baseline with XGBoost.
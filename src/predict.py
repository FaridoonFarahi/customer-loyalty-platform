"""Load the redemption model and score households.

Batch scoring for a planned campaign:

    python -m src.predict --start-day 712 --campaign-type TypeA --top-share 0.2

This builds each household's features from the 90 days before ``--start-day`` and writes a
ranked list. It is the batch form of deployment; ``src/api.py`` is the online form.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config, features


def load_model(source: str = "file"):
    """``file`` reads models/redemption_model.joblib; ``registry`` reads the MLflow champion."""
    if source == "file":
        if not config.MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"{config.MODEL_PATH} not found. Run `python -m src.pipeline` first."
            )
        return joblib.load(config.MODEL_PATH)
    if source == "registry":
        import mlflow.sklearn

        from src import tracking

        tracking.setup_mlflow()
        return mlflow.sklearn.load_model(
            f"models:/{config.REGISTERED_MODEL_NAME}@{config.CHAMPION_ALIAS}"
        )
    raise ValueError("source must be 'file' or 'registry'")


def score(model, frame: pd.DataFrame) -> np.ndarray:
    """Redemption probability for each row. Rows need the columns in config.MODEL_FEATURES."""
    missing = [column for column in config.MODEL_FEATURES if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing model inputs: {missing}")
    prepared = features.prepare_features(frame)
    return model.predict_proba(prepared[config.MODEL_FEATURES])[:, 1]


def rank_households(scored: pd.DataFrame, top_share: float) -> pd.DataFrame:
    """Sort by probability and flag the share of households to contact."""
    if not 0 < top_share <= 1:
        raise ValueError("top_share must be between 0 and 1")
    ranked = scored.sort_values(
        ["redemption_probability", "household_key"], ascending=[False, True]
    ).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["contact"] = ranked["rank"] <= int(round(len(ranked) * top_share))
    return ranked


def score_planned_campaign(
    start_day: int, campaign_type: str, top_share: float = 0.2,
    model_source: str = "file", raw_dir: Path | None = None,
) -> pd.DataFrame:
    table = features.build_scoring_table(start_day, campaign_type, raw_dir)
    table["redemption_probability"] = score(load_model(model_source), table)
    columns = ["household_key", "DESCRIPTION", "START_DAY", *config.NUMERIC_FEATURES, "redemption_probability"]
    return rank_households(table[columns], top_share)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score every household for a planned campaign.")
    parser.add_argument("--start-day", type=int, required=True, help="Planned first day of the campaign.")
    parser.add_argument("--campaign-type", choices=config.CAMPAIGN_TYPES, required=True)
    parser.add_argument("--top-share", type=float, default=0.2, help="Share of households to contact.")
    parser.add_argument("--model-source", choices=["file", "registry"], default="file")
    args = parser.parse_args()

    ranked = score_planned_campaign(args.start_day, args.campaign_type, args.top_share, args.model_source)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.REPORTS_DIR / f"scores_day{args.start_day}_{args.campaign_type}.csv"
    ranked.to_csv(out, index=False)

    contacted = ranked.loc[ranked["contact"]]
    print(f"Households scored: {len(ranked):,}")
    print(f"Households to contact (top {args.top_share:.0%}): {len(contacted):,}")
    print(f"Mean predicted redemption, contacted: {contacted['redemption_probability'].mean():.1%}")
    print(f"Mean predicted redemption, everyone: {ranked['redemption_probability'].mean():.1%}")
    print(f"Saved: {out.relative_to(config.OUTPUT_ROOT)}")


if __name__ == "__main__":
    main()

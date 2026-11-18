"""Scoring service for the redemption model.

    uvicorn src.api:app --reload        # then open http://127.0.0.1:8000/docs

``POST /predict`` takes the six purchase-history features and the campaign type for one or more
households and returns each household's redemption probability. The features must describe the
90 days BEFORE the campaign starts; ``src/features.py`` shows how they are built.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from src import config, predict

MAX_HOUSEHOLDS_PER_REQUEST = 5000
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once, when the service starts."""
    try:
        state["model"] = predict.load_model("file")
        state["metadata"] = (
            json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
            if config.MODEL_METADATA_PATH.is_file() else {}
        )
    except FileNotFoundError:
        state["model"] = None
        state["metadata"] = {}
    yield
    state.clear()


app = FastAPI(
    title="Coupon redemption scoring service",
    description="Probability that a targeted household redeems at least one coupon during a campaign.",
    version="1.0.0",
    lifespan=lifespan,
)


class HouseholdFeatures(BaseModel):
    """Purchase history for the 90 days before the campaign starts."""

    household_key: int | None = Field(None, description="Your identifier for the household. Returned unchanged.")
    campaign_type: Literal["TypeA", "TypeB", "TypeC"]
    baskets_90d: int = Field(..., ge=0, description="Distinct shopping baskets.")
    retailer_sales_90d: float = Field(..., ge=0, description="Total recorded sales.")
    active_days_90d: int = Field(..., ge=0, le=config.HISTORY_WINDOW_DAYS, description="Distinct days with a purchase.")
    recency_days_90d: float | None = Field(
        None, ge=1, le=config.HISTORY_WINDOW_DAYS,
        description="Days from the last purchase to the campaign start. Leave empty if there were no purchases.",
    )

    @model_validator(mode="after")
    def history_is_consistent(self) -> HouseholdFeatures:
        if self.baskets_90d == 0:
            if self.retailer_sales_90d > 0 or self.active_days_90d > 0 or self.recency_days_90d is not None:
                raise ValueError("With no baskets, sales and active days must be 0 and recency must be empty.")
        else:
            if self.recency_days_90d is None:
                raise ValueError("recency_days_90d is required when the household has baskets.")
            if self.active_days_90d == 0 or self.active_days_90d > self.baskets_90d:
                raise ValueError("active_days_90d must be between 1 and baskets_90d when the household has baskets.")
        return self

    def to_model_row(self) -> dict:
        has_history = self.baskets_90d > 0
        return {
            "baskets_90d": self.baskets_90d,
            "retailer_sales_90d": self.retailer_sales_90d,
            "recency_days_90d": self.recency_days_90d,
            "has_transactions_90d": int(has_history),
            "avg_retailer_sales_per_basket_90d": self.retailer_sales_90d / self.baskets_90d if has_history else None,
            "active_days_90d": self.active_days_90d,
            "DESCRIPTION": self.campaign_type,
        }


class PredictRequest(BaseModel):
    households: list[HouseholdFeatures] = Field(..., min_length=1, max_length=MAX_HOUSEHOLDS_PER_REQUEST)


class Prediction(BaseModel):
    household_key: int | None
    redemption_probability: float


class PredictResponse(BaseModel):
    model: str
    trained_at_utc: str | None
    predictions: list[Prediction]


@app.get("/health")
def health() -> dict:
    metadata = state.get("metadata", {})
    return {
        "status": "ok" if state.get("model") is not None else "model_not_loaded",
        "model": metadata.get("selected_model"),
        "trained_at_utc": metadata.get("trained_at_utc"),
        "registered_version": metadata.get("mlflow_registered_version"),
        "data_fingerprint": metadata.get("data_fingerprint"),
    }


@app.post("/predict", response_model=PredictResponse)
def predict_redemption(request: PredictRequest) -> PredictResponse:
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run `python -m src.pipeline` first.")
    frame = pd.DataFrame([household.to_model_row() for household in request.households])
    probabilities = predict.score(model, frame)
    metadata = state.get("metadata", {})
    return PredictResponse(
        model=metadata.get("selected_model", config.SELECTED_MODEL),
        trained_at_utc=metadata.get("trained_at_utc"),
        predictions=[
            Prediction(household_key=household.household_key, redemption_probability=round(float(probability), 6))
            for household, probability in zip(request.households, probabilities, strict=True)
        ],
    )

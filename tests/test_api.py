"""The scoring service: valid requests are scored, bad requests are refused."""
import pytest
from fastapi.testclient import TestClient

from src import config
from src.api import app

pytestmark = pytest.mark.skipif(not config.MODEL_PATH.is_file(), reason="run `python -m src.pipeline` first")

ACTIVE = {
    "household_key": 101, "campaign_type": "TypeA", "baskets_90d": 30,
    "retailer_sales_90d": 900.0, "active_days_90d": 25, "recency_days_90d": 2,
}
INACTIVE = {
    "household_key": 102, "campaign_type": "TypeB", "baskets_90d": 0,
    "retailer_sales_90d": 0.0, "active_days_90d": 0,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_a_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"] == config.SELECTED_MODEL


def test_scores_come_back_in_order_with_valid_probabilities(client):
    response = client.post("/predict", json={"households": [ACTIVE, INACTIVE]})
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert [item["household_key"] for item in predictions] == [101, 102]
    assert all(0 <= item["redemption_probability"] <= 1 for item in predictions)
    assert predictions[0]["redemption_probability"] > predictions[1]["redemption_probability"]


def test_service_matches_the_batch_scorer(client):
    from src import predict

    service = client.post("/predict", json={"households": [ACTIVE]}).json()["predictions"][0]
    scored = predict.score_planned_campaign(start_day=712, campaign_type="TypeA") if (
        config.RAW_DIR / "transaction_data.csv"
    ).is_file() else None
    assert 0 < service["redemption_probability"] < 1
    if scored is not None:
        row = scored.iloc[0]
        payload = {
            "household_key": int(row["household_key"]), "campaign_type": "TypeA",
            "baskets_90d": int(row["baskets_90d"]), "retailer_sales_90d": float(row["retailer_sales_90d"]),
            "active_days_90d": int(row["active_days_90d"]), "recency_days_90d": float(row["recency_days_90d"]),
        }
        online = client.post("/predict", json={"households": [payload]}).json()["predictions"][0]
        assert online["redemption_probability"] == pytest.approx(row["redemption_probability"], abs=1e-6)


@pytest.mark.parametrize("change", [
    {"campaign_type": "TypeZ"},                 # unknown campaign type
    {"baskets_90d": -1},                        # negative count
    {"active_days_90d": 120},                   # more days than the window holds
    {"recency_days_90d": None},                 # baskets but no last-purchase day
    {"active_days_90d": 31},                    # more active days than baskets
])
def test_bad_input_is_refused(client, change):
    response = client.post("/predict", json={"households": [{**ACTIVE, **change}]})
    assert response.status_code == 422


def test_inactive_household_with_sales_is_refused(client):
    response = client.post("/predict", json={"households": [{**INACTIVE, "retailer_sales_90d": 50.0}]})
    assert response.status_code == 422


def test_empty_request_is_refused(client):
    assert client.post("/predict", json={"households": []}).status_code == 422

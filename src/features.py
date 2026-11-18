"""Build the modeling table: one row per household-campaign pair.

Every feature uses only transactions from the 90 days BEFORE the campaign starts
(``DAY`` from ``START_DAY - 90`` to ``START_DAY - 1``). The label uses redemptions recorded
during the campaign. Nothing from the campaign period can reach a feature.

The same feature query serves training (pairs that were really targeted) and scoring
(any list of households with a planned start day), so the two cannot drift apart.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src import config


def connect() -> duckdb.DuckDBPyConnection:
    """An in-memory DuckDB database."""
    return duckdb.connect()


def load_source_tables(con: duckdb.DuckDBPyConnection, raw_dir: Path | None = None) -> None:
    """Load the four raw files the redemption model needs into DuckDB tables."""
    raw_dir = Path(raw_dir or config.RAW_DIR)
    required = ["transaction_data.csv", "campaign_desc.csv", "campaign_table.csv", "coupon_redempt.csv"]
    missing = [name for name in required if not (raw_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing raw files in {raw_dir}: {missing}. Download dunnhumby's 'The Complete Journey' "
            "and place the CSV files in data/raw/ (see README)."
        )
    con.execute(f"""
        CREATE OR REPLACE TABLE transactions AS
        SELECT household_key, BASKET_ID, DAY, SALES_VALUE
        FROM read_csv('{(raw_dir / "transaction_data.csv").as_posix()}')
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE campaigns AS
        SELECT CAMPAIGN, DESCRIPTION, START_DAY, END_DAY
        FROM read_csv('{(raw_dir / "campaign_desc.csv").as_posix()}')
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE campaign_households AS
        SELECT household_key, CAMPAIGN
        FROM read_csv('{(raw_dir / "campaign_table.csv").as_posix()}')
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE redemptions AS
        SELECT household_key, CAMPAIGN, DAY
        FROM read_csv('{(raw_dir / "coupon_redempt.csv").as_posix()}')
    """)


def register_source_tables(
    con: duckdb.DuckDBPyConnection,
    transactions: pd.DataFrame,
    campaigns: pd.DataFrame,
    campaign_households: pd.DataFrame,
    redemptions: pd.DataFrame,
) -> None:
    """Use pandas frames as the source tables. Tests use this with small hand-made data."""
    con.register("transactions_frame", transactions)
    con.register("campaigns_frame", campaigns)
    con.register("campaign_households_frame", campaign_households)
    con.register("redemptions_frame", redemptions)
    columns = {
        "transactions": "household_key, BASKET_ID, DAY, SALES_VALUE",
        "campaigns": "CAMPAIGN, DESCRIPTION, START_DAY, END_DAY",
        "campaign_households": "household_key, CAMPAIGN",
        "redemptions": "household_key, CAMPAIGN, DAY",
    }
    for table_name, column_list in columns.items():
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT {column_list} FROM {table_name}_frame")


def create_training_pairs(
    con: duckdb.DuckDBPyConnection, outcome_cutoff_day: int = config.OUTCOME_CUTOFF_DAY
) -> None:
    """Create ``pairs``: targeted household-campaign pairs with the redemption label."""
    con.execute(
        """
        CREATE OR REPLACE TABLE pairs AS
        SELECT
            h.household_key,
            h.CAMPAIGN,
            c.DESCRIPTION,
            c.START_DAY,
            c.END_DAY,
            CASE WHEN EXISTS (
                SELECT 1
                FROM redemptions AS r
                WHERE r.household_key = h.household_key
                  AND r.CAMPAIGN = h.CAMPAIGN
                  AND r.DAY BETWEEN c.START_DAY AND c.END_DAY
            ) THEN 1 ELSE 0 END AS redeemed
        FROM campaign_households AS h
        JOIN campaigns AS c USING (CAMPAIGN)
        WHERE c.END_DAY <= ?
        """,
        [outcome_cutoff_day],
    )


def create_scoring_pairs(con: duckdb.DuckDBPyConnection, start_day: int, campaign_type: str) -> None:
    """Create ``pairs`` for a planned campaign: every household seen before ``start_day``."""
    if campaign_type not in config.CAMPAIGN_TYPES:
        raise ValueError(f"campaign_type must be one of {config.CAMPAIGN_TYPES}")
    con.execute(
        """
        CREATE OR REPLACE TABLE pairs AS
        SELECT DISTINCT
            household_key,
            0 AS CAMPAIGN,
            ? AS DESCRIPTION,
            CAST(? AS BIGINT) AS START_DAY,
            CAST(NULL AS BIGINT) AS END_DAY
        FROM transactions
        WHERE DAY < ?
        """,
        [campaign_type, start_day, start_day],
    )


def add_history_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Join ``pairs`` to the 90 days of transactions before each start day."""
    window = config.HISTORY_WINDOW_DAYS
    table = con.execute(
        f"""
        WITH history AS (
            SELECT
                p.household_key,
                p.CAMPAIGN,
                COUNT(DISTINCT t.BASKET_ID) AS baskets_90d,
                COALESCE(SUM(t.SALES_VALUE), 0) AS retailer_sales_90d,
                MAX(p.START_DAY) - MAX(t.DAY) AS recency_days_90d,
                COUNT(DISTINCT t.DAY) AS active_days_90d
            FROM pairs AS p
            LEFT JOIN transactions AS t
                ON t.household_key = p.household_key
               AND t.DAY BETWEEN p.START_DAY - {window} AND p.START_DAY - 1
            GROUP BY p.household_key, p.CAMPAIGN
        )
        SELECT
            p.*,
            p.START_DAY - {window} AS history_start_day,
            p.START_DAY - 1 AS history_end_day,
            h.baskets_90d,
            h.retailer_sales_90d,
            h.recency_days_90d,
            CASE WHEN h.baskets_90d > 0 THEN 1 ELSE 0 END AS has_transactions_90d,
            CASE WHEN h.baskets_90d > 0 THEN h.retailer_sales_90d / h.baskets_90d END
                AS avg_retailer_sales_per_basket_90d,
            h.active_days_90d
        FROM pairs AS p
        JOIN history AS h USING (household_key, CAMPAIGN)
        ORDER BY p.START_DAY, p.CAMPAIGN, p.household_key
        """
    ).df()
    return prepare_features(table)


def prepare_features(table: pd.DataFrame) -> pd.DataFrame:
    """Give the model inputs fixed types. Training and serving both pass through here."""
    table = table.copy()
    for column in config.NUMERIC_FEATURES:
        table[column] = pd.to_numeric(table[column], errors="coerce").astype("float64")
    table["DESCRIPTION"] = table["DESCRIPTION"].astype(str)
    return table


def build_training_table(raw_dir: Path | None = None) -> pd.DataFrame:
    """Raw files in, labeled modeling table out."""
    con = connect()
    try:
        load_source_tables(con, raw_dir)
        create_training_pairs(con)
        return add_history_features(con)
    finally:
        con.close()


def build_scoring_table(start_day: int, campaign_type: str, raw_dir: Path | None = None) -> pd.DataFrame:
    """Features for every household as of the day before a planned campaign starts."""
    con = connect()
    try:
        load_source_tables(con, raw_dir)
        create_scoring_pairs(con, start_day, campaign_type)
        return add_history_features(con)
    finally:
        con.close()

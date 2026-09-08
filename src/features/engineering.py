"""
Leakage-safe feature engineering.

Every feature here uses only information available strictly BEFORE the
prediction date (enforced via .shift(1) before any rolling/expanding
window). This was validated in notebooks/ against known values before
being ported here.

NOTE ON MODEL SELECTION: features here were built to feed a candidate
gradient-boosted regressor, which was evaluated against a weekday-aware
baseline and did NOT outperform it (see docs/experiment_log.md). The
baseline (see src/risk/forecast.py) is the model actually used in
production. This module is retained because (a) it documents a
properly-run experiment, and (b) some of its outputs (SKU volatility,
category rolling demand) are reused as evidence fields for the LLM
explanations, not for forecasting itself.
"""

import numpy as np
import pandas as pd


def add_baseline_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the three baseline demand forecasts plus the winning
    weekday-aware baseline, all leakage-safe (shifted before any
    aggregation).
    """
    out = df.sort_values(["sku_id", "date"]).copy()
    out["weekday"] = out["date"].dt.dayofweek

    out["baseline_naive"] = out.groupby("sku_id")["units_sold"].shift(1)

    out["baseline_rolling7"] = out.groupby("sku_id")["units_sold"].transform(
        lambda s: s.shift(1).rolling(window=7, min_periods=3).mean()
    )

    # Winning baseline: expanding mean of demand on the same weekday,
    # using only past occurrences.
    out["baseline_weekday"] = out.groupby(["sku_id", "weekday"])["units_sold"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )

    return out


def add_gbm_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Add the full feature set used in the (rejected) GBM experiment.
    Kept for the documented experiment trail and because
    sku_avg_demand / sku_demand_std / category_rolling_mean_7 are
    reused as LLM evidence fields.
    """
    out = df.copy()

    for lag in (1, 7, 14):
        out[f"lag_{lag}"] = out.groupby("sku_id")["units_sold"].shift(lag)

    out["rolling_mean_7"] = out.groupby("sku_id")["units_sold"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=3).mean()
    )
    out["rolling_mean_14"] = out.groupby("sku_id")["units_sold"].transform(
        lambda s: s.shift(1).rolling(14, min_periods=5).mean()
    )
    out["rolling_std_7"] = out.groupby("sku_id")["units_sold"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=3).std()
    )

    out["weekday_expanding_mean"] = out["baseline_weekday"]

    out["recent_receipt_7"] = out.groupby("sku_id")["units_received"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).sum()
    )
    out["prev_closing_stock"] = out.groupby("sku_id")["closing_stock"].shift(1)

    out["month"] = out["date"].dt.month
    out["day_of_month"] = out["date"].dt.day
    out["category_code"] = out["category"].astype("category").cat.codes
    out["sku_code"] = out["sku_id"].astype("category").cat.codes

    out["sku_avg_demand"] = out.groupby("sku_id")["units_sold"].transform(
        lambda s: s.shift(1).expanding(min_periods=5).mean()
    )
    out["sku_demand_std"] = out.groupby("sku_id")["units_sold"].transform(
        lambda s: s.shift(1).expanding(min_periods=5).std()
    )

    cat_daily = (
        out.groupby(["category", "date"])["units_sold"].sum().reset_index()
        .rename(columns={"units_sold": "category_total"})
        .sort_values(["category", "date"])
    )
    cat_daily["category_rolling_mean_7"] = cat_daily.groupby("category")["category_total"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=3).mean()
    )
    out = out.merge(cat_daily[["category", "date", "category_rolling_mean_7"]], on=["category", "date"], how="left")

    out["recent_deviation"] = out["lag_1"] - out["rolling_mean_7"]

    feature_cols = [
        "lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
        "weekday_expanding_mean", "recent_receipt_7", "prev_closing_stock",
        "weekday", "month", "day_of_month", "category_code", "sku_code", "lead_time_days",
        "sku_avg_demand", "sku_demand_std", "category_rolling_mean_7", "recent_deviation",
    ]
    return out, feature_cols

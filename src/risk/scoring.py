"""
Inventory risk / decision layer.

This is the ANALYTICAL layer: it computes risk tiers and confidence
independently of any LLM. The LLM (src/llm/) only explains what this
module has already decided — it never determines risk itself.

Target formulation (confirmed): forecast demand over each SKU's
lead-time replenishment horizon using the weekday-aware baseline,
compare against current stock, and express the gap as a coverage
ratio. Risk and confidence are deliberately separate: a SKU can be
high risk with high confidence (clean data, genuinely short) or high
risk with low confidence (new SKU, or data-quality issues in its
history).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

RISK_TIER_THRESHOLDS = {
    "high_max": 0.7,      # coverage_ratio < 0.7 -> High
    "medium_max": 1.0,    # 0.7 <= coverage_ratio < 1.0 -> Medium
    "overstock_min": 3.0, # coverage_ratio > 3.0 -> Overstock
}

FLOW_MISMATCH_CONFIDENCE_THRESHOLD = 0.15  # >15% of history with flow inconsistencies -> Medium confidence


def risk_tier(coverage_ratio: float) -> str:
    """
    NOTE: coverage_ratio can be NaN (0 stock / 0 forecasted demand — a SKU
    with no history and no stock) or inf (positive stock / 0 forecasted
    demand — genuinely overstocked relative to zero expected demand).
    NaN must NOT silently fall through to the final 'else' branch below,
    since every comparison against NaN evaluates False in Python, which
    would wrongly label it 'Overstock' and hide a SKU that may actually
    need review. inf is handled correctly by the normal comparisons
    (inf > overstock_min is True) and legitimately means Overstock.
    """
    if pd.isna(coverage_ratio):
        return "Unknown"
    if coverage_ratio < RISK_TIER_THRESHOLDS["high_max"]:
        return "High"
    elif coverage_ratio < RISK_TIER_THRESHOLDS["medium_max"]:
        return "Medium"
    elif coverage_ratio <= RISK_TIER_THRESHOLDS["overstock_min"]:
        return "Low"
    else:
        return "Overstock"


def compute_weekday_avg_demand(est_df: pd.DataFrame) -> pd.DataFrame:
    """Per-SKU, per-weekday average demand using full available history."""
    return (
        est_df.groupby(["sku_id", "weekday"])["units_sold"]
        .mean()
        .reset_index()
        .rename(columns={"units_sold": "weekday_avg_demand"})
    )


def forecast_horizon_demand(sku_id: str, lead_time_days: int, as_of_date: pd.Timestamp,
                              weekday_avg: pd.DataFrame) -> float:
    """Sum expected demand over the next `lead_time_days` calendar days,
    using the per-SKU per-weekday average as the daily estimate."""
    future_dates = pd.date_range(as_of_date + pd.Timedelta(days=1), periods=lead_time_days)
    future_weekdays = future_dates.dayofweek
    sku_wd = weekday_avg[weekday_avg["sku_id"] == sku_id].set_index("weekday")["weekday_avg_demand"]
    fallback = sku_wd.mean()
    return float(sum(sku_wd.get(wd, fallback) for wd in future_weekdays))


def flow_mismatch_rate(df: pd.DataFrame) -> pd.Series:
    """Per-SKU rate of inventory-flow-equation mismatches
    (closing_stock[t] != closing_stock[t-1] + received[t] - sold[t]),
    used purely as a confidence signal, not for risk itself."""
    d = df.sort_values(["sku_id", "date"]).copy()
    d["prev_closing"] = d.groupby("sku_id")["closing_stock"].shift(1)
    d["expected_closing"] = d["prev_closing"] + d["units_received"] - d["units_sold"]
    d["flow_diff"] = d["closing_stock"] - d["expected_closing"]
    comparable = d.dropna(subset=["prev_closing", "units_received", "units_sold", "closing_stock"])
    comparable = comparable.assign(mismatch=comparable["flow_diff"].abs() >= 0.5)
    return comparable.groupby("sku_id")["mismatch"].mean().rename("flow_mismatch_rate")


def confidence_for_established(mismatch_rate: float) -> tuple[str, str]:
    if pd.notna(mismatch_rate) and mismatch_rate > FLOW_MISMATCH_CONFIDENCE_THRESHOLD:
        return "Medium", f"{mismatch_rate*100:.0f}% of history shows inventory-flow inconsistencies"
    return "High", "Clean data history"


def score_established_skus(est_df: pd.DataFrame) -> pd.DataFrame:
    """Full risk scoring for established (non-cold-start) SKUs."""
    today = est_df["date"].max()
    weekday_avg = compute_weekday_avg_demand(est_df)

    latest_state = (
        est_df.sort_values("date").groupby("sku_id").last()[["category", "closing_stock", "lead_time_days"]]
        .reset_index().rename(columns={"closing_stock": "current_stock"})
    )

    rows = []
    for _, row in latest_state.iterrows():
        horizon = int(row["lead_time_days"])
        forecast = forecast_horizon_demand(row["sku_id"], horizon, today, weekday_avg)
        rows.append({"sku_id": row["sku_id"], "horizon_days": horizon, "forecast_demand_horizon": forecast})
    horizon_df = pd.DataFrame(rows)

    risk_df = latest_state.merge(horizon_df, on="sku_id")
    risk_df["shortfall"] = risk_df["forecast_demand_horizon"] - risk_df["current_stock"]
    risk_df["coverage_ratio"] = risk_df["current_stock"] / risk_df["forecast_demand_horizon"]
    risk_df["risk_tier"] = risk_df["coverage_ratio"].apply(risk_tier)

    mismatch = flow_mismatch_rate(est_df).reset_index()
    risk_df = risk_df.merge(mismatch, on="sku_id", how="left")
    risk_df["flow_mismatch_rate"] = risk_df["flow_mismatch_rate"].fillna(0)

    conf = risk_df["flow_mismatch_rate"].apply(confidence_for_established)
    risk_df["confidence"] = conf.apply(lambda t: t[0])
    risk_df["confidence_reason"] = conf.apply(lambda t: t[1])
    risk_df["is_new_sku"] = False

    return risk_df


def score_cold_start_skus(new_sku_df: pd.DataFrame, est_df: pd.DataFrame) -> pd.DataFrame:
    """
    Risk scoring for new/cold-start SKUs. Never uses the SKU's own
    demand history (unreliable: <2 weeks, zero replenishment observed,
    unstable lead_time_days). Falls back to category-average daily
    demand from established SKUs, and hardcodes Low confidence.
    """
    state = (
        new_sku_df.sort_values("date").groupby("sku_id").last()[["category", "closing_stock", "lead_time_days"]]
        .reset_index().rename(columns={"closing_stock": "current_stock", "lead_time_days": "lead_time_days"})
    )
    category_avg = est_df.groupby("category")["units_sold"].mean()
    state["forecast_demand_horizon"] = state.apply(
        lambda r: category_avg.get(r["category"], category_avg.mean()) * r["lead_time_days"], axis=1
    )
    state["coverage_ratio"] = state["current_stock"] / state["forecast_demand_horizon"]
    state["risk_tier"] = state["coverage_ratio"].apply(risk_tier)
    state["confidence"] = "Low"
    state["confidence_reason"] = (
        "Cold-start: <2 weeks history, zero replenishment observed, unstable lead-time data"
    )
    state["is_new_sku"] = True
    state["flow_mismatch_rate"] = np.nan
    return state


def build_master_risk_table(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Top-level entry point: takes the cleaned full dataset (both
    established and new SKUs, with is_new_sku already flagged) and
    returns the unified risk table for all SKUs."""
    est_df = df_clean[~df_clean["is_new_sku"]].copy()
    est_df["weekday"] = est_df["date"].dt.dayofweek
    new_df = df_clean[df_clean["is_new_sku"]].copy()

    established_scored = score_established_skus(est_df)
    cold_start_scored = score_cold_start_skus(new_df, est_df)

    master = pd.concat([established_scored, cold_start_scored], ignore_index=True, sort=False)
    return master

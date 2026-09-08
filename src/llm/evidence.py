"""
Structured evidence construction for LLM-generated explanations.

The LLM never sees raw data and never calculates anything itself. It
receives only the structured dict produced by `build_evidence()` below
and is instructed to explain ONLY what that evidence supports. This
keeps risk determination strictly in the analytical layer
(src/risk/scoring.py) and the LLM strictly downstream, per the brief's
architecture requirement.
"""

import pandas as pd

from src.data.load_clean import stockout_mask


def compute_evidence_supplement(est_df: pd.DataFrame) -> pd.DataFrame:
    """
    Additional per-SKU evidence fields not already in the risk table:
    recent demand trend, volatility (CV), historical stockout count,
    recent receipts, ABC tier. Established SKUs only.
    """
    last_date = est_df["date"].max()
    recent_30 = est_df[est_df["date"] > last_date - pd.Timedelta(days=30)]
    prior_30 = est_df[
        (est_df["date"] <= last_date - pd.Timedelta(days=30))
        & (est_df["date"] > last_date - pd.Timedelta(days=60))
    ]

    recent_avg = recent_30.groupby("sku_id")["units_sold"].mean().rename("recent_30d_avg")
    prior_avg = prior_30.groupby("sku_id")["units_sold"].mean().rename("prior_30d_avg")
    trend = pd.concat([recent_avg, prior_avg], axis=1)
    trend["trend_pct"] = (trend["recent_30d_avg"] - trend["prior_30d_avg"]) / trend["prior_30d_avg"] * 100

    cv_stats = est_df.groupby("sku_id")["units_sold"].agg(["mean", "std"])
    cv_stats["cv"] = cv_stats["std"] / cv_stats["mean"]

    stockout_hist = est_df[stockout_mask(est_df)].groupby("sku_id").size().rename("historical_stockout_days")

    recent_receipts_30 = recent_30.groupby("sku_id")["units_received"].sum().rename("recent_receipts_30d")

    sku_total_demand = est_df.groupby("sku_id")["units_sold"].sum().sort_values(ascending=False)
    cum_pct = sku_total_demand.cumsum() / sku_total_demand.sum() * 100
    abc_tier = cum_pct.apply(lambda p: "A" if p <= 80 else ("B" if p <= 95 else "C")).rename("abc_tier")

    supplement = (
        pd.concat([trend[["trend_pct"]], cv_stats[["cv"]], stockout_hist, recent_receipts_30, abc_tier], axis=1)
        .reset_index().rename(columns={"index": "sku_id"})
    )
    supplement["historical_stockout_days"] = supplement["historical_stockout_days"].fillna(0)
    supplement["recent_receipts_30d"] = supplement["recent_receipts_30d"].fillna(0)
    return supplement


def build_evidence(sku_row: pd.Series) -> dict:
    """
    Build the structured evidence object for one SKU. This dict is the
    ONLY input the LLM receives for generating an explanation — it
    must not calculate or invent anything beyond what's here.
    """
    # Cast every numeric value to native Python types — pandas/numpy scalars
    # (np.float64 etc.) stringify ugly (e.g. "np.float64(0.0)") if the
    # evidence dict is ever displayed directly or logged, so we normalize
    # here at the one place all evidence is built.
    def _f(x, nd=0):
        return None if pd.isna(x) else round(float(x), nd)

    ev = {
        "sku_id": sku_row["sku_id"],
        "category": sku_row["category"],
        "risk_tier": sku_row["risk_tier"],
        "confidence": sku_row["confidence"],
        "current_stock_units": _f(sku_row["current_stock"]),
        "lead_time_days": int(sku_row["lead_time_days"]),
        "forecasted_demand_over_lead_time": _f(sku_row["forecast_demand_horizon"]),
        "shortfall_or_surplus_units": _f(sku_row["forecast_demand_horizon"] - sku_row["current_stock"]),
        "coverage_ratio": _f(sku_row["coverage_ratio"], nd=2),
        "is_new_sku": bool(sku_row["is_new_sku"]),
        "confidence_reason": sku_row["confidence_reason"],
    }
    if sku_row["is_new_sku"]:
        ev["note"] = "New SKU: demand forecast uses category-average as a proxy, not this SKU's own history."
    else:
        ev["demand_trend_last_30d_pct"] = _f(sku_row.get("trend_pct"), nd=1)
        ev["demand_volatility_cv"] = _f(sku_row.get("cv"), nd=2)
        ev["historical_stockout_days_last_6mo"] = int(sku_row.get("historical_stockout_days", 0) or 0)
        ev["recent_receipts_last_30d"] = int(sku_row.get("recent_receipts_30d", 0) or 0)
        ev["abc_tier"] = sku_row.get("abc_tier")
    return ev

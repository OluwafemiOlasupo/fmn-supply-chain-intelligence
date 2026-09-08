import numpy as np
import pandas as pd
import pytest

from src.risk.scoring import risk_tier, score_cold_start_skus, score_established_skus


def test_risk_tier_boundaries():
    assert risk_tier(0.0) == "High"
    assert risk_tier(0.69) == "High"
    assert risk_tier(0.7) == "Medium"
    assert risk_tier(0.99) == "Medium"
    assert risk_tier(1.0) == "Low"
    assert risk_tier(3.0) == "Low"
    assert risk_tier(3.01) == "Overstock"


def test_risk_tier_handles_nan_and_inf_without_silently_mislabeling():
    """
    coverage_ratio can be NaN (0/0) or inf (positive/0) if forecast_demand_horizon
    is ever zero. Every comparison against NaN is False in Python, so an
    unguarded risk_tier() would fall through to the 'else' branch and wrongly
    label a NaN-coverage SKU as 'Overstock' — the most dangerous possible
    mislabel (hides a SKU that might actually need urgent attention).
    This must not happen silently.
    """
    result_nan = risk_tier(float("nan"))
    result_inf = risk_tier(float("inf"))
    assert result_nan == "Unknown", f"NaN coverage_ratio must map to 'Unknown', got {result_nan!r}"
    assert result_inf == "Overstock", f"Infinite coverage (real signal: stock but zero demand) should be Overstock, got {result_inf!r}"


def _make_est_df(sku_id="SKU-TEST", n_days=40, base_demand=100, category="Snacks", lead_time=7):
    dates = pd.date_range("2026-01-01", periods=n_days)
    return pd.DataFrame({
        "date": dates,
        "sku_id": sku_id,
        "category": category,
        "units_sold": [base_demand] * n_days,
        "units_received": [0] * n_days,
        "closing_stock": [500] * n_days,
        "lead_time_days": lead_time,
        "weekday": dates.dayofweek,
    })


def test_score_established_skus_normal_case_runs_without_error():
    est = _make_est_df()
    result = score_established_skus(est)
    assert len(result) == 1
    assert result.iloc[0]["risk_tier"] in {"High", "Medium", "Low", "Overstock"}


def test_score_established_skus_handles_zero_demand_sku_without_crashing():
    """
    Edge case: a SKU that legitimately sold zero units for its whole history
    (forecast_demand_horizon = 0) must not crash the pipeline with a
    ZeroDivisionError / produce inf silently mislabeled as Overstock.
    """
    est = _make_est_df(base_demand=0)
    result = score_established_skus(est)
    assert len(result) == 1
    # current_stock=500, forecast=0 -> coverage_ratio = inf -> should resolve to Overstock (real signal: has stock, no demand)
    assert result.iloc[0]["risk_tier"] == "Overstock"
    assert np.isinf(result.iloc[0]["coverage_ratio"])


def test_score_cold_start_skus_never_uses_own_history():
    new_df = pd.DataFrame({
        "date": pd.date_range("2026-06-18", periods=3),
        "sku_id": "SKU-2000",
        "category": "Snacks",
        "units_sold": [50, 30, 0],
        "units_received": [0, 0, 0],
        "closing_stock": [200, 100, 0],
        "lead_time_days": [7, 5, 10],  # deliberately inconsistent, as observed in real data
    })
    est = _make_est_df(category="Snacks", base_demand=100)
    result = score_cold_start_skus(new_df, est)
    assert result.iloc[0]["confidence"] == "Low"
    # forecast should come from est's category average (100/day), not new_df's own (~27/day avg)
    assert result.iloc[0]["forecast_demand_horizon"] == pytest.approx(100 * 10)  # last-seen lead_time_days = 10

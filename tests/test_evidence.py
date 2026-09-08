import numpy as np
import pandas as pd
import pytest

from src.llm.evidence import build_evidence


def _base_row(**overrides):
    row = {
        "sku_id": "SKU-1000",
        "category": "Snacks",
        "risk_tier": "High",
        "confidence": "High",
        "current_stock": 0.0,
        "lead_time_days": 7,
        "forecast_demand_horizon": 763.0,
        "coverage_ratio": 0.0,
        "is_new_sku": False,
        "confidence_reason": "Clean data history",
        "trend_pct": -4.0,
        "cv": 0.29,
        "historical_stockout_days": 6,
        "recent_receipts_30d": 1824,
        "abc_tier": "C",
    }
    row.update(overrides)
    return pd.Series(row)


def test_build_evidence_returns_native_python_types_not_numpy_scalars():
    """
    Evidence dicts are string-interpolated into LLM prompts. np.float64
    values stringify as 'np.float64(0.0)' instead of '0.0', which is
    confusing if ever displayed and looks unprofessional in logs/prompts.
    """
    row = _base_row(current_stock=np.float64(0.0), coverage_ratio=np.float64(0.0))
    ev = build_evidence(row)
    assert type(ev["current_stock_units"]) is float
    assert type(ev["coverage_ratio"]) is float
    assert type(ev["lead_time_days"]) is int


def test_build_evidence_handles_missing_trend_gracefully():
    """
    A SKU without 60 days of history yet (new-ish but not cold-start,
    or a data gap) will have NaN trend_pct/cv. This must become a clean
    None, not NaN (which is not valid JSON and confuses the LLM prompt).
    """
    row = _base_row(trend_pct=np.nan, cv=np.nan)
    ev = build_evidence(row)
    assert ev["demand_trend_last_30d_pct"] is None
    assert ev["demand_volatility_cv"] is None


def test_build_evidence_new_sku_has_distinct_shape():
    row = _base_row(is_new_sku=True, confidence="Low",
                     confidence_reason="Cold-start: <2 weeks history")
    ev = build_evidence(row)
    assert ev["is_new_sku"] is True
    assert "note" in ev
    # New SKU evidence should NOT include established-SKU-only fields
    assert "abc_tier" not in ev
    assert "demand_trend_last_30d_pct" not in ev

import pandas as pd
import pytest

from src.data.load_clean import clean, stockout_mask


def test_clean_drops_exact_duplicates():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02"]),
        "sku_id": ["SKU-1000", "SKU-1000", "SKU-1000"],
        "category": ["Snacks", "Snacks", "Snacks"],
        "units_sold": [10, 10, 20],
        "units_received": [0, 0, 0],
        "closing_stock": [100, 100, 90],
        "lead_time_days": [7, 7, 7],
    })
    out = clean(df)
    assert len(out) == 2


def test_clean_normalizes_category_casing():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "sku_id": ["SKU-1000", "SKU-1001"],
        "category": ["SNACKS", "snacks"],
        "units_sold": [10, 20],
        "units_received": [0, 0],
        "closing_stock": [100, 90],
        "lead_time_days": [7, 7],
    })
    out = clean(df)
    assert set(out["category"].unique()) == {"Snacks"}


def test_clean_flags_new_skus():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01", "2026-01-01"]),
        "sku_id": ["SKU-1000", "SKU-2000"],
        "category": ["Snacks", "Snacks"],
        "units_sold": [10, 5],
        "units_received": [0, 0],
        "closing_stock": [100, 50],
        "lead_time_days": [7, 7],
    })
    out = clean(df)
    assert out.set_index("sku_id")["is_new_sku"].to_dict() == {"SKU-1000": False, "SKU-2000": True}


def test_stockout_mask_excludes_missing_not_treats_as_zero():
    """The key correction from Step 1: missing closing_stock must NOT count as a stockout."""
    df = pd.DataFrame({"closing_stock": [0, 5, None, 0]})
    mask = stockout_mask(df)
    assert mask.tolist() == [True, False, False, True]
    assert mask.sum() == 2  # not 3 — the missing value must not be counted
